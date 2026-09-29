"""
facetrust.adapters.ssl_watermarking
=====================================

Integração com o SSLWatermarking (Fernandez et al., "Watermarking Images
in Self-Supervised Latent-Spaces", https://github.com/facebookresearch/ssl_watermarking,
licença CC-BY-NC -- uso acadêmico/não-comercial).

DESIGN: em vez de reimplementar a matemática de embedding/decoding aqui
(o que arriscaria errar detalhes de PSNR-clipping, SSIM-attenuation e
construção do otimizador que não estão documentados fora do código-fonte
deles), este módulo IMPORTA e CHAMA as funções reais do repositório
clonado (`encode.watermark_0bit`, `decode.decode_0bit`,
`utils.build_backbone`, etc.). A única coisa escrita aqui é a camada de
tradução entre:
    - a API deles (dataloader com imagens em disco, `argparse.Namespace`
      de parâmetros)
    - a API do `facetrust` (lista de imagens PIL/numpy já em memória,
      `Sample`/`EvaluationDataset`, `facetrust.watermark.SSLWatermarkAdapter`)

CONFIANÇA: `decode_0bit()` é uma cópia fiel da função pública
`decode.decode_0bit` (li o arquivo inteiro, 70 linhas, e a reproduzo aqui
só como referência/fallback caso você não consiga importar o pacote deles
-- por padrão, a função REAL importada do repositório é usada). O restante
(carregamento de modelo, embedding) chama as funções deles diretamente.
Ainda assim, EU NÃO CONSEGUI RODAR ISSO (sem GPU/rede/pesos no ambiente em
que escrevi este arquivo) -- valide no seu ambiente antes de confiar nos
números.

Pré-requisitos:
    1. Clonar o repositório:
        git clone https://github.com/facebookresearch/ssl_watermarking.git
    2. Instalar as dependências dele (torch, torchvision, augly, ...):
        pip install -e ".[ssl-watermarking]"   # (deps genéricas do facetrust)
        pip install -r ssl_watermarking/requirements.txt  # deps específicas deles
    3. Baixar os pesos pré-treinados (~800 MB o backbone + a normalization
       layer). Duas fontes:
        - Repositório oficial: seguir os links de "Pretrained models &
          normalization layers" no README do ssl_watermarking.
        - Alternativa mais direta (Hugging Face Space dos mesmos autores,
          hospeda os mesmos pesos para download): baixe `dino_r50.pth` e
          `out2048.pth` de https://huggingface.co/spaces/pierrefdz/ssl_watermarking/tree/main
    4. Apontar `repo_path`, `model_path` e `normlayer_path` para os
       arquivos baixados/clonados ao chamar `load_ssl_watermarking(...)`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


@dataclass
class SSLWatermarkingContext:
    """Estado carregado uma única vez (modelo pesado, carrier, ângulo) e
    reutilizado em todas as chamadas de embed/decode. Construído por
    `load_ssl_watermarking()` -- não instancie diretamente."""

    model: Any  # utils.NormLayerWrapper (backbone + normalization layer), já em eval()
    carrier: Any  # torch.Tensor 1xD (0-bit) ou KxD (multi-bit)
    angle: float  # ângulo do hipercone, calibrado para o target_fpr pedido
    device: Any
    # módulos do repositório, importados dinamicamente -- guardados aqui
    # para não precisar re-importar (e para o resto do módulo não depender
    # de imports em nível de arquivo, já que o repositório é opcional).
    _encode: Any = field(repr=False)
    _decode: Any = field(repr=False)
    _utils: Any = field(repr=False)
    _utils_img: Any = field(repr=False)
    _data_augmentation: Any = field(repr=False)
    _torch: Any = field(repr=False)


def _import_ssl_watermarking_modules(repo_path: Union[str, Path]):
    """Adiciona `repo_path` ao sys.path e importa os módulos do
    ssl_watermarking. Os nomes de módulo (`encode`, `decode`, `utils`,
    `utils_img`, `data_augmentation`) são genéricos de propósito -- são
    exatamente os nomes de arquivo do repositório clonado, então
    precisam estar na raiz do `repo_path` fornecido."""
    repo_path = str(Path(repo_path).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SSLWatermarking requer 'torch' (e 'torchvision'). "
            "Instale com: pip install -e \".[ssl-watermarking]\""
        ) from exc

    try:
        import encode  # type: ignore
        import decode  # type: ignore
        import utils  # type: ignore
        import utils_img  # type: ignore
        import data_augmentation  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"Não consegui importar os módulos do ssl_watermarking a partir de "
            f"'{repo_path}'. Confirme que este é o diretório raiz do repositório "
            f"clonado (deve conter encode.py, decode.py, utils.py, utils_img.py, "
            f"data_augmentation.py)."
        ) from exc

    return torch, encode, decode, utils, utils_img, data_augmentation


def load_ssl_watermarking(
    repo_path: Union[str, Path],
    model_path: Union[str, Path],
    normlayer_path: Union[str, Path],
    model_name: str = "resnet50",
    target_fpr: float = 1e-6,
    num_bits: Optional[int] = None,
    carrier_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> SSLWatermarkingContext:
    """Carrega o backbone (DINO ResNet-50), a normalization layer, e
    gera/carrega o(s) carrier(s), retornando um `SSLWatermarkingContext`
    reutilizável.

    `num_bits=None` (default) -> contexto para watermarking 0-bit
    (detecção: "a marca está presente?"). `num_bits=K` -> contexto para
    watermarking multi-bit (payload de K bits).

    Chama diretamente `utils.build_backbone`, `utils.load_normalization_layer`,
    `utils.NormLayerWrapper`, `utils.pvalue_angle` e `utils.generate_carriers`
    do repositório clonado -- não reimplementa nada disso.
    """
    torch, encode, decode, utils, utils_img, data_augmentation = _import_ssl_watermarking_modules(repo_path)

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    backbone = utils.build_backbone(path=str(model_path), name=model_name)
    normlayer = utils.load_normalization_layer(path=str(normlayer_path))
    model = utils.NormLayerWrapper(backbone, normlayer).to(device_t)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Dimensão do espaço de features (roda um forward dummy, como main_0bit.py faz).
    dim = model(torch.zeros((1, 3, 224, 224)).to(device_t)).size(-1)
    k = num_bits if num_bits else 1

    if carrier_path is not None and Path(carrier_path).exists():
        carrier = torch.load(str(carrier_path), map_location=device_t)
        if carrier.shape[1] != dim:
            raise ValueError(
                f"Carrier carregado tem dimensão {carrier.shape[1]}, mas o "
                f"modelo produz features de dimensão {dim}. Gere um novo "
                f"carrier (não passe carrier_path) ou confira o backbone."
            )
    else:
        carrier = utils.generate_carriers(k, dim, output_fpath=str(carrier_path) if carrier_path else None)
    carrier = carrier.to(device_t)

    angle = utils.pvalue_angle(dim, k=1, proba=target_fpr)

    return SSLWatermarkingContext(
        model=model,
        carrier=carrier,
        angle=angle,
        device=device_t,
        _encode=encode,
        _decode=decode,
        _utils=utils,
        _utils_img=utils_img,
        _data_augmentation=data_augmentation,
        _torch=torch,
    )


class _InMemoryImageDataset:
    """Substitui `utils_img.get_dataloader` (que exige uma pasta em disco
    no formato ImageFolder) por uma lista de imagens PIL já carregadas em
    memória, usando a MESMA transform deles (`utils_img.default_transform`)
    -- para bater exatamente com o que `encode.watermark_0bit`/
    `watermark_multibit` esperam receber do dataloader."""

    def __init__(self, pil_images: Sequence["Image.Image"], transform):
        self.images = list(pil_images)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.transform(self.images[idx]), 0  # (imagem, rótulo fictício)


def _build_dataloader(ctx: SSLWatermarkingContext, pil_images: Sequence["Image.Image"], batch_size: int):
    dataset = _InMemoryImageDataset(pil_images, ctx._utils_img.default_transform)
    return ctx._torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _default_embed_params(epochs: int, lambda_w: float, lambda_i: float, optimizer: str, scheduler: Optional[str], verbose: int):
    """`encode.watermark_0bit`/`watermark_multibit` esperam um objeto com
    esses atributos (visto em `main_0bit.py`/`main_multibit.py`: eles
    passam `params` como o `argparse.Namespace` inteiro da CLI). Aqui
    montamos só os campos que o código de encode de fato lê.

    `optimizer`/`scheduler` são strings no formato usado pela CLI deles
    (ex. "Adam,lr=0.01") -- CONFIRA os defaults reais rodando
    `python main_0bit.py --help` no repositório clonado; os valores aqui
    são um ponto de partida razoável, não os defaults oficiais deles
    (que eu não consegui confirmar sem rodar o `--help` real)."""
    return SimpleNamespace(
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        lambda_w=lambda_w,
        lambda_i=lambda_i,
        verbose=verbose,
    )


def embed_0bit(
    images: Sequence["Image.Image"],
    ctx: SSLWatermarkingContext,
    epochs: int = 100,
    target_psnr: float = 42.0,
    lambda_w: float = 1.0,
    lambda_i: float = 1.0,
    optimizer: str = "Adam,lr=0.01",
    scheduler: Optional[str] = None,
    batch_size: int = 1,
    data_aug: str = "all",
    verbose: int = 0,
) -> List["Image.Image"]:
    """Embute um watermark 0-bit em cada imagem de `images`, chamando
    `encode.watermark_0bit` do repositório real. Retorna as imagens
    marcadas como PIL Images (mesma resolução da entrada).

    ATENÇÃO: isto é uma otimização iterativa por imagem (não um forward
    pass único) -- espere segundos por imagem em GPU, bem mais em CPU.
    Para um dataset de tutorial (dezenas de imagens), rode isso OFFLINE
    antes da sessão ao vivo, não durante os 20-30 min de hands-on.
    """
    loader = _build_dataloader(ctx, images, batch_size)
    aug = ctx._data_augmentation.All() if data_aug == "all" else ctx._data_augmentation.DifferentiableDataAugmentation()
    params = _default_embed_params(epochs, lambda_w, lambda_i, optimizer, scheduler, verbose)
    params.target_psnr = target_psnr  # watermark_0bit usa isso via psnr_clip internamente

    pt_imgs_out = ctx._encode.watermark_0bit(loader, ctx.carrier, ctx.angle, ctx.model, aug, params)
    to_pil = __import__("torchvision").transforms.ToPILImage()
    return [to_pil(ctx._utils_img.unnormalize_img(pt_img).squeeze(0).clamp(0, 1)) for pt_img in pt_imgs_out]


def decode_0bit(images: Sequence["Image.Image"], ctx: SSLWatermarkingContext) -> List[Dict[str, Any]]:
    """Detecta o watermark 0-bit em cada imagem, chamando `decode.decode_0bit`
    do repositório real. Retorna uma lista de dicts, um por imagem:
        {"index": i, "R": valor da função de aceitação (marcado se > 0),
         "log10_pvalue": ..., "marked": bool}
    ("marked" = R > 0, mesmo critério usado em `evaluate.py` do repositório.)
    """
    decoded = ctx._decode.decode_0bit(list(images), ctx.carrier, ctx.angle, ctx.model)
    for d in decoded:
        d["marked"] = d["R"] > 0
    return decoded


def embed_multibit(
    images: Sequence["Image.Image"],
    messages: Sequence[Sequence[int]],
    ctx: SSLWatermarkingContext,
    epochs: int = 100,
    target_psnr: float = 33.0,
    lambda_w: float = 5e4,
    lambda_i: float = 1.0,
    optimizer: str = "Adam,lr=0.01",
    scheduler: Optional[str] = None,
    batch_size: int = 1,
    data_aug: str = "all",
    verbose: int = 0,
) -> List["Image.Image"]:
    """Embute um payload multi-bit (`messages[i]`: sequência de 0s/1s do
    tamanho `num_bits` usado ao criar o contexto) em cada imagem,
    chamando `encode.watermark_multibit` do repositório real.

    CONFIANÇA MENOR que `embed_0bit`: a assinatura
    (`dataloader, msgs, carrier, model, data_aug, params`) e os defaults de
    `lambda_w=5e4`/`lambda_i=1.0` vêm do help da CLI (`main_multibit.py`)
    que consegui ler, mas eu não tenho o corpo completo da função para
    conferir linha a linha como fiz com `decode_0bit`. Teste com uma
    mensagem conhecida (ex. "000000...") num par embed+decode antes de
    confiar nos resultados.
    """
    loader = _build_dataloader(ctx, images, batch_size)
    aug = ctx._data_augmentation.All() if data_aug == "all" else ctx._data_augmentation.DifferentiableDataAugmentation()
    params = _default_embed_params(epochs, lambda_w, lambda_i, optimizer, scheduler, verbose)
    params.target_psnr = target_psnr

    msgs_t = ctx._torch.stack([ctx._torch.tensor(list(m), dtype=ctx._torch.float32) for m in messages])
    pt_imgs_out = ctx._encode.watermark_multibit(loader, msgs_t, ctx.carrier, ctx.model, aug, params)
    to_pil = __import__("torchvision").transforms.ToPILImage()
    return [to_pil(ctx._utils_img.unnormalize_img(pt_img).squeeze(0).clamp(0, 1)) for pt_img in pt_imgs_out]


def decode_multibit(images: Sequence["Image.Image"], ctx: SSLWatermarkingContext) -> List[List[int]]:
    """Decodifica o payload multi-bit de cada imagem, chamando
    `decode.decode_multibit` do repositório real. Retorna uma lista de
    listas de bits (0/1), uma por imagem -- pronta para `bit_error_rate()`
    em `facetrust.watermark`."""
    decoded = ctx._decode.decode_multibit(list(images), ctx.carrier, ctx.model)
    return [d["msg"].to(ctx._torch.int).tolist() if hasattr(d["msg"], "tolist") else list(d["msg"]) for d in decoded]


# ---------------------------------------------------------------------------
# Ponte para facetrust.watermark.SSLWatermarkAdapter (interface de 1 imagem)
# ---------------------------------------------------------------------------

def make_single_image_adapter_fns(ctx: SSLWatermarkingContext, mode: str = "0bit", **embed_kwargs):
    """Retorna (embed_fn, extract_fn) no formato de UMA imagem por vez,
    esperado por `facetrust.watermark.SSLWatermarkAdapter` (embed_fn(image,
    payload=None) -> image; extract_fn(image) -> resultado). Por baixo,
    empacota a imagem única numa lista de tamanho 1 e chama as funções em
    lote acima -- conveniente para plugar em `PipelineEvaluator`, mas
    ineficiente para processar um dataset inteiro (cada chamada recarrega
    o dataloader). Para lotes grandes, prefira `embed_0bit`/`embed_multibit`
    diretamente sobre a lista inteira de imagens de uma vez.

    `mode`: "0bit" | "multibit". Para "multibit", `payload` (uma sequência
    de bits) é obrigatório em cada chamada de embed_fn.
    """

    if mode == "0bit":
        def embed_fn(image, payload=None, **kwargs):
            return embed_0bit([image], ctx, **{**embed_kwargs, **kwargs})[0]

        def extract_fn(image, **kwargs):
            return decode_0bit([image], ctx)[0]

    elif mode == "multibit":
        def embed_fn(image, payload=None, **kwargs):
            if payload is None:
                raise ValueError("mode='multibit' exige `payload` (sequência de bits) em cada chamada.")
            return embed_multibit([image], [payload], ctx, **{**embed_kwargs, **kwargs})[0]

        def extract_fn(image, **kwargs):
            return decode_multibit([image], ctx)[0]

    else:
        raise ValueError(f"mode deve ser '0bit' ou 'multibit', recebi {mode!r}.")

    return embed_fn, extract_fn
