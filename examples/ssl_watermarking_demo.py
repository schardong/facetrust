"""
Exemplo: usa o SSLWatermarking de verdade (facebookresearch/ssl_watermarking,
CC-BY-NC) para marcar as imagens "clean" do seu dataset com um watermark
0-bit, decodifica de volta, e alimenta os resultados no `facetrust`
(detection_metrics, PSNR/SSIM, robustness_table).

Pré-requisitos (ver facetrust/adapters/ssl_watermarking.py para detalhes):
    1. git clone https://github.com/facebookresearch/ssl_watermarking.git
    2. pip install -e ".[ssl-watermarking]"
       pip install -r ssl_watermarking/requirements.txt
    3. Baixar os pesos (dino_r50*.pth + normlayer out2048*.pth) -- ver
       README do ssl_watermarking ou o Space no Hugging Face dos autores.

Rode com:
    python examples/ssl_watermarking_demo.py \\
        --ssl-repo /caminho/para/ssl_watermarking \\
        --model-path /caminho/para/dino_r50_plus.pth \\
        --normlayer-path /caminho/para/out2048_yfcc_orig.pth \\
        --clean-dir data/clean \\
        --out-dir data/watermarked
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from facetrust.adapters.ssl_watermarking import (
    load_ssl_watermarking,
    embed_0bit,
    decode_0bit,
)
from facetrust.image_quality import image_quality_report
from facetrust.watermark import detection_metrics


def main(
    ssl_repo: str,
    model_path: str,
    normlayer_path: str,
    clean_dir: str,
    out_dir: str,
    epochs: int,
    target_psnr: float,
    target_fpr: float,
):
    clean_dir = Path(clean_dir)
    out_dir = Path(out_dir)

    # Carrega TODAS as imagens "clean" (qualquer estrutura de subpastas),
    # mantendo o caminho relativo para nomear a saída de forma correspondente.
    image_paths = sorted(clean_dir.rglob("*.png")) + sorted(clean_dir.rglob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"Nenhuma imagem encontrada em {clean_dir}")
    print(f"Carregando {len(image_paths)} imagens de {clean_dir}...")
    images = [Image.open(p).convert("RGB") for p in image_paths]

    print(
        ">>> Carregando backbone + normalization layer (pode demorar a primeira vez)..."
    )
    ctx = load_ssl_watermarking(
        repo_path=ssl_repo,
        model_path=model_path,
        normlayer_path=normlayer_path,
        target_fpr=target_fpr,
    )

    print(
        f">>> Marcando {len(images)} imagens (0-bit, target_psnr={target_psnr}, epochs={epochs})..."
    )
    watermarked = embed_0bit(images, ctx, epochs=epochs, target_psnr=target_psnr)

    # Salva as imagens marcadas espelhando a estrutura de pastas de entrada,
    # já na convenção esperada por facetrust.pipeline (subject_id = nome
    # da subpasta em data/clean/<subject_id>/...).
    print(f">>> Salvando imagens marcadas em {out_dir}...")
    saved_paths = []
    for src_path, wm_img in zip(image_paths, watermarked):
        rel = src_path.relative_to(clean_dir)
        dst_path = out_dir / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        wm_img.save(dst_path)
        saved_paths.append(dst_path)

    print(">>> Decodificando (detecção 0-bit) para conferir...")
    decoded = decode_0bit(watermarked, ctx)

    # Também decodifica as imagens LIMPAS (deveriam vir como "não marcadas" --
    # essa é a checagem de falso-positivo do detector).
    decoded_clean = decode_0bit(images, ctx)

    labels_true = [1] * len(watermarked) + [0] * len(images)
    detected = [int(d["marked"]) for d in decoded] + [
        int(d["marked"]) for d in decoded_clean
    ]
    metrics = detection_metrics(labels_true, detected)
    print("\n[detecção 0-bit]", metrics)

    print("\n[qualidade de imagem por amostra]")
    for src_path, orig_img, wm_img in zip(image_paths, images, watermarked):
        report = image_quality_report(np.array(orig_img), np.array(wm_img))
        print(
            f"  {src_path.relative_to(clean_dir)}: psnr={report['psnr']:.2f} dB, ssim={report['ssim']:.4f}"
        )

    print(f"\nPronto. Imagens marcadas em: {out_dir.resolve()}")
    print(
        "Próximo passo: rode examples/demo_full_pipeline.py apontando "
        "--data-dir para a pasta que contém clean/ e este watermarked/, "
        "para ver PSNR/SSIM/embedding-drift/robustness_table completos."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ssl-repo",
        required=True,
        help="Caminho do repositório ssl_watermarking clonado",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Caminho do backbone (ex. dino_r50_plus.pth)",
    )
    parser.add_argument(
        "--normlayer-path",
        required=True,
        help="Caminho da normalization layer (ex. out2048_yfcc_orig.pth)",
    )
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--out-dir", default="data/watermarked")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Iterações de otimização por imagem"
    )
    parser.add_argument("--target-psnr", type=float, default=42.0)
    parser.add_argument("--target-fpr", type=float, default=1e-6)
    args = parser.parse_args()
    main(
        args.ssl_repo,
        args.model_path,
        args.normlayer_path,
        args.clean_dir,
        args.out_dir,
        args.epochs,
        args.target_psnr,
        args.target_fpr,
    )
