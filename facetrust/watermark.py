"""
facetrust.watermark
=====================

Métricas de confiabilidade de watermark (eixo "Watermark" dos slides:
Detection rate, BER, Payload, Robustness). Este módulo NÃO reimplementa
SSLWatermarking nem RoSTEALS -- ele consome os resultados que esses
métodos já produzem (bits decodificados, ou um score/flag de detecção) e
calcula as métricas de forma padronizada, para que possam ser combinadas
com as métricas biométricas e de qualidade de imagem no mesmo relatório.

Os dois métodos usados no tutorial têm filosofias diferentes:
    - SSLWatermarking (Fernandez et al.) é baseado em DETECÇÃO: o
      extrator devolve um score/estatística que indica se a marca está
      presente (0/1), sem necessariamente decodificar um payload.
    - RoSTEALS (Bui et al.) é baseado em PAYLOAD: o extrator decodifica uma
      sequência de bits que deve ser comparada à mensagem original (BER).

Os adaptadores abaixo dão uma interface comum aos dois; basta preencher as
chamadas às funções `embed`/`extract` de cada repositório.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def bit_error_rate(bits_true: Sequence[int], bits_decoded: Sequence[int]) -> float:
    """BER = fração de bits do payload que foram decodificados incorretamente."""
    a = np.asarray(bits_true, dtype=np.int8)
    b = np.asarray(bits_decoded, dtype=np.int8)
    if a.shape != b.shape:
        raise ValueError(f"Formatos incompatíveis: {a.shape} vs {b.shape}")
    return float(np.mean(a != b))


def detection_metrics(labels_true: Sequence[int], detected: Sequence[int]) -> Dict[str, float]:
    """Métricas de um detector binário (marca presente / ausente):
    taxa de detecção (recall), falso-positivo e acurácia. `labels_true`
    indica se a imagem de fato contém a marca (1) ou é limpa (0);
    `detected` é a saída do detector (1 = detectou marca)."""
    y = np.asarray(labels_true, dtype=int)
    p = np.asarray(detected, dtype=int)
    tp = int(np.sum((y == 1) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    detection_rate = tp / (tp + fn) if (tp + fn) else float("nan")
    false_positive_rate = fp / (fp + tn) if (fp + tn) else float("nan")
    accuracy = (tp + tn) / len(y) if len(y) else float("nan")
    return {
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "accuracy": accuracy,
    }


def robustness_table(results_per_transform: Dict[str, Dict[str, float]]) -> "pandas.DataFrame":  # noqa: F821
    """Monta a tabela "Robustness transformations: evaluate both systems"
    dos slides a partir de um dict:

        {"JPEG": {"detection_rate": .., "ber": .., "psnr": .., "embedding_drift": ..}, ...}

    Retorna um `pandas.DataFrame` (import feito localmente para manter
    `pandas` como dependência opcional deste módulo específico).
    """
    import pandas as pd

    return pd.DataFrame.from_dict(results_per_transform, orient="index")


# ---------------------------------------------------------------------------
# Adaptadores para os dois métodos usados no tutorial
# ---------------------------------------------------------------------------

class WatermarkMethod(abc.ABC):
    """Interface comum: embutir uma marca e extraí-la/detectá-la depois."""

    @abc.abstractmethod
    def embed(self, image: np.ndarray, payload: Optional[Sequence[int]] = None, **kwargs) -> np.ndarray:
        """Retorna a imagem marcada (mesma resolução da entrada)."""
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, image: np.ndarray, **kwargs):
        """Retorna o payload decodificado (SSLWatermarking) OU um
        score/flag de detecção (RoSTEALS-like), dependendo do método."""
        raise NotImplementedError


class SSLWatermarkAdapter(WatermarkMethod):
    """Adaptador fino para o SSLWatermarking (Fernandez et al., 2022).

    Preencha `embed_fn`/`extract_fn` com as funções reais do repositório
    (https://github.com/facebookresearch/ssl_watermarking, CC-BY-NC) já
    carregadas com seus pesos/keys. Isso evita duplicar código-fonte de
    terceiros aqui e mantém este pacote independente da versão instalada.

    Não escreva `embed_fn`/`extract_fn` do zero: use
    `facetrust.adapters.ssl_watermarking.make_single_image_adapter_fns(ctx)`,
    que já chama o `encode.watermark_0bit`/`decode.decode_0bit` reais do
    repositório clonado (ver esse módulo para o passo a passo de setup:
    clonar o repo, baixar os pesos, `load_ssl_watermarking(...)`).
    """

    def __init__(self, embed_fn, extract_fn):
        self._embed_fn = embed_fn
        self._extract_fn = extract_fn

    def embed(self, image: np.ndarray, payload: Optional[Sequence[int]] = None, **kwargs) -> np.ndarray:
        return self._embed_fn(image, **kwargs)

    def extract(self, image: np.ndarray, **kwargs):
        # SSLWatermarking é orientado à detecção: normalmente devolve um
        # score de similaridade contra a chave secreta e uma decisão
        # binária (marca presente/ausente).
        return self._extract_fn(image, **kwargs)


class RoSTEALSAdapter(WatermarkMethod):
    """Adaptador fino para o RoSTEALS (Bui et al., 2023).

    Preencha `embed_fn`/`extract_fn` com as funções reais do repositório
    (https://github.com/TuBui/RoSteALS) já carregadas com seus pesos.
    RoSTEALS decodifica um payload multi-bit, então `extract_fn` deve
    retornar a sequência de bits decodificada, comparável via
    `bit_error_rate`.
    """

    def __init__(self, embed_fn, extract_fn):
        self._embed_fn = embed_fn
        self._extract_fn = extract_fn

    def embed(self, image: np.ndarray, payload: Optional[Sequence[int]] = None, **kwargs) -> np.ndarray:
        return self._embed_fn(image, payload, **kwargs)

    def extract(self, image: np.ndarray, **kwargs) -> Sequence[int]:
        return self._extract_fn(image, **kwargs)
