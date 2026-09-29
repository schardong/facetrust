"""
facetrust.image_quality
=========================

Métricas de qualidade perceptual (eixo "Visual" dos slides: PSNR / SSIM,
mais LPIPS opcional). Usadas para comparar:
    - imagem limpa vs. imagem com watermark (custo perceptual da marca)
    - imagem limpa vs. imagem morphed (qualidade visual do ataque)
    - imagem antes/depois de uma transformação de robustez (JPEG, blur, etc.)

Importante (ver slide "Visual quality is not biometric utility"): estas
métricas NÃO substituem as métricas biométricas em `verification.py`. Uma
imagem com PSNR alto pode mesmo assim ter alta variação de embedding.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def psnr(image_a: np.ndarray, image_b: np.ndarray, data_range: int = 255) -> float:
    """Peak Signal-to-Noise Ratio entre duas imagens do mesmo tamanho."""
    a, b = _to_uint8(image_a), _to_uint8(image_b)
    return float(peak_signal_noise_ratio(a, b, data_range=data_range))


def ssim(image_a: np.ndarray, image_b: np.ndarray, data_range: int = 255) -> float:
    """Structural Similarity Index entre duas imagens do mesmo tamanho."""
    a, b = _to_uint8(image_a), _to_uint8(image_b)
    channel_axis = -1 if a.ndim == 3 else None
    return float(
        structural_similarity(a, b, data_range=data_range, channel_axis=channel_axis)
    )


_LPIPS_MODEL = None


def lpips_distance(image_a: np.ndarray, image_b: np.ndarray, net: str = "alex") -> float:
    """Distância perceptual LPIPS (opcional; requer `pip install lpips torch`).

    Espera imagens uint8 HWC em [0, 255]; faz a normalização internamente.
    """
    global _LPIPS_MODEL
    try:
        import torch  # type: ignore
        import lpips  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependência opcional
        raise ImportError(
            "lpips_distance requer 'torch' e 'lpips' (pip install torch lpips). "
            "Use apenas psnr()/ssim() se preferir não instalar essas dependências."
        ) from exc

    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net=net)

    def _prep(img: np.ndarray) -> "torch.Tensor":
        img = _to_uint8(img).astype(np.float32) / 127.5 - 1.0  # -> [-1, 1]
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        return tensor

    with torch.no_grad():
        dist = _LPIPS_MODEL(_prep(image_a), _prep(image_b))
    return float(dist.item())


def image_quality_report(
    reference: np.ndarray, other: np.ndarray, include_lpips: bool = False
) -> Dict[str, float]:
    """Relatório único {psnr, ssim, [lpips]} comparando `reference` a `other`.

    `reference` é tipicamente a imagem limpa (pré-ataque/pré-marca) e `other`
    a imagem processada (watermarked, morphed, ou pós-transformação).
    """
    report: Dict[str, float] = {
        "psnr": psnr(reference, other),
        "ssim": ssim(reference, other),
    }
    if include_lpips:
        try:
            report["lpips"] = lpips_distance(reference, other)
        except ImportError:
            report["lpips"] = float("nan")
    return report
