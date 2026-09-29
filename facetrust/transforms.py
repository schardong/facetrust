"""
facetrust.transforms
======================

Transformações "benignas" de processamento usadas na tabela "Robustness
transformations: evaluate both systems" dos slides (JPEG, Resize, Blur,
Crop, Rotation). O objetivo é aplicá-las ANTES de recomputar, para a mesma
imagem: (a) detecção/BER do watermark, (b) qualidade de imagem, e
(c) comportamento do reconhecedor (embedding drift / score).

"Screen recapture" (recaptura de tela/impressão) não é simulável de forma
confiável apenas com processamento de imagem; deixe esse caso para
captura física real e registre o resultado manualmente na mesma tabela.
"""

from __future__ import annotations

import io
from typing import Callable, Dict

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, rotate as ndi_rotate


def to_pil(image: np.ndarray) -> Image.Image:
    arr = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def jpeg_compress(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """Recompressão JPEG com qualidade `quality` (1-95)."""
    buf = io.BytesIO()
    to_pil(image).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def resize(image: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Reduz e depois volta ao tamanho original (perda de informação por
    reamostragem), mantendo a resolução comparável entre condições."""
    h, w = image.shape[:2]
    small = to_pil(image).resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    back = small.resize((w, h), Image.BILINEAR)
    return np.array(back)


def gaussian_blur(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    if image.ndim == 3:
        blurred = np.stack(
            [gaussian_filter(image[..., c].astype(np.float64), sigma=sigma) for c in range(image.shape[-1])],
            axis=-1,
        )
    else:
        blurred = gaussian_filter(image.astype(np.float64), sigma=sigma)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def center_crop(image: np.ndarray, crop_fraction: float = 0.9) -> np.ndarray:
    """Corta uma fração central da imagem e reamostra de volta ao tamanho
    original, para manter comparabilidade dimensional entre condições."""
    h, w = image.shape[:2]
    ch, cw = int(h * crop_fraction), int(w * crop_fraction)
    top, left = (h - ch) // 2, (w - cw) // 2
    cropped = image[top : top + ch, left : left + cw]
    return np.array(to_pil(cropped).resize((w, h), Image.BILINEAR))


def rotate(image: np.ndarray, degrees: float = 5.0) -> np.ndarray:
    rotated = ndi_rotate(image, angle=degrees, reshape=False, mode="reflect")
    return np.clip(rotated, 0, 255).astype(np.uint8)


# Conjunto padrão usado na tabela do tutorial. Ajuste os parâmetros
# conforme a severidade que quiser demonstrar aos participantes.
DEFAULT_TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "JPEG": lambda img: jpeg_compress(img, quality=50),
    "Resize": lambda img: resize(img, scale=0.5),
    "Blur": lambda img: gaussian_blur(img, sigma=1.5),
    "Crop": lambda img: center_crop(img, crop_fraction=0.9),
    "Rotation": lambda img: rotate(img, degrees=5.0),
    # "Screen recapture": deixado de fora -- requer captura física real.
}
