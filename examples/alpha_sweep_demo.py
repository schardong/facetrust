"""
Exemplo mínimo, equivalente ao "starter loop" do slide "Practical activity:
starter loop", mas usando as funções do pacote `facetrust` no lugar dos
placeholders (`arcface`, `embed_spatial_bits`, `psnr`, `cosine_similarity`,
`detect_watermark`) dos slides.

Você já tem, para uma imagem, uma família de versões com watermark em
diferentes níveis de força (alpha) -- por exemplo, geradas pelo
SSLWatermarking ou pelo RoSTEALS. Este script mede, para cada alpha:
    - PSNR / SSIM em relação à imagem limpa
    - embedding drift (1 - similaridade de cosseno)
    - opcionalmente, detecção/BER do watermark (plugue sua própria função)

E produz os dois plots pedidos no slide "Practical activity: plots to
produce": alpha vs. PSNR/SSIM, e alpha vs. embedding drift.

Ajuste as duas funções `load_image` e `get_embedder` para o seu ambiente.
"""

from __future__ import annotations

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from facetrust.embeddings import CallableEmbedder, FaceEmbedder
from facetrust.image_quality import psnr, ssim
from facetrust.verification import embedding_drift
from facetrust.plotting import plot_alpha_sweep


def load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def get_embedder() -> FaceEmbedder:
    """Troque por ArcFaceEmbedder() (requer insightface) ou por qualquer
    outro extrator já disponível no seu ambiente."""
    try:
        from facetrust.embeddings import ArcFaceEmbedder

        return ArcFaceEmbedder()
    except ImportError:
        print(
            "[aviso] insightface não encontrado -- usando um embedder de "
            "exemplo (média por canal) apenas para o pipeline rodar. "
            "Troque por ArcFace ou outro extrator real antes de tirar "
            "conclusões sobre os resultados."
        )
        return CallableEmbedder(lambda img: img.reshape(-1, img.shape[-1]).mean(axis=0))


def main(clean_path: str, watermarked_glob: str):
    """
    `clean_path`: caminho da imagem original.
    `watermarked_glob`: glob apontando para as versões com watermark, cujo
    nome de arquivo contenha o valor de alpha, ex.:
        "data/watermarked/subject01_alpha0.25.png"
        "data/watermarked/subject01_alpha0.5.png"
        ...
    """
    embedder = get_embedder()

    clean_img = load_image(clean_path)
    emb_clean = embedder.embed(clean_img)

    rows = []
    for path in sorted(glob.glob(watermarked_glob)):
        match = re.search(r"alpha([0-9.]+)", os.path.basename(path))
        alpha = float(match.group(1)) if match else float("nan")
        wm_img = load_image(path)
        emb_wm = embedder.embed(wm_img)

        rows.append(
            {
                "alpha": alpha,
                "psnr": psnr(clean_img, wm_img),
                "ssim": ssim(clean_img, wm_img),
                "embedding_drift": embedding_drift(emb_clean, emb_wm),
                # "detection": detect_watermark(wm_img, meta),  # plugue aqui
                # "ber": bit_error_rate(bits_true, bits_decoded),   # plugue aqui
            }
        )

    rows.sort(key=lambda r: r["alpha"])
    alphas = [r["alpha"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_alpha_sweep(alphas, [r["psnr"] for r in rows], ax=axes[0], ylabel="PSNR (dB)", label="PSNR")
    plot_alpha_sweep(alphas, [r["ssim"] for r in rows], ax=axes[0], ylabel="PSNR / SSIM", label="SSIM")
    plot_alpha_sweep(alphas, [r["embedding_drift"] for r in rows], ax=axes[1], ylabel="Embedding drift")
    axes[0].set_title("Plot A: alpha vs. PSNR/SSIM")
    axes[1].set_title("Plot B: alpha vs. embedding drift")
    fig.tight_layout()
    fig.savefig("alpha_sweep.png", dpi=150)
    print("Figura salva em alpha_sweep.png")

    for r in rows:
        print(r)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True, help="Caminho da imagem limpa")
    parser.add_argument(
        "--watermarked-glob",
        required=True,
        help='Glob das imagens com watermark, ex. "data/wm/*_alpha*.png"',
    )
    args = parser.parse_args()
    main(args.clean, args.watermarked_glob)
