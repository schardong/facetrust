#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Reorganiza pastas `watermarked_psnr_<XX>/` no layout lido por
`examples/demo_full_pipeline.py`.

Entrada (um diretório por PSNR alvo, espelhando a estrutura de `clean/`):
    <src-root>/watermarked_psnr_27/<subjid>/<nome>.png
    <src-root>/watermarked_psnr_32/<subjid>/<nome>.png
    <src-root>/watermarked_psnr_37/<subjid>/<nome>.png
    <src-root>/watermarked_psnr_42/<subjid>/<nome>.png

Saída (o que `load_dataset` espera; o valor após "__alpha" vira `Sample.alpha`):
    <out>/<subjid>/<nome>__alpha<valor>.png

O valor de alpha (a "força" usada no eixo X do alpha sweep) pode ser:
  --strength psnr (padrão): o próprio PSNR alvo em dB (27, 32, 37, 42).
      Atenção: PSNR MAIOR = watermark MAIS FRACO, ou seja, o eixo "alpha"
      fica invertido em relação à intuição de "força".
  --strength rms: amplitude RMS equivalente da perturbação em níveis de
      cinza (0-255), 255 * 10^(-PSNR/20). Cresce com a força do watermark,
      então o eixo X fica na direção intuitiva (mais forte = maior alpha).

Os arquivos são COPIADOS (não linkados), para que `data/` possa ser zipado.
"""

import argparse
import re
import shutil
from pathlib import Path

PSNR_DIR_RE = re.compile(r"^watermarked_psnr_(\d+(?:\.\d+)?)$")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("data"),
        help="Diretório que contém as pastas watermarked_psnr_XX."
        " (padrão: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Saída no layout do demo. (padrão: <src-root>/watermarked)",
    )
    parser.add_argument(
        "--strength",
        choices=["psnr", "rms"],
        default="psnr",
        help="Valor gravado em __alpha<valor>. (padrão: %(default)s)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Substitui arquivos já existentes no destino.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Apenas mostra o que seria feito, sem copiar nada.",
    )
    args = parser.parse_args()
    args.src_root = args.src_root.expanduser()
    args.out = (args.out or args.src_root / "watermarked").expanduser()
    return args


def format_alpha(psnr: float, strength: str) -> str:
    """Valor (como texto) que vai depois de "__alpha" no nome do arquivo."""
    if strength == "rms":
        return f"{255.0 * 10 ** (-psnr / 20.0):.2f}"
    return f"{psnr:g}"


def find_psnr_dirs(src_root: Path):
    """[(psnr, dir)] ordenado por PSNR crescente."""
    found = []
    for path in src_root.iterdir() if src_root.is_dir() else []:
        m = PSNR_DIR_RE.match(path.name)
        if path.is_dir() and m:
            found.append((float(m.group(1)), path))
    return sorted(found)


def process_dir(psnr, psnr_dir, out, strength, force, dry_run):
    """Retorna (copiados, pulados_existentes, pulados_sem_sujeito)."""
    alpha = format_alpha(psnr, strength)
    copied = skipped_exists = skipped_flat = 0
    for src in sorted(p for p in psnr_dir.rglob("*") if p.is_file()):
        if src.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = src.relative_to(psnr_dir)
        if len(rel.parts) < 2:
            # Sem subpasta de sujeito: não dá para saber o subject_id.
            print(f"  aviso: {src} está solto (sem subpasta <subjid>/); pulando.")
            skipped_flat += 1
            continue
        subjid = rel.parts[0]
        dest = out / subjid / f"{src.stem}__alpha{alpha}{src.suffix}"
        if dest.exists():
            if not force:
                skipped_exists += 1
                continue
            if not dry_run:
                dest.unlink()
        if dry_run:
            print(f"  {src} -> {dest}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        copied += 1
    return copied, skipped_exists, skipped_flat


def main():
    args = parse_args()
    psnr_dirs = find_psnr_dirs(args.src_root)
    if not psnr_dirs:
        raise SystemExit(
            f'Nenhuma pasta "watermarked_psnr_<XX>" encontrada em {args.src_root}.'
        )
    if args.dry_run:
        print("Modo dry-run: nenhum arquivo será copiado.")

    total = 0
    for psnr, psnr_dir in psnr_dirs:
        alpha = format_alpha(psnr, args.strength)
        print(f"{psnr_dir.name}  (PSNR {psnr:g} dB -> __alpha{alpha})")
        copied, exists, flat = process_dir(
            psnr, psnr_dir, args.out, args.strength, args.force, args.dry_run
        )
        print(
            f"  {copied} copiadas, {exists} já existiam (use -f para substituir)"
            + (f", {flat} soltas puladas" if flat else "")
        )
        total += copied
    print(f"Total: {total} imagens em {args.out}")


if __name__ == "__main__":
    main()
