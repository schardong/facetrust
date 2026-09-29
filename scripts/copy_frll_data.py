#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Copia as imagens do FRLL (limpas e morphed) para o diretório de dados
do facetrust.

Variante de `link_frll_data.py` que COPIA os arquivos em vez de criar
links simbólicos -- útil quando você precisa que `data/` seja uma cópia
autocontida (ex. para empacotar num `data.zip` e subir no Colab: links
simbólicos não sobrevivem a um `zip`/upload da forma esperada, viram
arquivos quebrados ou o próprio link como arquivo).

Estrutura gerada (idêntica à de `link_frll_data.py`):
    <datadir>/clean/<subjid>/<nome da imagem sem o prefixo de ID>
    <datadir>/morph/<tipo>/<nome da imagem com separadores trocados por "+">
"""

import argparse
import shutil
from pathlib import Path

DEFAULT_MORPH_TYPES = ["morph_stylegan2", "morph_stylegan3", "morph_diffae"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--clean-src",
        type=Path,
        default=Path("~/Pictures/frll_cropped"),
        help="Diretório com as imagens limpas, uma subpasta por expressão."
        " (padrão: %(default)s)",
    )
    parser.add_argument(
        "--morph-src",
        type=Path,
        default=Path("~/Pictures/morphs_frll_cropped"),
        help="Diretório com as imagens morphed, uma subpasta por tipo."
        " (padrão: %(default)s)",
    )
    parser.add_argument(
        "--datadir",
        type=Path,
        default=Path("data"),
        help="Diretório de dados de saída. (padrão: %(default)s)",
    )
    parser.add_argument(
        "--clean-out",
        type=Path,
        default=None,
        help="Saída das imagens limpas. (padrão: <datadir>/clean)",
    )
    parser.add_argument(
        "--morph-out",
        type=Path,
        default=None,
        help="Saída das imagens morphed. (padrão: <datadir>/morph)",
    )
    parser.add_argument(
        "-t",
        "--morph-types",
        nargs="+",
        default=DEFAULT_MORPH_TYPES,
        help="Subpastas de --morph-src a processar. (padrão: %(default)s)",
    )
    parser.add_argument(
        "--id-len",
        type=int,
        default=3,
        help="Número de caracteres do ID do sujeito no início do nome"
        " da imagem limpa. (padrão: %(default)s)",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Não copia as imagens limpas.",
    )
    parser.add_argument(
        "--skip-morph",
        action="store_true",
        help="Não copia as imagens morphed.",
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

    args.clean_src = args.clean_src.expanduser().resolve()
    args.morph_src = args.morph_src.expanduser().resolve()
    args.datadir = args.datadir.expanduser()
    if args.clean_out is None:
        args.clean_out = args.datadir / "clean"
    if args.morph_out is None:
        args.morph_out = args.datadir / "morph"
    args.clean_out = args.clean_out.expanduser()
    args.morph_out = args.morph_out.expanduser()
    return args


def make_copy(src: Path, dest: Path, force: bool, dry_run: bool) -> bool:
    """Copia `src` -> `dest` (conteúdo + metadados, via `shutil.copy2`).
    Retorna True se o arquivo foi (ou seria) copiado."""
    if dest.exists() or dest.is_symlink():
        if not force:
            return False
        if not dry_run:
            dest.unlink()
    if dry_run:
        print(f"  {src} -> {dest}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return True


def copy_clean(src: Path, outdir: Path, id_len: int, force: bool, dry_run: bool) -> int:
    """Cópias das imagens limpas: <outdir>/<subjid>/<nome sem o ID>."""
    if not src.is_dir():
        print(f'Erro: diretório "{src}" não existe. Pulando imagens limpas.')
        return 0

    count = 0
    for expression_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        print(expression_dir)
        for subjim in sorted(p for p in expression_dir.iterdir() if p.is_file()):
            subjid = subjim.stem[:id_len]
            # Remove o ID e o separador que vem logo após ele (ex. "001_")
            destpath = outdir / subjid / subjim.name[id_len + 1 :]
            count += make_copy(subjim, destpath, force, dry_run)
    return count


def fix_morph_name(name: str) -> str:
    """Troca o separador entre os IDs dos sujeitos ("-" ou "_") por "+"."""
    for sep in ("-", "_"):
        if sep in name:
            return name.replace(sep, "+")
    return name


def copy_morphs(
    src: Path, outdir: Path, morph_types: list, force: bool, dry_run: bool
) -> int:
    """Cópias das imagens morphed: <outdir>/<tipo>/<nome corrigido>."""
    count = 0
    for morphid in morph_types:
        morphdir = src / morphid
        if not morphdir.is_dir():
            print(f'Erro: diretório "{morphdir}" não existe. Pulando.')
            continue

        print(morphdir)
        trimmedmorphid = morphid.split("_")[-1]
        for morphim in sorted(p for p in morphdir.iterdir() if p.is_file()):
            fullid = f"{fix_morph_name(morphim.name)}"
            count += make_copy(
                morphim, outdir / trimmedmorphid / fullid, force, dry_run
            )
    return count


def main():
    args = parse_args()
    if args.dry_run:
        print("Modo dry-run: nenhum arquivo será copiado.")

    if not args.skip_clean:
        n = copy_clean(
            args.clean_src, args.clean_out, args.id_len, args.force, args.dry_run
        )
        print(f"{n} imagens limpas copiadas em {args.clean_out}")

    if not args.skip_morph:
        n = copy_morphs(
            args.morph_src, args.morph_out, args.morph_types, args.force, args.dry_run
        )
        print(f"{n} imagens morphed copiadas em {args.morph_out}")


if __name__ == "__main__":
    main()
