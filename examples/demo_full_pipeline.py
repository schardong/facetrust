"""
Demo completo do tutorial: carrega imagens já prontas (limpas, com
watermark e morphed), computa TODAS as métricas discutidas no artigo e
exporta as tabelas/figuras usadas nos slides.

Estrutura de pastas esperada (ajuste `load_dataset` se a sua for diferente):

    data/
      clean/<subject_id>/<sample_id>.png
      watermarked/<subject_id>/<sample_id>__alpha<alpha>.png
      morph/<algorithm>/<contributor_a>+<contributor_b>.png
      morph/<algorithm>/<contributor_a>+<contributor_b>__<variante>.png

`<algorithm>` é o nome do método de morphing usado (ex. "stylegan2",
"stylegan3", "diffae", "opencv") -- ele vira `Sample.morph_algorithm`, o
que permite comparar vários algoritmos entre si no mesmo par de
contribuidores via `image_quality_matrix()` (agrupando por
`morph_algorithm`) e `morph_attack_report(group_by_algorithm=True)`.
`<variante>` é opcional, para quando você tem mais de uma morph do MESMO
par pelo MESMO algoritmo (ex. seeds diferentes) -- serve só para manter o
nome de arquivo único, não é interpretado pelo pipeline.

Exemplo:
    data/clean/alice/cap01.png
    data/clean/alice/cap02.png
    data/watermarked/alice/cap01__alpha0.5.png
    data/morph/stylegan2/alice+bob.png
    data/morph/stylegan3/alice+bob.png
    data/morph/diffae/alice+bob.png
    data/morph/opencv/alice+bob__seed2.png

Rode com:
    python examples/demo_full_pipeline.py --data-dir data --out-dir outputs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from facetrust.embeddings import ArcFaceEmbedder, CallableEmbedder, FaceEmbedder
from facetrust.pipeline import Sample, EvaluationDataset, PipelineEvaluator
from facetrust.plotting import (
    plot_score_distributions,
    plot_roc,
    plot_det,
    plot_alpha_sweep,
    plot_map_curve,
)
from facetrust.verification import roc_curve_data, det_curve_data


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def get_embedder() -> FaceEmbedder:
    try:
        return ArcFaceEmbedder(det_size=(256, 256))
    except ImportError:
        print(
            "[aviso] insightface indisponível -- usando embedder de exemplo. "
            "Substitua por ArcFace (ou outro extrator real) para resultados "
            "com significado biométrico."
        )
        return CallableEmbedder(lambda img: img.reshape(-1, img.shape[-1]).mean(axis=0))


def load_dataset(data_dir: Path) -> EvaluationDataset:
    ds = EvaluationDataset()

    clean_dir = data_dir / "clean"
    for subject_dir in sorted(p for p in clean_dir.iterdir() if p.is_dir()):
        subject_id = subject_dir.name
        for img_path in sorted(subject_dir.glob("*.*")):
            sample_id = f"clean__{subject_id}__{img_path.stem}"
            ds.add(
                Sample(
                    sample_id=sample_id,
                    image=load_image(img_path),
                    subject_id=subject_id,
                    condition="clean",
                    stage="capture",
                )
            )

    wm_dir = data_dir / "watermarked"
    if wm_dir.exists():
        for subject_dir in sorted(p for p in wm_dir.iterdir() if p.is_dir()):
            subject_id = subject_dir.name
            for img_path in sorted(subject_dir.glob("*.*")):
                alpha = None
                if "__alpha" in img_path.stem:
                    try:
                        alpha = float(img_path.stem.split("__alpha")[-1])
                    except ValueError:
                        alpha = None
                # "03__alpha42" -> imagem clean de origem "clean/<subjid>/03.*".
                # Sem isto, PSNR/SSIM/drift comparariam com outra imagem clean
                # do mesmo sujeito (ex. outra expressão).
                base_stem = img_path.stem.split("__alpha")[0]
                sample_id = f"wm__{subject_id}__{img_path.stem}"
                ds.add(
                    Sample(
                        sample_id=sample_id,
                        image=load_image(img_path),
                        subject_id=subject_id,
                        condition="watermarked",
                        stage="processing",
                        alpha=alpha,
                        reference_id=f"clean__{subject_id}__{base_stem}",
                    )
                )

    morph_dir = data_dir / "morph"
    if morph_dir.exists():
        # Convenção: morph/<algorithm>/<contrib_a>+<contrib_b>[__variante].ext
        # O algoritmo é a subpasta, não o nome do arquivo -- isso evita
        # ambiguidade quando o nome do algoritmo tem "_" (ex. "diff_ae") e
        # deixa explícito que o algoritmo é um EIXO DE COMPARAÇÃO
        # (Sample.morph_algorithm), não a identidade da amostra.
        for algorithm_dir in sorted(p for p in morph_dir.iterdir() if p.is_dir()):
            algorithm = algorithm_dir.name
            for img_path in sorted(algorithm_dir.glob("*.*")):
                stem = img_path.stem
                pair_str, _, _variant = stem.partition(
                    "__"
                )  # variante é só para nome único
                contributors = pair_str.split("+") if pair_str else []
                ds.add(
                    Sample(
                        sample_id=f"morph__{algorithm}__{stem}",
                        image=load_image(img_path),
                        subject_id="+".join(contributors) if contributors else stem,
                        condition="morphed",
                        stage="enrollment",
                        contributors=contributors,
                        morph_algorithm=algorithm,
                    )
                )

    return ds


def main(
    data_dir: str,
    out_dir: str,
    embeddings_path: str = None,
    recompute_embeddings: bool = False,
):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = get_embedder()
    dataset = load_dataset(data_dir)
    print(f"Carregadas {len(dataset.samples)} amostras.")
    # Cache de embeddings: por padrão <data-dir>/embeddings.npz. Se existir, é
    # carregado e só o que faltar (ou cuja imagem mudou) é calculado; ao final
    # o cache é (re)salvo. Distribua esse arquivo junto com o data.zip.
    cache_path = Path(embeddings_path) if embeddings_path else data_dir / "embeddings.npz"
    dataset.compute_embeddings(
        embedder, cache_path=cache_path, recompute=recompute_embeddings
    )
    if dataset.embedder_description and dataset.embedder_description != embedder.describe():
        print(
            f"[aviso] embeddings vindos de '{dataset.embedder_description}', mas o embedder "
            f"ativo é '{embedder.describe()}': a tabela de robustez (seção 6) recalcula "
            f"embeddings com o embedder ativo, então os números não serão comparáveis."
        )

    evaluator = PipelineEvaluator(dataset)

    # 1) Qualidade de imagem: clean vs. watermarked / morphed
    iq = evaluator.image_quality_matrix()
    iq.to_csv(out_dir / "image_quality.csv", index=False)
    print("\n[image_quality]\n", iq)

    # 2) Verificação: clean vs. watermarked, isoladamente
    for condition in ["clean", "watermarked"]:
        if not dataset.by_condition(condition):
            continue
        try:
            report = evaluator.verification_report(condition)
        except ValueError as exc:
            # Tipicamente: só 1 imagem por identidade nesta condição, então
            # não existe nenhum par genuine (EER/TAR@FAR não têm o que
            # medir). Pula esta condição em vez de derrubar o script
            # inteiro -- as outras seções (qualidade de imagem, morphing)
            # não dependem disto.
            print(f"\n[verification:{condition}] pulado -- {exc}")
            continue
        pairs = report.pop("scores")
        print(f"\n[verification:{condition}]", {k: v for k, v in report.items()})

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        plot_score_distributions(pairs.genuine, pairs.impostor, ax=axes[0])
        fpr, tpr, _ = roc_curve_data(pairs.genuine, pairs.impostor)
        plot_roc(fpr, tpr, ax=axes[1], label=condition)
        fpr, frr, _ = det_curve_data(pairs.genuine, pairs.impostor)
        plot_det(fpr, frr, ax=axes[2], label=condition)
        axes[0].set_title(f"Distribuições ({condition})")
        axes[1].set_title("ROC")
        axes[2].set_title("DET")
        fig.tight_layout()
        fig.savefig(out_dir / f"verification_{condition}.png", dpi=150)
        plt.close(fig)

    # 3) Quatro cenários de comparação
    if dataset.by_condition("watermarked"):
        scenarios = evaluator.four_scenario_report()
        with open(out_dir / "four_scenarios.txt", "w") as f:
            for name, pairs in scenarios.items():
                line = (
                    f"{name}: genuine_mean={pairs.genuine.mean() if len(pairs.genuine) else float('nan'):.4f} "
                    f"impostor_mean={pairs.impostor.mean() if len(pairs.impostor) else float('nan'):.4f}\n"
                )
                print("[four_scenarios]", line.strip())
                f.write(line)

    # 4) Embedding drift vs. alpha (se houver watermark com alpha variando)
    if dataset.by_condition("watermarked"):
        drift_df = evaluator.embedding_drift_report("watermarked")
        drift_df.to_csv(out_dir / "embedding_drift.csv", index=False)
        print("\n[embedding_drift]\n", drift_df)
        if drift_df["alpha"].notna().any():
            grouped = (
                drift_df.dropna(subset=["alpha"])
                .groupby("alpha")["embedding_drift"]
                .mean()
            )
            fig, ax = plt.subplots(figsize=(5, 4))
            plot_alpha_sweep(
                grouped.index.values,
                grouped.values,
                ax=ax,
                ylabel="Embedding drift (médio)",
            )
            fig.tight_layout()
            fig.savefig(out_dir / "alpha_vs_drift.png", dpi=150)
            plt.close(fig)

    # 5) Potencial de ataque de morphing
    if dataset.by_condition("morphed"):
        morph_df = evaluator.morph_attack_report()
        morph_df.to_csv(out_dir / "morph_attack_potential.csv", index=False)
        print("\n[morph_attack_potential]\n", morph_df)
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_map_curve(
            morph_df["target_far"],
            morph_df["mmpmr"],
            ax=ax,
            metric_name="MMPMR",
            label="MMPMR",
        )
        plot_map_curve(
            morph_df["target_far"],
            morph_df["fmmpmr"],
            ax=ax,
            metric_name="MMPMR/FMMPMR",
            label="FMMPMR",
        )
        fig.tight_layout()
        fig.savefig(out_dir / "morph_attack_potential.png", dpi=150)
        plt.close(fig)

        # Se houver mais de um algoritmo de morphing no dataset, compare-os
        # diretamente (mesma varredura de FARs, uma curva por algoritmo).
        algorithms = {s.morph_algorithm for s in dataset.by_condition("morphed")}
        if len(algorithms) > 1:
            morph_by_alg = evaluator.morph_attack_report(group_by_algorithm=True)
            morph_by_alg.to_csv(
                out_dir / "morph_attack_potential_by_algorithm.csv", index=False
            )
            print("\n[morph_attack_potential_by_algorithm]\n", morph_by_alg)

            fig, ax = plt.subplots(figsize=(6, 4.5))
            for algorithm, group in morph_by_alg.groupby("morph_algorithm"):
                plot_map_curve(
                    group["target_far"],
                    group["mmpmr"],
                    ax=ax,
                    metric_name="MMPMR",
                    label=algorithm,
                )
            fig.tight_layout()
            fig.savefig(out_dir / "morph_attack_potential_by_algorithm.png", dpi=150)
            plt.close(fig)

            # Qualidade de imagem também comparável por algoritmo.
            iq_morph = iq[iq["condition"] == "morphed"]
            if not iq_morph.empty:
                print(
                    "\n[image_quality_by_algorithm]\n",
                    iq_morph.groupby("morph_algorithm")[["psnr", "ssim"]].mean(),
                )

    # 6) Robustez a transformações (tabela do tutorial)
    if dataset.by_condition("watermarked"):
        robustness_df = evaluator.robustness_table(embedder, condition="watermarked")
        robustness_df.to_csv(out_dir / "robustness_table.csv")
        print("\n[robustness_table]\n", robustness_df)

    # 7) Resumo (joint evaluation matrix)
    summary = evaluator.joint_evaluation_summary()
    summary.to_csv(out_dir / "joint_evaluation_summary.csv")
    print("\n[joint_evaluation_summary]\n", summary)

    print(f"\nRelatórios e figuras salvos em: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument(
        "--embeddings",
        default=None,
        help="Arquivo de cache de embeddings (.npz). Padrão: <data-dir>/embeddings.npz",
    )
    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Ignora o cache e recalcula todos os embeddings.",
    )
    args = parser.parse_args()
    main(args.data_dir, args.out_dir, args.embeddings, args.recompute_embeddings)
