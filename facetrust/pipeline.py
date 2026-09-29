"""
facetrust.pipeline
=====================

Orquestra os módulos de embeddings, qualidade de imagem, verificação,
morphing e watermarking ao longo dos estágios do pipeline biométrico
discutidos no artigo/slides:

    captura -> processamento -> enrollment -> reconhecimento (decisão)

Este módulo não impõe uma estrutura de pastas específica: você descreve o
seu conjunto de dados como uma lista de `Sample`, e o `PipelineEvaluator`
computa os relatórios (tabelas comparáveis às dos slides "Robustness
transformations" e "Joint evaluation matrix").

Convenção sugerida de `condition`:
    "clean"        - imagem original, sem ataque nem defesa
    "watermarked"  - imagem com watermark (SSLWatermarking ou RoSTEALS)
    "morphed"      - imagem resultante de um ataque de morphing

Convenção sugerida de `stage` (estágio do pipeline em que a amostra entra):
    "capture", "processing", "enrollment", "recognition"
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import __version__
from .embeddings import (
    FaceEmbedder,
    image_fingerprint,
    load_embeddings_file,
    save_embeddings_file,
)
from .image_quality import image_quality_report
from .verification import (
    ScorePairs,
    build_score_pairs,
    cross_condition_scores,
    eer,
    tar_at_far,
    embedding_drift,
)
from .morphing import MorphScores, mmpmr_at_far
from .transforms import DEFAULT_TRANSFORMS


@dataclass
class Sample:
    """Uma amostra de imagem em algum ponto do pipeline.

    Para `condition="morphed"`, NÃO existe um único "sujeito": a imagem
    resulta de dois ou mais `contributors`. Por isso, para morphs:
      - `subject_id` é apenas um rótulo de conveniência (sugestão:
        `"+".join(contributors)`, ex. "alice+bob") -- NÃO é usado para
        achar uma referência "clean" única, porque essa referência não
        existe para uma morph.
      - `morph_algorithm` identifica QUAL algoritmo gerou a morph
        (ex. "stylegan2", "stylegan3", "diffae", "opencv"). Isso é o que
        permite comparar vários algoritmos entre si no mesmo par de
        contribuidores -- `image_quality_matrix()` e
        `morph_attack_report(group_by_algorithm=True)` agrupam por este
        campo.
    """

    sample_id: str
    image: np.ndarray
    subject_id: str
    condition: str = "clean"  # clean | watermarked | morphed | ...
    stage: str = "capture"  # capture | processing | enrollment | recognition
    contributors: Optional[Sequence[str]] = None  # apenas para condition="morphed"
    morph_algorithm: Optional[str] = None  # apenas para condition="morphed"
    alpha: Optional[float] = None  # força do watermark, se aplicável
    # sample_id da imagem "clean" da qual esta amostra foi derivada (ex. a
    # versão com watermark de "clean__001__03"). Quando informado, as
    # métricas de referência (PSNR/SSIM, embedding drift) comparam contra
    # ESSA imagem. Sem ele, cai no comportamento antigo: uma "clean" por
    # `subject_id` (a última carregada), o que só é correto se cada
    # sujeito tem UMA imagem clean.
    reference_id: Optional[str] = None


@dataclass
class EvaluationDataset:
    """Coleção de amostras + embeddings já computados (cache).

    Os embeddings podem ser salvos em disco (`save_embeddings`) e carregados
    depois (`load_embeddings`), para não recalculá-los -- útil quando o
    extrator é lento (ArcFace) e os embeddings são distribuídos junto com
    os dados (ex. `data/embeddings.npz` dentro do data.zip do tutorial).
    """

    samples: Dict[str, Sample] = field(default_factory=dict)
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    # Extrator que gerou os embeddings (ver FaceEmbedder.describe()); vem do
    # cache carregado ou do embedder usado em compute_embeddings().
    embedder_description: Optional[str] = None

    def add(self, sample: Sample) -> None:
        self.samples[sample.sample_id] = sample

    def by_condition(self, condition: str) -> List[Sample]:
        return [s for s in self.samples.values() if s.condition == condition]

    # -- Cache de embeddings em disco ---------------------------------------

    def save_embeddings(self, path: Union[str, Path]) -> None:
        """Salva os embeddings atuais em `path` (.npz), junto com o hash do
        conteúdo de cada imagem e a descrição do extrator."""
        fingerprints = {
            sid: image_fingerprint(self.samples[sid].image)
            for sid in self.embeddings
            if sid in self.samples
        }
        embeddings = {sid: self.embeddings[sid] for sid in fingerprints}
        save_embeddings_file(
            path,
            embeddings,
            fingerprints,
            meta={"embedder": self.embedder_description, "facetrust_version": __version__},
        )

    def load_embeddings(self, path: Union[str, Path]) -> Dict[str, int]:
        """Carrega embeddings salvos por `save_embeddings`. Só aceita o
        embedding de uma amostra se a imagem dela for a MESMA de quando foi
        calculado (hash do conteúdo); embeddings de imagens que mudaram são
        descartados ("stale"), e os de amostras que não existem mais neste
        dataset são ignorados ("extra").

        Retorna {"loaded": n, "stale": n, "extra": n}. Amostras sem
        embedding no cache continuam sem embedding -- use
        `compute_embeddings` para calculá-las."""
        cached, fingerprints, meta = load_embeddings_file(path)
        loaded = stale = extra = 0
        for sid, vec in cached.items():
            sample = self.samples.get(sid)
            if sample is None:
                extra += 1
            elif fingerprints.get(sid) != image_fingerprint(sample.image):
                stale += 1
            else:
                self.embeddings[sid] = vec
                loaded += 1
        if loaded and meta.get("embedder"):
            self.embedder_description = meta["embedder"]
        return {"loaded": loaded, "stale": stale, "extra": extra}

    def compute_embeddings(
        self,
        embedder: Optional[FaceEmbedder] = None,
        cache_path: Optional[Union[str, Path]] = None,
        recompute: bool = False,
        checkpoint_every: int = 25,
        verbose: bool = True,
    ) -> Dict[str, int]:
        """Garante que toda amostra tenha embedding.

        - Com `cache_path` existente (e `recompute=False`), carrega o cache
          primeiro e só calcula o que faltar (amostras novas ou cujas imagens
          mudaram).
        - Com `cache_path`, salva o cache atualizado ao final -- e a cada
          `checkpoint_every` amostras calculadas, para não perder horas de
          trabalho se algo falhar no meio.
        - `embedder=None` só funciona se o cache cobrir TODAS as amostras
          (caso do Colab: os embeddings vêm prontos no data.zip e nem é
          preciso instalar/carregar o ArcFace).

        Retorna {"loaded": n, "stale": n, "extra": n, "computed": n}.
        """
        summary = {"loaded": 0, "stale": 0, "extra": 0, "computed": 0}
        cache_exists = cache_path is not None and Path(cache_path).exists()
        if recompute:
            self.embeddings.clear()
        elif cache_exists:
            summary.update(self.load_embeddings(cache_path))
            if verbose:
                print(
                    f"Cache de embeddings: {summary['loaded']} carregados, "
                    f"{summary['stale']} defasados (imagem mudou), "
                    f"{summary['extra']} de amostras que não estão neste dataset."
                )

        missing = [sid for sid in self.samples if sid not in self.embeddings]
        if not missing:
            return summary
        if embedder is None:
            raise ValueError(
                f"{len(missing)} amostras sem embedding (ex.: {missing[0]!r}) e nenhum "
                f"embedder foi passado. Passe um `embedder`, ou gere o cache completo "
                f"(`compute_embeddings(embedder, cache_path=...)`) na máquina que tem o modelo."
            )

        description = embedder.describe()
        if summary["loaded"] and self.embedder_description not in (None, description):
            warnings.warn(
                f"O cache foi gerado por '{self.embedder_description}', mas as {len(missing)} "
                f"amostras faltantes serão calculadas com '{description}' -- os embeddings "
                f"ficarão misturados e as métricas não serão comparáveis. Use "
                f"recompute=True para recalcular tudo com o mesmo extrator.",
                stacklevel=2,
            )
        self.embedder_description = description

        for k, sid in enumerate(missing, start=1):
            self.embeddings[sid] = embedder.embed(self.samples[sid].image)
            summary["computed"] += 1
            if verbose and (k % 10 == 0 or k == len(missing)):
                print(f"  embeddings calculados: {k}/{len(missing)}")
            if cache_path is not None and k % checkpoint_every == 0 and k < len(missing):
                self.save_embeddings(cache_path)
        if cache_path is not None:
            self.save_embeddings(cache_path)
            if verbose:
                print(f"Embeddings salvos em {cache_path}")
        return summary

    def labels_for(self, sample_ids: Sequence[str]) -> Dict[str, str]:
        return {sid: self.samples[sid].subject_id for sid in sample_ids}

    def embeddings_for(self, sample_ids: Sequence[str]) -> Dict[str, np.ndarray]:
        return {sid: self.embeddings[sid] for sid in sample_ids}


class PipelineEvaluator:
    """Camada de conveniência que produz os relatórios/tabelas do tutorial
    a partir de um `EvaluationDataset` já com embeddings calculados."""

    def __init__(self, dataset: EvaluationDataset):
        self.dataset = dataset

    # -- Qualidade de imagem ------------------------------------------------

    def _find_clean_reference(self, sample: Sample, clean_by_subject: Dict[str, Sample]) -> Optional[Sample]:
        """Imagem "clean" de referência de `sample`: a indicada por
        `sample.reference_id` (se existir como "clean" no dataset); senão,
        a "clean" do mesmo `subject_id` (comportamento antigo)."""
        if sample.reference_id is not None:
            reference = self.dataset.samples.get(sample.reference_id)
            if reference is not None and reference.condition == "clean":
                return reference
        return clean_by_subject.get(sample.subject_id)

    def image_quality_matrix(self, include_lpips: bool = False) -> pd.DataFrame:
        """Compara cada amostra não-limpa à(s) sua(s) referência(s) "clean".

        - "watermarked" (e condições análogas de sujeito único): 1 linha,
          comparada à imagem "clean" de origem (`Sample.reference_id`) ou,
          se não informada, à "clean" do MESMO `subject_id`.
        - "morphed": 1 linha POR CONTRIBUIDOR (via `sample.contributors`),
          já que uma morph não tem um único sujeito de referência. Cada
          linha traz `contributor_id` e `morph_algorithm`, permitindo
          `df.groupby(['morph_algorithm'])[['psnr','ssim']].mean()` para
          comparar vários algoritmos de morphing entre si.
        """
        clean_by_subject = {
            s.subject_id: s for s in self.dataset.by_condition("clean")
        }
        rows = []
        for sample in self.dataset.samples.values():
            if sample.condition == "clean":
                continue

            if sample.condition == "morphed":
                if not sample.contributors:
                    continue
                for contributor_id in sample.contributors:
                    reference = clean_by_subject.get(contributor_id)
                    if reference is None:
                        continue
                    report = image_quality_report(
                        reference.image, sample.image, include_lpips=include_lpips
                    )
                    report.update(
                        {
                            "sample_id": sample.sample_id,
                            "subject_id": sample.subject_id,
                            "contributor_id": contributor_id,
                            "morph_algorithm": sample.morph_algorithm,
                            "condition": sample.condition,
                            "stage": sample.stage,
                            "alpha": sample.alpha,
                        }
                    )
                    rows.append(report)
                continue

            reference = self._find_clean_reference(sample, clean_by_subject)
            if reference is None:
                continue
            report = image_quality_report(reference.image, sample.image, include_lpips=include_lpips)
            report.update(
                {
                    "sample_id": sample.sample_id,
                    "subject_id": sample.subject_id,
                    "contributor_id": None,
                    "morph_algorithm": None,
                    "condition": sample.condition,
                    "stage": sample.stage,
                    "alpha": sample.alpha,
                }
            )
            rows.append(report)
        return pd.DataFrame(rows)

    # -- Verificação biométrica ---------------------------------------------

    def verification_report(self, condition: str, target_fars: Sequence[float] = (1e-1, 1e-2, 1e-3)) -> Dict:
        """EER, ROC/DET (via ScorePairs) e TAR@FAR para uma condição
        específica (ex.: "clean" ou "watermarked")."""
        sample_ids = [s.sample_id for s in self.dataset.by_condition(condition)]
        embeddings = self.dataset.embeddings_for(sample_ids)
        labels = self.dataset.labels_for(sample_ids)
        pairs = build_score_pairs(embeddings, labels)
        eer_value, eer_threshold = eer(pairs.genuine, pairs.impostor)
        report = {
            "condition": condition,
            "n_samples": len(sample_ids),
            "eer": eer_value,
            "eer_threshold": eer_threshold,
            "scores": pairs,
        }
        for far in target_fars:
            tar, thr = tar_at_far(pairs.genuine, pairs.impostor, far)
            report[f"tar@far={far:g}"] = tar
        return report

    def four_scenario_report(self) -> Dict[str, ScorePairs]:
        """Implementa os "Four core comparison scenarios": Clean<->Clean,
        Clean<->Watermarked, Watermarked<->Clean, Watermarked<->Watermarked."""
        clean_ids = [s.sample_id for s in self.dataset.by_condition("clean")]
        wm_ids = [s.sample_id for s in self.dataset.by_condition("watermarked")]
        clean_emb = self.dataset.embeddings_for(clean_ids)
        wm_emb = self.dataset.embeddings_for(wm_ids)
        clean_lbl = self.dataset.labels_for(clean_ids)
        wm_lbl = self.dataset.labels_for(wm_ids)

        return {
            "clean-clean": build_score_pairs(clean_emb, clean_lbl),
            "clean-watermarked": cross_condition_scores(clean_emb, wm_emb, clean_lbl, wm_lbl),
            "watermarked-clean": cross_condition_scores(wm_emb, clean_emb, wm_lbl, clean_lbl),
            "watermarked-watermarked": build_score_pairs(wm_emb, wm_lbl),
        }

    def embedding_drift_report(self, condition: str = "watermarked") -> pd.DataFrame:
        """Embedding drift entre cada amostra da condição dada e a sua
        imagem "clean" de origem (`Sample.reference_id`; sem ele, a "clean"
        do mesmo sujeito), por nível de alpha (se houver)."""
        clean_by_subject = {s.subject_id: s for s in self.dataset.by_condition("clean")}
        rows = []
        for sample in self.dataset.by_condition(condition):
            reference = self._find_clean_reference(sample, clean_by_subject)
            if reference is None:
                continue
            drift = embedding_drift(
                self.dataset.embeddings[reference.sample_id],
                self.dataset.embeddings[sample.sample_id],
            )
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "subject_id": sample.subject_id,
                    "alpha": sample.alpha,
                    "embedding_drift": drift,
                }
            )
        return pd.DataFrame(rows)

    # -- Potencial de ataque de morphing -------------------------------------

    def _build_morph_scores(self) -> Tuple[MorphScores, Dict[str, Optional[str]]]:
        """Constrói os scores morph-vs-contribuidor (para MMPMR/FMMPMR/RMMR)
        e, junto, um mapa sample_id -> morph_algorithm (pode ser None se não
        informado). Fatorado para ser reutilizado tanto pelo relatório
        agregado quanto pela quebra por algoritmo."""
        clean_by_subject = {s.subject_id: s for s in self.dataset.by_condition("clean")}

        morph_scores: MorphScores = {}
        algorithm_by_sample: Dict[str, Optional[str]] = {}
        for sample in self.dataset.by_condition("morphed"):
            if not sample.contributors:
                continue
            per_contributor: Dict[str, List[float]] = {}
            for contributor_id in sample.contributors:
                ref = clean_by_subject.get(contributor_id)
                if ref is None:
                    continue
                score = float(
                    np.dot(
                        self.dataset.embeddings[sample.sample_id],
                        self.dataset.embeddings[ref.sample_id],
                    )
                    / (
                        np.linalg.norm(self.dataset.embeddings[sample.sample_id])
                        * np.linalg.norm(self.dataset.embeddings[ref.sample_id])
                    )
                )
                per_contributor[contributor_id] = [score]
            if per_contributor:
                morph_scores[sample.sample_id] = per_contributor
                algorithm_by_sample[sample.sample_id] = sample.morph_algorithm
        return morph_scores, algorithm_by_sample

    def morph_attack_report(
        self,
        target_fars: Sequence[float] = (1e-1, 1e-2, 1e-3, 1e-4),
        group_by_algorithm: bool = False,
    ) -> pd.DataFrame:
        """Constrói `MorphScores` a partir das amostras "morphed" (usando
        `contributors` de cada Sample) e reporta MMPMR/FMMPMR/ProdAvg-MMPMR/
        RMMR na varredura de FARs, com o threshold calibrado na distribuição
        impostor "clean" do sistema.

        RMMR precisa de comparações genuine (mesmo sujeito, capturas
        diferentes) para medir FRR -- ou seja, exige PELO MENOS 2 amostras
        "clean" por identidade no dataset. Se seu dataset tem só 1 imagem
        "clean" por sujeito, a coluna "rmmr" volta como NaN (com um aviso
        em `warnings`) em vez de quebrar o cálculo; mmpmr/fmmpmr/
        prodavg_mmpmr continuam calculados normalmente, pois não dependem
        disso.

        `group_by_algorithm=True` reporta uma linha por
        `(morph_algorithm, target_far)` em vez de agregar todas as morphs
        juntas -- use isto para comparar vários algoritmos de morphing
        (ex. "stylegan2" vs. "stylegan3" vs. "diffae") entre si, desde que
        `Sample.morph_algorithm` tenha sido preenchido ao montar o dataset.
        Amostras sem `morph_algorithm` definido caem no grupo "(unknown)".
        """
        clean_ids = [s.sample_id for s in self.dataset.by_condition("clean")]
        clean_emb = self.dataset.embeddings_for(clean_ids)
        clean_lbl = self.dataset.labels_for(clean_ids)
        bona_fide_pairs = build_score_pairs(clean_emb, clean_lbl)

        morph_scores, algorithm_by_sample = self._build_morph_scores()

        if not group_by_algorithm:
            rows = [
                mmpmr_at_far(bona_fide_pairs.genuine, bona_fide_pairs.impostor, morph_scores, far, variant="all")
                for far in target_fars
            ]
            return pd.DataFrame(rows)

        algorithms = sorted({alg or "(unknown)" for alg in algorithm_by_sample.values()})
        rows = []
        for algorithm in algorithms:
            subset = {
                sid: scores
                for sid, scores in morph_scores.items()
                if (algorithm_by_sample.get(sid) or "(unknown)") == algorithm
            }
            for far in target_fars:
                row = mmpmr_at_far(bona_fide_pairs.genuine, bona_fide_pairs.impostor, subset, far, variant="all")
                row["morph_algorithm"] = algorithm
                row["n_morphs"] = len(subset)
                rows.append(row)
        return pd.DataFrame(rows)

    # -- Robustez a transformações --------------------------------------------

    def robustness_table(
        self,
        embedder: FaceEmbedder,
        condition: str = "watermarked",
        transforms: Optional[Dict[str, Callable[[np.ndarray], np.ndarray]]] = None,
        watermark_detector: Optional[Callable[[np.ndarray], Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """Recria a tabela "Robustness transformations: evaluate both
        systems": para cada transformação, mede degradação de qualidade de
        imagem, embedding drift e (se `watermark_detector` for fornecido)
        detecção/BER do watermark, MÉDIA sobre todas as amostras de
        `condition`.

        `watermark_detector(image) -> {"detection_rate": .., "ber": ..}`
        deve encapsular a chamada de extração do método usado (ver
        `facetrust.watermark`).
        """
        transforms = transforms or DEFAULT_TRANSFORMS
        rows = []
        for name, transform_fn in transforms.items():
            quality_vals, drift_vals, wm_rows = [], [], []
            for sample in self.dataset.by_condition(condition):
                transformed = transform_fn(sample.image)
                quality_vals.append(image_quality_report(sample.image, transformed))
                emb_before = self.dataset.embeddings[sample.sample_id]
                emb_after = embedder.embed(transformed)
                drift_vals.append(embedding_drift(emb_before, emb_after))
                if watermark_detector is not None:
                    wm_rows.append(watermark_detector(transformed))

            row = {
                "transform": name,
                "psnr_mean": float(np.mean([q["psnr"] for q in quality_vals])),
                "ssim_mean": float(np.mean([q["ssim"] for q in quality_vals])),
                "embedding_drift_mean": float(np.mean(drift_vals)),
            }
            if wm_rows:
                for key in wm_rows[0]:
                    row[f"wm_{key}_mean"] = float(np.mean([r[key] for r in wm_rows]))
            rows.append(row)
        return pd.DataFrame(rows).set_index("transform")

    # -- Relatório consolidado -------------------------------------------------

    def joint_evaluation_summary(self) -> pd.DataFrame:
        """Monta a "Joint evaluation matrix" dos slides como uma tabela
        Dimensão -> Métricas calculadas neste dataset (quando disponíveis)."""
        summary = {
            "Qualidade perceptual": "ver image_quality_matrix() [PSNR, SSIM, LPIPS]",
            "Confiabilidade do watermark": "ver watermark_metrics / robustness_table() [Detection rate, BER]",
            "Robustez": "ver robustness_table() [sobrevivência a JPEG/resize/blur/crop/rotação]",
            "Representação": "ver embedding_drift_report() [1 - cos(f(I), f(I_w))]",
            "Verificação": "ver verification_report() / four_scenario_report() [EER, TAR@FAR]",
            "Ameaça de morphing": "ver morph_attack_report() [MMPMR / FMMPMR / ProdAvg-MMPMR / RMMR; group_by_algorithm=True para comparar algoritmos]",
            "Confiança (trust)": "provenance/integridade -- fora do escopo numérico deste pacote",
        }
        return pd.DataFrame.from_dict(summary, orient="index", columns=["Onde calcular"])
