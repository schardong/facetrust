"""
Testes de sanidade com dados sintéticos (sem depender de modelos externos
de face nem de watermarking). Rode com:

    python -m pytest tests/  (ou apenas `python tests/test_metrics_synthetic.py`)
"""

import numpy as np

from facetrust.verification import (
    cosine_similarity,
    embedding_drift,
    build_score_pairs,
    eer,
    tar_at_far,
    threshold_at_far,
)
from facetrust.morphing import mmpmr, fmmpmr, prodavg_mmpmr, rmmr, mmpmr_at_far
from facetrust.image_quality import psnr, ssim
from facetrust.identification import cmc_curve, rank_n_accuracy


def test_cosine_and_drift_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9
    assert abs(embedding_drift(v, v)) < 1e-9


def test_image_quality_identical_images():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    assert psnr(img, img) == float("inf")
    assert abs(ssim(img, img) - 1.0) < 1e-9


def test_build_score_pairs_and_eer_separated_distributions():
    rng = np.random.default_rng(0)
    embeddings, labels = {}, {}
    # 5 sujeitos, 4 amostras cada, embeddings bem separados por sujeito
    for subj in range(5):
        base = rng.normal(size=8)
        base /= np.linalg.norm(base)
        for k in range(4):
            noisy = base + rng.normal(scale=0.02, size=8)
            sid = f"s{subj}_{k}"
            embeddings[sid] = noisy
            labels[sid] = f"subject_{subj}"

    pairs = build_score_pairs(embeddings, labels)
    assert len(pairs.genuine) > 0 and len(pairs.impostor) > 0
    assert pairs.genuine.mean() > pairs.impostor.mean()

    eer_value, _ = eer(pairs.genuine, pairs.impostor)
    assert 0.0 <= eer_value <= 0.2  # distribuições bem separadas -> EER baixo

    tar, _ = tar_at_far(pairs.genuine, pairs.impostor, target_far=0.1)
    assert 0.0 <= tar <= 1.0


def test_threshold_at_far_matches_expected_quantile():
    impostor = np.linspace(0, 1, 1000)
    tau = threshold_at_far(impostor, target_far=0.01)
    empirical_far = float((impostor >= tau).mean())
    assert abs(empirical_far - 0.01) < 0.005


def test_eer_and_tar_at_far_raise_clear_error_without_genuine():
    """Caso real: condição do dataset com só 1 amostra por identidade ->
    nenhum par genuine existe. Antes disso levantava um ValueError críptico
    do numpy (All-NaN slice); agora deve ser um erro claro e acionável."""
    empty_genuine = np.array([])
    impostor = np.array([0.1, 0.2, 0.3])

    try:
        eer(empty_genuine, impostor)
        assert False, "deveria ter levantado ValueError"
    except ValueError as exc:
        assert "genuine" in str(exc)

    try:
        tar_at_far(empty_genuine, impostor, target_far=0.01)
        assert False, "deveria ter levantado ValueError"
    except ValueError as exc:
        assert "genuine" in str(exc)

    # mesmo teste para impostor vazio
    genuine = np.array([0.8, 0.9])
    empty_impostor = np.array([])
    try:
        eer(genuine, empty_impostor)
        assert False, "deveria ter levantado ValueError"
    except ValueError as exc:
        assert "impostor" in str(exc)


def test_mmpmr_variants_basic_behavior():
    # 1 morph com dois contribuintes, 1 prova cada; scores altos -> "passa"
    scores_high = {"m1": {"a": [0.9], "b": [0.85]}}
    # aqui o contribuinte "b" não tem NENHUMA prova acima do threshold
    scores_low = {"m1": {"a": [0.9], "b": [0.3]}}

    assert mmpmr(scores_high, threshold=0.5) == 1.0
    assert mmpmr(scores_low, threshold=0.5) == 0.0  # "b" nunca é vencido
    assert fmmpmr(scores_high, threshold=0.5) == 1.0
    assert fmmpmr(scores_low, threshold=0.5) == 0.0

    # Com múltiplas provas por contribuinte, MMPMR (>= 1 prova basta) e
    # FMMPMR (todas as provas têm que bater) devem DIVERGIR -- essa é
    # exatamente a distinção que a versão anterior deste teste não cobria.
    scores_mixed = {"m1": {"a": [0.95, 0.40], "b": [0.9, 0.9]}}
    # "a" tem uma prova boa (0.95) e uma ruim (0.40); "b" tem as duas boas.
    assert mmpmr(scores_mixed, threshold=0.5) == 1.0  # "a" passa pela prova 0.95
    assert fmmpmr(scores_mixed, threshold=0.5) == 0.0  # mas a prova 0.40 de "a" reprova


def test_rmmr_equals_mmpmr_plus_frr():
    # RMMR(tau) = MMPMR(tau) + FRR(tau) -- não é proporção de contribuintes.
    genuine = np.array([0.9, 0.8, 0.3, 0.95])  # FRR(0.5) = 1/4 = 0.25
    scores_high = {"m1": {"a": [0.9], "b": [0.85]}}  # MMPMR(0.5) = 1.0

    result = rmmr(genuine, scores_high, threshold=0.5)
    assert abs(result - (1.0 + 0.25)) < 1e-9


def _reference_minmax_mmpmr(mdata, threshold):
    """Reimplementação independente do algoritmo MinMax-MMPMR descrito
    publicamente em Scherhag et al. (2017) e no README de
    https://github.com/dasec/mvr (não é uma cópia do arquivo deles -- a
    licença customizada da Hochschule Darmstadt não foi conferida, então
    este teste reimplementa o ALGORITMO a partir da descrição pública, só
    para validar `facetrust.morphing.mmpmr` de forma independente).
    `mdata`: {morph_id: {contributor_id: [scores...]}}.
    """
    reduced = {
        m: {c: max(vals) for c, vals in contributors.items()}
        for m, contributors in mdata.items()
    }
    return sum(min(c.values()) >= threshold for c in reduced.values()) / len(reduced)


def _reference_prodavg_mmpmr(mdata, threshold):
    """Idem, para ProdAvg-MMPMR."""
    total = 0.0
    for contributors in mdata.values():
        hit_rates = [np.mean([s >= threshold for s in vals]) for vals in contributors.values()]
        prod = 1.0
        for hr in hit_rates:
            prod *= hr
        total += prod
    return total / len(mdata)


def test_mmpmr_and_prodavg_match_independent_reference_reimplementation():
    """Valida `mmpmr()` e `prodavg_mmpmr()` contra uma reimplementação
    independente do algoritmo público de Scherhag et al. -- não a mesma
    função (evita duplicar viés de implementação), múltiplos morphs e
    múltiplas provas por contribuinte para exercitar os dois caminhos."""
    rng = np.random.default_rng(42)
    morph_scores = {
        f"m{i}": {
            "a": rng.uniform(0.3, 1.0, size=rng.integers(1, 4)).tolist(),
            "b": rng.uniform(0.3, 1.0, size=rng.integers(1, 4)).tolist(),
        }
        for i in range(20)
    }
    threshold = 0.7

    assert abs(mmpmr(morph_scores, threshold) - _reference_minmax_mmpmr(morph_scores, threshold)) < 1e-9
    assert abs(
        prodavg_mmpmr(morph_scores, threshold) - _reference_prodavg_mmpmr(morph_scores, threshold)
    ) < 1e-9


def test_mmpmr_at_far_returns_expected_keys():
    rng = np.random.default_rng(1)
    bona_fide_genuine = rng.uniform(0.4, 1.0, size=500)
    bona_fide_impostor = rng.uniform(0, 0.6, size=500)
    morph_scores = {"m1": {"a": [0.95], "b": [0.9]}}
    result = mmpmr_at_far(
        bona_fide_genuine, bona_fide_impostor, morph_scores, target_far=0.01, variant="all"
    )
    assert set(["threshold", "target_far", "mmpmr", "fmmpmr", "prodavg_mmpmr", "rmmr"]).issubset(result.keys())


def test_mmpmr_at_far_degrades_gracefully_without_genuine_scores():
    """Caso real: dataset com só 1 imagem 'clean' por sujeito -> nenhum par
    genuine existe. variant='all' deve continuar retornando mmpmr/fmmpmr/
    prodavg_mmpmr normalmente, com rmmr=NaN (não uma exceção)."""
    import warnings

    rng = np.random.default_rng(2)
    bona_fide_impostor = rng.uniform(0, 0.6, size=200)
    empty_genuine = np.array([])  # simula ausência de pares genuine
    morph_scores = {"m1": {"a": [0.95], "b": [0.9]}}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mmpmr_at_far(empty_genuine, bona_fide_impostor, morph_scores, target_far=0.01, variant="all")
        assert any("RMMR" in str(w.message) for w in caught)

    assert result["mmpmr"] == 1.0  # não deve ser afetado
    assert np.isnan(result["rmmr"])

    # Pedir "rmmr" explicitamente, sem genuine, ainda deve levantar erro --
    # foi pedido especificamente e não há como calculá-lo silenciosamente.
    try:
        mmpmr_at_far(empty_genuine, bona_fide_impostor, morph_scores, target_far=0.01, variant="rmmr")
        assert False, "deveria ter levantado ValueError"
    except ValueError:
        pass


def test_cmc_and_rank_accuracy():
    gallery = {"g1": np.array([1.0, 0.0]), "g2": np.array([0.0, 1.0])}
    gallery_labels = {"g1": "alice", "g2": "bob"}
    probes = {"p1": np.array([0.9, 0.1])}
    probe_labels = {"p1": "alice"}

    curve = cmc_curve(gallery, gallery_labels, probes, probe_labels, max_rank=2)
    assert curve[0] == 1.0  # rank-1 já acerta
    assert rank_n_accuracy(gallery, gallery_labels, probes, probe_labels, n=1) == 1.0


def test_image_quality_matrix_handles_morphs_per_contributor():
    """Regressão: uma morph não tem um único `subject_id` de referência --
    `image_quality_matrix` deve gerar uma linha POR CONTRIBUIDOR, e nunca
    tentar (e falhar silenciosamente) achar uma referência "clean" pelo
    subject_id sintético da morph."""
    from facetrust.pipeline import Sample, EvaluationDataset, PipelineEvaluator

    def make_img(seed, size=(16, 16, 3)):
        r = np.random.default_rng(seed)
        return (r.uniform(0, 255, size=size)).astype(np.uint8)

    ds = EvaluationDataset()
    ds.add(Sample("alice_clean", make_img(1), subject_id="alice", condition="clean"))
    ds.add(Sample("bob_clean", make_img(2), subject_id="bob", condition="clean"))
    ds.add(
        Sample(
            "morph_ab_stylegan2",
            make_img(3),
            subject_id="alice+bob",
            condition="morphed",
            contributors=["alice", "bob"],
            morph_algorithm="stylegan2",
        )
    )
    ds.add(
        Sample(
            "morph_ab_diffae",
            make_img(4),
            subject_id="alice+bob",
            condition="morphed",
            contributors=["alice", "bob"],
            morph_algorithm="diffae",
        )
    )

    evaluator = PipelineEvaluator(ds)
    iq = evaluator.image_quality_matrix()

    # 2 morphs x 2 contribuidores cada = 4 linhas (antes da correção: 0)
    morph_rows = iq[iq["condition"] == "morphed"]
    assert len(morph_rows) == 4
    assert set(morph_rows["contributor_id"]) == {"alice", "bob"}
    assert set(morph_rows["morph_algorithm"]) == {"stylegan2", "diffae"}
    # cada algoritmo aparece em exatamente 2 linhas (uma por contribuidor)
    assert (morph_rows.groupby("morph_algorithm").size() == 2).all()


def test_morph_attack_report_group_by_algorithm():
    from facetrust.embeddings import CallableEmbedder
    from facetrust.pipeline import Sample, EvaluationDataset, PipelineEvaluator

    rng = np.random.default_rng(5)

    def make_img(seed, size=(16, 16, 3)):
        r = np.random.default_rng(seed)
        return (r.uniform(0, 255, size=size)).astype(np.uint8)

    ds = EvaluationDataset()
    subjects = {"alice": 1, "bob": 2, "carol": 3}
    for name, seed in subjects.items():
        for k in range(3):
            ds.add(Sample(f"{name}_clean_{k}", make_img(seed), subject_id=name, condition="clean"))

    ds.add(
        Sample(
            "morph_ab_algoA",
            make_img(1),  # próxima de alice -> deve "passar" contra alice, não contra bob
            subject_id="alice+bob",
            condition="morphed",
            contributors=["alice", "bob"],
            morph_algorithm="algoA",
        )
    )
    ds.add(
        Sample(
            "morph_ab_algoB",
            make_img(99),  # longe de ambos -> não deve passar
            subject_id="alice+bob",
            condition="morphed",
            contributors=["alice", "bob"],
            morph_algorithm="algoB",
        )
    )

    embedder = CallableEmbedder(lambda img: img.flatten()[:32])
    ds.compute_embeddings(embedder)

    evaluator = PipelineEvaluator(ds)
    report = evaluator.morph_attack_report(target_fars=[0.5], group_by_algorithm=True)

    assert set(report["morph_algorithm"]) == {"algoA", "algoB"}
    assert (report["n_morphs"] == 1).all()  # 1 morph por algoritmo neste dataset


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
