"""
facetrust.verification
========================

Métricas de verificação biométrica (Seção "Biometric Verification Metrics"
do artigo / slides "FAR, FRR, and threshold trade-offs", "ROC, DET, EER,
and operational points", "Embedding drift", "Four core comparison
scenarios").

Fluxo típico:
    1. Extraia embeddings de todas as imagens (ver `facetrust.embeddings`).
    2. Construa pares "genuine" (mesmo sujeito) e "impostor" (sujeitos
       diferentes) com `build_score_pairs` ou `cross_condition_scores`.
    3. Sumarize com `eer`, `tar_at_far`, `roc_curve_data`, `det_curve_data`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_curve as _sk_roc_curve


# ---------------------------------------------------------------------------
# Similaridade e embedding drift
# ---------------------------------------------------------------------------

def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Similaridade de cosseno entre dois embeddings (não precisa estar
    normalizado de antemão)."""
    a = np.asarray(emb_a, dtype=np.float64).reshape(-1)
    b = np.asarray(emb_b, dtype=np.float64).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-10
    return float(np.dot(a, b) / denom)


def embedding_drift(emb_clean: np.ndarray, emb_other: np.ndarray) -> float:
    """d(I, I_w) = 1 - cos(f(I), f(I_w))  (slide "Embedding drift").

    Mede o quanto uma transformação (watermark, morphing, compressão, ...)
    deslocou a representação da face no espaço de embeddings. Útil, mas
    insuficiente sozinho: o efeito operacional depende das distribuições
    genuine/impostor (ver `eer`, `tar_at_far`).
    """
    return 1.0 - cosine_similarity(emb_clean, emb_other)


# ---------------------------------------------------------------------------
# Construção de pares genuine / impostor
# ---------------------------------------------------------------------------

@dataclass
class ScorePairs:
    genuine: np.ndarray
    impostor: np.ndarray


def build_score_pairs(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, str],
    max_impostor_pairs: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> ScorePairs:
    """Gera scores genuine/impostor a partir de um único conjunto de amostras.

    `embeddings`: sample_id -> vetor
    `labels`: sample_id -> identidade (mesma identidade = comparação genuine)

    Todas as combinações de pares (i, j), i != j, são usadas para genuine;
    para impostor, opcionalmente sub-amostra até `max_impostor_pairs` (o
    número de pares impostor cresce quadraticamente e costuma dominar).
    """
    ids = list(embeddings.keys())
    genuine: List[float] = []
    impostor: List[float] = []

    for id_a, id_b in combinations(ids, 2):
        score = cosine_similarity(embeddings[id_a], embeddings[id_b])
        if labels[id_a] == labels[id_b]:
            genuine.append(score)
        else:
            impostor.append(score)

    impostor_arr = np.array(impostor, dtype=np.float64)
    if max_impostor_pairs is not None and len(impostor_arr) > max_impostor_pairs:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(impostor_arr), size=max_impostor_pairs, replace=False)
        impostor_arr = impostor_arr[idx]

    return ScorePairs(genuine=np.array(genuine, dtype=np.float64), impostor=impostor_arr)


def cross_condition_scores(
    reference: Dict[str, np.ndarray],
    probe: Dict[str, np.ndarray],
    reference_labels: Dict[str, str],
    probe_labels: Dict[str, str],
) -> ScorePairs:
    """Scores genuine/impostor comparando dois CONJUNTOS distintos.

    Isto implementa os "Four core comparison scenarios" do tutorial:
    Clean<->Clean, Clean<->Watermarked, Watermarked<->Clean e
    Watermarked<->Watermarked. Basta passar, por exemplo,
    `reference=embeddings_clean` (base já enrolada) e
    `probe=embeddings_watermarked` (novas amostras de verificação).
    """
    genuine: List[float] = []
    impostor: List[float] = []
    for rid, remb in reference.items():
        for pid, pemb in probe.items():
            score = cosine_similarity(remb, pemb)
            if reference_labels[rid] == probe_labels[pid]:
                genuine.append(score)
            else:
                impostor.append(score)
    return ScorePairs(genuine=np.array(genuine), impostor=np.array(impostor))


# ---------------------------------------------------------------------------
# FAR / FRR / ROC / DET / EER / TAR@FAR
# ---------------------------------------------------------------------------

def far_frr(genuine: np.ndarray, impostor: np.ndarray, thresholds: Optional[np.ndarray] = None):
    """FAR(tau) = P(s >= tau | impostor); FRR(tau) = P(s < tau | genuine).

    Retorna (thresholds, far, frr) como arrays alinhados.
    """
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if thresholds is None:
        thresholds = np.linspace(
            min(genuine.min(initial=0.0), impostor.min(initial=0.0)),
            max(genuine.max(initial=1.0), impostor.max(initial=1.0)),
            500,
        )
    far = np.array([(impostor >= t).mean() if len(impostor) else np.nan for t in thresholds])
    frr = np.array([(genuine < t).mean() if len(genuine) else np.nan for t in thresholds])
    return thresholds, far, frr


def eer(genuine: np.ndarray, impostor: np.ndarray) -> Tuple[float, float]:
    """Equal Error Rate: ponto de operação onde FAR == FRR.

    Retorna (eer_value, threshold_no_eer).
    """
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if genuine.size == 0:
        raise ValueError(
            "genuine vazio: EER precisa de comparações genuine (mesmo "
            "sujeito, capturas diferentes). Isso geralmente significa que "
            "esta condição do dataset tem só 1 imagem por identidade -- "
            "adicione pelo menos 2 amostras por sujeito."
        )
    if impostor.size == 0:
        raise ValueError(
            "impostor vazio: EER precisa de comparações impostor "
            "(sujeitos diferentes) -- verifique se há pelo menos 2 "
            "identidades distintas nesta condição do dataset."
        )
    thresholds, far, frr = far_frr(genuine, impostor)
    diff = np.abs(far - frr)
    idx = int(np.nanargmin(diff))
    eer_value = float((far[idx] + frr[idx]) / 2.0)
    return eer_value, float(thresholds[idx])


def tar_at_far(genuine: np.ndarray, impostor: np.ndarray, target_far: float) -> Tuple[float, float]:
    """TAR (= 1 - FRR) no menor threshold que ainda satisfaz FAR <= target_far.

    Retorna (tar, threshold_usado). Comum reportar TAR@FAR=1e-2, 1e-3 etc.
    (regime de baixo falso aceite, ver slide "ROC, DET, EER, and
    operational points").
    """
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if genuine.size == 0:
        raise ValueError(
            "genuine vazio: TAR@FAR precisa de comparações genuine (mesmo "
            "sujeito, capturas diferentes). Isso geralmente significa que "
            "esta condição do dataset tem só 1 imagem por identidade -- "
            "adicione pelo menos 2 amostras por sujeito."
        )
    if impostor.size == 0:
        raise ValueError(
            "impostor vazio: TAR@FAR precisa de comparações impostor "
            "(sujeitos diferentes) -- verifique se há pelo menos 2 "
            "identidades distintas nesta condição do dataset."
        )
    thresholds, far, frr = far_frr(genuine, impostor)
    valid = np.where(far <= target_far)[0]
    if len(valid) == 0:
        return 0.0, float(thresholds[-1])
    # threshold mais permissivo (menor) entre os que satisfazem o alvo,
    # pois FAR decresce e FRR cresce com o threshold.
    idx = valid[0]
    tar = float(1.0 - frr[idx])
    return tar, float(thresholds[idx])


def roc_curve_data(genuine: np.ndarray, impostor: np.ndarray):
    """ROC via sklearn: retorna (fpr, tpr, thresholds)."""
    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    y_score = np.concatenate([genuine, impostor])
    fpr, tpr, thr = _sk_roc_curve(y_true, y_score)
    return fpr, tpr, thr


def det_curve_data(genuine: np.ndarray, impostor: np.ndarray):
    """DET: mesma base do ROC, mas em termos de FAR (=FPR) x FRR (=1-TPR)."""
    fpr, tpr, thr = roc_curve_data(genuine, impostor)
    frr = 1.0 - tpr
    return fpr, frr, thr


def threshold_at_far(impostor: np.ndarray, target_far: float) -> float:
    """Utilitário: encontra o threshold operacional que produz `target_far`
    na distribuição impostor fornecida (tipicamente a distribuição
    bona-fide/limpa do sistema). Reutilizado por `facetrust.morphing` para
    avaliar potencial de ataque no ponto de operação do reconhecedor."""
    impostor = np.asarray(impostor, dtype=np.float64)
    sorted_scores = np.sort(impostor)[::-1]
    n = len(sorted_scores)
    if n == 0:
        raise ValueError("Distribuição impostor vazia.")
    k = max(int(np.ceil(target_far * n)) - 1, 0)
    k = min(k, n - 1)
    return float(sorted_scores[k])
