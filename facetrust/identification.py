"""
facetrust.identification
==========================

Métricas do cenário de IDENTIFICAÇÃO (1:N), complementares às de
verificação (1:1) em `verification.py`. Ver definição no artigo:
"em identificação, um sistema recebe uma amostra única e busca em uma base
de amostras conhecidas o par mais próximo".

Útil para demonstrar o efeito de watermarking/morphing quando a "vítima"
está armazenada em uma galeria (enrollment database) e a amostra de
verificação (probe) é limpa, marcada ou morphed.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .verification import cosine_similarity


def _rank_of_correct_match(
    probe_emb: np.ndarray,
    probe_label: str,
    gallery: Dict[str, np.ndarray],
    gallery_labels: Dict[str, str],
) -> int:
    """Posição (1-indexada) da primeira entrada da galeria com a identidade
    correta, ordenando por similaridade decrescente. Retorna len(gallery)+1
    se a identidade não estiver na galeria."""
    scored = [
        (cosine_similarity(probe_emb, gemb), gallery_labels[gid])
        for gid, gemb in gallery.items()
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    for rank, (_, label) in enumerate(scored, start=1):
        if label == probe_label:
            return rank
    return len(gallery) + 1


def cmc_curve(
    gallery: Dict[str, np.ndarray],
    gallery_labels: Dict[str, str],
    probes: Dict[str, np.ndarray],
    probe_labels: Dict[str, str],
    max_rank: int = 10,
) -> np.ndarray:
    """Cumulative Match Characteristic: fração de probes cuja identidade
    correta aparece até o rank k, para k = 1..max_rank."""
    ranks = np.array(
        [
            _rank_of_correct_match(probes[pid], probe_labels[pid], gallery, gallery_labels)
            for pid in probes
        ]
    )
    cmc = np.array([(ranks <= k).mean() for k in range(1, max_rank + 1)])
    return cmc


def rank_n_accuracy(
    gallery: Dict[str, np.ndarray],
    gallery_labels: Dict[str, str],
    probes: Dict[str, np.ndarray],
    probe_labels: Dict[str, str],
    n: int = 1,
) -> float:
    """Atalho para CMC[n-1] (ex.: rank_n_accuracy(..., n=1) = rank-1
    accuracy)."""
    curve = cmc_curve(gallery, gallery_labels, probes, probe_labels, max_rank=n)
    return float(curve[-1])
