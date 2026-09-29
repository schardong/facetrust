"""
facetrust.morphing
=====================

Métricas de "Morphing Attack Potential" (Scherhag et al., 2017 -- MMPMR e
RMMR; Venkatesh et al., 2020 -- FMMPMR). O ponto central do artigo/tutorial
é que o sucesso do ataque deve ser medido CONTRA o sistema alvo e o
threshold operacional (não apenas por qualidade visual) -- por isso este
módulo é construído sobre `facetrust.verification` (`threshold_at_far`).

Definições (confirmadas contra a literatura original, não só memorizadas):

- MMPMR: uma morph "vence" um contribuinte se PELO MENOS UMA das provas
  desse contribuinte bate o threshold; o ataque como um todo só conta como
  sucesso se TODOS os contribuintes forem vencidos (cada um por pelo menos
  uma prova). Em símbolos, por contribuinte k: passa_k = max(scores_k) >= tau;
  MMPMR = média sobre morphs de [ AND_k passa_k ].

- FMMPMR: variante mais estrita -- a morph só conta como sucesso se
  TODAS as provas de TODOS os contribuintes baterem o threshold (nenhuma
  folga: nem "a melhor foto de cada um" basta, tem que ser qualquer foto).

- RMMR: NÃO é uma proporção intermediária entre MMPMR e FMMPMR (isso era
  um erro numa versão anterior deste módulo). RMMR = MMPMR(tau) + FRR(tau),
  onde FRR(tau) é a taxa de falso-rejeite do sistema na população
  bona-fide NO MESMO threshold. A ideia: um MMPMR baixo não significa nada
  de bom por si só, se o mesmo threshold também rejeitar um monte de
  usuários genuínos -- RMMR "ancora" o número de vulnerabilidade ao custo
  de usabilidade pago para obtê-lo.

- ProdAvg-MMPMR: variante alternativa ao MinMax-MMPMR (`mmpmr()`) para
  múltiplas provas por contribuinte, menos enviesada (ver `prodavg_mmpmr`).

Proveniência: MMPMR (variante MinMax), ProdAvg-MMPMR e RMMR (=MMPMR+FRR)
foram conferidos linha a linha contra `metrics.py` da implementação de
referência dos próprios autores (Ulrich Scherhag, Hochschule Darmstadt --
https://github.com/dasec/mvr), não só contra a descrição em prosa dos
artigos. FMMPMR é de um artigo diferente (Venkatesh et al., 2020), que não
tem implementação de referência pública conhecida -- validado aqui contra
a definição em prosa de múltiplas fontes secundárias, com um pouco mais de
cautela do que as outras três. `facetrust` usa a convenção `>=` para
"aceita" (consistente com `facetrust.verification`), enquanto a
implementação de referência usa `>` estrito -- a diferença só importa no
caso de um score empatar exatamente com o threshold, algo com
probabilidade essencialmente nula em scores de similaridade contínuos.

Convenção de dados: para cada imagem morphed m, existe um conjunto de
"contribuintes" (as identidades combinadas na morph). Comparamos a morph
contra uma ou mais amostras de referência (probe) de CADA contribuinte.

    scores[m] = {contributor_id: [score_probe_1, score_probe_2, ...], ...}
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

from .verification import threshold_at_far


MorphScores = Dict[str, Dict[str, Sequence[float]]]


def mmpmr(scores: MorphScores, threshold: float) -> float:
    """Mated Morph Presentation Match Rate.

    Uma morph vence um contribuinte se PELO MENOS UMA prova desse
    contribuinte bate o threshold (o atacante só precisa de uma foto boa
    de cada cúmplice). O ataque conta como sucesso se TODOS os
    contribuintes forem vencidos:

        MMPMR(tau) = (1/M) * sum_m [ AND_k ( max_i score(m,k,i) >= tau ) ]

    threshold tipicamente vem de `verification.threshold_at_far` calculado
    sobre a distribuição impostor "bona fide" do sistema, para refletir o
    ponto de operação real (ver `mmpmr_at_far`).
    """
    if not scores:
        return float("nan")
    hits = []
    for contributors in scores.values():
        contributor_passes = [float(np.max(vals)) >= threshold for vals in contributors.values()]
        hits.append(1.0 if all(contributor_passes) else 0.0)
    return float(np.mean(hits))


def fmmpmr(scores: MorphScores, threshold: float) -> float:
    """Fully-MMPMR (Venkatesh et al., 2020): variante mais estrita do
    MMPMR. Exige que TODAS as provas de TODOS os contribuintes batam o
    threshold -- não basta a melhor foto de cada um, qualquer foto
    apresentada tem que funcionar:

        FMMPMR(tau) = (1/M) * sum_m [ AND sobre TODAS as comparações
                       individuais (score >= tau) ]

    Como usa "todas as comparações" em vez de "pelo menos uma por
    contribuinte", FMMPMR <= MMPMR sempre, para o mesmo threshold e mesmos
    dados. As duas coincidem quando há só uma prova por contribuinte.
    """
    if not scores:
        return float("nan")
    hits = []
    for contributors in scores.values():
        all_pass = all(
            (np.asarray(vals) >= threshold).all() for vals in contributors.values()
        )
        hits.append(1.0 if all_pass else 0.0)
    return float(np.mean(hits))


def prodavg_mmpmr(scores: MorphScores, threshold: float) -> float:
    """ProdAvg-MMPMR (Scherhag et al., 2017): variante alternativa ao
    MinMax-MMPMR (`mmpmr()`) para quando há múltiplas provas por
    contribuinte. A literatura mais recente recomenda ProdAvg sobre
    MinMax: MinMax tende a INFLAR artificialmente a vulnerabilidade
    conforme aumenta o número de provas por contribuinte (basta UMA prova
    boa entre muitas tentativas para "vencer" o contribuinte), enquanto
    ProdAvg dá um resultado contínuo e mais estável.

    Para cada contribuidor, calcula a FRAÇÃO de suas provas que passam o
    threshold (um "hit rate" em [0,1], não um booleano); multiplica essas
    frações entre os contribuidores da morph; e tira a média sobre as
    morphs:

        ProdAvg-MMPMR(tau) = (1/M) * sum_m [ PROD_k hit_rate(m,k,tau) ]
        hit_rate(m,k,tau) = fração das provas do contribuinte k que
                             passam tau, na morph m

    Ao contrário de `mmpmr()`/`fmmpmr()`, o resultado por morph já é
    contínuo (não 0/1) antes da média final.
    """
    if not scores:
        return float("nan")
    per_morph = []
    for contributors in scores.values():
        hit_rates = [float(np.mean(np.asarray(vals) >= threshold)) for vals in contributors.values()]
        per_morph.append(float(np.prod(hit_rates)) if hit_rates else 0.0)
    return float(np.mean(per_morph))


def rmmr(genuine_scores: np.ndarray, morph_scores: MorphScores, threshold: float) -> float:
    """Relative Morph Match Rate (Scherhag et al., 2017):

        RMMR(tau) = MMPMR(tau) + FRR(tau)

    onde FRR(tau) é a taxa de falso-rejeite do sistema, medida na
    distribuição GENUINE bona-fide (comparações mesmo-sujeito, sem
    morphing) NO MESMO threshold usado para o MMPMR.

    Motivação: um MMPMR baixo pode ser só um artefato de um threshold
    extremamente rígido -- que também rejeitaria um monte de usuários
    legítimos. RMMR ancora a vulnerabilidade relatada ao custo de
    usabilidade (FRR) pago para alcançá-la, no MESMO ponto de operação.
    Não é uma probabilidade (pode passar de 1); é um índice comparativo.

    IMPORTANTE sobre o dataset: `genuine_scores` vem de comparações
    MESMO-SUJEITO (a mesma pessoa contra si mesma, em capturas
    diferentes) -- portanto exige PELO MENOS 2 amostras "clean" por
    identidade no seu dataset. Se cada sujeito só tem 1 foto "clean", não
    existe nenhum par genuine possível e RMMR não pode ser calculado
    (não há como medir FRR sem repetir alguém).
    """
    genuine = np.asarray(genuine_scores, dtype=np.float64)
    if genuine.size == 0:
        raise ValueError(
            "genuine_scores vazio: RMMR precisa de comparações genuine "
            "(mesmo sujeito, capturas diferentes) para medir FRR. Isso "
            "geralmente significa que seu dataset tem só 1 imagem 'clean' "
            "por identidade -- adicione pelo menos 2 capturas por sujeito "
            "em data/clean/<subject_id>/, ou calcule mmpmr()/fmmpmr()/"
            "prodavg_mmpmr() isoladamente (não precisam de genuine)."
        )
    frr = float(np.mean(genuine < threshold))
    return mmpmr(morph_scores, threshold) + frr


def mmpmr_at_far(
    bona_fide_genuine_scores: np.ndarray,
    bona_fide_impostor_scores: np.ndarray,
    morph_scores: MorphScores,
    target_far: float,
    variant: str = "mmpmr",
) -> Dict[str, float]:
    """Conveniência: obtém o threshold operacional do sistema (a partir da
    distribuição impostor bona-fide) para o FAR alvo e calcula o(s) MAP
    metric(s) nesse ponto de operação -- exatamente o que o tutorial pede:
    "o sucesso do ataque deve ser medido contra o sistema alvo e o
    threshold operacional".

    `bona_fide_genuine_scores` só é necessário para `variant` incluindo
    "rmmr" (RMMR precisa de FRR, que vem da distribuição genuine -- que
    por sua vez exige pelo menos 2 amostras "clean" por identidade no
    dataset). Com `variant="all"` e genuine vazio/None, RMMR é pulado
    (fica NaN, com um aviso) em vez de derrubar mmpmr/fmmpmr/prodavg, que
    não dependem disso. Com `variant="rmmr"` explicitamente, um genuine
    vazio ainda levanta erro -- foi pedido especificamente e não há como
    calculá-lo.

    `variant`: "mmpmr" | "fmmpmr" | "prodavg" | "rmmr" | "all"
    """
    tau = threshold_at_far(bona_fide_impostor_scores, target_far)
    results: Dict[str, float] = {"threshold": tau, "target_far": target_far}
    if variant in ("mmpmr", "all"):
        results["mmpmr"] = mmpmr(morph_scores, tau)
    if variant in ("fmmpmr", "all"):
        results["fmmpmr"] = fmmpmr(morph_scores, tau)
    if variant in ("prodavg", "all"):
        results["prodavg_mmpmr"] = prodavg_mmpmr(morph_scores, tau)
    if variant in ("rmmr", "all"):
        genuine = np.asarray(bona_fide_genuine_scores, dtype=np.float64) if bona_fide_genuine_scores is not None else np.array([])
        if genuine.size == 0 and variant == "all":
            warnings.warn(
                "RMMR pulado (ficou NaN): sem comparações genuine bona-fide. "
                "Isso geralmente significa que seu dataset tem só 1 imagem "
                "'clean' por identidade -- adicione pelo menos 2 capturas "
                "por sujeito em data/clean/<subject_id>/ para poder medir "
                "FRR e calcular RMMR.",
                stacklevel=2,
            )
            results["rmmr"] = float("nan")
        else:
            results["rmmr"] = rmmr(genuine, morph_scores, tau)
    return results


def map_curve(
    bona_fide_genuine_scores: np.ndarray,
    bona_fide_impostor_scores: np.ndarray,
    morph_scores: MorphScores,
    target_fars: Sequence[float] = (1e-1, 1e-2, 1e-3, 1e-4),
    variant: str = "mmpmr",
) -> List[Dict[str, float]]:
    """MMPMR/FMMPMR/RMMR calculados numa varredura de FARs operacionais --
    a "curva de potencial de ataque" mencionada no tutorial. Retorna uma
    lista de dicts, um por FAR alvo (pronto para virar `pandas.DataFrame`)."""
    return [
        mmpmr_at_far(bona_fide_genuine_scores, bona_fide_impostor_scores, morph_scores, far, variant=variant)
        for far in target_fars
    ]
