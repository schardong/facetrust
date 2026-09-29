"""
facetrust.plotting
=====================

Funções de plot correspondentes às figuras usadas no tutorial:
distribuições genuine/impostor, ROC/DET, varredura de alpha (PSNR/SSIM e
embedding drift vs. alpha) e curva de potencial de ataque de morphing
(MMPMR vs. FAR alvo).

Cada função aceita um `ax` opcional (para compor grids de subplots) e
retorna o `Axes` usado, para facilitar customização posterior.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_score_distributions(genuine: np.ndarray, impostor: np.ndarray, ax=None, bins: int = 40):
    ax = ax or plt.gca()
    ax.hist(impostor, bins=bins, alpha=0.6, density=True, label="impostor", color="tab:red")
    ax.hist(genuine, bins=bins, alpha=0.6, density=True, label="genuine", color="tab:blue")
    ax.set_xlabel("Similaridade (cosseno)")
    ax.set_ylabel("Densidade")
    ax.legend()
    return ax


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, ax=None, label: Optional[str] = None):
    ax = ax or plt.gca()
    ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("FAR (False Accept Rate)")
    ax.set_ylabel("TAR (True Accept Rate)")
    if label:
        ax.legend()
    return ax


def plot_det(fpr: np.ndarray, frr: np.ndarray, ax=None, label: Optional[str] = None, log_scale: bool = True):
    ax = ax or plt.gca()
    ax.plot(fpr, frr, label=label)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("FAR (False Accept Rate)")
    ax.set_ylabel("FRR (False Reject Rate)")
    if label:
        ax.legend()
    return ax


def plot_alpha_sweep(alphas: Sequence[float], values: Sequence[float], ax=None, ylabel: str = "", label: Optional[str] = None):
    """Plot genérico de "watermark strength alpha" vs. uma métrica
    qualquer (PSNR, SSIM, embedding drift, BER, ...) -- reutilize para
    todos os plots de varredura de alpha pedidos na atividade prática."""
    ax = ax or plt.gca()
    ax.plot(alphas, values, marker="o", label=label)
    ax.set_xlabel(r"Força do watermark ($\alpha$)")
    ax.set_ylabel(ylabel)
    if label:
        ax.legend()
    return ax


def plot_map_curve(target_fars: Sequence[float], metric_values: Sequence[float], ax=None, metric_name: str = "MMPMR", label: Optional[str] = None):
    """Curva de potencial de ataque de morphing: métrica (MMPMR/FMMPMR/RMMR)
    em função do FAR operacional alvo do sistema de reconhecimento."""
    ax = ax or plt.gca()
    ax.plot(target_fars, metric_values, marker="o", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("FAR alvo do sistema (escala log)")
    ax.set_ylabel(metric_name)
    if label:
        ax.legend()
    return ax
