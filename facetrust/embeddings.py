"""
facetrust.embeddings
=====================

Camada de abstração para extração de embeddings faciais.

O tutorial usa ArcFace como modelo representativo (ver slide "Embedding-based
face recognition"), mas o pipeline de avaliação não depende de nenhum modelo
específico: qualquer extrator que mapeie uma imagem para um vetor pode ser
usado, inclusive embeddings já pré-computados por vocês.
"""

from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np


def l2_normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Normaliza um vetor (ou matriz NxD, linha a linha) para norma unitária."""
    vec = np.asarray(vec, dtype=np.float64)
    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        return vec / max(norm, eps)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vec / norms


class FaceEmbedder(abc.ABC):
    """Interface mínima que qualquer extrator de embeddings deve seguir."""

    @abc.abstractmethod
    def embed(self, image: np.ndarray) -> np.ndarray:
        """Recebe uma imagem (H, W, 3) já alinhada/recortada e retorna um
        vetor de embedding 1D."""
        raise NotImplementedError

    def embed_batch(self, images: Iterable[np.ndarray]) -> np.ndarray:
        """Implementação default (loop); sobrescreva para inferência em lote."""
        return np.stack([self.embed(img) for img in images], axis=0)

    def describe(self) -> str:
        """Identificação curta do extrator, gravada junto dos embeddings
        salvos em disco (para avisar se um cache foi gerado por outro
        extrator). Sobrescreva se houver parâmetros que mudam o resultado."""
        return type(self).__name__


class CallableEmbedder(FaceEmbedder):
    """Encapsula qualquer função `image -> vetor` como um FaceEmbedder.

    Útil quando o usuário já possui uma função de extração (ex. um wrapper
    fino em torno do ArcFace do `insightface`, ou de outro backbone).
    """

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], normalize: bool = True):
        self._fn = fn
        self._normalize = normalize

    def embed(self, image: np.ndarray) -> np.ndarray:
        vec = np.asarray(self._fn(image), dtype=np.float64).reshape(-1)
        return l2_normalize(vec) if self._normalize else vec


class ArcFaceEmbedder(FaceEmbedder):
    """Wrapper opcional em torno do `insightface` (modelo ArcFace).

    Requer `pip install insightface onnxruntime` (ou `onnxruntime-gpu`) e
    conexão à internet na primeira execução, para baixar os pesos do
    modelo (pacote `buffalo_l`, por exemplo). Este wrapper é apenas
    conveniência: qualquer outro extrator pode ser usado via
    `CallableEmbedder`.
    """

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = -1, det_size=(640, 640)):
        try:
            import insightface  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependência opcional
            raise ImportError(
                "ArcFaceEmbedder requer o pacote 'insightface' "
                "(pip install insightface onnxruntime). "
                "Alternativamente, use CallableEmbedder com seu próprio "
                "extrator de embeddings."
            ) from exc

        self.model_name = model_name
        self.det_size = tuple(det_size)
        self._app = insightface.app.FaceAnalysis(name=model_name)
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    def describe(self) -> str:
        return f"ArcFaceEmbedder(model={self.model_name}, det_size={self.det_size})"

    def embed(self, image: np.ndarray) -> np.ndarray:
        faces = self._app.get(image)
        if not faces:
            raise ValueError(
                "Nenhuma face detectada na imagem. Certifique-se de que a "
                "imagem já está no formato esperado pelo detector, ou "
                "faça o crop/alinhamento antes de chamar embed()."
            )
        # Usa a maior face detectada (evita pegar rostos de fundo).
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return l2_normalize(face.normed_embedding)


@dataclass
class EmbeddingStore:
    """Cache simples id -> embedding, com utilitários de import/export.

    Pensado para poupar reprocessamento: uma vez extraídos os embeddings de
    todas as imagens (limpas, marcadas, morphed) de um conjunto, salve-os em
    disco e reutilize entre as diferentes análises do tutorial.
    """

    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, sample_id: str, embedding: np.ndarray) -> None:
        self.embeddings[sample_id] = np.asarray(embedding, dtype=np.float64).reshape(-1)

    def compute_and_add(self, sample_id: str, image: np.ndarray, embedder: FaceEmbedder) -> np.ndarray:
        vec = embedder.embed(image)
        self.add(sample_id, vec)
        return vec

    def get(self, sample_id: str) -> Optional[np.ndarray]:
        return self.embeddings.get(sample_id)

    def save(self, path: str) -> None:
        ids = list(self.embeddings.keys())
        mat = np.stack([self.embeddings[i] for i in ids], axis=0)
        np.savez(path, ids=np.array(ids, dtype=object), embeddings=mat)

    @classmethod
    def load(cls, path: str) -> "EmbeddingStore":
        data = np.load(path, allow_pickle=True)
        store = cls()
        for sample_id, vec in zip(data["ids"], data["embeddings"]):
            store.add(str(sample_id), vec)
        return store


# ---------------------------------------------------------------------------
# Salvar / carregar embeddings (cache em disco)
# ---------------------------------------------------------------------------

def image_fingerprint(image: np.ndarray) -> str:
    """Hash do CONTEÚDO da imagem (forma, tipo e pixels). Gravado junto de
    cada embedding para detectar cache defasado: se a imagem mudou desde que
    o embedding foi calculado, o hash não bate e o embedding é recalculado."""
    arr = np.ascontiguousarray(image)
    h = hashlib.sha1()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def save_embeddings_file(
    path: Union[str, Path],
    embeddings: Dict[str, np.ndarray],
    fingerprints: Dict[str, str],
    meta: Optional[dict] = None,
) -> None:
    """Grava embeddings num `.npz` (formato NumPy, sem pickle -- seguro de
    distribuir e de carregar). Campos: `ids`, `embeddings` (N x D),
    `fingerprints` (hash da imagem de cada id) e `meta` (JSON)."""
    ids = sorted(embeddings)
    if not ids:
        raise ValueError("Nenhum embedding para salvar.")
    matrix = np.stack([np.asarray(embeddings[i], dtype=np.float64).reshape(-1) for i in ids])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")  # grava num temporário e renomeia: nunca deixa arquivo pela metade
    np.savez_compressed(
        tmp,
        ids=np.array(ids, dtype=str),
        embeddings=matrix,
        fingerprints=np.array([fingerprints.get(i, "") for i in ids], dtype=str),
        meta=np.array(json.dumps(meta or {})),
    )
    tmp.replace(path)


def load_embeddings_file(
    path: Union[str, Path],
) -> Tuple[Dict[str, np.ndarray], Dict[str, str], dict]:
    """Inverso de `save_embeddings_file`: retorna (embeddings, fingerprints, meta)."""
    with np.load(Path(path), allow_pickle=False) as data:
        ids = [str(i) for i in data["ids"]]
        matrix = data["embeddings"]
        fps = [str(f) for f in data["fingerprints"]]
        meta = json.loads(str(data["meta"]))
    return (
        {i: matrix[k] for k, i in enumerate(ids)},
        {i: fps[k] for k, i in enumerate(ids)},
        meta,
    )
