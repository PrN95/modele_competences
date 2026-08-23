"""Adaptateur local et injectable pour l'encodage BGE-M3."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from src.config import (
    MODEL_BATCH_SIZE,
    MODEL_DEVICE,
    MODEL_MAX_LENGTH,
    MODEL_NORMALIZE_EMBEDDINGS,
    MODEL_PATH,
    MODEL_USE_FP16,
)
from src.domain import Competence


class ModeleBGE(Protocol):
    """Partie de l'API FlagEmbedding utilisée par l'adaptateur."""

    def encode(self, sentences: list[str], **kwargs: Any) -> Mapping[str, Any]: ...


class EncodeurCompetences(Protocol):
    """Interface injectable consommée par le rapprochement sémantique."""

    def encoder(self, competences: Sequence[Competence]) -> SortieEncodage: ...


ModeleFactory = Callable[..., ModeleBGE]


@dataclass(frozen=True, slots=True)
class SortieEncodage:
    """Représentations dense et sparse, sans représentation ColBERT."""

    vecteurs_dense: tuple[tuple[float, ...], ...]
    poids_sparse: tuple[Mapping[str, float], ...]

    def __post_init__(self) -> None:
        if len(self.vecteurs_dense) != len(self.poids_sparse):
            raise ValueError("Les sorties dense et sparse doivent avoir la même taille.")
        if self.vecteurs_dense:
            dimension = len(self.vecteurs_dense[0])
            if dimension == 0 or any(len(vector) != dimension for vector in self.vecteurs_dense):
                raise ValueError("Les vecteurs dense doivent partager une dimension non nulle.")


class AdaptateurBGEM3:
    """Charge une seule fois un modèle BGE-M3 local, au premier encodage."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        *,
        model_factory: ModeleFactory | None = None,
    ) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"Modèle BGE-M3 local absent: le dossier '{path}' est introuvable. "
                "Aucun téléchargement automatique n'est autorisé."
            )

        self._model_path = path
        self._model_factory = model_factory or _creer_modele_flagembedding
        self._model: ModeleBGE | None = None
        self._load_lock = Lock()

    @property
    def est_charge(self) -> bool:
        return self._model is not None

    def encoder(self, competences: Sequence[Competence]) -> SortieEncodage:
        """Encode les textes exacts des compétences en dense et sparse."""

        if not competences:
            return SortieEncodage(vecteurs_dense=(), poids_sparse=())

        model = self._obtenir_modele()
        raw_output = model.encode(
            [competence.texte for competence in competences],
            batch_size=MODEL_BATCH_SIZE,
            max_length=MODEL_MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return _normaliser_sortie(raw_output, len(competences))

    def _obtenir_modele(self) -> ModeleBGE:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    _forcer_mode_hors_ligne()
                    self._model = self._model_factory(
                        str(self._model_path),
                        devices=[MODEL_DEVICE],
                        use_fp16=MODEL_USE_FP16,
                        normalize_embeddings=MODEL_NORMALIZE_EMBEDDINGS,
                        batch_size=MODEL_BATCH_SIZE,
                        query_max_length=MODEL_MAX_LENGTH,
                        passage_max_length=MODEL_MAX_LENGTH,
                        return_dense=True,
                        return_sparse=True,
                        return_colbert_vecs=False,
                        trust_remote_code=False,
                    )
        return self._model


def _forcer_mode_hors_ligne() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _creer_modele_flagembedding(model_path: str, **kwargs: Any) -> ModeleBGE:
    # Import différé : l'import de FlagEmbedding ne déclenche pas le chargement
    # du modèle et le projet n'importe jamais sentence-transformers directement.
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(model_path, **kwargs)


def _normaliser_sortie(raw_output: Mapping[str, Any], expected_count: int) -> SortieEncodage:
    if raw_output.get("colbert_vecs") is not None:
        raise ValueError("Une sortie ColBERT a été reçue alors qu'elle est désactivée.")
    if "dense_vecs" not in raw_output or "lexical_weights" not in raw_output:
        raise ValueError("FlagEmbedding doit retourner dense_vecs et lexical_weights.")

    dense_array = np.asarray(raw_output["dense_vecs"], dtype=float)
    if dense_array.ndim != 2 or dense_array.shape[0] != expected_count:
        raise ValueError("Le nombre de vecteurs dense ne correspond pas aux compétences.")

    sparse_values = raw_output["lexical_weights"]
    if not isinstance(sparse_values, Sequence) or len(sparse_values) != expected_count:
        raise ValueError("Le nombre de poids sparse ne correspond pas aux compétences.")

    dense = tuple(tuple(float(value) for value in row) for row in dense_array)
    sparse = tuple(
        MappingProxyType({str(token): float(weight) for token, weight in weights.items()})
        for weights in sparse_values
    )
    return SortieEncodage(vecteurs_dense=dense, poids_sparse=sparse)
