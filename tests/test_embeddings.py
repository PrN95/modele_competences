import os
from pathlib import Path

import numpy as np
import pytest

from src.domain import Competence
from src.embeddings import AdaptateurBGEM3


class FauxModele:
    def __init__(self) -> None:
        self.encode_calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, sentences, **kwargs):
        self.encode_calls.append((sentences, kwargs))
        return {
            "dense_vecs": np.array([[1.0, 0.0] for _ in sentences]),
            "lexical_weights": [{"token": 0.5} for _ in sentences],
            "colbert_vecs": None,
        }


def competence(niveau: int = 3) -> Competence:
    return Competence("C-1", "Intitulé fictif", "Description fictive", niveau)


def test_missing_local_model_directory_is_rejected(tmp_path: Path) -> None:
    factory_called = False

    def factory(*args, **kwargs):
        nonlocal factory_called
        factory_called = True
        return FauxModele()

    with pytest.raises(FileNotFoundError, match="Aucun téléchargement automatique"):
        AdaptateurBGEM3(tmp_path / "bge-m3-absent", model_factory=factory)

    assert factory_called is False


def test_loading_is_deferred_and_done_once(tmp_path: Path) -> None:
    model_path = tmp_path / "models" / "bge-m3"
    model_path.mkdir(parents=True)
    model = FauxModele()
    factory_calls = []

    def factory(*args, **kwargs):
        factory_calls.append((args, kwargs))
        return model

    adapter = AdaptateurBGEM3(model_path, model_factory=factory)
    assert adapter.est_charge is False
    assert factory_calls == []

    adapter.encoder((competence(),))
    adapter.encoder((competence(),))

    assert adapter.est_charge is True
    assert len(factory_calls) == 1
    assert len(model.encode_calls) == 2


def test_flagembedding_parameters_and_exact_text(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "models" / "bge-m3"
    model_path.mkdir(parents=True)
    model = FauxModele()
    constructor = {}

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    def factory(*args, **kwargs):
        constructor["args"] = args
        constructor["kwargs"] = kwargs
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        return model

    adapter = AdaptateurBGEM3(model_path, model_factory=factory)
    output = adapter.encoder((competence(niveau=3),))

    assert constructor["args"] == (str(model_path.resolve()),)
    assert constructor["kwargs"] == {
        "devices": ["cpu"],
        "use_fp16": False,
        "normalize_embeddings": True,
        "batch_size": 1,
        "query_max_length": 256,
        "passage_max_length": 256,
        "return_dense": True,
        "return_sparse": True,
        "return_colbert_vecs": False,
        "trust_remote_code": False,
    }
    texts, encode_kwargs = model.encode_calls[0]
    assert texts == ["Intitulé fictif. Description fictive"]
    assert "3" not in texts[0]
    assert encode_kwargs == {
        "batch_size": 1,
        "max_length": 256,
        "return_dense": True,
        "return_sparse": True,
        "return_colbert_vecs": False,
    }
    assert output.vecteurs_dense == ((1.0, 0.0),)
    assert dict(output.poids_sparse[0]) == {"token": 0.5}
    assert not hasattr(output, "colbert_vecs")


def test_unexpected_colbert_output_is_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "models" / "bge-m3"
    model_path.mkdir(parents=True)

    class FauxModeleColbert(FauxModele):
        def encode(self, sentences, **kwargs):
            output = super().encode(sentences, **kwargs)
            output["colbert_vecs"] = [np.array([[1.0]])]
            return output

    adapter = AdaptateurBGEM3(model_path, model_factory=lambda *args, **kwargs: FauxModeleColbert())

    with pytest.raises(ValueError, match="ColBERT"):
        adapter.encoder((competence(),))
