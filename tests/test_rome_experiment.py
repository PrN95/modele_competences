from pathlib import Path

import numpy as np
import pandas as pd

from src.rome import RomeRelations
from src.rome_experiment import (
    RemoteOpenAIEmbeddings,
    RomeDenseCache,
    RomeHybridCache,
    load_or_create_embeddings,
    load_or_create_hybrid_embeddings,
    load_or_create_hybrid_similarity_matrices,
    load_or_create_similarity_matrix,
    prepare_rome_corpus,
    rome_campaign_output_directory,
    run_dense_grid,
    run_hybrid_auc_roc,
    sparse_weights_to_matrix,
    write_dense_experiment_outputs,
    write_hybrid_auc_outputs,
)


class _Response:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            import requests
            raise requests.HTTPError("refus")

    def json(self):
        return self.payload


class _Session:
    def __init__(self):
        self.posts = []

    def get(self, url, **kwargs):
        return _Response({"data": [{"id": "modele-annonce"}]})

    def post(self, url, **kwargs):
        self.posts.append(kwargs["json"]["input"])
        inputs = kwargs["json"]["input"]
        if len(inputs) > 2:
            return _Response({}, status=413)
        return _Response({"data": [{"index": index, "embedding": [float(index + 1), 1.0]} for index in range(len(inputs))]})


def _prepared() -> tuple:
    corpus = pd.DataFrame(
        {
            "code_rome": ["M1000", "M2000", "M3000"],
            "intitule": ["A", "B", "C"],
            "competences": ["python\ndata", "python\nstatistiques", "maçonnerie"],
            "statut_validation_pdf": ["valide", "valide", "valide"],
        }
    )
    prepared = prepare_rome_corpus(corpus)
    relations = RomeRelations({"M1000": ("A",), "M2000": ("B",), "M3000": ("C",)}, frozenset({frozenset(("M1000", "M2000"))}))
    return prepared, relations


def test_dense_grid_uses_symmetric_coverages_and_selects_f1_best() -> None:
    prepared, relations = _prepared()
    vectors = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [-1, 0]], dtype=float)
    matrix = vectors @ vectors.T

    experiment = run_dense_grid(prepared, relations, matrix, thresholds=(0.8, 0.95), campaign_id="test", progress=lambda _: None)

    assert len(experiment.summary) == 4
    assert experiment.summary["meilleure_configuration"].sum() == 1
    best = experiment.best_configuration
    assert best["f1"] == 1.0
    assert best["accuracy"] == 1.0
    detail = experiment.details[(experiment.details["code_rome_a"] == "M1000") & (experiment.details["code_rome_b"] == "M2000")].iloc[0]
    assert detail["couverture_a_vers_b"] == 1.0
    assert detail["couverture_b_vers_a"] == 1.0
    assert detail["couverture_symetrique"] == 1.0


def test_dense_grid_uses_only_explicit_pairs_from_a_truth_table() -> None:
    prepared, _ = _prepared()
    relations = RomeRelations(
        {"M1000": ("A",), "M2000": ("B",), "M3000": ("C",)},
        frozenset({frozenset(("M1000", "M2000"))}),
        frozenset({frozenset(("M1000", "M3000"))}),
    )
    vectors = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [-1, 0]], dtype=float)

    experiment = run_dense_grid(prepared, relations, vectors @ vectors.T, thresholds=(0.8,), progress=lambda _: None)

    assert len(experiment.details) == 2
    assert set(experiment.details["relation_reference"]) == {True, False}


def test_cache_reuses_embeddings_and_similarity_matrix(tmp_path: Path) -> None:
    prepared, _ = _prepared()
    cache = RomeDenseCache(prepared, "bge-m3", tmp_path)
    calls = 0

    def encoder(texts):
        nonlocal calls
        calls += 1
        return np.array([[1, 0], [0, 1], [1, 0], [0, 1], [-1, 0]], dtype=float)

    vectors, cached = load_or_create_embeddings(cache, encoder, progress=lambda _: None)
    assert not cached and calls == 1
    _, cached = load_or_create_embeddings(cache, encoder, progress=lambda _: None)
    assert cached and calls == 1
    matrix, cached = load_or_create_similarity_matrix(cache, vectors, progress=lambda _: None)
    assert not cached
    matrix_again, cached = load_or_create_similarity_matrix(cache, vectors, progress=lambda _: None)
    assert cached and np.array_equal(matrix, matrix_again)


def test_remote_encoder_discovers_model_and_retries_with_smaller_batches() -> None:
    session = _Session()
    client = RemoteOpenAIEmbeddings("http://embedding/v1", session=session, max_retries=1)

    vectors = client.encode(("a", "b", "c"), batch_size=4, progress=lambda _: None)

    assert client.model_id == "modele-annonce"
    assert [len(batch) for batch in session.posts] == [3, 1, 2]
    assert vectors.shape == (3, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)


def test_experiment_outputs_include_expected_files(tmp_path: Path) -> None:
    prepared, relations = _prepared()
    vectors = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [-1, 0]], dtype=float)
    experiment = run_dense_grid(prepared, relations, vectors @ vectors.T, thresholds=(0.8,), campaign_id="test", progress=lambda _: None)
    cache = RomeDenseCache(prepared, "bge-m3", tmp_path / "cache")

    paths = write_dense_experiment_outputs(experiment, prepared, relations, cache, "models/bge-m3", tmp_path / "resultats")

    assert all(path.exists() for path in paths)
    assert "non_relation_reference" in paths[3].read_text(encoding="utf-8")


def test_campaign_output_directory_is_scoped_to_the_new_rome_root() -> None:
    directory = rome_campaign_output_directory("bge_m3_dense_20260913T120000")

    assert directory.as_posix() == "outputs/rome/experiments_2/bge_m3_dense_20260913T120000"


def test_hybrid_auc_uses_continuous_symmetric_scores_and_marks_best_weight() -> None:
    prepared, relations = _prepared()
    dense = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [-1, 0]], dtype=float)
    sparse = sparse_weights_to_matrix((
        {"python": 1.0}, {"data": 1.0}, {"python": 1.0}, {"data": 1.0}, {"maconnerie": 1.0},
    ))

    experiment = run_hybrid_auc_roc(prepared, relations, dense @ dense.T, sparse @ sparse.T, dense_weights=(1.0, 0.0), campaign_id="test", progress=lambda _: None)

    assert list(experiment.summary["auc_roc"]) == [0.75, 1.0]
    assert len(experiment.best_configurations) == 1
    assert experiment.best_configurations.iloc[0]["coefficient_sparse"] == 1.0
    detail = experiment.details[(experiment.details["coefficient_sparse"] == 1.0) & (experiment.details["code_rome_a"] == "M1000") & (experiment.details["code_rome_b"] == "M2000")].iloc[0]
    assert detail["score_continu_a_vers_b"] == 1.0
    assert detail["score_continu_b_vers_a"] == 1.0
    assert detail["score_continu_hybride"] == 1.0


def test_hybrid_cache_reuses_dense_sparse_embeddings_and_matrices(tmp_path: Path) -> None:
    prepared, _ = _prepared()
    cache = RomeHybridCache(prepared, "bge-m3-local", tmp_path / "cache")
    calls = 0

    def encoder(texts):
        nonlocal calls
        calls += 1
        return np.array([[1, 0], [0, 1], [1, 0], [0, 1], [-1, 0]], dtype=float), ({"a": 1.0}, {"b": 1.0}, {"a": 1.0}, {"b": 1.0}, {"c": 1.0})

    dense, lexical, cached = load_or_create_hybrid_embeddings(cache, encoder, progress=lambda _: None)
    assert not cached and calls == 1
    dense_again, lexical_again, cached = load_or_create_hybrid_embeddings(cache, encoder, progress=lambda _: None)
    assert cached and calls == 1 and np.array_equal(dense, dense_again)
    assert (lexical != lexical_again).nnz == 0
    _, _, cached = load_or_create_hybrid_similarity_matrices(cache, dense, lexical, progress=lambda _: None)
    assert not cached
    _, _, cached = load_or_create_hybrid_similarity_matrices(cache, dense, lexical, progress=lambda _: None)
    assert cached


def test_hybrid_outputs_include_auc_report_and_metadata(tmp_path: Path) -> None:
    prepared, relations = _prepared()
    dense = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [-1, 0]], dtype=float)
    sparse = sparse_weights_to_matrix(({"a": 1.0}, {"b": 1.0}, {"a": 1.0}, {"b": 1.0}, {"c": 1.0}))
    experiment = run_hybrid_auc_roc(prepared, relations, dense @ dense.T, sparse @ sparse.T, dense_weights=(1.0,), campaign_id="test", progress=lambda _: None)
    cache = RomeHybridCache(prepared, "bge-m3-local", tmp_path / "cache")

    paths = write_hybrid_auc_outputs(experiment, prepared, relations, cache, "models/bge-m3", tmp_path / "resultats", model_name="bge-m3-local", reference_table="reference.xlsx", embeddings_cached=False, matrices_cached=False)

    assert all(path.exists() for path in paths)
    assert "AUC-ROC" in paths[3].read_text(encoding="utf-8")
