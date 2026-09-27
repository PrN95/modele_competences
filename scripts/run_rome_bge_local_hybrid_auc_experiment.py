"""Lance la campagne 2 ROME avec BGE-M3 local dense+sparse.

L'encodage reste séquentiel pour garder une empreinte mémoire compatible avec
la machine locale. Les onze pondérations réutilisent les mêmes caches.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_PATH, ROME_REFERENCE_TABLE
from src.domain import Competence
from src.embeddings import AdaptateurBGEM3
from src.rome import read_rome_relations
from src.rome_experiment import (
    RomeHybridCache,
    load_or_create_hybrid_embeddings,
    load_or_create_hybrid_similarity_matrices,
    prepare_rome_corpus,
    rome_campaign_output_directory,
    run_hybrid_auc_roc,
    write_hybrid_auc_outputs,
)


def _local_hybrid_encoder(model_path: Path):
    """Retourne un encodeur BGE-M3 dense+sparse, compétence par compétence."""
    adapter = AdaptateurBGEM3(model_path)

    def encode(texts: tuple[str, ...]):
        dense: list[tuple[float, ...]] = []
        sparse: list[dict[str, float]] = []
        total = len(texts)
        started = time.perf_counter()
        for index, text in enumerate(texts, start=1):
            result = adapter.encoder((Competence(text, None, 1),))
            dense.extend(result.vecteurs_dense)
            sparse.extend(dict(weights) for weights in result.poids_sparse)
            if index == total or index % 25 == 0:
                print(f"Embeddings BGE-M3 locaux dense+sparse : {index}/{total} compétences, lot 1, durée {time.perf_counter() - started:.1f} s")
        return np.asarray(dense, dtype=np.float32), sparse

    return encode


def main() -> int:
    parser = argparse.ArgumentParser(description="Campagne 2 ROME BGE-M3 locale hybride : balayage AUC-ROC des pondérations dense+sparse.")
    parser.add_argument("--model-path", default=str(MODEL_PATH), help="Dossier local du modèle BGE-M3.")
    parser.add_argument("--reference", default=str(ROME_REFERENCE_TABLE), help="Table de vérité ROME.")
    parser.add_argument("--campaign-name", default=f"campagne_hybride_auc_roc_bge_m3_local_{time.strftime('%Y%m%dT%H%M%S')}", help="Nom du nouveau dossier sous outputs/rome/experiments_2/.")
    args = parser.parse_args()
    started = time.perf_counter()
    output = rome_campaign_output_directory(args.campaign_name)
    if output.exists():
        raise FileExistsError(f"Le dossier de campagne existe déjà : {output}")

    model_path = Path(args.model_path).expanduser().resolve()
    prepared = prepare_rome_corpus(pd.read_csv("outputs/rome/corpus_rome.csv"))
    relations, diagnostics = read_rome_relations(args.reference)
    errors = [item for item in diagnostics if item.severity == "erreur"]
    if errors:
        raise RuntimeError("Les relations ROME comportent des erreurs; relancez d'abord la validation du corpus.")

    print(f"BGE-M3 local : {model_path} (encodage séquentiel dense+sparse, lot 1)")
    cache = RomeHybridCache(prepared, f"bge-m3-local-{model_path}", output / "cache")
    dense, sparse, embeddings_cached = load_or_create_hybrid_embeddings(cache, _local_hybrid_encoder(model_path), print)
    dense_matrix, sparse_matrix, matrices_cached = load_or_create_hybrid_similarity_matrices(cache, dense, sparse, print)
    experiment = run_hybrid_auc_roc(prepared, relations, dense_matrix, sparse_matrix, progress=print, model_name="bge-m3-local")
    outputs = write_hybrid_auc_outputs(
        experiment,
        prepared,
        relations,
        cache,
        model_path,
        output,
        model_name="bge-m3-local",
        reference_table=args.reference,
        embeddings_cached=embeddings_cached,
        matrices_cached=matrices_cached,
    )
    best = experiment.best_configurations
    print(f"Campagne terminée en {time.perf_counter() - started:.2f}s ({'embeddings en cache' if embeddings_cached else 'embeddings calculés'}, {'matrices en cache' if matrices_cached else 'matrices calculées'}).")
    for row in best.itertuples():
        print(f"Meilleure pondération : dense={row.coefficient_dense:.1f}, sparse={row.coefficient_sparse:.1f}, AUC-ROC={row.auc_roc:.4f}")
    print("Résultats : " + ", ".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
