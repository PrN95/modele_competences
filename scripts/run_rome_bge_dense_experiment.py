"""Lance explicitement la première campagne BGE-M3 dense sur ROME."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROME_EXPERIMENT_BATCH_SIZE, ROME_REFERENCE_TABLE
from src.rome import read_rome_relations
from src.rome_experiment import (
    RomeDenseCache,
    load_or_create_embeddings,
    load_or_create_similarity_matrix,
    prepare_rome_corpus,
    rome_campaign_output_directory,
    RemoteOpenAIEmbeddings,
    run_dense_grid,
    write_dense_experiment_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Campagne ROME BGE-M3 dense-only distante.")
    parser.add_argument("--batch-size", type=int, default=None, help="Taille de lot expérimentale (défaut : ROME_EXPERIMENT_BATCH_SIZE ou 32).")
    parser.add_argument("--reference", default=str(ROME_REFERENCE_TABLE), help="Table de vérité ROME (défaut : ROME 6).")
    parser.add_argument("--campaign-name", default=f"bge_m3_dense_{time.strftime('%Y%m%dT%H%M%S')}", help="Nom du nouveau dossier sous outputs/rome/experiments_2/.")
    args = parser.parse_args()
    start = time.perf_counter()
    corpus = pd.read_csv("outputs/rome/corpus_rome.csv")
    prepared = prepare_rome_corpus(corpus)
    relations, diagnostics = read_rome_relations(args.reference)
    errors = [item for item in diagnostics if item.severity == "erreur"]
    if errors:
        raise RuntimeError("Les relations ROME comportent des erreurs; relancez d'abord la validation du corpus.")
    client = RemoteOpenAIEmbeddings()
    model_id = client.discover_model()
    batch_size = args.batch_size if args.batch_size is not None else ROME_EXPERIMENT_BATCH_SIZE
    if batch_size < 1:
        raise ValueError("--batch-size doit être strictement positif.")
    output = rome_campaign_output_directory(args.campaign_name)
    if output.exists():
        raise FileExistsError(f"Le dossier de campagne existe déjà : {output}")
    print(f"BGE distant joignable : {client.base_url} (modèle annoncé : {model_id})")
    cache = RomeDenseCache(prepared, f"{model_id}-{client.base_url}", output / "cache")
    vectors, embeddings_cached = load_or_create_embeddings(cache, lambda texts: client.encode(texts, batch_size=batch_size, progress=print), print)
    matrix, matrix_cached = load_or_create_similarity_matrix(cache, vectors, print)
    experiment = run_dense_grid(prepared, relations, matrix, progress=print)
    outputs = write_dense_experiment_outputs(experiment, prepared, relations, cache, model_id, output, endpoint=client.base_url, batch_size=batch_size, reference_table=args.reference)
    best = experiment.best_configuration
    print(f"Campagne terminée en {time.perf_counter() - start:.2f}s (lot demandé : {batch_size}; {'embeddings en cache' if embeddings_cached else 'embeddings calculés'}, {'matrice en cache' if matrix_cached else 'matrice calculée'}).")
    print(f"Meilleure configuration : seuil_sim={best['seuil_sim']:.2f}, seuil_couv={best['seuil_couv']:.2f}, F1={best['f1']:.4f}")
    print("Résultats : " + ", ".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
