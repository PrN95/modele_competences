"""Lance la campagne ROME Qwen3-Embedding-8B dense-only distante."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import QWEN_API_KEY, QWEN_BASE_URL, ROME_EXPERIMENT_BATCH_SIZE, ROME_REFERENCE_TABLE
from src.rome import read_rome_relations
from src.rome_experiment import (
    RemoteOpenAIEmbeddings, RomeDenseCache, load_or_create_embeddings,
    load_or_create_similarity_matrix, prepare_rome_corpus, run_dense_grid,
    rome_campaign_output_directory, write_dense_experiment_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Campagne ROME Qwen dense-only distante.")
    parser.add_argument("--batch-size", type=int, default=None, help="Taille de lot expérimentale (défaut : ROME_EXPERIMENT_BATCH_SIZE ou 32).")
    parser.add_argument("--reference", default=str(ROME_REFERENCE_TABLE), help="Table de vérité ROME (défaut : ROME 6).")
    parser.add_argument("--campaign-name", default=f"qwen3_embedding_8b_dense_{time.strftime('%Y%m%dT%H%M%S')}", help="Nom du nouveau dossier sous outputs/rome/experiments_2/.")
    args = parser.parse_args()
    batch_size = args.batch_size if args.batch_size is not None else ROME_EXPERIMENT_BATCH_SIZE
    if batch_size < 1:
        raise ValueError("--batch-size doit être strictement positif.")
    started = time.perf_counter()
    prepared = prepare_rome_corpus(pd.read_csv("outputs/rome/corpus_rome.csv"))
    relations, diagnostics = read_rome_relations(args.reference)
    if any(item.severity == "erreur" for item in diagnostics):
        raise RuntimeError("Les relations ROME comportent des erreurs; relancez d'abord la validation du corpus.")
    output = rome_campaign_output_directory(args.campaign_name)
    if output.exists():
        raise FileExistsError(f"Le dossier de campagne existe déjà : {output}")
    client = RemoteOpenAIEmbeddings(QWEN_BASE_URL, api_key=QWEN_API_KEY)
    model_id = client.discover_model()
    print(f"Qwen distant joignable : {client.base_url} (modèle annoncé : {model_id})")
    cache = RomeDenseCache(prepared, f"{model_id}-{client.base_url}", output / "cache")
    vectors, embeddings_cached = load_or_create_embeddings(cache, lambda texts: client.encode(texts, batch_size=batch_size, progress=print), print)
    matrix, matrix_cached = load_or_create_similarity_matrix(cache, vectors, print)
    experiment = run_dense_grid(prepared, relations, matrix, progress=print, model_name="qwen3-embedding-8b")
    outputs = write_dense_experiment_outputs(experiment, prepared, relations, cache, model_id, output, endpoint=client.base_url, batch_size=batch_size, model_name="qwen3-embedding-8b", report_title="Qwen3-Embedding-8B", reference_table=args.reference)
    best = experiment.best_configuration
    print(f"Campagne terminée en {time.perf_counter() - started:.2f}s (lot demandé : {batch_size}; {'embeddings en cache' if embeddings_cached else 'embeddings calculés'}, {'matrice en cache' if matrix_cached else 'matrice calculée'}).")
    print(f"Meilleure configuration : seuil_sim={best['seuil_sim']:.2f}, seuil_couv={best['seuil_couv']:.2f}, F1={best['f1']:.4f}")
    print("Résultats : " + ", ".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
