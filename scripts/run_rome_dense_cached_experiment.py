"""Relance une grille ROME dense-only à partir d'un cache déjà calculé.

Cette commande n'instancie aucun client distant : elle échoue si les vecteurs ou
la matrice attendus ne sont pas déjà disponibles et validés pour le corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rome import read_rome_relations
from src.config import ROME_REFERENCE_TABLE
from src.rome_experiment import (
    RomeDenseCache,
    prepare_rome_corpus,
    rome_campaign_output_directory,
    run_dense_grid,
    write_dense_experiment_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Relancer une campagne ROME dense-only sans appel d'embedding.")
    parser.add_argument("--reference", default=str(ROME_REFERENCE_TABLE), help="Table Excel de relations ROME validée (défaut : ROME 6).")
    parser.add_argument("--campaign-name", required=True, help="Nom du nouveau dossier sous outputs/rome/experiments_2/.")
    parser.add_argument("--source-cache", required=True, help="Répertoire de cache de la campagne antérieure.")
    parser.add_argument("--model-id", required=True, help="Identifiant exact ayant servi à nommer le cache.")
    parser.add_argument("--endpoint", required=True, help="Endpoint historique, conservé comme métadonnée.")
    parser.add_argument("--model-name", required=True, help="Nom restitué dans les sorties.")
    parser.add_argument("--report-title", required=True, help="Titre du rapport de synthèse.")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille de lot historique, pour traçabilité.")
    args = parser.parse_args()
    output = rome_campaign_output_directory(args.campaign_name)
    if output.exists():
        raise FileExistsError(f"Le dossier de campagne existe déjà : {output}")

    started = time.perf_counter()
    prepared = prepare_rome_corpus(pd.read_csv("outputs/rome/corpus_rome.csv"))
    relations, diagnostics = read_rome_relations(args.reference)
    errors = [item for item in diagnostics if item.severity == "erreur"]
    if errors:
        raise RuntimeError("Les relations ROME comportent des erreurs : " + "; ".join(item.message for item in errors))

    cache = RomeDenseCache(prepared, f"{args.model_id}-{args.endpoint}", args.source_cache)
    vectors = cache.load_embeddings()
    matrix = cache.load_matrix()
    if vectors is None or matrix is None:
        missing = []
        if vectors is None:
            missing.append(str(cache.embedding_path))
        if matrix is None:
            missing.append(str(cache.matrix_path))
        raise RuntimeError("Cache dense absent ou incompatible ; aucun recalcul n'est autorisé : " + ", ".join(missing))

    print(f"Embeddings réutilisés : {cache.embedding_path}")
    print(f"Matrice de similarités réutilisée : {cache.matrix_path}")
    experiment = run_dense_grid(prepared, relations, matrix, progress=print, model_name=args.model_name)
    outputs = write_dense_experiment_outputs(
        experiment,
        prepared,
        relations,
        cache,
        args.model_id,
        output,
        endpoint=args.endpoint,
        batch_size=args.batch_size,
        model_name=args.model_name,
        report_title=args.report_title,
        reference_table=args.reference,
    )
    metadata_path = output / "metadonnees.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "embeddings_reutilises_depuis_cache": True,
        "matrice_similarites_reutilisee_depuis_cache": True,
        "repertoire_cache_source": str(Path(args.source_cache)),
        "appel_serveur_embedding": False,
    })
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best = experiment.best_configuration
    print(f"Campagne terminée en {time.perf_counter() - started:.2f}s, sans appel serveur.")
    print(f"Meilleure configuration : seuil_sim={best['seuil_sim']:.2f}, seuil_couv={best['seuil_couv']:.2f}, F1={best['f1']:.4f}")
    print("Résultats : " + ", ".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
