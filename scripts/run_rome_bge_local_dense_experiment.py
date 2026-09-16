"""Lance la campagne ROME dense-only avec le BGE-M3 installé localement.

Le lot est intentionnellement fixé à une compétence : la campagne est conçue
pour les postes disposant de peu de mémoire et privilégie la stabilité au temps
d'exécution. Les sorties sparse sont demandées au modèle local conformément à
son contrat, mais ne sont pas utilisées par cette campagne dense-only.
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
    RomeDenseCache,
    load_or_create_embeddings,
    load_or_create_similarity_matrix,
    prepare_rome_corpus,
    rome_campaign_output_directory,
    run_dense_grid,
    write_dense_experiment_outputs,
)


def _local_dense_encoder(model_path: Path):
    """Retourne un encodeur séquentiel afin de borner la mémoire vive."""
    adapter = AdaptateurBGEM3(model_path)

    def encode(texts: tuple[str, ...]) -> np.ndarray:
        vectors: list[tuple[float, ...]] = []
        total = len(texts)
        started = time.perf_counter()
        for index, text in enumerate(texts, start=1):
            # Le niveau n'intervient pas dans la campagne ROME.
            result = adapter.encoder((Competence(text, None, 1),))
            vectors.extend(result.vecteurs_dense)
            if index == total or index % 25 == 0:
                print(f"Embeddings BGE-M3 locaux : {index}/{total} compétences, lot 1, durée {time.perf_counter() - started:.1f} s")
        return np.asarray(vectors, dtype=np.float32)

    return encode


def main() -> int:
    parser = argparse.ArgumentParser(description="Campagne ROME BGE-M3 dense-only locale, à faible consommation mémoire.")
    parser.add_argument("--model-path", default=str(MODEL_PATH), help="Dossier local du modèle BGE-M3.")
    parser.add_argument("--reference", default=str(ROME_REFERENCE_TABLE), help="Table de vérité ROME (défaut : ROME 6).")
    parser.add_argument("--campaign-name", default=f"campagne_dense_only_bge_m3_local_{time.strftime('%Y%m%dT%H%M%S')}", help="Nom du nouveau dossier sous outputs/rome/experiments_2/.")
    args = parser.parse_args()
    start = time.perf_counter()
    model_path = Path(args.model_path).expanduser().resolve()
    corpus = pd.read_csv("outputs/rome/corpus_rome.csv")
    prepared = prepare_rome_corpus(corpus)
    relations, diagnostics = read_rome_relations(args.reference)
    errors = [item for item in diagnostics if item.severity == "erreur"]
    if errors:
        raise RuntimeError("Les relations ROME comportent des erreurs; relancez d'abord la validation du corpus.")
    output = rome_campaign_output_directory(args.campaign_name)
    if output.exists():
        raise FileExistsError(f"Le dossier de campagne existe déjà : {output}")
    print(f"BGE-M3 local : {model_path} (encodage séquentiel, lot 1)")
    cache = RomeDenseCache(prepared, f"bge-m3-local-{model_path}", output / "cache")
    vectors, embeddings_cached = load_or_create_embeddings(cache, _local_dense_encoder(model_path), print)
    matrix, matrix_cached = load_or_create_similarity_matrix(cache, vectors, print)
    experiment = run_dense_grid(prepared, relations, matrix, progress=print, model_name="bge-m3-local")
    outputs = write_dense_experiment_outputs(
        experiment, prepared, relations, cache, model_path, output,
        endpoint="local", batch_size=1, model_name="bge-m3-local",
        report_title="BGE-M3 local", reference_table=args.reference,
    )
    best = experiment.best_configuration
    print(f"Campagne terminée en {time.perf_counter() - start:.2f}s (lot 1; {'embeddings en cache' if embeddings_cached else 'embeddings calculés'}, {'matrice en cache' if matrix_cached else 'matrice calculée'}).")
    print(f"Meilleure configuration : seuil_sim={best['seuil_sim']:.2f}, seuil_couv={best['seuil_couv']:.2f}, F1={best['f1']:.4f}")
    print("Résultats : " + ", ".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
