"""Lance la campagne 3 ROME depuis le cache hybride local de campagne 2."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODEL_PATH, ROME_REFERENCE_TABLE
from src.rome import read_rome_relations
from src.rome_experiment import RomeHybridCache, prepare_rome_corpus, rome_campaign_output_directory, run_hybrid_grid, write_hybrid_final_outputs

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--reference", default=str(ROME_REFERENCE_TABLE))
    parser.add_argument("--campaign-name", default=f"campagne_hybride_final_bge_m3_local_{time.strftime('%Y%m%dT%H%M%S')}")
    args = parser.parse_args()
    output = rome_campaign_output_directory(args.campaign_name)
    if output.exists(): raise FileExistsError(f"Le dossier existe déjà : {output}")
    prepared = prepare_rome_corpus(pd.read_csv("outputs/rome/corpus_rome.csv"))
    relations, diagnostics = read_rome_relations(args.reference)
    if any(item.severity == "erreur" for item in diagnostics): raise RuntimeError("Table de référence invalide.")
    cache = RomeHybridCache(prepared, f"bge-m3-local-{Path(args.model_path).expanduser().resolve()}", args.source_cache)
    dense, sparse = cache.load_matrix(), cache.load_sparse_matrix()
    if dense is None or sparse is None: raise RuntimeError("Cache dense+sparse absent ou incompatible.")
    experiment = run_hybrid_grid(prepared, relations, dense, sparse, dense_weight=0.7, sparse_weight=0.3, progress=print)
    paths = write_hybrid_final_outputs(experiment, prepared, relations, cache, output, dense_weight=0.7, sparse_weight=0.3, reference_table=args.reference)
    best = experiment.best_configuration
    print(f"Meilleure configuration : seuil_sim={best['seuil_sim']:.2f}, seuil_couv={best['seuil_couv']:.2f}, F1={best['f1']:.4f}")
    print("Résultats : " + ", ".join(map(str, paths)))
    return 0
if __name__ == "__main__": raise SystemExit(main())
