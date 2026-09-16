"""Commande locale de validation et de génération du corpus ROME."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Autorise l'exécution directe depuis la racine du dépôt, sans installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rome import build_rome_corpus, write_rome_outputs
from src.config import ROME_REFERENCE_TABLE


def main() -> int:
    parser = argparse.ArgumentParser(description="Valider les données ROME et générer le corpus expérimental.")
    parser.add_argument("--excel", default=str(ROME_REFERENCE_TABLE))
    parser.add_argument("--pdf-dir", default="data/rome/Fiches emploi ROME")
    parser.add_argument("--output-dir", default="outputs/rome")
    args = parser.parse_args()
    result = build_rome_corpus(args.excel, args.pdf_dir)
    corpus, report = write_rome_outputs(result, args.output_dir)
    print(f"Corpus généré : {corpus}")
    print(f"Rapport généré : {report}")
    print(f"Diagnostics : {len(result.diagnostics)}")
    return 1 if result.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
