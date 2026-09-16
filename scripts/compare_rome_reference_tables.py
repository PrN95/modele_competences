"""Valide et compare deux tables de relations ROME sans modifier le corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rome import ROME_REQUIRED_COLUMNS, normalize_rome_code
from src.config import ROME_REFERENCE_TABLE


CURRENT_CODE = "Code ROME emploi actuel"
TARGET_CODE = "Code ROME emploi cible"


def _read_pairs(path: Path) -> tuple[set[tuple[str, str]], set[str], list[dict[str, object]], list[str], list[str], list[str]]:
    dataframe = pd.read_excel(path, engine="openpyxl")
    columns = [str(column) for column in dataframe.columns]
    missing = [column for column in ROME_REQUIRED_COLUMNS if column not in columns]
    errors: list[str] = []
    pairs: set[tuple[str, str]] = set()
    codes: set[str] = set()
    details: list[dict[str, object]] = []
    duplicates: list[str] = []
    if CURRENT_CODE not in columns or TARGET_CODE not in columns:
        errors.append("Colonnes de codes ROME nécessaires absentes ; les relations ne peuvent pas être lues.")
        return pairs, codes, details, missing, errors, duplicates
    for index, row in dataframe.iterrows():
        current, target = normalize_rome_code(row[CURRENT_CODE]), normalize_rome_code(row[TARGET_CODE])
        line = int(index) + 2
        if current is None or target is None:
            errors.append(f"Ligne {line} : code ROME actuel ou cible absent/invalide.")
            continue
        codes.update((current, target))
        if current == target:
            errors.append(f"Ligne {line} : relation réflexive {current} ignorée.")
            continue
        pair = tuple(sorted((current, target)))
        if pair in pairs:
            duplicates.append(f"ligne {line} : {pair[0]} — {pair[1]}")
        details.append({"ligne": line, "code_rome_a": pair[0], "code_rome_b": pair[1]})
        pairs.add(pair)
    return pairs, codes, details, missing, errors, duplicates


def compare(old_path: Path, new_path: Path, corpus_path: Path) -> tuple[str, pd.DataFrame, bool]:
    corpus = pd.read_csv(corpus_path)
    corpus_codes = set(corpus["code_rome"].astype(str))
    old_pairs, old_codes, _, old_missing, old_errors, old_duplicates = _read_pairs(old_path)
    new_pairs, new_codes, _, new_missing, new_errors, new_duplicates = _read_pairs(new_path)
    added, removed, kept = new_pairs - old_pairs, old_pairs - new_pairs, old_pairs & new_pairs
    rows = ([{"changement": "ajoutee", "code_rome_a": a, "code_rome_b": b} for a, b in sorted(added)]
            + [{"changement": "supprimee", "code_rome_a": a, "code_rome_b": b} for a, b in sorted(removed)]
            + [{"changement": "conservee", "code_rome_a": a, "code_rome_b": b} for a, b in sorted(kept)])
    details = pd.DataFrame(rows, columns=("changement", "code_rome_a", "code_rome_b"))
    new_extra, new_absent = sorted(new_codes - corpus_codes), sorted(corpus_codes - new_codes)
    blocking = bool(new_missing or new_errors or new_extra or new_absent)
    added_lines = [f"- {a} — {b}" for a, b in sorted(added)] or ["- Aucune."]
    removed_lines = [f"- {a} — {b}" for a, b in sorted(removed)] or ["- Aucune."]
    kept_lines = [f"- {a} — {b}" for a, b in sorted(kept)] or ["- Aucune."]
    lines = [
        "# Comparaison des tables de référence ROME",
        "",
        f"- Ancienne table : `{old_path}`",
        f"- Nouvelle table : `{new_path}`",
        f"- Corpus comparé : `{corpus_path}` ({len(corpus_codes)} fiches)",
        "",
        "## Validation de la nouvelle table",
        "",
        f"- Colonnes attendues : {', '.join(f'`{column}`' for column in ROME_REQUIRED_COLUMNS)}.",
        f"- Colonnes absentes : {', '.join(f'`{column}`' for column in new_missing) if new_missing else 'aucune'}.",
        f"- Codes présents dans la nouvelle table mais sans fiche du corpus : {', '.join(new_extra) if new_extra else 'aucun'}.",
        f"- Fiches du corpus sans code dans la nouvelle table : {', '.join(new_absent) if new_absent else 'aucune'}.",
        f"- Anomalies de normalisation/réflexivité : {' ; '.join(new_errors) if new_errors else 'aucune'}.",
        f"- Doublons non directionnels : {' ; '.join(new_duplicates) if new_duplicates else 'aucun'}. Ils sont dédoublonnés pour l'évaluation.",
        "",
        "## Relations symétriques dédoublonnées",
        "",
        f"- Ancienne : {len(old_pairs)} relations de référence ({len(old_codes)} codes).",
        f"- Nouvelle : {len(new_pairs)} relations de référence ({len(new_codes)} codes).",
        f"- Ajoutées : {len(added)} ; supprimées : {len(removed)} ; conservées : {len(kept)}.",
        "",
        "## Relations ajoutées",
        "",
        *added_lines,
        "",
        "## Relations supprimées",
        "",
        *removed_lines,
        "",
        "## Relations conservées",
        "",
        *kept_lines,
        "",
        "## Décision",
        "",
        ("**BLOQUANT — campagnes non lancées.** Corriger les colonnes/anomalies listées puis relancer cette comparaison."
         if blocking else "**VALIDE — la nouvelle table peut être utilisée pour les campagnes.**"),
    ]
    if old_missing or old_errors:
        lines += ["", "Note ancienne table : " + "; ".join([*(f"colonnes absentes : {', '.join(old_missing)}" for _ in [0] if old_missing), *old_errors])]
    if old_duplicates:
        lines += ["", "Note ancienne table : doublons non directionnels dédoublonnés — " + "; ".join(old_duplicates) + "."]
    return "\n".join(lines) + "\n", details, blocking


def main() -> int:
    parser = argparse.ArgumentParser(description="Comparer une ancienne table explicitement fournie à la référence ROME courante.")
    parser.add_argument("--old", required=True, help="Ancienne table à comparer (doit être fournie explicitement).")
    parser.add_argument("--new", default=str(ROME_REFERENCE_TABLE))
    parser.add_argument("--corpus", default="outputs/rome/corpus_rome.csv")
    parser.add_argument("--report", default="outputs/rome/comparaison_tables_reference_rome.md")
    parser.add_argument("--details", default="outputs/rome/comparaison_tables_reference_rome.csv")
    args = parser.parse_args()
    report, details, blocking = compare(Path(args.old), Path(args.new), Path(args.corpus))
    report_path, detail_path = Path(args.report), Path(args.details)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    details.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"Rapport : {report_path}")
    print(f"Détail : {detail_path}")
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
