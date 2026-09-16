from pathlib import Path
import importlib.util

import pandas as pd


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "compare_rome_reference_tables.py"
    spec = importlib.util.spec_from_file_location("compare_rome_reference_tables", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_comparison_reports_missing_required_title_column_but_compares_code_pairs(tmp_path: Path) -> None:
    module = _load_module()
    old = tmp_path / "old.xlsx"
    new = tmp_path / "new.xlsx"
    corpus = tmp_path / "corpus.csv"
    pd.DataFrame({
        "Intitulé emploi actuel": ["A", "B"],
        "Code ROME emploi actuel": ["m1000", "M2000"],
        "Intitulé emploi cible": ["B", "C"],
        "Code ROME emploi cible": ["M2000", "M3000"],
    }).to_excel(old, index=False)
    pd.DataFrame({
        "Intitulé emploi actuel": ["A", "B"],
        "Code ROME emploi actuel": ["M1000", "M2000"],
        "Intitulé emploi proche": ["C", "A"],
        "Code ROME emploi cible": ["M3000", "M1000"],
    }).to_excel(new, index=False)
    pd.DataFrame({"code_rome": ["M1000", "M2000", "M3000"]}).to_csv(corpus, index=False)

    report, details, blocking = module.compare(old, new, corpus)

    assert blocking
    assert "Colonnes absentes : `Intitulé emploi cible`" in report
    assert "Ajoutées : 1 ; supprimées : 1 ; conservées : 1" in report
    assert set(details["changement"]) == {"ajoutee", "supprimee", "conservee"}
