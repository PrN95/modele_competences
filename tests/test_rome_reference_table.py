import pandas as pd

from src.config import ROME_REFERENCE_TABLE
from src.rome import read_rome_relations


def test_rome_6_truth_table_has_expected_balanced_unique_pairs() -> None:
    table = pd.read_excel(ROME_REFERENCE_TABLE, engine="openpyxl")

    assert len(table) == 150
    assert table["similarite"].value_counts().to_dict() == {"Oui": 75, "Non": 75}
    pairs = table.apply(
        lambda row: frozenset((str(row["code_emploi_a"]).strip().upper(), str(row["code_emploi_b"]).strip().upper())),
        axis=1,
    )
    assert pairs.nunique() == 150

    relations, diagnostics = read_rome_relations(ROME_REFERENCE_TABLE)
    assert len(relations.pairs) == 75
    assert len(relations.negative_pairs) == 75
    assert not relations.pairs & relations.negative_pairs
    assert not [diagnostic for diagnostic in diagnostics if diagnostic.severity == "erreur"]
