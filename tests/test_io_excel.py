import pytest

from src.io_excel import read_emploi, read_emploi_pair
from src.validation import ExcelValidationError


def test_read_valid_pair(write_workbook, valid_current_rows, valid_target_rows) -> None:
    current_path = write_workbook("actuel.xlsx", valid_current_rows)
    target_path = write_workbook("cible.xlsx", valid_target_rows)

    pair = read_emploi_pair(current_path, target_path)

    assert pair.actuel.type == "actuel"
    assert pair.actuel.effectif == 24
    assert pair.cible.type == "cible"
    assert pair.cible.effectif is None


def test_wrong_sheet_name_is_rejected(write_workbook, valid_current_rows) -> None:
    path = write_workbook("mauvaise_feuille.xlsx", valid_current_rows, sheet_name="competences")

    with pytest.raises(ExcelValidationError) as caught:
        read_emploi(path)

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.fichier == "mauvaise_feuille.xlsx"
    assert diagnostic.feuille == "Competences"
    assert diagnostic.colonne == "feuille"
    assert diagnostic.ligne_excel is None


def test_non_xlsx_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "emploi.csv"

    with pytest.raises(ExcelValidationError) as caught:
        read_emploi(path)

    assert caught.value.diagnostics[0].regle == "le format obligatoire est .xlsx"


def test_missing_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "absent.xlsx"

    with pytest.raises(ExcelValidationError) as caught:
        read_emploi(path)

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.fichier == "absent.xlsx"
    assert "classeur illisible" in diagnostic.regle


def test_pair_rejects_reversed_job_types(write_workbook, valid_current_rows, valid_target_rows) -> None:
    target_path = write_workbook("premier.xlsx", valid_target_rows)
    current_path = write_workbook("second.xlsx", valid_current_rows)

    with pytest.raises(ExcelValidationError) as caught:
        read_emploi_pair(target_path, current_path)

    diagnostic = next(item for item in caught.value.diagnostics if item.colonne == "emploi_type")
    assert diagnostic.ligne_excel == 2
    assert "type 'actuel'" in diagnostic.regle

