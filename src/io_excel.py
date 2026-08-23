"""Lecture des fichiers Excel de la phase 1."""

from pathlib import Path

import pandas as pd

from src.config import COMPETENCES_SHEET
from src.domain import CoupleEmplois, Emploi, EmploiType
from src.validation import ExcelValidationError, ValidationDiagnostic, validate_competences_dataframe


def read_emploi(path: str | Path, *, expected_type: EmploiType | None = None) -> Emploi:
    """Lit et valide un fichier ``.xlsx`` représentant un emploi."""

    source = Path(path)
    if source.suffix.lower() != ".xlsx":
        raise _io_error(source, "fichier", "le format obligatoire est .xlsx")

    try:
        workbook = pd.ExcelFile(source, engine="openpyxl")
    except (FileNotFoundError, OSError, ValueError) as error:
        raise _io_error(source, "fichier", f"classeur illisible: {error}") from error

    if COMPETENCES_SHEET not in workbook.sheet_names:
        raise _io_error(
            source,
            "feuille",
            f"la feuille obligatoire '{COMPETENCES_SHEET}' est absente",
        )

    try:
        dataframe = pd.read_excel(workbook, sheet_name=COMPETENCES_SHEET)
    except (OSError, ValueError) as error:
        raise _io_error(source, "feuille", f"feuille illisible: {error}") from error

    return validate_competences_dataframe(
        dataframe,
        source,
        expected_type=expected_type,
        feuille=COMPETENCES_SHEET,
    )


def read_emploi_pair(actuel_path: str | Path, cible_path: str | Path) -> CoupleEmplois:
    """Lit le couple un emploi actuel / un emploi cible du premier incrément."""

    return CoupleEmplois(
        actuel=read_emploi(actuel_path, expected_type="actuel"),
        cible=read_emploi(cible_path, expected_type="cible"),
    )


def _io_error(source: Path, colonne: str, regle: str) -> ExcelValidationError:
    return ExcelValidationError(
        [
            ValidationDiagnostic(
                fichier=source.name,
                feuille=COMPETENCES_SHEET,
                colonne=colonne,
                regle=regle,
            )
        ]
    )

