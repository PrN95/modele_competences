"""Validation du schéma Excel et construction des objets métier."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    COMPETENCE_LEVELS,
    COMPETENCES_SHEET,
    EMPLOI_TYPES,
    REQUIRED_COLUMNS,
)
from src.domain import Competence, Emploi, EmploiType


@dataclass(frozen=True, slots=True)
class ValidationDiagnostic:
    """Description contextualisée d'une règle non respectée."""

    fichier: str
    feuille: str
    colonne: str
    regle: str
    ligne_excel: int | None = None

    def __str__(self) -> str:
        ligne = str(self.ligne_excel) if self.ligne_excel is not None else "non applicable"
        return (
            f"fichier='{self.fichier}', feuille='{self.feuille}', "
            f"ligne Excel={ligne}, colonne='{self.colonne}', règle non respectée: {self.regle}"
        )


class ExcelValidationError(ValueError):
    """Erreur regroupant un ou plusieurs diagnostics de validation."""

    def __init__(self, diagnostics: list[ValidationDiagnostic] | tuple[ValidationDiagnostic, ...]):
        self.diagnostics = tuple(diagnostics)
        message = "Fichier Excel invalide:\n" + "\n".join(
            f"- {diagnostic}" for diagnostic in self.diagnostics
        )
        super().__init__(message)


def _is_missing(value: Any) -> bool:
    if isinstance(value, str):
        return not value.strip()
    return bool(pd.isna(value))


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _as_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric_value = float(value)
    if not numeric_value.is_integer():
        return None
    return int(numeric_value)


def _excel_row(dataframe_index: int) -> int:
    return dataframe_index + 2


def validate_competences_dataframe(
    dataframe: pd.DataFrame,
    fichier: str | Path,
    *,
    expected_type: EmploiType | None = None,
    feuille: str = COMPETENCES_SHEET,
) -> Emploi:
    """Valide une feuille ``Competences`` et construit l'emploi associé."""

    fichier_nom = Path(fichier).name
    diagnostics: list[ValidationDiagnostic] = []

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        diagnostics.extend(
            ValidationDiagnostic(
                fichier=fichier_nom,
                feuille=feuille,
                colonne=column,
                regle="colonne obligatoire absente",
            )
            for column in missing_columns
        )
        raise ExcelValidationError(diagnostics)

    if dataframe.empty:
        raise ExcelValidationError(
            [
                ValidationDiagnostic(
                    fichier=fichier_nom,
                    feuille=feuille,
                    colonne="competence_intitule",
                    regle="le fichier doit contenir au moins une compétence",
                )
            ]
        )

    text_columns = ("emploi_intitule", "competence_intitule")
    valid_text_rows: dict[str, set[int]] = {column: set() for column in text_columns}

    for index, row in dataframe.iterrows():
        row_index = int(index)
        excel_row = _excel_row(row_index)

        for column in text_columns:
            value = row[column]
            if not _is_nonempty_text(value):
                diagnostics.append(
                    ValidationDiagnostic(
                        fichier=fichier_nom,
                        feuille=feuille,
                        ligne_excel=excel_row,
                        colonne=column,
                        regle="valeur texte obligatoire; une valeur composée uniquement d'espaces est vide",
                    )
                )
            else:
                valid_text_rows[column].add(row_index)

        emploi_type = row["emploi_type"]
        if emploi_type not in EMPLOI_TYPES:
            diagnostics.append(
                ValidationDiagnostic(
                    fichier=fichier_nom,
                    feuille=feuille,
                    ligne_excel=excel_row,
                    colonne="emploi_type",
                    regle="valeur exacte attendue: 'actuel' ou 'cible'",
                )
            )

        niveau = _as_integer(row["competence_niveau"])
        if niveau not in COMPETENCE_LEVELS:
            diagnostics.append(
                ValidationDiagnostic(
                    fichier=fichier_nom,
                    feuille=feuille,
                    ligne_excel=excel_row,
                    colonne="competence_niveau",
                    regle="entier obligatoire compris entre 1 et 4",
                )
            )

    _validate_constant_column(
        dataframe,
        "emploi_intitule",
        fichier_nom,
        feuille,
        diagnostics,
        valid_text_rows["emploi_intitule"],
    )
    _validate_constant_column(
        dataframe,
        "emploi_type",
        fichier_nom,
        feuille,
        diagnostics,
        {
            int(index)
            for index, value in dataframe["emploi_type"].items()
            if value in EMPLOI_TYPES
        },
    )
    effectifs = _validate_effectif(dataframe, fichier_nom, feuille, diagnostics)

    first_type = dataframe.iloc[0]["emploi_type"]
    if first_type in EMPLOI_TYPES and expected_type is not None and first_type != expected_type:
        diagnostics.append(
            ValidationDiagnostic(
                fichier=fichier_nom,
                feuille=feuille,
                ligne_excel=2,
                colonne="emploi_type",
                regle=f"ce fichier doit décrire un emploi de type '{expected_type}'",
            )
        )

    if diagnostics:
        raise ExcelValidationError(diagnostics)

    competences = tuple(
        Competence(
            intitule=row["competence_intitule"],
            description=_optional_text(row.get("competence_description")),
            niveau=int(row["competence_niveau"]),
            id=_optional_text(row.get("competence_id")),
        )
        for _, row in dataframe.iterrows()
    )
    effectif = effectifs[0] if effectifs else None

    return Emploi(
        intitule=dataframe.iloc[0]["emploi_intitule"],
        type=first_type,
        effectif=effectif,
        competences=competences,
        fichier_source=fichier_nom,
        feuille_source=feuille,
        id=_optional_text(dataframe.iloc[0].get("emploi_id")),
    )


def _validate_constant_column(
    dataframe: pd.DataFrame,
    column: str,
    fichier: str,
    feuille: str,
    diagnostics: list[ValidationDiagnostic],
    valid_rows: set[int],
) -> None:
    if len(valid_rows) != len(dataframe):
        return
    reference = dataframe.iloc[0][column]
    for index, value in dataframe[column].items():
        if value != reference:
            diagnostics.append(
                ValidationDiagnostic(
                    fichier=fichier,
                    feuille=feuille,
                    ligne_excel=_excel_row(int(index)),
                    colonne=column,
                    regle="la valeur doit être identique sur toutes les lignes du fichier",
                )
            )


def _optional_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    return str(value).strip()


def _validate_effectif(
    dataframe: pd.DataFrame,
    fichier: str,
    feuille: str,
    diagnostics: list[ValidationDiagnostic],
) -> list[int]:
    emploi_type = dataframe.iloc[0]["emploi_type"]
    converted: list[int] = []
    missing_rows: list[int] = []

    for index, value in dataframe["emploi_effectif"].items():
        row_index = int(index)
        if _is_missing(value):
            missing_rows.append(row_index)
            if emploi_type == "actuel":
                diagnostics.append(
                    ValidationDiagnostic(
                        fichier=fichier,
                        feuille=feuille,
                        ligne_excel=_excel_row(row_index),
                        colonne="emploi_effectif",
                        regle="entier strictement positif obligatoire pour un emploi actuel",
                    )
                )
            continue

        integer = _as_integer(value)
        if integer is None:
            diagnostics.append(
                ValidationDiagnostic(
                    fichier=fichier,
                    feuille=feuille,
                    ligne_excel=_excel_row(row_index),
                    colonne="emploi_effectif",
                    regle="la valeur renseignée doit être un entier",
                )
            )
            continue
        if emploi_type == "actuel" and integer <= 0:
            diagnostics.append(
                ValidationDiagnostic(
                    fichier=fichier,
                    feuille=feuille,
                    ligne_excel=_excel_row(row_index),
                    colonne="emploi_effectif",
                    regle="entier strictement positif obligatoire pour un emploi actuel",
                )
            )
            continue
        converted.append(integer)

    if emploi_type == "cible" and missing_rows and len(missing_rows) != len(dataframe):
        for row_index in missing_rows:
            diagnostics.append(
                ValidationDiagnostic(
                    fichier=fichier,
                    feuille=feuille,
                    ligne_excel=_excel_row(row_index),
                    colonne="emploi_effectif",
                    regle="pour une cible, l'effectif doit être vide partout ou renseigné partout",
                )
            )

    if len(converted) > 1:
        reference = converted[0]
        for index, value in dataframe["emploi_effectif"].items():
            integer = None if _is_missing(value) else _as_integer(value)
            if integer is not None and integer != reference:
                diagnostics.append(
                    ValidationDiagnostic(
                        fichier=fichier,
                        feuille=feuille,
                        ligne_excel=_excel_row(int(index)),
                        colonne="emploi_effectif",
                        regle="la valeur doit être identique sur toutes les lignes du fichier",
                    )
                )

    return converted
