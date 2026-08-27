import pandas as pd
import pytest

from src.validation import ExcelValidationError, validate_competences_dataframe


def test_valid_current_dataframe_builds_domain_object(valid_current_rows) -> None:
    emploi = validate_competences_dataframe(pd.DataFrame(valid_current_rows), "actuel.xlsx")

    assert emploi.id == "EMP-ACT-01"
    assert emploi.type == "actuel"
    assert emploi.effectif == 24
    assert len(emploi.competences) == 2
    assert emploi.fichier_source == "actuel.xlsx"
    assert emploi.feuille_source == "Competences"


def test_target_empty_effectif_becomes_none(valid_target_rows) -> None:
    emploi = validate_competences_dataframe(pd.DataFrame(valid_target_rows), "cible.xlsx")

    assert emploi.effectif is None
    assert emploi.effectif != "nan"


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("emploi_intitule", "\t"),
        ("competence_intitule", "\n"),
    ],
)
def test_whitespace_only_text_is_rejected(valid_current_rows, column, value) -> None:
    valid_current_rows[0][column] = value

    with pytest.raises(ExcelValidationError) as caught:
        validate_competences_dataframe(pd.DataFrame(valid_current_rows), "espaces.xlsx")

    diagnostic = next(item for item in caught.value.diagnostics if item.colonne == column)
    assert diagnostic.fichier == "espaces.xlsx"
    assert diagnostic.feuille == "Competences"
    assert diagnostic.ligne_excel == 2
    assert "espaces" in diagnostic.regle


def test_identifier_columns_are_optional(valid_current_rows) -> None:
    dataframe = pd.DataFrame(valid_current_rows).drop(
        columns=["emploi_id", "competence_id"]
    )

    emploi = validate_competences_dataframe(dataframe, "sans_ids.xlsx")

    assert emploi.id is None
    assert all(competence.id is None for competence in emploi.competences)


def test_blank_identifiers_are_treated_as_absent(valid_current_rows) -> None:
    for row in valid_current_rows:
        row["emploi_id"] = " "
        row["competence_id"] = None

    emploi = validate_competences_dataframe(pd.DataFrame(valid_current_rows), "ids_vides.xlsx")

    assert emploi.id is None
    assert all(competence.id is None for competence in emploi.competences)


def test_description_is_optional(valid_current_rows) -> None:
    dataframe = pd.DataFrame(valid_current_rows).drop(columns=["competence_description"])

    emploi = validate_competences_dataframe(dataframe, "sans_description.xlsx")

    assert emploi.competences[0].description is None
    assert emploi.competences[0].texte == emploi.competences[0].intitule


@pytest.mark.parametrize("effectif", [None, 0, -1, 1.5, "12"])
def test_current_effectif_must_be_a_strictly_positive_integer(valid_current_rows, effectif) -> None:
    valid_current_rows[0]["emploi_effectif"] = effectif

    with pytest.raises(ExcelValidationError) as caught:
        validate_competences_dataframe(pd.DataFrame(valid_current_rows), "effectif.xlsx")

    assert any(item.colonne == "emploi_effectif" for item in caught.value.diagnostics)


def test_current_effectif_must_be_constant(valid_current_rows) -> None:
    valid_current_rows[1]["emploi_effectif"] = 25

    with pytest.raises(ExcelValidationError) as caught:
        validate_competences_dataframe(pd.DataFrame(valid_current_rows), "effectif.xlsx")

    diagnostic = next(
        item
        for item in caught.value.diagnostics
        if item.colonne == "emploi_effectif" and "identique" in item.regle
    )
    assert diagnostic.ligne_excel == 3


def test_target_effectif_can_be_a_constant_integer(valid_target_rows) -> None:
    for row in valid_target_rows:
        row["emploi_effectif"] = 12

    emploi = validate_competences_dataframe(pd.DataFrame(valid_target_rows), "cible.xlsx")

    assert emploi.effectif == 12


def test_target_effectif_cannot_mix_empty_and_populated_rows(valid_target_rows) -> None:
    valid_target_rows[0]["emploi_effectif"] = 12

    with pytest.raises(ExcelValidationError) as caught:
        validate_competences_dataframe(pd.DataFrame(valid_target_rows), "cible.xlsx")

    assert any("vide partout" in item.regle for item in caught.value.diagnostics)


def test_expertise_level_is_accepted(valid_current_rows) -> None:
    valid_current_rows[0]["competence_niveau"] = 4

    emploi = validate_competences_dataframe(pd.DataFrame(valid_current_rows), "expertise.xlsx")

    assert emploi.competences[0].niveau == 4


@pytest.mark.parametrize("niveau", [-1, 0, 5, 1.5, "2", None])
def test_competence_level_must_be_an_integer_from_one_to_four(valid_current_rows, niveau) -> None:
    valid_current_rows[0]["competence_niveau"] = niveau

    with pytest.raises(ExcelValidationError) as caught:
        validate_competences_dataframe(pd.DataFrame(valid_current_rows), "niveau.xlsx")

    assert any(item.colonne == "competence_niveau" for item in caught.value.diagnostics)


def test_job_type_must_be_exact(valid_current_rows) -> None:
    valid_current_rows[0]["emploi_type"] = " Actuel "

    with pytest.raises(ExcelValidationError) as caught:
        validate_competences_dataframe(pd.DataFrame(valid_current_rows), "type.xlsx")

    assert any("valeur exacte" in item.regle for item in caught.value.diagnostics)


def test_missing_required_column_is_rejected(valid_current_rows) -> None:
    dataframe = pd.DataFrame(valid_current_rows).drop(columns=["competence_intitule"])

    with pytest.raises(ExcelValidationError) as caught:
        validate_competences_dataframe(dataframe, "colonne.xlsx")

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.colonne == "competence_intitule"
    assert diagnostic.ligne_excel is None
    assert "colonne obligatoire absente" == diagnostic.regle


def test_empty_sheet_is_rejected() -> None:
    columns = [
        "emploi_intitule",
        "emploi_type",
        "emploi_effectif",
        "competence_intitule",
        "competence_niveau",
    ]

    with pytest.raises(ExcelValidationError) as caught:
        validate_competences_dataframe(pd.DataFrame(columns=columns), "vide.xlsx")

    assert "au moins une compétence" in caught.value.diagnostics[0].regle


def test_diagnostic_string_contains_all_context(valid_current_rows) -> None:
    valid_current_rows[0]["competence_niveau"] = 0

    with pytest.raises(ExcelValidationError) as caught:
        validate_competences_dataframe(pd.DataFrame(valid_current_rows), "contexte.xlsx")

    message = str(caught.value)
    assert "fichier='contexte.xlsx'" in message
    assert "feuille='Competences'" in message
    assert "ligne Excel=2" in message
    assert "colonne='competence_niveau'" in message
    assert "règle non respectée:" in message
