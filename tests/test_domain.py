import pytest

from src.domain import Competence


def test_competence_text_uses_only_title_and_description() -> None:
    competence = Competence(
        intitule="Intitulé fictif",
        description="Description fictive",
        niveau=2,
        id="COMP-01",
    )

    assert competence.texte == "Intitulé fictif. Description fictive"
    assert str(competence.niveau) not in competence.texte


def test_competence_can_be_created_without_identifier_or_description() -> None:
    competence = Competence(intitule="Intitulé complet", description=None, niveau=4)

    assert competence.id is None
    assert competence.texte == "Intitulé complet"


@pytest.mark.parametrize("niveau", [-1, 0, 5])
def test_domain_rejects_levels_outside_one_to_four(niveau: int) -> None:
    with pytest.raises(ValueError, match="compris entre 1 et 4"):
        Competence(intitule="Compétence", description=None, niveau=niveau)


def test_optional_identifier_is_excluded_from_domain_equality() -> None:
    sans_id = Competence("Compétence", None, 2)
    avec_id = Competence("Compétence", None, 2, id="META-01")

    assert sans_id == avec_id
