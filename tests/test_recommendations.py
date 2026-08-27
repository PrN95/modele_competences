import pytest

from src.domain import Competence, CorrespondanceFournie
from src.matching import analyser_correspondance
from src.recommendations import determiner_besoin_formation


def competence(identifier: str, niveau: int) -> Competence:
    return Competence(identifier, "Description fictive", niveau, id=identifier)


def test_absent_competence_trains_from_zero_to_required_level() -> None:
    match = analyser_correspondance(
        CorrespondanceFournie(competence("CIBLE", 4), None, 0.69, 0.69)
    )

    need = determiner_besoin_formation(match)

    assert need is not None
    assert need.niveau_depart == 0
    assert need.niveau_cible == 4
    assert need.ecart_niveau == 4
    assert need.motif == "competence_absente"
    assert need.recommandation == "formation_complete"


@pytest.mark.parametrize(("niveau_depart", "ecart"), [(1, 2), (1, 3)])
def test_large_gap_recommends_important_training_path(
    niveau_depart: int, ecart: int
) -> None:
    match = analyser_correspondance(
        CorrespondanceFournie(
            competence("CIBLE", niveau_depart + ecart),
            competence("ACTUELLE", niveau_depart),
            0.71,
            0.71,
        )
    )

    need = determiner_besoin_formation(match)

    assert need is not None
    assert need.niveau_depart == niveau_depart
    assert need.niveau_cible == niveau_depart + ecart
    assert need.ecart_niveau == ecart
    assert need.motif == "niveau_insuffisant"
    assert need.recommandation == "parcours_formation_important"


def test_one_level_gap_recommends_single_level_progression() -> None:
    match = analyser_correspondance(
        CorrespondanceFournie(
            competence("CIBLE", 4), competence("ACTUELLE", 3), 0.8, 0.8
        )
    )

    need = determiner_besoin_formation(match)

    assert need is not None
    assert need.recommandation == "progression_un_niveau"


def test_sufficient_level_has_no_training_need() -> None:
    match = analyser_correspondance(
        CorrespondanceFournie(
            competence("CIBLE", 2), competence("ACTUELLE", 3), 0.70, 0.70
        )
    )

    assert determiner_besoin_formation(match) is None
