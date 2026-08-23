from src.domain import Competence, CorrespondanceFournie
from src.matching import analyser_correspondance
from src.recommendations import determiner_besoin_formation


def competence(identifier: str, niveau: int) -> Competence:
    return Competence(identifier, identifier, "Description fictive", niveau)


def test_absent_competence_trains_from_zero_to_required_level() -> None:
    match = analyser_correspondance(
        CorrespondanceFournie(competence("CIBLE", 3), None, 0.69, 0.69)
    )

    need = determiner_besoin_formation(match)

    assert need is not None
    assert need.niveau_depart == 0
    assert need.niveau_cible == 3
    assert need.ecart_niveau == 3
    assert need.motif == "competence_absente"


def test_insufficient_level_trains_only_missing_progression() -> None:
    match = analyser_correspondance(
        CorrespondanceFournie(
            competence("CIBLE", 3), competence("ACTUELLE", 1), 0.71, 0.71
        )
    )

    need = determiner_besoin_formation(match)

    assert need is not None
    assert need.niveau_depart == 1
    assert need.niveau_cible == 3
    assert need.ecart_niveau == 2
    assert need.motif == "niveau_insuffisant"


def test_sufficient_level_has_no_training_need() -> None:
    match = analyser_correspondance(
        CorrespondanceFournie(
            competence("CIBLE", 2), competence("ACTUELLE", 3), 0.70, 0.70
        )
    )

    assert determiner_besoin_formation(match) is None

