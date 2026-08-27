import pytest

from src.config import DENSE_WEIGHT, SEMANTIC_MATCH_THRESHOLD, SPARSE_WEIGHT
from src.domain import Competence, CorrespondanceFournie
from src.matching import analyser_correspondance, calculer_score_hybride


def competence(identifier: str, niveau: int) -> Competence:
    return Competence(
        intitule=f"Compétence fictive {identifier}",
        description="Description fictive",
        niveau=niveau,
        id=identifier,
    )


def test_weights_and_threshold_are_exact_fractions() -> None:
    assert DENSE_WEIGHT.numerator == 2 and DENSE_WEIGHT.denominator == 3
    assert SPARSE_WEIGHT.numerator == 1 and SPARSE_WEIGHT.denominator == 3
    assert SEMANTIC_MATCH_THRESHOLD.numerator == 7
    assert SEMANTIC_MATCH_THRESHOLD.denominator == 10


def test_hybrid_score_uses_exact_normalized_weights() -> None:
    assert calculer_score_hybride(1.0, 0.0) == pytest.approx(2 / 3)
    assert calculer_score_hybride(0.0, 1.0) == pytest.approx(1 / 3)
    assert calculer_score_hybride(0.9, 0.3) == 0.70


@pytest.mark.parametrize(
    ("score_sparse", "expected_hybrid", "expected_recognized"),
    [
        (0.2999999999999997, 0.6999999999999999, False),
        (0.3, 0.7, True),
        (0.3000000000000003, 0.7000000000000001, True),
    ],
    ids=["immediately-below", "exactly-equal", "immediately-above"],
)
def test_mixed_scores_around_exact_threshold(
    score_sparse: float,
    expected_hybrid: float,
    expected_recognized: bool,
) -> None:
    supplied = CorrespondanceFournie(
        competence_cible=competence("CIBLE", 2),
        competence_actuelle=competence("ACTUELLE", 2),
        score_dense=0.9,
        score_sparse=score_sparse,
    )

    result = analyser_correspondance(supplied)

    assert result.score_hybride == expected_hybrid
    assert result.reconnue is expected_recognized


@pytest.mark.parametrize(
    ("score", "expected_recognized"),
    [(0.69, False), (0.70, True), (0.71, True)],
)
def test_semantic_thresholds(score: float, expected_recognized: bool) -> None:
    supplied = CorrespondanceFournie(
        competence_cible=competence("CIBLE", 2),
        competence_actuelle=competence("ACTUELLE", 2),
        score_dense=score,
        score_sparse=score,
    )

    result = analyser_correspondance(supplied)

    assert result.score_hybride == pytest.approx(score)
    assert result.reconnue is expected_recognized


def test_level_never_changes_hybrid_score() -> None:
    cible = competence("CIBLE", 3)
    low_level = analyser_correspondance(
        CorrespondanceFournie(cible, competence("BAS", 1), 0.8, 0.5)
    )
    high_level = analyser_correspondance(
        CorrespondanceFournie(cible, competence("HAUT", 3), 0.8, 0.5)
    )

    assert low_level.score_hybride == high_level.score_hybride
    assert low_level.niveau_actuel == 1
    assert high_level.niveau_actuel == 3


def test_absent_competence_has_level_zero_and_full_gap() -> None:
    result = analyser_correspondance(
        CorrespondanceFournie(
            competence("CIBLE", 3),
            competence("CANDIDATE", 3),
            0.69,
            0.69,
        )
    )

    assert result.reconnue is False
    assert result.niveau_actuel == 0
    assert result.ecart_niveau == 3
    assert result.statut == "absente"


def test_expertise_target_gap_is_calculated_after_semantic_matching() -> None:
    result = analyser_correspondance(
        CorrespondanceFournie(
            competence("CIBLE", 4),
            competence("ACTUELLE", 1),
            0.8,
            0.8,
        )
    )

    assert result.reconnue is True
    assert result.ecart_niveau == 3


def test_recognized_match_requires_current_competence() -> None:
    supplied = CorrespondanceFournie(competence("CIBLE", 2), None, 0.70, 0.70)

    with pytest.raises(ValueError, match="compétence actuelle"):
        analyser_correspondance(supplied)
