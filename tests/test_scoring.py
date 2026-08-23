import pytest

from src.domain import Competence, CorrespondanceFournie, CoupleEmplois, Emploi
from src.scoring import analyser_couple


def competence(identifier: str, niveau: int) -> Competence:
    return Competence(identifier, f"Compétence fictive {identifier}", "Description fictive", niveau)


def emploi(identifier: str, job_type: str, skills, effectif=None) -> Emploi:
    return Emploi(
        id=identifier,
        intitule=f"Emploi fictif {identifier}",
        type=job_type,
        effectif=effectif,
        competences=tuple(skills),
        fichier_source=f"{identifier}.xlsx",
        feuille_source="Competences",
    )


def test_complete_manually_verifiable_example() -> None:
    current_skills = (
        competence("ACT-ABS", 3),
        competence("ACT-LOW", 1),
        competence("ACT-OK", 3),
    )
    target_skills = (
        competence("CIB-ABS", 3),
        competence("CIB-LOW", 2),
        competence("CIB-OK", 2),
    )
    pair = CoupleEmplois(
        actuel=emploi("ACTUEL", "actuel", current_skills, effectif=24),
        cible=emploi("CIBLE", "cible", target_skills),
    )
    supplied = (
        CorrespondanceFournie(target_skills[0], current_skills[0], 0.69, 0.69),
        CorrespondanceFournie(target_skills[1], current_skills[1], 0.70, 0.70),
        CorrespondanceFournie(target_skills[2], current_skills[2], 0.71, 0.71),
    )

    result = analyser_couple(pair, supplied)

    # Reconnues: 2/3. Niveau suffisant: 1/3.
    assert result.couverture_semantique == pytest.approx(2 / 3)
    assert result.score_global == pytest.approx(1 / 3)
    # Contributions D_n: absente 3/3 = 1, insuffisante 1/2, suffisante 0/2 = 0.
    assert result.ecart_moyen_normalise == pytest.approx((1 + 0.5 + 0) / 3)
    assert result.ecart_moyen_normalise == pytest.approx(0.5)
    assert result.besoin_collectif == pytest.approx(24 * 0.5)
    assert result.besoin_collectif == pytest.approx(12.0)
    assert [item.statut for item in result.correspondances] == [
        "absente",
        "niveau_insuffisant",
        "niveau_suffisant",
    ]
    assert len(result.besoins_formation) == 2


def test_every_target_competence_must_have_exactly_one_supplied_match() -> None:
    current_skill = competence("ACT", 2)
    target_skills = (competence("CIB-1", 2), competence("CIB-2", 2))
    pair = CoupleEmplois(
        actuel=emploi("ACTUEL", "actuel", (current_skill,), 12),
        cible=emploi("CIBLE", "cible", target_skills),
    )

    with pytest.raises(ValueError, match="CIB-2"):
        analyser_couple(
            pair,
            (CorrespondanceFournie(target_skills[0], current_skill, 0.8, 0.8),),
        )


def test_supplied_current_competence_must_belong_to_current_job() -> None:
    current_skill = competence("ACT", 2)
    target_skill = competence("CIB", 2)
    foreign_skill = competence("AUTRE", 2)
    pair = CoupleEmplois(
        actuel=emploi("ACTUEL", "actuel", (current_skill,), 12),
        cible=emploi("CIBLE", "cible", (target_skill,)),
    )

    with pytest.raises(ValueError, match="n'appartient pas"):
        analyser_couple(
            pair,
            (CorrespondanceFournie(target_skill, foreign_skill, 0.8, 0.8),),
        )


def test_results_follow_target_competence_order() -> None:
    current_skills = (competence("ACT-1", 2), competence("ACT-2", 2))
    target_skills = (competence("CIB-1", 2), competence("CIB-2", 2))
    pair = CoupleEmplois(
        actuel=emploi("ACTUEL", "actuel", current_skills, 12),
        cible=emploi("CIBLE", "cible", target_skills),
    )
    supplied_in_reverse_order = (
        CorrespondanceFournie(target_skills[1], current_skills[1], 0.8, 0.8),
        CorrespondanceFournie(target_skills[0], current_skills[0], 0.8, 0.8),
    )

    result = analyser_couple(pair, supplied_in_reverse_order)

    assert [item.competence_cible.id for item in result.correspondances] == [
        "CIB-1",
        "CIB-2",
    ]
