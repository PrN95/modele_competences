import pytest

from src.domain import Competence, CorrespondanceFournie, CoupleEmplois, Emploi
from src.scoring import analyser_couple


def competence(identifier: str, niveau: int) -> Competence:
    return Competence(
        intitule=f"Compétence fictive {identifier}",
        description="Description fictive",
        niveau=niveau,
        id=identifier,
    )


def emploi(identifier: str, job_type: str, skills, effectif=None) -> Emploi:
    return Emploi(
        intitule=f"Emploi fictif {identifier}",
        type=job_type,
        effectif=effectif,
        competences=tuple(skills),
        fichier_source=f"{identifier}.xlsx",
        feuille_source="Competences",
        id=identifier,
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

    # Reconnues: G_ef = 2/3. Entièrement satisfaites: Gs_ef = 1/3.
    assert result.couverture_semantique == pytest.approx(2 / 3)
    assert result.score_global == pytest.approx(2 / 3)
    assert result.score_strict == pytest.approx(1 / 3)
    # Ecart_moyen_ef = (3×3 + 1×2 + 0×2) / (3 + 2 + 2) = 11/7.
    assert result.ecart_moyen == pytest.approx(11 / 7)
    assert result.admissible is False
    assert [item.statut for item in result.correspondances] == [
        "absente",
        "niveau_insuffisant",
        "niveau_suffisant",
    ]
    assert result.besoins_formation == ()


def test_ecart_moyen_est_non_calculable_si_aucun_niveau_n_est_connu() -> None:
    actuelle = competence("ACT", None)
    cible = competence("CIB", None)
    result = analyser_couple(
        CoupleEmplois(actuel=emploi("ACT", "actuel", (actuelle,)), cible=emploi("CIB", "cible", (cible,))),
        (CorrespondanceFournie(cible, actuelle, 0.8, 0.8),),
    )

    assert result.couverture_semantique == 1.0
    assert result.ecart_moyen is None
    assert result.correspondances[0].statut == "niveau_non_renseigne"


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


def test_scoring_does_not_depend_on_optional_identifiers() -> None:
    actuelle = Competence("Compétence actuelle", None, 2)
    cible = Competence("Compétence cible", None, 3)
    pair = CoupleEmplois(
        actuel=Emploi(
            "Emploi actuel",
            "actuel",
            None,
            (actuelle,),
            "actuel.pdf",
            None,
        ),
        cible=Emploi(
            "Emploi cible",
            "cible",
            None,
            (cible,),
            "cible.pdf",
            None,
        ),
    )

    result = analyser_couple(
        pair, (CorrespondanceFournie(cible, actuelle, 0.8, 0.8),)
    )

    assert result.score_global == 1.0
    assert result.score_strict == 0.0
    assert result.ecart_moyen == 1.0


def test_recognized_target_with_all_levels_insufficient_has_full_coverage() -> None:
    actuelles = (competence("ACT-1", 1), competence("ACT-2", 2))
    cibles = (competence("CIB-1", 3), competence("CIB-2", 4))
    pair = CoupleEmplois(
        actuel=emploi("ACTUEL", "actuel", actuelles),
        cible=emploi("CIBLE", "cible", cibles),
    )

    result = analyser_couple(
        pair,
        (
            CorrespondanceFournie(cibles[0], actuelles[0], 0.8, 0.8),
            CorrespondanceFournie(cibles[1], actuelles[1], 0.8, 0.8),
        ),
    )

    assert result.g_ef == 1.0
    assert result.gs_ef == 0.0
    assert result.admissible is True
    assert result.ecart_moyen == pytest.approx((2 * 3 + 2 * 4) / (3 + 4))
