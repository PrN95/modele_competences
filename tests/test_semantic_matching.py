from collections.abc import Sequence

import pytest

from src.domain import Competence, CoupleEmplois, Emploi
from src.embeddings import SortieEncodage
from src.matching import (
    calculer_score_sparse,
    generer_correspondances_semantiques,
)
from src.scoring import analyser_couple_semantiquement


def competence(identifier: str, niveau: int = 2) -> Competence:
    return Competence(identifier, identifier, "Description fictive", niveau)


def emploi(identifier: str, job_type: str, skills, effectif=None) -> Emploi:
    return Emploi(
        id=identifier,
        intitule=identifier,
        type=job_type,
        effectif=effectif,
        competences=tuple(skills),
        fichier_source=f"{identifier}.xlsx",
        feuille_source="Competences",
    )


class FauxEncodeur:
    def __init__(self, outputs: dict[tuple[str, ...], SortieEncodage]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[str, ...]] = []

    def encoder(self, competences: Sequence[Competence]) -> SortieEncodage:
        ids = tuple(item.id for item in competences)
        self.calls.append(ids)
        return self.outputs[ids]


def sortie(dense, sparse=None) -> SortieEncodage:
    sparse = sparse or [{} for _ in dense]
    return SortieEncodage(
        vecteurs_dense=tuple(tuple(row) for row in dense),
        poids_sparse=tuple(sparse),
    )


def test_sparse_score_is_dot_product_on_common_tokens() -> None:
    assert calculer_score_sparse({"a": 0.5, "b": 0.2}, {"a": 0.4, "c": 1.0}) == pytest.approx(0.2)


def test_best_current_competence_is_selected_for_each_target() -> None:
    actuelles = (competence("A1"), competence("A2"))
    cibles = (competence("T1"), competence("T2"))
    couple = CoupleEmplois(
        emploi("ACT", "actuel", actuelles, 12),
        emploi("CIB", "cible", cibles),
    )
    encodeur = FauxEncodeur(
        {
            ("A1", "A2"): sortie(((1.0, 0.0), (0.0, 1.0))),
            ("T1", "T2"): sortie(((0.9, 0.1), (0.1, 0.9))),
        }
    )

    correspondances = generer_correspondances_semantiques(couple, encodeur)

    assert [item.competence_actuelle.id for item in correspondances] == ["A1", "A2"]
    assert encodeur.calls == [("A1", "A2"), ("T1", "T2")]


def test_same_current_competence_can_cover_multiple_targets() -> None:
    actuelles = (competence("A1"), competence("A2"))
    cibles = (competence("T1"), competence("T2"))
    couple = CoupleEmplois(
        emploi("ACT", "actuel", actuelles, 12),
        emploi("CIB", "cible", cibles),
    )
    encodeur = FauxEncodeur(
        {
            ("A1", "A2"): sortie(((1.0, 0.0), (0.0, 1.0))),
            ("T1", "T2"): sortie(((0.9, 0.1), (0.8, 0.2))),
        }
    )

    correspondances = generer_correspondances_semantiques(couple, encodeur)

    assert [item.competence_actuelle.id for item in correspondances] == ["A1", "A1"]


def test_candidate_below_threshold_is_kept_and_sent_to_business_scoring() -> None:
    actuelle = competence("A1", niveau=3)
    cible = competence("T1", niveau=2)
    couple = CoupleEmplois(
        emploi("ACT", "actuel", (actuelle,), 12),
        emploi("CIB", "cible", (cible,)),
    )
    encodeur = FauxEncodeur(
        {
            ("A1",): sortie(((1.0, 0.0),), ({"x": 1.0},)),
            ("T1",): sortie(((0.6, 0.8),), ({"x": 0.6},)),
        }
    )

    supplied = generer_correspondances_semantiques(couple, encodeur)
    result = analyser_couple_semantiquement(couple, encodeur)

    assert supplied[0].competence_actuelle == actuelle
    assert supplied[0].score_dense == pytest.approx(0.6)
    assert supplied[0].score_sparse == pytest.approx(0.6)
    assert result.correspondances[0].competence_actuelle == actuelle
    assert result.correspondances[0].reconnue is False
    assert result.correspondances[0].niveau_actuel == 0
    assert result.couverture_semantique == 0.0
    assert result.score_global == 0.0
    assert result.besoins_formation[0].niveau_depart == 0


def test_recognized_candidate_is_transmitted_to_phase_two() -> None:
    actuelle = competence("A1", niveau=3)
    cible = competence("T1", niveau=2)
    couple = CoupleEmplois(
        emploi("ACT", "actuel", (actuelle,), 12),
        emploi("CIB", "cible", (cible,)),
    )
    encodeur = FauxEncodeur(
        {
            ("A1",): sortie(((1.0, 0.0),), ({"x": 1.0},)),
            ("T1",): sortie(((0.9, 0.1),), ({"x": 0.9},)),
        }
    )

    result = analyser_couple_semantiquement(couple, encodeur)

    assert result.correspondances[0].reconnue is True
    assert result.correspondances[0].niveau_actuel == 3
    assert result.score_global == 1.0
    assert result.besoins_formation == ()

