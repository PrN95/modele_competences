from collections.abc import Sequence

import pytest

from src.domain import Competence, CoupleEmplois, Emploi
from src.embeddings import SortieEncodage
from src.matching import (
    calculer_score_sparse,
    generer_correspondances_semantiques,
)
from src.scoring import analyser_couple, analyser_couple_semantiquement


def competence(identifier: str, niveau: int = 2) -> Competence:
    return Competence(identifier, "Description fictive", niveau, id=identifier)


def emploi(identifier: str, job_type: str, skills, effectif=None) -> Emploi:
    return Emploi(
        intitule=identifier,
        type=job_type,
        effectif=effectif,
        competences=tuple(skills),
        fichier_source=f"{identifier}.xlsx",
        feuille_source="Competences",
        id=identifier,
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
    assert result.score_strict == 0.0
    assert result.besoins_formation == ()


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
    assert result.score_strict == 1.0
    assert result.besoins_formation == ()


def test_exact_hybrid_tie_is_broken_by_highest_current_level() -> None:
    actuelles = (competence("A-BAS", 1), competence("A-HAUT", 4))
    cible = competence("T", 2)
    couple = CoupleEmplois(
        emploi("ACT", "actuel", actuelles),
        emploi("CIB", "cible", (cible,)),
    )
    encodeur = FauxEncodeur(
        {
            ("A-BAS", "A-HAUT"): sortie(
                ((1.0, 0.0), (0.0, 1.0)),
                ({"x": 1.0}, {"y": 1.0}),
            ),
            ("T",): sortie(((0.8, 0.8),), ({"x": 0.8, "y": 0.8},)),
        }
    )

    result = generer_correspondances_semantiques(couple, encodeur)[0]

    assert result.competence_actuelle == actuelles[1]
    assert result.detail_egalite is None


def test_exact_hybrid_and_level_tie_is_broken_by_highest_sparse_score() -> None:
    actuelles = (competence("A-SPARSE-BAS", 2), competence("A-SPARSE-HAUT", 2))
    cible = competence("T", 2)
    couple = CoupleEmplois(
        emploi("ACT", "actuel", actuelles),
        emploi("CIB", "cible", (cible,)),
    )
    encodeur = FauxEncodeur(
        {
            ("A-SPARSE-BAS", "A-SPARSE-HAUT"): sortie(
                ((1.0, 0.0), (0.0, 1.0)),
                ({"bas": 0.3}, {"haut": 0.6}),
            ),
            ("T",): sortie(
                ((0.9, 0.75),),
                ({"bas": 1.0, "haut": 1.0},),
            ),
        }
    )

    result = generer_correspondances_semantiques(couple, encodeur)[0]

    assert result.score_dense == pytest.approx(0.75)
    assert result.score_sparse == pytest.approx(0.6)
    assert result.competence_actuelle == actuelles[1]
    assert result.detail_egalite is None


def test_persistent_candidate_tie_is_deterministic_and_reported() -> None:
    actuelles = (competence("Zulu", 2), competence("Alpha", 2))
    cible = competence("T", 2)
    couple = CoupleEmplois(
        emploi("ACT", "actuel", actuelles),
        emploi("CIB", "cible", (cible,)),
    )
    encodeur = FauxEncodeur(
        {
            ("Zulu", "Alpha"): sortie(
                ((1.0, 0.0), (0.0, 1.0)),
                ({"z": 0.5}, {"a": 0.5}),
            ),
            ("T",): sortie(((0.8, 0.8),), ({"z": 1.0, "a": 1.0},)),
        }
    )

    supplied = generer_correspondances_semantiques(couple, encodeur)[0]
    detailed = analyser_couple(couple, (supplied,)).correspondances[0]

    assert supplied.competence_actuelle == actuelles[1]
    assert supplied.detail_egalite is not None
    assert "Égalité exacte persistante" in supplied.detail_egalite
    assert "choix déterministe : « Alpha »" in supplied.detail_egalite
    assert detailed.detail_egalite == supplied.detail_egalite
