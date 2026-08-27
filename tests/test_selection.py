from collections.abc import Sequence

import pytest

from src.domain import Competence, CorrespondanceFournie, CoupleEmplois, Emploi
from src.embeddings import SortieEncodage
from src.matching import controler_competences_actuelles_non_reprises
from src.scoring import analyser_couple, selectionner_cibles


def competence(nom: str, niveau: int) -> Competence:
    return Competence(nom, None, niveau)


def emploi(nom: str, type_emploi: str, competences) -> Emploi:
    return Emploi(
        intitule=nom,
        type=type_emploi,
        effectif=None,
        competences=tuple(competences),
        fichier_source=f"{nom}.pdf",
        feuille_source=None,
    )


def analyser_cible(actuel: Emploi, cible: Emploi, scores) -> object:
    correspondances = tuple(
        CorrespondanceFournie(competence_cible, competence_actuelle, dense, sparse)
        for competence_cible, competence_actuelle, dense, sparse in scores
    )
    return analyser_couple(CoupleEmplois(actuel, cible), correspondances)


def test_equal_global_scores_are_broken_by_lower_average_gap() -> None:
    actuelle = competence("Compétence actuelle", 3)
    actuel = emploi("Actuel", "actuel", (actuelle,))
    cible_proche = emploi("Cible proche", "cible", (competence("Cible A", 4),))
    cible_lointaine = emploi("Cible lointaine", "cible", (competence("Cible B", 4),))
    proche = analyser_cible(
        actuel,
        cible_proche,
        ((cible_proche.competences[0], actuelle, 0.8, 0.8),),
    )
    lointaine = analyser_cible(
        actuel,
        cible_lointaine,
        ((cible_lointaine.competences[0], actuelle, 0.6, 0.6),),
    )

    selection = selectionner_cibles((lointaine, proche))

    assert proche.score_global == lointaine.score_global == 0.0
    assert proche.ecart_moyen == 1.0
    assert lointaine.ecart_moyen == 4.0
    assert selection.meilleures_analyses == (proche,)


def test_highest_global_score_is_selected_first() -> None:
    actuelle = competence("Compétence actuelle", 2)
    actuel = emploi("Actuel", "actuel", (actuelle,))
    cible_satisfaite = emploi(
        "Cible satisfaite", "cible", (competence("Exigence satisfaite", 2),)
    )
    cible_non_satisfaite = emploi(
        "Cible non satisfaite", "cible", (competence("Exigence non satisfaite", 2),)
    )
    satisfaite = analyser_cible(
        actuel,
        cible_satisfaite,
        ((cible_satisfaite.competences[0], actuelle, 0.8, 0.8),),
    )
    non_satisfaite = analyser_cible(
        actuel,
        cible_non_satisfaite,
        ((cible_non_satisfaite.competences[0], actuelle, 0.6, 0.6),),
    )

    selection = selectionner_cibles((non_satisfaite, satisfaite))

    assert satisfaite.score_global == 1.0
    assert non_satisfaite.score_global == 0.0
    assert selection.meilleures_analyses == (satisfaite,)


def test_persistent_tie_keeps_and_flags_all_targets() -> None:
    actuelle = competence("Compétence actuelle", 2)
    actuel = emploi("Actuel", "actuel", (actuelle,))
    cibles = tuple(
        emploi(nom, "cible", (competence(f"Compétence {nom}", 2),))
        for nom in ("Cible A", "Cible B")
    )
    analyses = tuple(
        analyser_cible(
            actuel,
            cible,
            ((cible.competences[0], actuelle, 0.8, 0.8),),
        )
        for cible in cibles
    )

    selection = selectionner_cibles(analyses)

    assert selection.ex_aequo is True
    assert selection.meilleures_analyses == analyses
    assert selection.alerte is None


def test_low_global_score_target_is_kept_with_required_alert() -> None:
    actuelle = competence("Compétence actuelle", 1)
    actuel = emploi("Actuel", "actuel", (actuelle,))
    cible = emploi("Cible", "cible", (competence("Compétence cible", 4),))
    analyse = analyser_cible(
        actuel,
        cible,
        ((cible.competences[0], actuelle, 0.8, 0.8),),
    )

    selection = selectionner_cibles((analyse,))

    assert analyse.score_global == 0.0
    assert selection.meilleures_analyses == (analyse,)
    assert selection.alerte == "correspondance globale faible"


class FauxEncodeur:
    def __init__(self, sorties: dict[tuple[str, ...], SortieEncodage]) -> None:
        self.sorties = sorties

    def encoder(self, competences: Sequence[Competence]) -> SortieEncodage:
        return self.sorties[tuple(item.intitule for item in competences)]


def sortie(dense, sparse) -> SortieEncodage:
    return SortieEncodage(
        vecteurs_dense=tuple(tuple(row) for row in dense),
        poids_sparse=tuple(sparse),
    )


def test_inverse_control_distinguishes_other_target_and_all_targets_without_scores() -> None:
    actuelles = (
        competence("Actuelle sélectionnée", 2),
        competence("Actuelle autre cible", 2),
        competence("Actuelle non reprise", 2),
    )
    emploi_actuel = emploi("Actuel", "actuel", actuelles)
    cible_selectionnee = emploi(
        "Cible sélectionnée", "cible", (competence("Cible sélectionnée", 2),)
    )
    autre_cible = emploi("Autre cible", "cible", (competence("Cible autre", 2),))
    encodeur = FauxEncodeur(
        {
            tuple(item.intitule for item in actuelles): sortie(
                ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                ({"a": 1}, {"b": 1}, {"c": 1}),
            ),
            ("Cible sélectionnée",): sortie(((1, 0, 0),), ({"a": 1},)),
            ("Cible autre",): sortie(((0, 1, 0),), ({"b": 1},)),
        }
    )

    analyse_selectionnee = analyser_cible(
        emploi_actuel,
        cible_selectionnee,
        ((cible_selectionnee.competences[0], actuelles[0], 1.0, 1.0),),
    )
    analyse_autre = analyser_cible(
        emploi_actuel,
        autre_cible,
        ((autre_cible.competences[0], actuelles[1], 1.0, 1.0),),
    )
    scores_avant = (
        analyse_selectionnee.score_global,
        analyse_selectionnee.ecart_moyen,
        analyse_autre.score_global,
        analyse_autre.ecart_moyen,
    )
    selection_avant = selectionner_cibles((analyse_selectionnee, analyse_autre))

    signalements = controler_competences_actuelles_non_reprises(
        emploi_actuel,
        (cible_selectionnee, autre_cible),
        (cible_selectionnee,),
        encodeur,
    )

    assert [item.competence_actuelle for item in signalements] == list(actuelles[1:])
    assert signalements[0].type_non_reprise == "presente_dans_une_autre_cible"
    assert signalements[0].meilleur_emploi_cible == autre_cible
    assert signalements[1].type_non_reprise == "absente_de_toutes_les_cibles"
    assert (
        signalements[1].message
        == "compétence actuelle non reprise dans les emplois cibles analysés"
    )
    assert scores_avant == (
        analyse_selectionnee.score_global,
        analyse_selectionnee.ecart_moyen,
        analyse_autre.score_global,
        analyse_autre.ecart_moyen,
    )
    assert selectionner_cibles((analyse_selectionnee, analyse_autre)) == selection_avant


def test_pytest_guard_prevents_real_bge_model_loading(tmp_path) -> None:
    from src.embeddings import AdaptateurBGEM3

    model_path = tmp_path / "bge-m3"
    model_path.mkdir()
    adapter = AdaptateurBGEM3(model_path)

    with pytest.raises(AssertionError, match="ne doit jamais être chargé"):
        adapter.encoder((competence("Compétence", 2),))
