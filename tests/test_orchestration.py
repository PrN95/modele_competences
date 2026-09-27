from collections.abc import Sequence

import pytest

from app import (
    construire_details_competences,
    construire_details_orchestration,
    construire_matrice_couples,
    construire_synthese_orchestration,
)
from src.domain import Competence, Emploi
from src.embeddings import SortieEncodage
from src.orchestration import calculer_r_epfq, orchestrer_emplois
from src.scoring import MESSAGE_AUCUNE_CIBLE, MESSAGE_ARBITRAGE_RH


def competence(nom: str, niveau: int = 2) -> Competence:
    return Competence(nom, None, niveau, id=nom)


def emploi(nom: str, type_emploi: str, competences) -> Emploi:
    return Emploi(
        intitule=nom,
        type=type_emploi,
        effectif=None,
        competences=tuple(competences),
        fichier_source=f"{nom}.pdf",
        feuille_source=None,
    )


def sortie(vecteurs, poids) -> SortieEncodage:
    return SortieEncodage(
        vecteurs_dense=tuple(tuple(vecteur) for vecteur in vecteurs),
        poids_sparse=tuple(poids),
    )


class FauxEncodeur:
    def __init__(self, sorties: dict[tuple[str, ...], SortieEncodage]) -> None:
        self.sorties = sorties
        self.appels: list[tuple[str, ...]] = []

    def encoder(self, competences: Sequence[Competence]) -> SortieEncodage:
        cle = tuple(competence.id for competence in competences)
        self.appels.append(cle)
        return self.sorties[cle]


def test_orchestration_analyse_toute_la_matrice_et_autorise_la_meme_cible() -> None:
    competences_p1 = (competence("A1", 1), competence("A2", 4))
    competences_p2 = (competence("A3", 3),)
    emploi_p1 = emploi("Emploi actuel P1", "actuel", competences_p1)
    emploi_p2 = emploi("Emploi actuel P2", "actuel", competences_p2)
    cible_q1 = emploi("Emploi cible Q1", "cible", (competence("C1", 2),))
    cible_q2 = emploi("Emploi cible Q2", "cible", (competence("C2", 2),))
    encodeur = FauxEncodeur(
        {
            ("A1", "A2"): sortie(((1, 0, 0), (0, 1, 0)), ({"x": 1}, {"y": 1})),
            ("A3",): sortie(((1, 0, 0),), ({"x": 1},)),
            ("C1",): sortie(((1, 0, 0),), ({"x": 1},)),
            ("C2",): sortie(((0, 0, 1),), ({"z": 1},)),
        }
    )

    resultat = orchestrer_emplois(
        (emploi_p1, emploi_p2),
        (cible_q1, cible_q2),
        encodeur,
        seuil_sim=0.7,
        seuil_couv=0.7,
    )

    assert len(resultat.resultats_emplois) == 2
    assert all(len(item.selection.analyses_classees) == 2 for item in resultat.resultats_emplois)
    assert [
        item.cibles_retenues[0].analyse.emploi_cible
        for item in resultat.resultats_emplois
    ] == [cible_q1, cible_q1]
    assert resultat.resultats_emplois[0].cibles_retenues[0].r_epfq == 0.5
    assert resultat.resultats_emplois[1].cibles_retenues[0].r_epfq == 1.0
    assert calculer_r_epfq(emploi_p1, cible_q1, encodeur, seuil_sim=0.7) == 0.5

    synthese = construire_synthese_orchestration(resultat)
    matrice = construire_matrice_couples(resultat)
    details = construire_details_orchestration(resultat)
    assert len(synthese) == 2
    assert len(matrice) == 4
    assert {ligne["Emploi_Cible_Retenu"] for ligne in details} == {
        cible_q1.intitule
    }
    assert all(ligne["R_epfq"] in {0.5, 1.0} for ligne in details)
    assert all(
        ligne["R_epfq"] is None
        for ligne in matrice
        if not ligne["Est_Selectionne"]
    )


def test_orchestration_sans_cible_admissible_ne_calcule_aucun_r_epfq() -> None:
    actuelle = competence("A")
    emploi_actuel = emploi("Emploi actuel", "actuel", (actuelle,))
    cibles = (
        emploi("Emploi cible Q1", "cible", (competence("C1"),)),
        emploi("Emploi cible Q2", "cible", (competence("C2"),)),
    )
    encodeur = FauxEncodeur(
        {
            ("A",): sortie(((1, 0, 0),), ({"a": 1},)),
            ("C1",): sortie(((0, 1, 0),), ({"b": 1},)),
            ("C2",): sortie(((0, 0, 1),), ({"c": 1},)),
        }
    )

    resultat = orchestrer_emplois((emploi_actuel,), cibles, encodeur)
    resultat_emploi = resultat.resultats_emplois[0]

    assert resultat_emploi.selection.alerte == MESSAGE_AUCUNE_CIBLE
    assert resultat_emploi.selection.meilleures_analyses == ()
    assert resultat_emploi.cibles_retenues == ()
    synthese = construire_synthese_orchestration(resultat)
    assert synthese[0]["R_epfq"] is None
    assert synthese[0]["Cible_La_Plus_Proche"] == "Emploi cible Q1"
    assert synthese[0]["Score_Global_Cible_La_Plus_Proche"] == 0.0
    assert construire_details_orchestration(resultat) == []
    matrice = construire_matrice_couples(resultat)
    assert len(matrice) == 2
    assert all(ligne["Statut_Selection"] == "non retenue" for ligne in matrice)
    assert [ligne["Est_Cible_La_Plus_Proche"] for ligne in matrice] == [True, False]
    assert all(
        ligne["Raison_Non_Selection"] == "couverture inférieure au seuil"
        for ligne in matrice
    )
    cible_plus_proche = resultat_emploi.selection.analyses_classees[0]
    details = construire_details_competences(
        cible_plus_proche,
        seuil_sim=resultat.seuil_sim,
        poids_dense=resultat.poids_dense,
        poids_sparse=resultat.poids_sparse,
    )
    assert details
    assert all(ligne["Recommandation de Formation"] == "" for ligne in details)


def test_orchestration_conserve_la_triple_egalite_et_calcule_r_par_cible() -> None:
    actuelle = competence("A", 1)
    emploi_actuel = emploi("Emploi actuel", "actuel", (actuelle,))
    cibles = (
        emploi("Emploi cible Q1", "cible", (competence("C1", 2),)),
        emploi("Emploi cible Q2", "cible", (competence("C2", 2),)),
    )
    encodeur = FauxEncodeur(
        {
            ("A",): sortie(((1, 0),), ({"x": 1},)),
            ("C1",): sortie(((1, 0),), ({"x": 1},)),
            ("C2",): sortie(((1, 0),), ({"x": 1},)),
        }
    )

    resultat = orchestrer_emplois((emploi_actuel,), cibles, encodeur)
    resultat_emploi = resultat.resultats_emplois[0]

    assert resultat_emploi.selection.alerte == MESSAGE_ARBITRAGE_RH
    assert len(resultat_emploi.cibles_retenues) == 2
    assert {
        item.analyse.emploi_cible for item in resultat_emploi.cibles_retenues
    } == set(cibles)
    assert all(item.r_epfq == pytest.approx(1.0) for item in resultat_emploi.cibles_retenues)
    assert all(item.analyse.besoins_formation for item in resultat_emploi.cibles_retenues)


def test_orchestration_avec_faux_encodeur_utilise_et_restitue_les_poids() -> None:
    actuelle = competence("A", 2)
    cible_competence = competence("C", 2)
    emploi_actuel = emploi("Actuel", "actuel", (actuelle,))
    emploi_cible = emploi("Cible", "cible", (cible_competence,))
    sorties = {
        ("A",): sortie(((1.0, 0.0),), ({"commun": 1.0},)),
        ("C",): sortie(((0.6, 0.8),), ({"commun": 1.0},)),
    }

    resultat_defaut = orchestrer_emplois(
        (emploi_actuel,),
        (emploi_cible,),
        FauxEncodeur(sorties),
        seuil_sim=0.0,
        seuil_couv=0.0,
    )
    resultat_moitie = orchestrer_emplois(
        (emploi_actuel,),
        (emploi_cible,),
        FauxEncodeur(sorties),
        seuil_sim=0.0,
        seuil_couv=0.0,
        poids_dense=0.5,
        poids_sparse=0.5,
    )

    score_defaut = resultat_defaut.resultats_emplois[0].selection.analyses_classees[
        0
    ].correspondances[0].score_hybride
    score_moitie = resultat_moitie.resultats_emplois[0].selection.analyses_classees[
        0
    ].correspondances[0].score_hybride
    assert resultat_defaut.poids_dense == pytest.approx(2 / 3)
    assert resultat_defaut.poids_sparse == pytest.approx(1 / 3)
    assert score_defaut == pytest.approx((2 / 3 * 0.6) + (1 / 3 * 1.0))
    assert resultat_moitie.poids_dense == 0.5
    assert resultat_moitie.poids_sparse == 0.5
    assert score_moitie == pytest.approx(0.8)
    assert construire_synthese_orchestration(resultat_moitie)[0]["Poids_Dense"] == 0.5
    assert construire_matrice_couples(resultat_moitie)[0]["Poids_Sparse"] == 0.5
    assert construire_details_orchestration(resultat_moitie)[0]["Poids_Dense"] == 0.5


def test_orchestration_rejette_les_poids_avant_d_appeler_le_faux_encodeur() -> None:
    emploi_actuel = emploi("Actuel", "actuel", (competence("A"),))
    emploi_cible = emploi("Cible", "cible", (competence("C"),))
    encodeur = FauxEncodeur({})

    with pytest.raises(ValueError, match="somme.*doit être égale à 1"):
        orchestrer_emplois(
            (emploi_actuel,),
            (emploi_cible,),
            encodeur,
            poids_dense=0.8,
            poids_sparse=0.3,
        )

    assert encodeur.appels == []
