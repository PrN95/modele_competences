import csv
import io

from app import (
    cle_couple_emplois,
    construire_details_competences,
    construire_details_orchestration,
    construire_lignes_export,
    convertir_csv,
    indexer_analyses_selectionnees,
)
from src.domain import Competence, CorrespondanceFournie, CoupleEmplois, Emploi
from src.orchestration import (
    CibleRetenueAvecReutilisation,
    ResultatOrchestration,
    ResultatOrchestrationEmploi,
)
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


def analyser(actuel: Emploi, cible: Emploi, scores) -> object:
    correspondances = tuple(
        CorrespondanceFournie(
            competence_cible,
            competence_actuelle,
            score_dense,
            score_sparse,
            detail_egalite,
        )
        for (
            competence_cible,
            competence_actuelle,
            score_dense,
            score_sparse,
            detail_egalite,
        ) in scores
    )
    return analyser_couple(CoupleEmplois(actuel, cible), correspondances)


def test_export_recommends_only_selected_target_and_marks_it_selected() -> None:
    actuelle = competence("Compétence actuelle", 1)
    actuel = emploi("Emploi actuel", "actuel", (actuelle,))
    cible_retenue = emploi(
        "Emploi cible retenu",
        "cible",
        (competence("Compétence à renforcer", 2),),
    )
    cible_exclue = emploi(
        "Emploi cible exclu",
        "cible",
        (competence("Compétence absente", 4),),
    )
    retenue = analyser(
        actuel,
        cible_retenue,
        ((cible_retenue.competences[0], actuelle, 0.8, 0.8, None),),
    )
    exclue = analyser(
        actuel,
        cible_exclue,
        ((cible_exclue.competences[0], actuelle, 0.6, 0.6, None),),
    )
    selection = selectionner_cibles((exclue, retenue))

    lignes = construire_lignes_export(selection, seuil_sim=0.7, seuil_couv=0.7)
    ligne_retenue = next(item for item in lignes if item["Emploi_Cible"] == cible_retenue.intitule)
    ligne_exclue = next(item for item in lignes if item["Emploi_Cible"] == cible_exclue.intitule)

    assert ligne_retenue["Est_Selectionne"] is True
    assert ligne_retenue["Recommandation_Formation"] == "Formation légère pour progresser d’un niveau"
    assert ligne_exclue["Est_Selectionne"] is False
    assert ligne_exclue["Recommandation_Formation"] == ""
    assert ligne_exclue["Seuil_Sim"] == 0.7
    assert ligne_exclue["Seuil_Couv"] == 0.7


def test_selection_uses_pair_identity_instead_of_enriched_analysis_equality() -> None:
    actuelle = competence("Compétence actuelle", 1)
    actuel = emploi("Emploi actuel", "actuel", (actuelle,))
    cible = emploi("Emploi cible", "cible", (competence("Compétence cible", 2),))
    analyse_initiale = analyser(
        actuel,
        cible,
        ((cible.competences[0], actuelle, 0.8, 0.8, None),),
    )

    selection = selectionner_cibles((analyse_initiale,))
    analyse_enrichie = selection.meilleures_analyses[0]
    index = indexer_analyses_selectionnees(selection)

    assert analyse_initiale != analyse_enrichie
    assert cle_couple_emplois(analyse_initiale) == cle_couple_emplois(analyse_enrichie)
    assert index[cle_couple_emplois(analyse_initiale)] is analyse_enrichie


def test_absent_competence_keeps_its_own_recommendation_in_details_and_export() -> None:
    actuelle = competence("Compétence actuelle", 2)
    actuel = emploi("Emploi actuel", "actuel", (actuelle,))
    cibles = tuple(competence(f"Compétence cible {index}", 2) for index in range(4))
    cible = emploi("Emploi cible", "cible", cibles)
    analyse = analyser(
        actuel,
        cible,
        (
            (cibles[0], actuelle, 0.8, 0.8, None),
            (cibles[1], actuelle, 0.8, 0.8, None),
            (cibles[2], actuelle, 0.8, 0.8, None),
            (cibles[3], actuelle, 0.6, 0.6, None),
        ),
    )
    selection = selectionner_cibles((analyse,))
    retenue = selection.meilleures_analyses[0]

    correspondance_absente = retenue.correspondances[3]
    valeurs_metier_avant = (
        retenue.g_ef,
        retenue.gs_ef,
        retenue.ecart_moyen,
        correspondance_absente.niveau_actuel,
        correspondance_absente.ecart_niveau,
        correspondance_absente.score_hybride,
    )
    assert valeurs_metier_avant == (0.75, 0.75, 0.5, 0, 2, 0.6)

    details = construire_details_competences(retenue, seuil_sim=0.7)
    detail_absence = next(item for item in details if item["Compétence Cible"] == cibles[3].intitule)
    export = construire_lignes_export(selection, seuil_sim=0.7, seuil_couv=0.7)
    export_absence = next(item for item in export if item["Competence_Cible"] == cibles[3].intitule)

    assert details[0]["Compétence actuelle la plus proche"] == actuelle.intitule
    assert details[0]["Niveau Actuel"] == 2
    assert details[0]["Écart Niveau"] == 0
    assert details[0]["Recommandation de Formation"] == ""
    assert detail_absence["Compétence actuelle la plus proche"] == (
        "Compétence actuelle (plus proche, non reconnue)"
    )
    assert detail_absence["Niveau Actuel"] == "N/A"
    assert detail_absence["Écart Niveau"] == "N/A"
    assert detail_absence["Statut"] == "Absente"
    assert detail_absence["Score Hybride (H_ac)"] == "0.6000"
    assert detail_absence["Seuil de similarité utilisé"] == 0.7
    assert detail_absence["Recommandation de Formation"] == "Formation complète nécessaire pour acquérir la compétence"
    assert export_absence["Compétence actuelle la plus proche"] == (
        "Compétence actuelle (plus proche, non reconnue)"
    )
    assert export_absence["Niveau_Actuel"] == "N/A"
    assert export_absence["Ecart_Niveau"] == "N/A"
    assert export_absence["Statut"] == "Absente"
    assert export_absence["Score_Hybride_H_ac"] == 0.6
    assert export_absence["Seuil_Sim"] == 0.7
    assert export_absence["Recommandation_Formation"] == "Formation complète nécessaire pour acquérir la compétence"

    resultat = ResultatOrchestration(
        resultats_emplois=(
            ResultatOrchestrationEmploi(
                emploi_actuel=actuel,
                selection=selection,
                cibles_retenues=(
                    CibleRetenueAvecReutilisation(analyse=retenue, r_epfq=1.0),
                ),
            ),
        ),
        seuil_sim=0.7,
        seuil_couv=0.7,
    )
    export_consolide = construire_details_orchestration(resultat)
    ligne_consolidee = next(
        item
        for item in export_consolide
        if item["Competence_Cible"] == cibles[3].intitule
    )
    assert ligne_consolidee["Compétence actuelle la plus proche"] == (
        "Compétence actuelle (plus proche, non reconnue)"
    )
    assert ligne_consolidee["Niveau_Actuel"] == "N/A"
    assert ligne_consolidee["Ecart_Niveau"] == "N/A"
    assert ligne_consolidee["Statut"] == "Absente"
    assert ligne_consolidee["Score_Hybride_H_ac"] == 0.6
    assert ligne_consolidee["Seuil_Sim"] == 0.7
    assert ligne_consolidee["Recommandation_Formation"] == (
        "Formation complète nécessaire pour acquérir la compétence"
    )

    lignes_csv = list(csv.DictReader(io.StringIO(convertir_csv(export_consolide))))
    ligne_csv = next(
        item for item in lignes_csv if item["Competence_Cible"] == cibles[3].intitule
    )
    assert ligne_csv["Niveau_Actuel"] == "N/A"
    assert ligne_csv["Ecart_Niveau"] == "N/A"
    assert ligne_csv["Statut"] == "Absente"
    assert ligne_csv["Score_Hybride_H_ac"] == "0.6"
    assert ligne_csv["Seuil_Sim"] == "0.7"
    assert ligne_csv["Recommandation_Formation"] == (
        "Formation complète nécessaire pour acquérir la compétence"
    )

    assert valeurs_metier_avant == (
        retenue.g_ef,
        retenue.gs_ef,
        retenue.ecart_moyen,
        correspondance_absente.niveau_actuel,
        correspondance_absente.ecart_niveau,
        correspondance_absente.score_hybride,
    )
    assert correspondance_absente.niveau_actuel == 0
    assert correspondance_absente.ecart_niveau == 2


def test_tie_detail_is_displayed_exported_and_all_triple_ties_stay_selected() -> None:
    actuelle = competence("Compétence actuelle", 1)
    actuel = emploi("Emploi actuel", "actuel", (actuelle,))
    detail = "Égalité exacte persistante ; choix déterministe documenté."
    cibles = (
        emploi("Emploi cible A", "cible", (competence("Compétence A", 2),)),
        emploi("Emploi cible B", "cible", (competence("Compétence B", 2),)),
    )
    analyses = tuple(
        analyser(
            actuel,
            cible,
            ((cible.competences[0], actuelle, 0.8, 0.8, detail),),
        )
        for cible in cibles
    )
    selection = selectionner_cibles(analyses)

    assert len(selection.meilleures_analyses) == 2
    for analyse_retenue in selection.meilleures_analyses:
        assert construire_details_competences(analyse_retenue)[0]["Détail Égalité"] == detail

    lignes = construire_lignes_export(selection, seuil_sim=0.7, seuil_couv=0.7)
    assert {item["Emploi_Cible"] for item in lignes if item["Est_Selectionne"]} == {
        cible.intitule for cible in cibles
    }
    assert all(item["Detail_Egalite"] == detail for item in lignes)
    assert all(item["Recommandation_Formation"] == "Formation légère pour progresser d’un niveau" for item in lignes)
