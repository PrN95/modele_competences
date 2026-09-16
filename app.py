import streamlit as st
import pandas as pd
import tempfile
import io
import hashlib
from pathlib import Path
from typing import Sequence
import warnings

from src.config import (
    BGE_BASE_URL,
    BGE_MODEL,
    DENSE_WEIGHT,
    GEMMA_BASE_URL,
    GEMMA_MODEL,
    MODEL_PATH,
    QWEN_BASE_URL,
    QWEN_MODEL,
    REMOTE_TIMEOUT_SECONDS,
    SEUIL_COUV,
    SEUIL_SIM,
    SPARSE_WEIGHT,
)
from src.domain import (
    Competence,
    CorrespondanceCompetence,
    Emploi,
    ResultatAnalyseCouple,
    ResultatSelectionCibles,
)
from src.embeddings import AdaptateurBGEM3, SortieEncodage
from src.io_pdf import PDFExtractionError, PDFExtractionWarning, read_emploi_pdf
from src.orchestration import (
    ResultatOrchestration,
    orchestrer_emplois,
    orchestrer_emplois_llm,
)
from src.scoring import MESSAGE_AUCUNE_CIBLE, MESSAGE_ARBITRAGE_RH
from src.matching import (
    controler_competences_actuelles_non_reprises,
    valider_poids_hybrides,
)
from src.comparison import ResultatExecutionModele, executer_comparaison
from src.remote_models import (
    AdaptateurBGEDistantEmbeddings,
    AdaptateurGemma4,
    AdaptateurQwenEmbeddings,
)


MODELE_BGE = "BGE-M3 local"
MODELE_BGE_DISTANT = "BGE-M3 distant"
MODELE_QWEN = "Qwen3-Embedding-8B distant"
MODELE_GEMMA = "Gemma 4 distant"
MODELE_FAUX = "Faux encodeur (Démo/Tests)"
MODE_COMPARAISON_DISTANTE = "Comparaison des trois modèles distant"


def formater_score(
    value: float | None,
    *,
    poids: float | None = None,
    composante: str | None = None,
) -> str:
    if value is not None:
        return f"{value:.4f}"
    if poids == 0 and composante is not None:
        return f"Non calculé, poids {composante} = 0"
    return "N/A"


def formater_niveau(niveau: int | None) -> int | str:
    """Rend explicite l'absence de niveau sans lui attribuer une valeur."""

    return niveau if niveau is not None else "Non renseigné"


def formater_ecart_moyen(ecart_moyen: float | None) -> str:
    return (
        f"{ecart_moyen:.2f}"
        if ecart_moyen is not None
        else "Non calculable, niveau absent"
    )


def cible_la_plus_proche(
    selection: ResultatSelectionCibles,
) -> ResultatAnalyseCouple | None:
    """Retourne la meilleure cible analysée, y compris si elle est exclue."""

    return selection.analyses_classees[0] if selection.analyses_classees else None


def etat_niveau(correspondance: CorrespondanceCompetence) -> str:
    if not correspondance.reconnue:
        return "Sans correspondance sémantique"
    if correspondance.ecart_niveau is None:
        return "Correspondance, niveau non comparable"
    return "Correspondance, écart calculé"


def cle_couple_emplois(analyse: ResultatAnalyseCouple) -> tuple[int, int]:
    """Identifie un couple sans comparer les copies enrichies de l'analyse."""

    return (id(analyse.emploi_actuel), id(analyse.emploi_cible))


def indexer_analyses_selectionnees(
    selection: ResultatSelectionCibles,
) -> dict[tuple[int, int], ResultatAnalyseCouple]:
    """Indexe toutes les cibles retenues, y compris les triples ex aequo."""

    return {
        cle_couple_emplois(analyse): analyse
        for analyse in selection.meilleures_analyses
    }


def recommandation_pour_correspondance(
    analyse_selectionnee: ResultatAnalyseCouple | None,
    competence_cible: Competence,
) -> str:
    """Retourne une recommandation uniquement pour une analyse sélectionnée."""

    if analyse_selectionnee is not None:
        correspondance = next(
            (item for item in analyse_selectionnee.correspondances if item.competence_cible is competence_cible or item.competence_cible == competence_cible),
            None,
        )
        if correspondance is not None and correspondance.recommandation_llm is not None:
            return correspondance.recommandation_llm
    if analyse_selectionnee is None:
        return ""
    besoin = next(
        (
            item
            for item in analyse_selectionnee.besoins_formation
            if item.competence_cible is competence_cible
            or item.competence_cible == competence_cible
        ),
        None,
    )
    return besoin.commentaire if besoin is not None else ""


def libelle_competence_actuelle_la_plus_proche(
    correspondance: CorrespondanceCompetence,
) -> str:
    """Présente la candidate sans la faire passer pour une correspondance reconnue."""

    actuelle = correspondance.competence_actuelle
    if actuelle is None:
        return "N/A"
    if not correspondance.reconnue:
        return f"{actuelle.intitule} (plus proche, non reconnue)"
    return actuelle.intitule


def construire_details_competences(
    analyse_selectionnee: ResultatAnalyseCouple,
    *,
    seuil_sim: float | None = float(SEUIL_SIM),
    poids_dense: float | None = float(DENSE_WEIGHT),
    poids_sparse: float | None = float(SPARSE_WEIGHT),
    modele: str = MODELE_BGE,
    type_score: str = "score hybride d'embedding",
) -> list[dict[str, object]]:
    """Construit la restitution détaillée d'une cible réellement retenue."""

    details: list[dict[str, object]] = []
    categoriel_gemma = type_score == "décision catégorielle LLM"
    for correspondance in analyse_selectionnee.correspondances:
        ligne: dict[str, object] = {
                "Modèle": modele,
                ("Méthode" if categoriel_gemma else "Type de score"): type_score,
                "Compétence Cible": correspondance.competence_cible.intitule,
                "Niveau Cible": formater_niveau(correspondance.niveau_requis),
                "Compétence actuelle correspondante": (
                    correspondance.competence_actuelle.intitule
                    if correspondance.reconnue
                    and correspondance.competence_actuelle is not None
                    else "N/A"
                ),
                "Niveau Actuel": (
                    formater_niveau(correspondance.niveau_actuel)
                    if correspondance.reconnue else "N/A"
                ),
                "Décision": "Reconnue" if correspondance.reconnue else "Absente",
                "Écart Niveau": (
                    correspondance.ecart_niveau
                    if correspondance.reconnue and correspondance.ecart_niveau is not None
                    else (
                        "Non calculable, niveau absent"
                        if correspondance.reconnue else "N/A"
                    )
                ),
                "État comparaison niveau": etat_niveau(correspondance),
                "Statut": correspondance.statut.replace("_", " ").title(),
                "Recommandation de Formation": recommandation_pour_correspondance(
                    analyse_selectionnee,
                    correspondance.competence_cible,
                ),
            }
        if not categoriel_gemma:
            ligne.update(
                {
                    "Compétence actuelle la plus proche": (
                        libelle_competence_actuelle_la_plus_proche(correspondance)
                    ),
                    "Score Dense (D_ac)": formater_score(
                        correspondance.score_dense,
                        poids=poids_dense,
                        composante="dense",
                    ),
                    "Score Sparse (L_ac)": formater_score(
                        correspondance.score_sparse,
                        poids=poids_sparse,
                        composante="sparse",
                    ),
                    "Score Hybride (H_ac)": formater_score(correspondance.score_hybride),
                    "Poids Dense utilisé": poids_dense if poids_dense is not None else "N/A",
                    "Poids Sparse utilisé": poids_sparse if poids_sparse is not None else "N/A",
                    "Seuil de similarité utilisé": seuil_sim,
                    "Détail Égalité": correspondance.detail_egalite or "",
                }
            )
        details.append(ligne)
    return details


def construire_lignes_export(
    selection: ResultatSelectionCibles,
    *,
    seuil_sim: float,
    seuil_couv: float,
    poids_dense: float = float(DENSE_WEIGHT),
    poids_sparse: float = float(SPARSE_WEIGHT),
    modele: str = MODELE_BGE,
) -> list[dict[str, object]]:
    """Exporte toutes les analyses, sans recommander les cibles non retenues."""

    selectionnees = indexer_analyses_selectionnees(selection)
    lignes: list[dict[str, object]] = []
    for analyse in selection.analyses_classees:
        analyse_selectionnee = selectionnees.get(cle_couple_emplois(analyse))
        est_selectionnee = analyse_selectionnee is not None
        for correspondance in analyse.correspondances:
            lignes.append(
                {
                    "modele": modele,
                    "Emploi_Actuel": analyse.emploi_actuel.intitule,
                    "Fichier_Source_Actuel": analyse.emploi_actuel.fichier_source,
                    "Emploi_Cible": analyse.emploi_cible.intitule,
                    "Fichier_Source_Cible": analyse.emploi_cible.fichier_source,
                    "Seuil_Sim": seuil_sim,
                    "Seuil_Couv": seuil_couv,
                    "Poids_Dense": poids_dense,
                    "Poids_Sparse": poids_sparse,
                    "G_ef_Couverture": float(analyse.g_ef),
                    "Gs_ef_Satisfaction": float(analyse.gs_ef),
                    "Ecart_Moyen_ef": analyse.ecart_moyen,
                    "Est_Admissible": analyse.admissible,
                    "Est_Selectionne": est_selectionnee,
                    "Competence_Cible": correspondance.competence_cible.intitule,
                    "Niveau_Cible": formater_niveau(correspondance.niveau_requis),
                    "Compétence actuelle la plus proche": (
                        libelle_competence_actuelle_la_plus_proche(correspondance)
                    ),
                    "Niveau_Actuel": (
                        formater_niveau(correspondance.niveau_actuel)
                        if correspondance.reconnue
                        else "N/A"
                    ),
                    "Score_Dense_D_ac": correspondance.score_dense,
                    "Score_Sparse_L_ac": correspondance.score_sparse,
                    "Score_Hybride_H_ac": correspondance.score_hybride,
                    "Score_Evaluation_LLM": correspondance.score_llm,
                    "Justification_Courte_LLM": correspondance.justification_courte or "",
                    "Reconnue": correspondance.reconnue,
                    "Ecart_Niveau": (
                        correspondance.ecart_niveau
                        if correspondance.reconnue and correspondance.ecart_niveau is not None
                        else "Non calculable, niveau absent"
                        if correspondance.reconnue else "N/A"
                    ),
                    "Etat_Comparaison_Niveau": etat_niveau(correspondance),
                    "Statut": (
                        "Absente"
                        if not correspondance.reconnue
                        else correspondance.statut
                    ),
                    "Detail_Egalite": correspondance.detail_egalite or "",
                    "Recommandation_Formation": recommandation_pour_correspondance(
                        analyse_selectionnee,
                        correspondance.competence_cible,
                    ),
                }
            )
    return lignes


def construire_synthese_orchestration(
    resultat: ResultatOrchestration,
) -> list[dict[str, object]]:
    """Construit une ligne de synthèse par emploi actuel."""

    lignes: list[dict[str, object]] = []
    for resultat_emploi in resultat.resultats_emplois:
        retenues = resultat_emploi.cibles_retenues
        if not retenues:
            ligne = {
                    "modele": resultat.modele,
                    "Type_Score": resultat.type_score,
                    "Emploi_Actuel": resultat_emploi.emploi_actuel.intitule,
                    "Fichier_Source_Actuel": resultat_emploi.emploi_actuel.fichier_source,
                    "Emplois_Cibles_Retenus": "",
                    "Statut": MESSAGE_AUCUNE_CIBLE,
                    "G_epfq": None,
                    "Gs_epfq": None,
                    "Ecart_Moyen_epfq": None,
                    "R_epfq": None,
                    "Reponse_Brute_Gemma": resultat_emploi.reponse_brute_gemma or "",
                }
            if resultat.type_score != "décision catégorielle LLM":
                ligne.update(
                    Seuil_Sim=resultat.seuil_sim,
                    Seuil_Couv=resultat.seuil_couv,
                    Poids_Dense=resultat.poids_dense,
                    Poids_Sparse=resultat.poids_sparse,
                )
            lignes.append(ligne)
            continue

        analyse_reference = retenues[0].analyse
        ligne = {
                "modele": resultat.modele,
                "Type_Score": resultat.type_score,
                "Emploi_Actuel": resultat_emploi.emploi_actuel.intitule,
                "Fichier_Source_Actuel": resultat_emploi.emploi_actuel.fichier_source,
                "Emplois_Cibles_Retenus": " | ".join(
                    cible.analyse.emploi_cible.intitule for cible in retenues
                ),
                "Statut": (
                    MESSAGE_ARBITRAGE_RH if len(retenues) > 1 else "Retenu"
                ),
                "G_epfq": float(analyse_reference.g_ef),
                "Gs_epfq": float(analyse_reference.gs_ef),
                "Ecart_Moyen_epfq": analyse_reference.ecart_moyen,
                "R_epfq": " | ".join(
                    f"{cible.analyse.emploi_cible.intitule}: {cible.r_epfq:.2%}"
                    for cible in retenues
                ),
                "Reponse_Brute_Gemma": resultat_emploi.reponse_brute_gemma or "",
            }
        if resultat.type_score != "décision catégorielle LLM":
            ligne.update(
                Seuil_Sim=resultat.seuil_sim,
                Seuil_Couv=resultat.seuil_couv,
                Poids_Dense=resultat.poids_dense,
                Poids_Sparse=resultat.poids_sparse,
            )
        lignes.append(ligne)
    return lignes


def construire_matrice_couples(
    resultat: ResultatOrchestration,
) -> list[dict[str, object]]:
    """Exporte les indicateurs de chaque couple de la matrice complète."""

    lignes: list[dict[str, object]] = []
    for resultat_emploi in resultat.resultats_emplois:
        reutilisation_par_cible = {
            cle_couple_emplois(cible.analyse): cible.r_epfq
            for cible in resultat_emploi.cibles_retenues
        }
        for analyse in resultat_emploi.selection.analyses_classees:
            cle = cle_couple_emplois(analyse)
            est_selectionnee = cle in reutilisation_par_cible
            raison_non_selection = ""
            if not est_selectionnee:
                raison_non_selection = (
                    "couverture inférieure au seuil"
                    if not analyse.admissible
                    else "score global inférieur à la meilleure cible admissible"
                )
            ligne = {
                    "modele": resultat.modele,
                    "Type_Score": resultat.type_score,
                    "Emploi_Actuel": analyse.emploi_actuel.intitule,
                    "Fichier_Source_Actuel": analyse.emploi_actuel.fichier_source,
                    "Emploi_Cible": analyse.emploi_cible.intitule,
                    "Fichier_Source_Cible": analyse.emploi_cible.fichier_source,
                    "G_epfq": float(analyse.g_ef),
                    "Gs_epfq": float(analyse.gs_ef),
                    "Ecart_Moyen_epfq": analyse.ecart_moyen,
                    "Est_Selectionne": est_selectionnee,
                    "Statut_Selection": "retenue" if est_selectionnee else "non retenue",
                    "Raison_Non_Selection": raison_non_selection,
                    "R_epfq": (
                        reutilisation_par_cible[cle] if est_selectionnee else None
                    ),
                    "Reponse_Brute_Gemma": resultat_emploi.reponse_brute_gemma or "",
                }
            if resultat.type_score != "décision catégorielle LLM":
                ligne.update(
                    Est_Admissible=analyse.admissible,
                    Seuil_Sim=resultat.seuil_sim,
                    Seuil_Couv=resultat.seuil_couv,
                    Poids_Dense=resultat.poids_dense,
                    Poids_Sparse=resultat.poids_sparse,
                )
            lignes.append(ligne)
    return lignes


def construire_details_orchestration(
    resultat: ResultatOrchestration,
) -> list[dict[str, object]]:
    """Exporte les détails et recommandations des seules cibles retenues."""

    lignes: list[dict[str, object]] = []
    for resultat_emploi in resultat.resultats_emplois:
        for cible_retenue in resultat_emploi.cibles_retenues:
            analyse = cible_retenue.analyse
            for correspondance in analyse.correspondances:
                ligne: dict[str, object] = {
                        "modele": resultat.modele,
                        (
                            "Methode"
                            if resultat.type_score == "décision catégorielle LLM"
                            else "Type_Score"
                        ): resultat.type_score,
                        "Emploi_Actuel": analyse.emploi_actuel.intitule,
                        "Fichier_Source_Actuel": analyse.emploi_actuel.fichier_source,
                        "Emploi_Cible_Retenu": analyse.emploi_cible.intitule,
                        "Fichier_Source_Cible": analyse.emploi_cible.fichier_source,
                        "G_epfq": float(analyse.g_ef),
                        "Gs_epfq": float(analyse.gs_ef),
                        "Ecart_Moyen_epfq": analyse.ecart_moyen,
                        "R_epfq": cible_retenue.r_epfq,
                        "Reponse_Brute_Gemma": resultat_emploi.reponse_brute_gemma or "",
                        "Competence_Cible": correspondance.competence_cible.intitule,
                        "Niveau_Cible": formater_niveau(correspondance.niveau_requis),
                        "Competence_Actuelle_Correspondante": (
                            correspondance.competence_actuelle.intitule
                            if correspondance.reconnue
                            and correspondance.competence_actuelle is not None
                            else "N/A"
                        ),
                        "Niveau_Actuel": (
                            formater_niveau(correspondance.niveau_actuel)
                            if correspondance.reconnue
                            else "N/A"
                        ),
                        "Decision": "Reconnue" if correspondance.reconnue else "Absente",
                        "Ecart_Niveau": (
                            correspondance.ecart_niveau
                            if correspondance.reconnue and correspondance.ecart_niveau is not None
                            else "Non calculable, niveau absent"
                            if correspondance.reconnue else "N/A"
                        ),
                        "Etat_Comparaison_Niveau": etat_niveau(correspondance),
                        "Statut": (
                            "Absente"
                            if not correspondance.reconnue
                            else correspondance.statut
                        ),
                        "Recommandation_Formation": (
                            recommandation_pour_correspondance(
                                analyse,
                                correspondance.competence_cible,
                            )
                        ),
                    }
                if resultat.type_score != "décision catégorielle LLM":
                    ligne.update(
                        {
                            "Seuil_Sim": resultat.seuil_sim,
                            "Seuil_Couv": resultat.seuil_couv,
                            "Poids_Dense": resultat.poids_dense,
                            "Poids_Sparse": resultat.poids_sparse,
                            "Compétence actuelle la plus proche": (
                                libelle_competence_actuelle_la_plus_proche(correspondance)
                            ),
                            "Score_Dense_D_ac": correspondance.score_dense,
                            "Score_Sparse_L_ac": (
                                None
                                if resultat.modele in {MODELE_BGE_DISTANT, MODELE_QWEN}
                                else correspondance.score_sparse
                            ),
                            "Score_Hybride_H_ac": correspondance.score_hybride,
                            "Reconnue": correspondance.reconnue,
                            "Detail_Egalite": correspondance.detail_egalite or "",
                        }
                    )
                lignes.append(ligne)
    return lignes


def construire_synthese_comparaison(
    executions: Sequence[ResultatExecutionModele],
    emplois_actuels: Sequence[Emploi],
) -> list[dict[str, object]]:
    """Construit la comparaison inter-modèles, y compris les échecs isolés."""

    lignes: list[dict[str, object]] = []
    for execution in executions:
        if execution.resultat is None:
            lignes.extend(
                {
                    "modele": execution.modele,
                    "Emploi_Actuel": emploi.intitule,
                    "Emploi_Cible_Retenu": "",
                    "G_epfq": None,
                    "Gs_epfq": None,
                    "Ecart_Moyen": None,
                    "R_epfq": None,
                    "Temps_Execution_s": execution.temps_execution_secondes,
                    "Statut": execution.statut,
                    "Detail_Egalites": "",
                    "Detail_Erreur": execution.detail_erreur or "",
                }
                for emploi in emplois_actuels
            )
            continue
        for resultat_emploi in execution.resultat.resultats_emplois:
            if not resultat_emploi.cibles_retenues:
                lignes.append(
                    {
                        "modele": execution.modele,
                        "Emploi_Actuel": resultat_emploi.emploi_actuel.intitule,
                        "Emploi_Cible_Retenu": "",
                        "G_epfq": None,
                        "Gs_epfq": None,
                        "Ecart_Moyen": None,
                        "R_epfq": None,
                        "Temps_Execution_s": execution.temps_execution_secondes,
                        "Statut": execution.statut,
                        "Detail_Egalites": "",
                        "Detail_Erreur": "",
                    }
                )
                continue
            for cible in resultat_emploi.cibles_retenues:
                analyse = cible.analyse
                details_egalites = " | ".join(
                    item.detail_egalite
                    for item in analyse.correspondances
                    if item.detail_egalite
                )
                if len(resultat_emploi.cibles_retenues) > 1:
                    details_egalites = " | ".join(
                        item
                        for item in (MESSAGE_ARBITRAGE_RH, details_egalites)
                        if item
                    )
                lignes.append(
                    {
                        "modele": execution.modele,
                        "Emploi_Actuel": resultat_emploi.emploi_actuel.intitule,
                        "Emploi_Cible_Retenu": analyse.emploi_cible.intitule,
                        "G_epfq": float(analyse.g_ef),
                        "Gs_epfq": float(analyse.gs_ef),
                        "Ecart_Moyen": analyse.ecart_moyen,
                        "R_epfq": cible.r_epfq,
                        "Temps_Execution_s": execution.temps_execution_secondes,
                        "Statut": execution.statut,
                        "Detail_Egalites": details_egalites,
                        "Detail_Erreur": "",
                    }
                )
    return lignes


def construire_details_comparaison(
    executions: Sequence[ResultatExecutionModele],
) -> list[dict[str, object]]:
    lignes: list[dict[str, object]] = []
    for execution in executions:
        if execution.resultat is None:
            continue
        for ligne in construire_details_orchestration(execution.resultat):
            lignes.append(
                {
                    **ligne,
                    "Temps_Execution_s": execution.temps_execution_secondes,
                    "Statut_Execution": execution.statut,
                }
            )
    return lignes


def convertir_csv(lignes: list[dict[str, object]]) -> str:
    """Sérialise des lignes consolidées en CSV UTF-8 avec BOM."""

    buffer = io.StringIO()
    pd.DataFrame(lignes).to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()


class FauxEncodeurStreamlit:
    """Encodeur fictif pour tester l'interface sans le modèle BGE-M3 local."""

    def encoder(self, competences: Sequence[Competence]) -> SortieEncodage:
        vecteurs_dense = []
        poids_sparse = []
        for comp in competences:
            # Génération d'un vecteur dense déterministe pseudo-aléatoire basé sur le texte
            h = hashlib.sha256(comp.texte.encode("utf-8")).digest()
            # Vecteur de dimension 1024
            vec = []
            for i in range(1024):
                val = (h[i % len(h)] / 255.0) - 0.5
                vec.append(val)
            # Normalisation du vecteur
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            vecteurs_dense.append(tuple(vec))

            # Poids sparse fictifs basés sur les mots clés du texte
            words = [w.lower() for w in comp.texte.split() if len(w) > 2]
            poids = {w: 0.5 for w in words} if words else {"default": 1.0}
            poids_sparse.append(poids)

        return SortieEncodage(tuple(vecteurs_dense), tuple(poids_sparse))


@st.cache_resource
def charger_encodeur_reel(model_path: str) -> AdaptateurBGEM3:
    return AdaptateurBGEM3(model_path)


@st.cache_resource
def charger_encodeur_fictif() -> FauxEncodeurStreamlit:
    return FauxEncodeurStreamlit()


@st.cache_resource
def charger_qwen(
    base_url: str,
    model: str,
    timeout: float,
    api_key: str,
) -> AdaptateurQwenEmbeddings:
    return AdaptateurQwenEmbeddings(
        base_url,
        model,
        timeout=timeout,
        api_key=api_key or None,
    )


@st.cache_resource
def charger_bge_distant(
    base_url: str,
    model: str,
    timeout: float,
    api_key: str,
) -> AdaptateurBGEDistantEmbeddings:
    return AdaptateurBGEDistantEmbeddings(
        base_url,
        model,
        timeout=timeout,
        api_key=api_key or None,
    )


@st.cache_resource
def charger_gemma(
    base_url: str,
    model: str,
    timeout: float,
    api_key: str,
) -> AdaptateurGemma4:
    return AdaptateurGemma4(
        base_url,
        model,
        timeout=timeout,
        api_key=api_key or None,
    )


def afficher_resultat_modele(
    execution: ResultatExecutionModele,
    emplois_cibles: Sequence[Emploi],
    moteur: object,
) -> None:
    """Affiche un résultat sans confondre embedding et jugement LLM."""

    st.markdown(f"## {execution.modele}")
    if execution.resultat is None:
        st.error(f"{execution.statut} : {execution.detail_erreur}")
        return
    resultat = execution.resultat
    if resultat.type_score == "décision catégorielle LLM":
        st.caption(
            f"Méthode : sélection directe et indicateurs métier informatifs par Gemma 4 ; "
            f"temps={execution.temps_execution_secondes:.3f} s."
        )
    else:
        poids = f"dense={resultat.poids_dense:.8f}, sparse={resultat.poids_sparse:.8f}"
        st.caption(
            f"Méthode : {resultat.type_score} ; seuil_sim={resultat.seuil_sim:.2f} ; "
            f"seuil_couv={resultat.seuil_couv:.2f} ; poids={poids} ; "
            f"temps={execution.temps_execution_secondes:.3f} s."
        )
    st.dataframe(
        pd.DataFrame(construire_synthese_orchestration(resultat)),
        use_container_width=True,
    )
    st.markdown("#### Comparatif de toutes les cibles analysées")
    st.dataframe(
        pd.DataFrame(construire_matrice_couples(resultat)),
        use_container_width=True,
    )
    for resultat_emploi in resultat.resultats_emplois:
        emploi_actuel = resultat_emploi.emploi_actuel
        st.markdown(f"#### Emploi actuel : **{emploi_actuel.intitule}**")
        cible_plus_proche = cible_la_plus_proche(resultat_emploi.selection)
        if cible_plus_proche is not None:
            st.markdown(
                "##### Emploi cible le plus proche : "
                f"**{cible_plus_proche.emploi_cible.intitule}**"
            )
        if not resultat_emploi.cibles_retenues:
            if resultat.type_score == "décision catégorielle LLM":
                st.warning("Aucune passerelle selon Gemma 4")
            if resultat_emploi.reponse_brute_gemma:
                with st.expander("Réponse JSON brute de Gemma"):
                    st.code(resultat_emploi.reponse_brute_gemma, language="json")
            if resultat.type_score == "décision catégorielle LLM":
                continue
            st.warning("Aucune cible retenue avec les seuils actuels")
            st.caption(
                f"Paramètres utilisés — seuil_sim={resultat.seuil_sim:.2f} ; "
                f"seuil_couv={resultat.seuil_couv:.2f} ; "
                f"poids dense={resultat.poids_dense:.8f} ; "
                f"poids sparse={resultat.poids_sparse:.8f}."
            )
            if cible_plus_proche is None:
                st.warning("Aucun emploi cible analysable pour cet emploi actuel")
                continue
            col_g, col_gs, col_ecart = st.columns(3)
            col_g.metric("G_epfq / couverture", f"{cible_plus_proche.g_ef:.2%}")
            col_gs.metric("Gs_epfq", f"{cible_plus_proche.gs_ef:.2%}")
            col_ecart.metric("Écart moyen", formater_ecart_moyen(cible_plus_proche.ecart_moyen))
            st.dataframe(
                pd.DataFrame(
                    construire_details_competences(
                        cible_plus_proche,
                        seuil_sim=resultat.seuil_sim,
                        poids_dense=resultat.poids_dense,
                        poids_sparse=resultat.poids_sparse,
                        modele=resultat.modele,
                        type_score=resultat.type_score,
                    )
                ),
                use_container_width=True,
            )
        if len(resultat_emploi.cibles_retenues) > 1:
            st.warning(MESSAGE_ARBITRAGE_RH)
        for cible_retenue in resultat_emploi.cibles_retenues:
            analyse = cible_retenue.analyse
            st.markdown(f"##### Cible retenue : **{analyse.emploi_cible.intitule}**")
            col_g, col_gs, col_ecart, col_r = st.columns(4)
            col_g.metric("G_epfq", f"{analyse.g_ef:.2%}")
            col_gs.metric("Gs_epfq", f"{analyse.gs_ef:.2%}")
            col_ecart.metric("Écart moyen", formater_ecart_moyen(analyse.ecart_moyen))
            col_r.metric("R_epfq", f"{cible_retenue.r_epfq:.2%}")
            st.dataframe(
                pd.DataFrame(
                    construire_details_competences(
                        analyse,
                        seuil_sim=resultat.seuil_sim,
                        poids_dense=resultat.poids_dense,
                        poids_sparse=resultat.poids_sparse,
                        modele=resultat.modele,
                        type_score=resultat.type_score,
                    )
                ),
                use_container_width=True,
            )
        if resultat.type_score == "décision catégorielle LLM" and resultat_emploi.reponse_brute_gemma:
            with st.expander("Réponse JSON brute de Gemma"):
                st.code(resultat_emploi.reponse_brute_gemma, language="json")
        selectionnees = tuple(cible.analyse.emploi_cible for cible in resultat_emploi.cibles_retenues)
        if resultat.type_score == "décision catégorielle LLM":
            reutilisees = {
                id(correspondance.competence_actuelle)
                for cible in resultat_emploi.cibles_retenues
                for correspondance in cible.analyse.correspondances
                if correspondance.reconnue and correspondance.competence_actuelle is not None
            }
            non_reprises = tuple(
                competence
                for competence in emploi_actuel.competences
                if id(competence) not in reutilisees
            )
            signalements = ()
        else:
            signalements = controler_competences_actuelles_non_reprises(
                emploi_actuel,
                emplois_cibles,
                selectionnees,
                moteur,
                seuil_sim=resultat.seuil_sim,
                poids_dense=resultat.poids_dense,
                poids_sparse=resultat.poids_sparse,
            )
        st.markdown("**Compétences actuelles non reprises :**")
        if resultat.type_score == "décision catégorielle LLM" and non_reprises:
            for competence in non_reprises:
                st.write(
                    f"- ℹ️ **{competence.intitule}** : compétence actuelle non reprise "
                    "dans l’emploi cible sélectionné par Gemma 4"
                )
        elif signalements:
            for signalement in signalements:
                st.write(
                    f"- ℹ️ **{signalement.competence_actuelle.intitule}** : "
                    f"{signalement.message}"
                )
        else:
            st.write("*Toutes les compétences actuelles sont réutilisées.*")


def main():
    st.set_page_config(
        page_title="PoC Aide à la Décision RH - Rapprochement d'Emplois",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("PoC Aide à la Décision RH")
    st.subheader("Rapprochement sémantique d'emplois, scoring et recommandations de formation")

    st.sidebar.header("Expérimentation comparative")
    mode_execution = st.sidebar.radio(
        "Mode d'exécution",
        ["Un seul modèle", MODE_COMPARAISON_DISTANTE],
    )
    modele_unique = st.sidebar.selectbox(
        "Modèle utilisé",
        [MODELE_BGE, MODELE_BGE_DISTANT, MODELE_QWEN, MODELE_GEMMA, MODELE_FAUX],
        disabled=(mode_execution == MODE_COMPARAISON_DISTANTE),
    )
    chemin_modele = st.sidebar.text_input(
        "Chemin du modèle BGE-M3 local",
        value=str(MODEL_PATH),
        disabled=(mode_execution == MODE_COMPARAISON_DISTANTE),
    )
    bge_distant_url = st.sidebar.text_input("URL BGE distant", value=BGE_BASE_URL)
    bge_distant_model = st.sidebar.text_input("Identifiant BGE distant", value=BGE_MODEL)
    qwen_url = st.sidebar.text_input("URL Qwen", value=QWEN_BASE_URL)
    qwen_model = st.sidebar.text_input("Identifiant Qwen", value=QWEN_MODEL)
    gemma_url = st.sidebar.text_input("URL Gemma", value=GEMMA_BASE_URL)
    gemma_model = st.sidebar.text_input("Identifiant Gemma", value=GEMMA_MODEL)
    timeout_distant = st.sidebar.number_input(
        "Timeout des serveurs distants (secondes)",
        min_value=1.0,
        max_value=600.0,
        value=float(REMOTE_TIMEOUT_SECONDS),
        step=5.0,
    )
    api_key = st.sidebar.text_input(
        "Clé API optionnelle",
        value="",
        type="password",
        help="Aucune clé n'est inscrite en dur ni exportée.",
    )
    col_test_bge, col_test_qwen, col_test_gemma = st.sidebar.columns(3)
    if col_test_bge.button("Tester BGE distant"):
        try:
            identifiant = charger_bge_distant(
                bge_distant_url, bge_distant_model, timeout_distant, api_key
            ).tester_connexion()
        except Exception as err:
            st.sidebar.error(f"BGE distant indisponible : {err}")
        else:
            st.sidebar.success(f"BGE distant connecté : {identifiant}")
    if col_test_qwen.button("Tester Qwen"):
        try:
            identifiant = charger_qwen(
                qwen_url, qwen_model, timeout_distant, api_key
            ).tester_connexion()
        except Exception as err:
            st.sidebar.error(f"Qwen indisponible : {err}")
        else:
            st.sidebar.success(f"Qwen connecté : {identifiant}")
    if col_test_gemma.button("Tester Gemma"):
        try:
            identifiant = charger_gemma(
                gemma_url, gemma_model, timeout_distant, api_key
            ).tester_connexion()
        except Exception as err:
            st.sidebar.error(f"Gemma indisponible : {err}")
        else:
            st.sidebar.success(f"Gemma connecté : {identifiant}")

    seuil_sim = float(SEUIL_SIM)
    seuil_couv = float(SEUIL_COUV)
    poids_dense = float(DENSE_WEIGHT)
    poids_sparse = float(SPARSE_WEIGHT)
    afficher_seuils = (
        mode_execution == MODE_COMPARAISON_DISTANTE
        or modele_unique != MODELE_GEMMA
    )
    afficher_poids = (
        mode_execution != MODE_COMPARAISON_DISTANTE
        and modele_unique in {MODELE_BGE, MODELE_FAUX}
    )
    if afficher_seuils:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Seuils configurables :**")
        seuil_sim = st.sidebar.slider(
            "Seuil de similarité compétence (seuil_sim)",
            min_value=0.0,
            max_value=1.0,
            value=float(SEUIL_SIM),
            step=0.05,
        )
        seuil_couv = st.sidebar.slider(
            "Seuil de couverture sémantique (seuil_couv)",
            min_value=0.0,
            max_value=1.0,
            value=float(SEUIL_COUV),
            step=0.05,
        )
    if afficher_poids:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Coefficients du score hybride BGE-M3 local :**")
        poids_dense = st.sidebar.number_input(
            "Coefficient dense (poids_dense)",
            min_value=0.0,
            max_value=1.0,
            value=float(DENSE_WEIGHT),
            step=0.01,
            format="%.8f",
            help="Valeur de référence exacte : 2/3.",
        )
        poids_sparse = st.sidebar.number_input(
            "Coefficient sparse (poids_sparse)",
            min_value=0.0,
            max_value=1.0,
            value=float(SPARSE_WEIGHT),
            step=0.01,
            format="%.8f",
            help="Valeur de référence exacte : 1/3.",
        )
    poids_valides = True
    if afficher_poids:
        try:
            valider_poids_hybrides(poids_dense, poids_sparse)
        except ValueError as err:
            poids_valides = False
            st.sidebar.error(str(err))
        st.sidebar.caption(
            "Ces poids s'appliquent uniquement à BGE-M3 local. BGE distant et Qwen "
            "utilisent dense=1,00 et sparse=0,00 ; Gemma 4 effectue une sélection "
            "catégorielle directe."
        )

    st.markdown("### 1. Chargement des documents PDF structurés")
    col_actuel, col_cible = st.columns(2)

    with col_actuel:
        st.markdown("#### Emplois actuels")
        fichiers_actuels = st.file_uploader(
            "Déposer un ou plusieurs PDF d'emplois actuels",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_actuel",
        )

    with col_cible:
        st.markdown("#### Emplois cibles")
        fichiers_cibles = st.file_uploader(
            "Déposer un ou plusieurs PDF d'emplois cibles",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_cible",
        )

    if not fichiers_actuels or not fichiers_cibles:
        st.info(
            "Veuillez charger au moins un emploi actuel et un emploi cible "
            "pour démarrer l'analyse."
        )
        return

    emplois_actuels: list[Emploi] = []
    emplois_cibles: list[Emploi] = []

    for f in fichiers_actuels:
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = Path(tmp.name)

            with warnings.catch_warnings(record=True) as extraction_warnings:
                warnings.simplefilter("always", PDFExtractionWarning)
                emploi = read_emploi_pdf(
                    tmp_path,
                    document_type="emploi_actuel",
                    original_filename=f.name,
                )
            emploi = Emploi(
                intitule=emploi.intitule,
                type=emploi.type,
                effectif=emploi.effectif,
                competences=emploi.competences,
                fichier_source=f.name,
                feuille_source=None,
                id=emploi.id,
                format_extraction=emploi.format_extraction,
            )
            emplois_actuels.append(emploi)
            st.caption(
                "Rôle choisi : emploi actuel ; "
                f"format { {'talentsoft': 'Talentsoft', 'rome': 'ROME', 'tableau': 'tableau'}.get(emploi.format_extraction or '', 'inconnu') } reconnu ; "
                f"intitulé : {emploi.intitule} ; compétences extraites : {len(emploi.competences)}."
            )
            for warning in extraction_warnings:
                st.warning(str(warning.message))
        except PDFExtractionError as err:
            st.error(
                f"Erreur d'extraction pour le fichier actuel '{f.name}' "
                f"(formats tentés : tableau, Talentsoft puis ROME) : {err}"
            )
        except Exception as e:
            st.error(f"Erreur inattendue pour '{f.name}' : {e}")
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    for f in fichiers_cibles:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = Path(tmp.name)

            with warnings.catch_warnings(record=True) as extraction_warnings:
                warnings.simplefilter("always", PDFExtractionWarning)
                emploi = read_emploi_pdf(
                    tmp_path,
                    document_type="metier_cible",
                    original_filename=f.name,
                )
            emploi = Emploi(
                intitule=emploi.intitule,
                type=emploi.type,
                effectif=emploi.effectif,
                competences=emploi.competences,
                fichier_source=f.name,
                feuille_source=None,
                id=emploi.id,
                format_extraction=emploi.format_extraction,
            )
            emplois_cibles.append(emploi)
            st.caption(
                "Rôle choisi : emploi cible ; "
                f"format { {'talentsoft': 'Talentsoft', 'rome': 'ROME', 'tableau': 'tableau'}.get(emploi.format_extraction or '', 'inconnu') } reconnu ; "
                f"intitulé : {emploi.intitule} ; compétences extraites : {len(emploi.competences)}."
            )
            for warning in extraction_warnings:
                st.warning(str(warning.message))
        except PDFExtractionError as err:
            st.error(
                f"Erreur d'extraction pour le fichier cible '{f.name}' "
                f"(formats tentés : tableau, Talentsoft puis ROME) : {err}"
            )
        except Exception as e:
            st.error(f"Erreur inattendue pour '{f.name}' : {e}")
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    if not emplois_actuels or not emplois_cibles:
        st.warning("Échec de l'extraction des profils. Veuillez vérifier les fichiers PDF.")
        return

    st.success(
        f"Extraction réussie : {len(emplois_actuels)} emploi(s) actuel(s) et "
        f"{len(emplois_cibles)} emploi(s) cible(s)."
    )

    st.markdown("---")
    st.markdown("### 2. Analyse globale emplois actuels × emplois cibles")

    poids_requis = (
        mode_execution != MODE_COMPARAISON_DISTANTE
        and modele_unique in {MODELE_BGE, MODELE_FAUX}
    )
    if st.button(
        "Lancer l'analyse globale",
        type="primary",
        disabled=poids_requis and not poids_valides,
    ):
        objets_moteurs: dict[str, object] = {}

        def lancer_bge() -> ResultatOrchestration:
            encodeur = charger_encodeur_reel(chemin_modele)
            objets_moteurs[MODELE_BGE] = encodeur
            return orchestrer_emplois(
                emplois_actuels,
                emplois_cibles,
                encodeur,
                seuil_sim=seuil_sim,
                seuil_couv=seuil_couv,
                poids_dense=poids_dense,
                poids_sparse=poids_sparse,
                modele=MODELE_BGE,
            )

        def lancer_qwen() -> ResultatOrchestration:
            encodeur = charger_qwen(qwen_url, qwen_model, timeout_distant, api_key)
            objets_moteurs[MODELE_QWEN] = encodeur
            return orchestrer_emplois(
                emplois_actuels,
                emplois_cibles,
                encodeur,
                seuil_sim=seuil_sim,
                seuil_couv=seuil_couv,
                poids_dense=1.0,
                poids_sparse=0.0,
                modele=MODELE_QWEN,
                type_score="score dense d'embedding (H=D)",
            )

        def lancer_bge_distant() -> ResultatOrchestration:
            encodeur = charger_bge_distant(
                bge_distant_url, bge_distant_model, timeout_distant, api_key
            )
            objets_moteurs[MODELE_BGE_DISTANT] = encodeur
            return orchestrer_emplois(
                emplois_actuels,
                emplois_cibles,
                encodeur,
                seuil_sim=seuil_sim,
                seuil_couv=seuil_couv,
                poids_dense=1.0,
                poids_sparse=0.0,
                modele=MODELE_BGE_DISTANT,
                type_score="score dense d'embedding (H=D)",
            )

        def lancer_gemma() -> ResultatOrchestration:
            evaluateur = charger_gemma(
                gemma_url,
                gemma_model,
                timeout_distant,
                api_key,
            )
            objets_moteurs[MODELE_GEMMA] = evaluateur
            return orchestrer_emplois_llm(
                emplois_actuels,
                emplois_cibles,
                evaluateur,
                modele=MODELE_GEMMA,
            )

        def lancer_faux() -> ResultatOrchestration:
            encodeur = charger_encodeur_fictif()
            objets_moteurs[MODELE_FAUX] = encodeur
            return orchestrer_emplois(
                emplois_actuels,
                emplois_cibles,
                encodeur,
                seuil_sim=seuil_sim,
                seuil_couv=seuil_couv,
                poids_dense=poids_dense,
                poids_sparse=poids_sparse,
                modele=MODELE_FAUX,
            )

        lanceurs = {
            MODELE_BGE: lancer_bge,
            MODELE_BGE_DISTANT: lancer_bge_distant,
            MODELE_QWEN: lancer_qwen,
            MODELE_GEMMA: lancer_gemma,
            MODELE_FAUX: lancer_faux,
        }
        selection_lanceurs = (
            {
                MODELE_BGE_DISTANT: lancer_bge_distant,
                MODELE_QWEN: lancer_qwen,
                MODELE_GEMMA: lancer_gemma,
            }
            if mode_execution == MODE_COMPARAISON_DISTANTE
            else {modele_unique: lanceurs[modele_unique]}
        )
        with st.spinner("Exécution des rapprochements en cours..."):
            executions = executer_comparaison(selection_lanceurs)

        st.markdown("### 3. Synthèse comparative")
        synthese_comparaison = construire_synthese_comparaison(
            executions,
            emplois_actuels,
        )
        st.dataframe(pd.DataFrame(synthese_comparaison), use_container_width=True)

        st.markdown("### 4. Résultats détaillés par modèle")
        for execution in executions:
            moteur = objets_moteurs.get(execution.modele)
            if moteur is None:
                st.markdown(f"## {execution.modele}")
                st.error(f"{execution.statut} : {execution.detail_erreur}")
                continue
            try:
                afficher_resultat_modele(execution, emplois_cibles, moteur)
            except Exception as err:
                st.warning(
                    "Le résultat principal reste disponible, mais le contrôle "
                    f"informatif complémentaire a échoué : {err}"
                )

        resultats_reussis = tuple(
            execution.resultat
            for execution in executions
            if execution.resultat is not None
        )
        matrice = [
            ligne
            for resultat in resultats_reussis
            for ligne in construire_matrice_couples(resultat)
        ]
        details = construire_details_comparaison(executions)
        st.markdown("### 5. Exports CSV comparatifs")
        st.download_button(
            "Télécharger la synthèse comparative",
            convertir_csv(synthese_comparaison),
            "comparaison_modeles.csv",
            "text/csv",
        )
        st.download_button(
            "Télécharger la matrice consolidée",
            convertir_csv(matrice),
            "matrice_modeles.csv",
            "text/csv",
        )
        st.download_button(
            "Télécharger les détails consolidés",
            convertir_csv(details),
            "details_modeles.csv",
            "text/csv",
        )


if __name__ == "__main__":
    main()
