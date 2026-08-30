import streamlit as st
import pandas as pd
import tempfile
import io
import hashlib
from pathlib import Path
from typing import Sequence

from src.config import MODEL_PATH, SEUIL_COUV, SEUIL_SIM
from src.domain import (
    Competence,
    CorrespondanceCompetence,
    Emploi,
    ResultatAnalyseCouple,
    ResultatSelectionCibles,
)
from src.embeddings import AdaptateurBGEM3, SortieEncodage, EncodeurCompetences
from src.io_pdf import read_emploi_pdf, PDFExtractionError
from src.orchestration import (
    ResultatOrchestration,
    orchestrer_emplois,
)
from src.scoring import MESSAGE_AUCUNE_CIBLE, MESSAGE_ARBITRAGE_RH
from src.matching import controler_competences_actuelles_non_reprises


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
    seuil_sim: float = float(SEUIL_SIM),
) -> list[dict[str, object]]:
    """Construit la restitution détaillée d'une cible réellement retenue."""

    details: list[dict[str, object]] = []
    for correspondance in analyse_selectionnee.correspondances:
        details.append(
            {
                "Compétence Cible": correspondance.competence_cible.intitule,
                "Niveau Cible": correspondance.niveau_requis,
                "Compétence actuelle la plus proche": (
                    libelle_competence_actuelle_la_plus_proche(correspondance)
                ),
                "Niveau Actuel": (
                    correspondance.niveau_actuel
                    if correspondance.reconnue
                    else "N/A"
                ),
                "Score Dense (D_ac)": f"{correspondance.score_dense:.4f}",
                "Score Sparse (L_ac)": f"{correspondance.score_sparse:.4f}",
                "Score Hybride (H_ac)": f"{correspondance.score_hybride:.4f}",
                "Seuil de similarité utilisé": seuil_sim,
                "Reconnue": "Oui" if correspondance.reconnue else "Non",
                "Écart Niveau": (
                    correspondance.ecart_niveau
                    if correspondance.reconnue
                    else "N/A"
                ),
                "Statut": correspondance.statut.replace("_", " ").title(),
                "Détail Égalité": correspondance.detail_egalite or "",
                "Recommandation de Formation": recommandation_pour_correspondance(
                    analyse_selectionnee,
                    correspondance.competence_cible,
                ),
            }
        )
    return details


def construire_lignes_export(
    selection: ResultatSelectionCibles,
    *,
    seuil_sim: float,
    seuil_couv: float,
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
                    "Emploi_Actuel": analyse.emploi_actuel.intitule,
                    "Fichier_Source_Actuel": analyse.emploi_actuel.fichier_source,
                    "Emploi_Cible": analyse.emploi_cible.intitule,
                    "Fichier_Source_Cible": analyse.emploi_cible.fichier_source,
                    "Seuil_Sim": seuil_sim,
                    "Seuil_Couv": seuil_couv,
                    "G_ef_Couverture": float(analyse.g_ef),
                    "Gs_ef_Satisfaction": float(analyse.gs_ef),
                    "Ecart_Moyen_ef": float(analyse.ecart_moyen),
                    "Est_Admissible": analyse.admissible,
                    "Est_Selectionne": est_selectionnee,
                    "Competence_Cible": correspondance.competence_cible.intitule,
                    "Niveau_Cible": correspondance.niveau_requis,
                    "Compétence actuelle la plus proche": (
                        libelle_competence_actuelle_la_plus_proche(correspondance)
                    ),
                    "Niveau_Actuel": (
                        correspondance.niveau_actuel
                        if correspondance.reconnue
                        else "N/A"
                    ),
                    "Score_Dense_D_ac": correspondance.score_dense,
                    "Score_Sparse_L_ac": correspondance.score_sparse,
                    "Score_Hybride_H_ac": correspondance.score_hybride,
                    "Reconnue": correspondance.reconnue,
                    "Ecart_Niveau": (
                        correspondance.ecart_niveau
                        if correspondance.reconnue
                        else "N/A"
                    ),
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
            lignes.append(
                {
                    "Emploi_Actuel": resultat_emploi.emploi_actuel.intitule,
                    "Fichier_Source_Actuel": resultat_emploi.emploi_actuel.fichier_source,
                    "Emplois_Cibles_Retenus": "",
                    "Statut": MESSAGE_AUCUNE_CIBLE,
                    "G_epfq": None,
                    "Gs_epfq": None,
                    "Ecart_Moyen_epfq": None,
                    "R_epfq": None,
                    "Seuil_Sim": resultat.seuil_sim,
                    "Seuil_Couv": resultat.seuil_couv,
                }
            )
            continue

        analyse_reference = retenues[0].analyse
        lignes.append(
            {
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
                "Ecart_Moyen_epfq": float(analyse_reference.ecart_moyen),
                "R_epfq": " | ".join(
                    f"{cible.analyse.emploi_cible.intitule}: {cible.r_epfq:.2%}"
                    for cible in retenues
                ),
                "Seuil_Sim": resultat.seuil_sim,
                "Seuil_Couv": resultat.seuil_couv,
            }
        )
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
            lignes.append(
                {
                    "Emploi_Actuel": analyse.emploi_actuel.intitule,
                    "Fichier_Source_Actuel": analyse.emploi_actuel.fichier_source,
                    "Emploi_Cible": analyse.emploi_cible.intitule,
                    "Fichier_Source_Cible": analyse.emploi_cible.fichier_source,
                    "G_epfq": float(analyse.g_ef),
                    "Gs_epfq": float(analyse.gs_ef),
                    "Ecart_Moyen_epfq": float(analyse.ecart_moyen),
                    "Est_Admissible": analyse.admissible,
                    "Est_Selectionne": est_selectionnee,
                    "R_epfq": (
                        reutilisation_par_cible[cle] if est_selectionnee else None
                    ),
                    "Seuil_Sim": resultat.seuil_sim,
                    "Seuil_Couv": resultat.seuil_couv,
                }
            )
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
                lignes.append(
                    {
                        "Emploi_Actuel": analyse.emploi_actuel.intitule,
                        "Fichier_Source_Actuel": analyse.emploi_actuel.fichier_source,
                        "Emploi_Cible_Retenu": analyse.emploi_cible.intitule,
                        "Fichier_Source_Cible": analyse.emploi_cible.fichier_source,
                        "G_epfq": float(analyse.g_ef),
                        "Gs_epfq": float(analyse.gs_ef),
                        "Ecart_Moyen_epfq": float(analyse.ecart_moyen),
                        "R_epfq": cible_retenue.r_epfq,
                        "Seuil_Sim": resultat.seuil_sim,
                        "Seuil_Couv": resultat.seuil_couv,
                        "Competence_Cible": correspondance.competence_cible.intitule,
                        "Niveau_Cible": correspondance.niveau_requis,
                        "Compétence actuelle la plus proche": (
                            libelle_competence_actuelle_la_plus_proche(
                                correspondance
                            )
                        ),
                        "Niveau_Actuel": (
                            correspondance.niveau_actuel
                            if correspondance.reconnue
                            else "N/A"
                        ),
                        "Score_Dense_D_ac": correspondance.score_dense,
                        "Score_Sparse_L_ac": correspondance.score_sparse,
                        "Score_Hybride_H_ac": correspondance.score_hybride,
                        "Reconnue": correspondance.reconnue,
                        "Ecart_Niveau": (
                            correspondance.ecart_niveau
                            if correspondance.reconnue
                            else "N/A"
                        ),
                        "Statut": (
                            "Absente"
                            if not correspondance.reconnue
                            else correspondance.statut
                        ),
                        "Detail_Egalite": correspondance.detail_egalite or "",
                        "Recommandation_Formation": (
                            recommandation_pour_correspondance(
                                analyse,
                                correspondance.competence_cible,
                            )
                        ),
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


def main():
    st.set_page_config(
        page_title="PoC Aide à la Décision RH - Rapprochement d'Emplois",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("PoC Aide à la Décision RH")
    st.subheader("Rapprochement sémantique d'emplois, scoring et recommandations de formation")

    st.sidebar.header("Configuration du modèle")
    mode_encodeur = st.sidebar.selectbox(
        "Moteur d'encodage sémantique",
        ["Modèle BGE-M3 Local", "Faux Encodeur (Démo/Tests)"],
        help=(
            "Le modèle BGE-M3 local est le mode normal. Le faux encodeur est "
            "réservé aux démonstrations et aux tests explicites."
        ),
    )

    chemin_modele = st.sidebar.text_input(
        "Chemin du modèle BGE-M3 local",
        value=str(MODEL_PATH),
        disabled=(mode_encodeur == "Faux Encodeur (Démo/Tests)"),
    )

    encodeur: EncodeurCompetences | None = None
    if mode_encodeur == "Modèle BGE-M3 Local":
        try:
            encodeur = charger_encodeur_reel(chemin_modele)
            st.sidebar.success("Modèle BGE-M3 configuré.")
        except Exception as e:
            st.sidebar.error(f"Erreur de configuration du modèle : {e}")
            st.error(
                "Le mode normal exige le modèle BGE-M3 local. Corrigez son "
                "chemin ou choisissez explicitement le mode de démonstration."
            )
            return
    else:
        encodeur = charger_encodeur_fictif()
        st.sidebar.info("Mode Faux Encodeur (Démo) actif.")

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

            emploi = read_emploi_pdf(tmp_path, document_type="emploi_actuel")
            emploi = Emploi(
                intitule=emploi.intitule,
                type=emploi.type,
                effectif=emploi.effectif,
                competences=emploi.competences,
                fichier_source=f.name,
                feuille_source=None,
                id=emploi.id,
            )
            emplois_actuels.append(emploi)
        except PDFExtractionError as err:
            st.error(f"Erreur d'extraction pour le fichier actuel '{f.name}' : {err}")
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

            emploi = read_emploi_pdf(tmp_path, document_type="metier_cible")
            emploi = Emploi(
                intitule=emploi.intitule,
                type=emploi.type,
                effectif=emploi.effectif,
                competences=emploi.competences,
                fichier_source=f.name,
                feuille_source=None,
                id=emploi.id,
            )
            emplois_cibles.append(emploi)
        except PDFExtractionError as err:
            st.error(f"Erreur d'extraction pour le fichier cible '{f.name}' : {err}")
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

    if st.button("Lancer l'analyse globale", type="primary"):
        with st.spinner("Encodage sémantique et calcul des scores en cours..."):
            try:
                resultat = orchestrer_emplois(
                    emplois_actuels,
                    emplois_cibles,
                    encodeur,
                    seuil_sim=seuil_sim,
                    seuil_couv=seuil_couv,
                )
            except Exception as err:
                st.error(f"L'analyse globale a échoué : {err}")
                return

        st.caption(
            f"Seuils utilisés — similarité : {seuil_sim:.2f} ; "
            f"couverture : {seuil_couv:.2f}"
        )

        synthese = construire_synthese_orchestration(resultat)
        matrice = construire_matrice_couples(resultat)
        details = construire_details_orchestration(resultat)

        st.markdown("#### Synthèse par emploi actuel")
        st.dataframe(pd.DataFrame(synthese), use_container_width=True)

        st.markdown("#### Matrice complète des couples analysés")
        st.dataframe(pd.DataFrame(matrice), use_container_width=True)

        st.markdown("---")
        st.markdown("### 3. Cibles retenues, compétences et recommandations")

        for resultat_emploi in resultat.resultats_emplois:
            emploi_actuel = resultat_emploi.emploi_actuel
            st.markdown(
                f"#### Emploi actuel : **{emploi_actuel.intitule}** "
                f"(`{emploi_actuel.fichier_source}`)"
            )
            if not resultat_emploi.cibles_retenues:
                st.error(MESSAGE_AUCUNE_CIBLE)
                continue
            if len(resultat_emploi.cibles_retenues) > 1:
                st.warning(MESSAGE_ARBITRAGE_RH)

            for cible_retenue in resultat_emploi.cibles_retenues:
                analyse = cible_retenue.analyse
                cible = analyse.emploi_cible
                st.markdown(
                    f"##### Emploi cible retenu : **{cible.intitule}** "
                    f"(`{cible.fichier_source}`)"
                )
                col_g, col_gs, col_ecart, col_r = st.columns(4)
                col_g.metric("G_epfq", f"{analyse.g_ef:.2%}")
                col_gs.metric("Gs_epfq", f"{analyse.gs_ef:.2%}")
                col_ecart.metric("Écart moyen", f"{analyse.ecart_moyen:.2f}")
                col_r.metric("R_epfq", f"{cible_retenue.r_epfq:.2%}")
                st.dataframe(
                    pd.DataFrame(
                        construire_details_competences(
                            analyse,
                            seuil_sim=resultat.seuil_sim,
                        )
                    ),
                    use_container_width=True,
                )

            signalements = controler_competences_actuelles_non_reprises(
                emploi_actuel=emploi_actuel,
                emplois_cibles=emplois_cibles,
                emplois_cibles_selectionnes=tuple(
                    cible.analyse.emploi_cible
                    for cible in resultat_emploi.cibles_retenues
                ),
                encodeur=encodeur,
                seuil_sim=seuil_sim,
            )
            st.markdown("**Compétences actuelles non reprises :**")
            if signalements:
                for signalement in signalements:
                    st.write(
                        f"- ℹ️ **{signalement.competence_actuelle.intitule}** : "
                        f"{signalement.message}"
                    )
            else:
                st.write("*Toutes les compétences actuelles sont réutilisées.*")

        st.markdown("---")
        st.markdown("### 4. Exports CSV consolidés")
        st.download_button(
            label="Télécharger la synthèse globale",
            data=convertir_csv(synthese),
            file_name="synthese_globale_emplois.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Télécharger la matrice complète des couples",
            data=convertir_csv(matrice),
            file_name="matrice_emplois_actuels_cibles.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Télécharger les détails des cibles retenues",
            data=convertir_csv(details),
            file_name="details_cibles_retenues.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
