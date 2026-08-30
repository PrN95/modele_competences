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
    Emploi,
    CoupleEmplois,
    ResultatAnalyseCouple,
    ResultatSelectionCibles,
    SignalementCompetenceActuelle,
)
from src.embeddings import AdaptateurBGEM3, SortieEncodage, EncodeurCompetences
from src.io_pdf import read_emploi_pdf, PDFExtractionError
from src.scoring import (
    analyser_couple_semantiquement,
    selectionner_cibles,
    MESSAGE_AUCUNE_CIBLE,
    MESSAGE_ARBITRAGE_RH,
)
from src.matching import controler_competences_actuelles_non_reprises


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

    # Barre latérale : Configuration du modèle
    st.sidebar.header("Configuration du Modèle")
    mode_encodeur = st.sidebar.selectbox(
        "Moteur d'encodage sémantique",
        ["Faux Encodeur (Démo/Tests)", "Modèle BGE-M3 Local"],
        help="Permet d'utiliser un simulateur d'embeddings si le modèle BGE-M3 n'est pas présent localement.",
    )

    chemin_modele = st.sidebar.text_input(
        "Chemin du modèle BGE-M3 local",
        value=str(MODEL_PATH),
        disabled=(mode_encodeur == "Faux Encodeur (Démo/Tests)"),
    )

    # Initialisation de l'encodeur
    encodeur: EncodeurCompetences | None = None
    if mode_encodeur == "Modèle BGE-M3 Local":
        try:
            encodeur = charger_encodeur_reel(chemin_modele)
            st.sidebar.success("Modèle BGE-M3 configuré.")
        except Exception as e:
            st.sidebar.error(f"Erreur de configuration du modèle : {e}")
            st.sidebar.info("Utilisation temporaire du Faux Encodeur de secours.")
            encodeur = charger_encodeur_fictif()
    else:
        encodeur = charger_encodeur_fictif()
        st.sidebar.info("Mode Faux Encodeur (Démo) actif.")

    # Affichage des seuils configurés
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Seuils appliqués (config) :**")
    st.sidebar.markdown(f"- Seuil de similarité (\(seuil\_sim\)) : `{float(SEUIL_SIM):.2f}`")
    st.sidebar.markdown(f"- Seuil de couverture (\(seuil\_couv\)) : `{float(SEUIL_COUV):.2f}`")

    # Zone de chargement des PDF
    st.markdown("### 1. Chargement des documents PDF structurés")
    col_actuel, col_cible = st.columns(2)

    with col_actuel:
        st.markdown("#### Zone Emploi Actuel")
        fichiers_actuels = st.file_uploader(
            "Télécharger le ou les PDF d'emploi actuel",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_actuel",
        )

    with col_cible:
        st.markdown("#### Zone Métier Cible")
        fichiers_cibles = st.file_uploader(
            "Télécharger le ou les PDF de métier cible",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_cible",
        )

    if not fichiers_actuels or not fichiers_cibles:
        st.info("Veuillez charger au moins un emploi actuel et un métier cible pour démarrer l'analyse.")
        return

    # Phase d'extraction des PDF
    emplois_actuels: list[Emploi] = []
    emplois_cibles: list[Emploi] = []

    # Extraction des emplois actuels
    for f in fichiers_actuels:
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = Path(tmp.name)
            
            emploi = read_emploi_pdf(tmp_path, document_type="emploi_actuel")
            # Restaurer le vrai nom du fichier original dans les métadonnées
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
            tmp_path.unlink()
        except PDFExtractionError as err:
            st.error(f"Erreur d'extraction pour le fichier actuel '{f.name}' : {err}")
        except Exception as e:
            st.error(f"Erreur inattendue pour '{f.name}' : {e}")

    # Extraction des métiers cibles
    for f in fichiers_cibles:
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = Path(tmp.name)
            
            emploi = read_emploi_pdf(tmp_path, document_type="metier_cible")
            # Restaurer le vrai nom du fichier original dans les métadonnées
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
            tmp_path.unlink()
        except PDFExtractionError as err:
            st.error(f"Erreur d'extraction pour le fichier cible '{f.name}' : {err}")
        except Exception as e:
            st.error(f"Erreur inattendue pour '{f.name}' : {e}")

    if not emplois_actuels or not emplois_cibles:
        st.warning("Échec de l'extraction des profils. Veuillez vérifier les fichiers PDF.")
        return

    st.success(
        f"Extraction réussie : {len(emplois_actuels)} profil(s) actuel(s) et {len(emplois_cibles)} métier(s) cible(s) disponible(s)."
    )

    # Sélection de l'emploi actuel à analyser s'il y en a plusieurs
    st.markdown("---")
    st.markdown("### 2. Analyse du rapprochement")
    
    if len(emplois_actuels) > 1:
        options_actuels = {f"{e.intitule} ({e.fichier_source})": e for e in emplois_actuels}
        selection_actuel_label = st.selectbox(
            "Sélectionner l'emploi actuel à analyser",
            list(options_actuels.keys()),
        )
        emploi_actuel_selectionne = options_actuels[selection_actuel_label]
    else:
        emploi_actuel_selectionne = emplois_actuels[0]
        st.write(f"**Emploi actuel analysé :** {emploi_actuel_selectionne.intitule} (`{emploi_actuel_selectionne.fichier_source}`)")

    # Lancement du calcul sémantique et scoring
    analyses: list[ResultatAnalyseCouple] = []
    
    # Bouton explicite pour lancer l'encodage (très utile si modèle lourd)
    if st.button("Lancer l'analyse comparative", type="primary"):
        with st.spinner("Encodage sémantique et calcul des scores en cours..."):
            for cible in emplois_cibles:
                try:
                    couple = CoupleEmplois(actuel=emploi_actuel_selectionne, cible=cible)
                    # Analyse comparative hybride
                    analyse = analyser_couple_semantiquement(couple, encodeur)
                    analyses.append(analyse)
                except Exception as e:
                    st.error(f"Erreur lors de la comparaison avec '{cible.intitule}' : {e}")

        if not analyses:
            st.error("Aucune comparaison n'a pu être menée.")
            return

        # Phase de sélection globale
        selection_resultat = selectionner_cibles(analyses)

        # Affichage de l'alerte générale de sélection
        if selection_resultat.alerte:
            if selection_resultat.alerte == MESSAGE_AUCUNE_CIBLE:
                st.error(f"### {MESSAGE_AUCUNE_CIBLE}")
            elif selection_resultat.alerte == MESSAGE_ARBITRAGE_RH:
                st.warning(f"### ⚠️ {MESSAGE_ARBITRAGE_RH}")
        else:
            st.success("### Métier cible identifié avec succès")

        # Affichage synthétique des cibles admissibles et non admissibles
        st.markdown("#### Synthèse des scores par métier cible")
        
        synthese_data = []
        for ans in selection_resultat.analyses_classees:
            est_retenue = ans in selection_resultat.meilleures_analyses
            status_text = "Retenu ⭐" if est_retenue else ("Admissible" if ans.admissible else "Exclu (couverture insuffisante)")
            synthese_data.append({
                "Métier Cible": ans.emploi_cible.intitule,
                "Couverture Sémantique (G_ef)": f"{float(ans.g_ef):.2%}",
                "Satisfaction Niveaux (Gs_ef)": f"{float(ans.gs_ef):.2%}",
                "Écart Moyen Pondéré": f"{float(ans.ecart_moyen):.2f}",
                "Statut": status_text,
                "Fichier Source": ans.emploi_cible.fichier_source,
            })
            
        st.table(pd.DataFrame(synthese_data))

        # Détail pour chaque métier cible retenu / admissible
        if selection_resultat.meilleures_analyses:
            st.markdown("---")
            st.markdown("### 3. Détails des compétences et Recommandations de formation")
            
            for idx, analyse_retenue in enumerate(selection_resultat.meilleures_analyses):
                cible = analyse_retenue.emploi_cible
                st.markdown(f"#### 🎯 Métier retenu : **{cible.intitule}** (`{cible.fichier_source}`)")
                
                # Tableau détaillé des compétences
                details_competences = []
                for comp_corresp in analyse_retenue.correspondances:
                    # Recherche de la recommandation de formation associée
                    besoin = next(
                        (b for b in analyse_retenue.besoins_formation if b.competence_cible == comp_corresp.competence_cible),
                        None,
                    )
                    recommandation_text = besoin.commentaire if besoin else "Aucun commentaire particulier"

                    actuelle_nom = comp_corresp.competence_actuelle.intitule if comp_corresp.competence_actuelle else "N/A"
                    details_competences.append({
                        "Compétence Cible": comp_corresp.competence_cible.intitule,
                        "Niveau Cible": comp_corresp.niveau_requis,
                        "Compétence Actuelle Associée": actuelle_nom,
                        "Niveau Actuel": comp_corresp.niveau_actuel if comp_corresp.reconnue else 0,
                        "Score Dense (D_ac)": f"{comp_corresp.score_dense:.4f}",
                        "Score Sparse (L_ac)": f"{comp_corresp.score_sparse:.4f}",
                        "Score Hybride (H_ac)": f"{comp_corresp.score_hybride:.4f}",
                        "Reconnue": "Oui" if comp_corresp.reconnue else "Non",
                        "Écart Niveau": comp_corresp.ecart_niveau,
                        "Statut": comp_corresp.statut.replace("_", " ").title(),
                        "Recommandation de Formation": recommandation_text,
                    })
                
                df_details = pd.DataFrame(details_competences)
                st.dataframe(df_details, use_container_width=True)

                # Restitution spécifique par catégories : absentes ou niveaux insuffisants
                col_abs, col_insuff = st.columns(2)
                
                with col_abs:
                    st.markdown("**Compétences absentes ou non reconnues :**")
                    absentes = [c.competence_cible.intitule for c in analyse_retenue.correspondances if not c.reconnue]
                    if absentes:
                        for a in absentes:
                            st.write(f"- 🔴 {a} (Recommandation : *{details_competences[absentes.index(a)]['Recommandation de Formation']}*)")
                    else:
                        st.write("*Aucune compétence absente.*")
                
                with col_insuff:
                    st.markdown("**Compétences reconnues mais à niveau insuffisant :**")
                    insuffisantes = [
                        c for c in analyse_retenue.correspondances 
                        if c.reconnue and c.statut == "niveau_insuffisant"
                    ]
                    if insuffisantes:
                        for ins in insuffisantes:
                            rec_text = next(
                                (b.commentaire for b in analyse_retenue.besoins_formation if b.competence_cible == ins.competence_cible),
                                "N/A"
                            )
                            st.write(f"- 🟡 **{ins.competence_cible.intitule}** : requis niveau {ins.niveau_requis}, actuel {ins.niveau_actuel} (Écart: {ins.ecart_niveau})")
                            st.caption(f"  *Recommandation : {rec_text}*")
                    else:
                        st.write("*Aucun niveau insuffisant.*")

            # Section 4: Contrôle Inverse (Compétences actuelles non reprises)
            st.markdown("---")
            st.markdown("### 4. Contrôle Inverse & Traçabilité")
            
            signalements = controler_competences_actuelles_non_reprises(
                emploi_actuel=emploi_actuel_selectionne,
                emplois_cibles=emplois_cibles,
                emplois_cibles_selectionnes=tuple(ans.emploi_cible for ans in selection_resultat.meilleures_analyses),
                encodeur=encodeur,
            )
            
            st.markdown("**Compétences de l'emploi actuel non reprises dans les métiers cibles analysés :**")
            if signalements:
                sig_data = []
                for sig in signalements:
                    st.write(f"- ℹ️ **{sig.competence_actuelle.intitule}** : {sig.message}")
                    sig_data.append({
                        "Compétence Actuelle": sig.competence_actuelle.intitule,
                        "Type de Signalement": sig.type_non_reprise.replace("_", " ").title(),
                        "Meilleure Cible": sig.meilleur_emploi_cible.intitule if sig.meilleur_emploi_cible else "N/A",
                        "Meilleure Compétence Cible": sig.meilleure_competence_cible.intitule if sig.meilleure_competence_cible else "N/A",
                        "Score Hybride": f"{sig.score_hybride:.4f}" if sig.score_hybride else "N/A",
                        "Message": sig.message,
                    })
                
                # Permettre l'exportation CSV de ce contrôle
                df_sig = pd.DataFrame(sig_data)
            else:
                st.write("*Toutes les compétences actuelles ont été valorisées dans au moins un des métiers cibles retenus.*")
                df_sig = pd.DataFrame()

            # Section 5: Exportation Globale CSV
            st.markdown("---")
            st.markdown("### 5. Exportation CSV des Résultats")
            
            # Construction d'un export CSV unifié contenant toutes les lignes de correspondances
            export_rows = []
            for ans in selection_resultat.analyses_classees:
                est_retenue = ans in selection_resultat.meilleures_analyses
                for comp_corresp in ans.correspondances:
                    besoin = next(
                        (b for b in ans.besoins_formation if b.competence_cible == comp_corresp.competence_cible),
                        None,
                    )
                    recommandation_text = besoin.commentaire if besoin else ("Aucun commentaire particulier" if comp_corresp.reconnue else "Formation complète nécessaire pour acquérir la compétence")
                    
                    export_rows.append({
                        "Emploi_Actuel": emploi_actuel_selectionne.intitule,
                        "Fichier_Source_Actuel": emploi_actuel_selectionne.fichier_source,
                        "Metier_Cible": ans.emploi_cible.intitule,
                        "Fichier_Source_Cible": ans.emploi_cible.fichier_source,
                        "G_ef_Couverture": float(ans.g_ef),
                        "Gs_ef_Satisfaction": float(ans.gs_ef),
                        "Ecart_Moyen_ef": float(ans.ecart_moyen),
                        "Est_Admissible": ans.admissible,
                        "Est_Selectionne": est_retenue,
                        "Competence_Cible": comp_corresp.competence_cible.intitule,
                        "Niveau_Cible": comp_corresp.niveau_requis,
                        "Competence_Actuelle_Associee": comp_corresp.competence_actuelle.intitule if comp_corresp.competence_actuelle else "N/A",
                        "Niveau_Actuel": comp_corresp.niveau_actuel if comp_corresp.reconnue else 0,
                        "Score_Dense_D_ac": comp_corresp.score_dense,
                        "Score_Sparse_L_ac": comp_corresp.score_sparse,
                        "Score_Hybride_H_ac": comp_corresp.score_hybride,
                        "Reconnue": comp_corresp.reconnue,
                        "Ecart_Niveau": comp_corresp.ecart_niveau,
                        "Statut": comp_corresp.statut,
                        "Recommandation_Formation": recommandation_text,
                    })
            
            df_export = pd.DataFrame(export_rows)
            
            # Création du buffer CSV en mémoire
            csv_buffer = io.StringIO()
            df_export.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="Télécharger le rapport détaillé au format CSV",
                data=csv_data,
                file_name=f"rapport_comparatif_{emploi_actuel_selectionne.intitule.replace(' ', '_').lower()}.csv",
                mime="text/csv",
            )
            
            if not df_sig.empty:
                csv_sig_buffer = io.StringIO()
                df_sig.to_csv(csv_sig_buffer, index=False, encoding="utf-8-sig")
                st.download_button(
                    label="Télécharger le rapport de contrôle inverse (Compétences non reprises) en CSV",
                    data=csv_sig_buffer.getvalue(),
                    file_name=f"controle_inverse_{emploi_actuel_selectionne.intitule.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()

