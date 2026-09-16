"""Orchestration de la matrice emplois actuels × emplois cibles."""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from src.config import DENSE_WEIGHT, SEUIL_COUV, SEUIL_SIM, SPARSE_WEIGHT
from src.domain import (
    CorrespondanceCompetence,
    CorrespondanceFournie,
    CoupleEmplois,
    Emploi,
    ResultatAnalyseCouple,
    ResultatSelectionCibles,
)
from src.embeddings import EncodeurCompetences
from src.matching import (
    analyser_correspondance,
    calculer_score_dense,
    calculer_score_hybride,
    calculer_score_sparse,
    valider_poids_hybrides,
)
from src.scoring import (
    analyser_couple_semantiquement,
    selectionner_cibles,
)
from src.remote_models import SelectionGemma


class SelecteurEmploiGemma(Protocol):
    def selectionner(
        self,
        emploi_actuel: Emploi,
        emplois_cibles: Sequence[Emploi],
    ) -> SelectionGemma: ...


@dataclass(frozen=True, slots=True)
class CibleRetenueAvecReutilisation:
    """Emploi cible retenu, enrichi de l'indicateur informatif ``R_epfq``."""

    analyse: ResultatAnalyseCouple
    r_epfq: float


@dataclass(frozen=True, slots=True)
class ResultatOrchestrationEmploi:
    """Toutes les analyses et la sélection associées à un emploi actuel."""

    emploi_actuel: Emploi
    selection: ResultatSelectionCibles
    cibles_retenues: tuple[CibleRetenueAvecReutilisation, ...]
    reponse_brute_gemma: str | None = None


@dataclass(frozen=True, slots=True)
class ResultatOrchestration:
    """Résultat consolidé de la matrice complète d'emplois."""

    resultats_emplois: tuple[ResultatOrchestrationEmploi, ...]
    seuil_sim: float | None
    seuil_couv: float | None
    poids_dense: float | None = float(DENSE_WEIGHT)
    poids_sparse: float | None = float(SPARSE_WEIGHT)
    modele: str = "BGE-M3 local"
    type_score: str = "score hybride d'embedding"


def orchestrer_emplois(
    emplois_actuels: Sequence[Emploi],
    emplois_cibles: Sequence[Emploi],
    encodeur: EncodeurCompetences,
    *,
    seuil_sim: float | Fraction | None = None,
    seuil_couv: float | Fraction | None = None,
    poids_dense: float | Fraction | None = None,
    poids_sparse: float | Fraction | None = None,
    modele: str = "BGE-M3 local",
    type_score: str = "score hybride d'embedding",
) -> ResultatOrchestration:
    """Analyse chaque emploi actuel contre chaque emploi cible, puis sélectionne."""

    actuels = tuple(emplois_actuels)
    cibles = tuple(emplois_cibles)
    _valider_emplois(actuels, cibles)
    seuil_sim_effectif = seuil_sim if seuil_sim is not None else SEUIL_SIM
    seuil_couv_effectif = seuil_couv if seuil_couv is not None else SEUIL_COUV
    poids_dense_effectif, poids_sparse_effectif = valider_poids_hybrides(
        poids_dense,
        poids_sparse,
    )

    resultats: list[ResultatOrchestrationEmploi] = []
    for emploi_actuel in actuels:
        analyses = tuple(
            analyser_couple_semantiquement(
                CoupleEmplois(actuel=emploi_actuel, cible=emploi_cible),
                encodeur,
                seuil_sim=seuil_sim_effectif,
                seuil_couv=seuil_couv_effectif,
                poids_dense=poids_dense_effectif,
                poids_sparse=poids_sparse_effectif,
            )
            for emploi_cible in cibles
        )
        selection = selectionner_cibles(analyses)
        cibles_retenues = tuple(
            CibleRetenueAvecReutilisation(
                analyse=analyse,
                r_epfq=calculer_r_epfq(
                    emploi_actuel,
                    analyse.emploi_cible,
                    encodeur,
                    seuil_sim=seuil_sim_effectif,
                    poids_dense=poids_dense_effectif,
                    poids_sparse=poids_sparse_effectif,
                ),
            )
            for analyse in selection.meilleures_analyses
        )
        resultats.append(
            ResultatOrchestrationEmploi(
                emploi_actuel=emploi_actuel,
                selection=selection,
                cibles_retenues=cibles_retenues,
            )
        )

    return ResultatOrchestration(
        resultats_emplois=tuple(resultats),
        seuil_sim=float(seuil_sim_effectif),
        seuil_couv=float(seuil_couv_effectif),
        poids_dense=float(poids_dense_effectif),
        poids_sparse=float(poids_sparse_effectif),
        modele=modele,
        type_score=type_score,
    )


def orchestrer_emplois_llm(
    emplois_actuels: Sequence[Emploi],
    emplois_cibles: Sequence[Emploi],
    selecteur: SelecteurEmploiGemma,
    *,
    modele: str = "Gemma 4 distant",
) -> ResultatOrchestration:
    """Restitue le raisonnement autonome de Gemma, sans le recalculer."""

    actuels = tuple(emplois_actuels)
    cibles = tuple(emplois_cibles)
    _valider_emplois(actuels, cibles)
    resultats: list[ResultatOrchestrationEmploi] = []
    for emploi_actuel in actuels:
        decision = selecteur.selectionner(emploi_actuel, cibles)
        if not any(decision.emploi_cible is cible for cible in cibles):
            raise ValueError("Gemma a sélectionné un emploi cible inconnu.")
        if None in (decision.g_epfq, decision.gs_epfq, decision.ecart_moyen, decision.r_epfq):
            raise ValueError("La réponse Gemma de passerelle est incomplète.")
        correspondances = tuple(
            CorrespondanceCompetence(
                item.competence_cible, item.competence_actuelle, None, None, None, None,
                item.statut == "Reconnue", item.niveau_actuel, item.ecart_niveau, item.statut_niveau,
                recommandation_llm=item.recommandation_formation,
            ) for item in decision.correspondances
        )
        analyse = ResultatAnalyseCouple(emploi_actuel, decision.emploi_cible, correspondances, (), decision.g_epfq, decision.g_epfq, decision.gs_epfq, decision.ecart_moyen, True)
        selection = ResultatSelectionCibles(
            analyses_classees=(analyse,),
            analyses_admissibles=(analyse,),
            meilleures_analyses=(analyse,),
            alerte=None,
        )
        retenues = (
            CibleRetenueAvecReutilisation(
                analyse=analyse,
                r_epfq=decision.r_epfq,
            ),
        )
        resultats.append(
            ResultatOrchestrationEmploi(emploi_actuel, selection, retenues, decision.reponse_brute)
        )
    return ResultatOrchestration(
        resultats_emplois=tuple(resultats),
        seuil_sim=None,
        seuil_couv=None,
        poids_dense=None,
        poids_sparse=None,
        modele=modele,
        type_score="décision catégorielle LLM",
    )


def calculer_r_epfq(
    emploi_actuel: Emploi,
    emploi_cible: Emploi,
    encodeur: EncodeurCompetences,
    *,
    seuil_sim: float | Fraction | None = None,
    poids_dense: float | Fraction | None = None,
    poids_sparse: float | Fraction | None = None,
) -> float:
    """Calcule la part des compétences actuelles réutilisées dans une cible retenue."""

    if emploi_actuel.type != "actuel" or not emploi_actuel.competences:
        raise ValueError("R_epfq exige un emploi actuel contenant des compétences.")
    if emploi_cible.type != "cible" or not emploi_cible.competences:
        raise ValueError("R_epfq exige un emploi cible contenant des compétences.")
    seuil_effectif = seuil_sim if seuil_sim is not None else SEUIL_SIM
    poids_dense_effectif, poids_sparse_effectif = valider_poids_hybrides(
        poids_dense,
        poids_sparse,
    )
    encodage_actuel = encodeur.encoder(emploi_actuel.competences)
    encodage_cible = encodeur.encoder(emploi_cible.competences)
    nombre_reutilisees = 0

    for index_actuel, competence_actuelle in enumerate(emploi_actuel.competences):
        candidates: list[CorrespondanceFournie] = []
        for index_cible, competence_cible in enumerate(emploi_cible.competences):
            score_dense = (
                calculer_score_dense(
                    encodage_actuel.vecteurs_dense[index_actuel],
                    encodage_cible.vecteurs_dense[index_cible],
                )
                if poids_dense_effectif > 0
                else None
            )
            score_sparse = (
                calculer_score_sparse(
                    encodage_actuel.poids_sparse[index_actuel],
                    encodage_cible.poids_sparse[index_cible],
                )
                if poids_sparse_effectif > 0
                else None
            )
            candidates.append(
                CorrespondanceFournie(
                    competence_cible=competence_cible,
                    competence_actuelle=competence_actuelle,
                    score_dense=score_dense,
                    score_sparse=score_sparse,
                )
            )

        meilleure = max(
            candidates,
            key=lambda item: calculer_score_hybride(
                item.score_dense,
                item.score_sparse,
                poids_dense_effectif,
                poids_sparse_effectif,
            ),
        )
        if analyser_correspondance(
            meilleure,
            seuil_effectif,
            poids_dense=poids_dense_effectif,
            poids_sparse=poids_sparse_effectif,
        ).reconnue:
            nombre_reutilisees += 1

    return nombre_reutilisees / len(emploi_actuel.competences)


def _valider_emplois(
    emplois_actuels: tuple[Emploi, ...],
    emplois_cibles: tuple[Emploi, ...],
) -> None:
    if not emplois_actuels:
        raise ValueError("Au moins un emploi actuel est obligatoire.")
    if not emplois_cibles:
        raise ValueError("Au moins un emploi cible est obligatoire.")
    if any(emploi.type != "actuel" for emploi in emplois_actuels):
        raise ValueError("La collection des emplois actuels contient un type invalide.")
    if any(emploi.type != "cible" for emploi in emplois_cibles):
        raise ValueError("La collection des emplois cibles contient un type invalide.")
