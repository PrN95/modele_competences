"""Orchestration de la matrice emplois actuels × emplois cibles."""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from src.config import SEUIL_COUV, SEUIL_SIM
from src.domain import (
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
)
from src.scoring import analyser_couple_semantiquement, selectionner_cibles


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


@dataclass(frozen=True, slots=True)
class ResultatOrchestration:
    """Résultat consolidé de la matrice complète d'emplois."""

    resultats_emplois: tuple[ResultatOrchestrationEmploi, ...]
    seuil_sim: float
    seuil_couv: float


def orchestrer_emplois(
    emplois_actuels: Sequence[Emploi],
    emplois_cibles: Sequence[Emploi],
    encodeur: EncodeurCompetences,
    *,
    seuil_sim: float | Fraction | None = None,
    seuil_couv: float | Fraction | None = None,
) -> ResultatOrchestration:
    """Analyse chaque emploi actuel contre chaque emploi cible, puis sélectionne."""

    actuels = tuple(emplois_actuels)
    cibles = tuple(emplois_cibles)
    _valider_emplois(actuels, cibles)
    seuil_sim_effectif = seuil_sim if seuil_sim is not None else SEUIL_SIM
    seuil_couv_effectif = seuil_couv if seuil_couv is not None else SEUIL_COUV

    resultats: list[ResultatOrchestrationEmploi] = []
    for emploi_actuel in actuels:
        analyses = tuple(
            analyser_couple_semantiquement(
                CoupleEmplois(actuel=emploi_actuel, cible=emploi_cible),
                encodeur,
                seuil_sim=seuil_sim_effectif,
                seuil_couv=seuil_couv_effectif,
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
    )


def calculer_r_epfq(
    emploi_actuel: Emploi,
    emploi_cible: Emploi,
    encodeur: EncodeurCompetences,
    *,
    seuil_sim: float | Fraction | None = None,
) -> float:
    """Calcule la part des compétences actuelles réutilisées dans une cible retenue."""

    if emploi_actuel.type != "actuel" or not emploi_actuel.competences:
        raise ValueError("R_epfq exige un emploi actuel contenant des compétences.")
    if emploi_cible.type != "cible" or not emploi_cible.competences:
        raise ValueError("R_epfq exige un emploi cible contenant des compétences.")
    seuil_effectif = seuil_sim if seuil_sim is not None else SEUIL_SIM
    encodage_actuel = encodeur.encoder(emploi_actuel.competences)
    encodage_cible = encodeur.encoder(emploi_cible.competences)
    nombre_reutilisees = 0

    for index_actuel, competence_actuelle in enumerate(emploi_actuel.competences):
        candidates: list[CorrespondanceFournie] = []
        for index_cible, competence_cible in enumerate(emploi_cible.competences):
            score_dense = calculer_score_dense(
                encodage_actuel.vecteurs_dense[index_actuel],
                encodage_cible.vecteurs_dense[index_cible],
            )
            score_sparse = calculer_score_sparse(
                encodage_actuel.poids_sparse[index_actuel],
                encodage_cible.poids_sparse[index_cible],
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
            ),
        )
        if analyser_correspondance(meilleure, seuil_effectif).reconnue:
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
