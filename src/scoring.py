"""Calcul des indicateurs d'un unique couple d'emplois."""

from collections.abc import Iterable

from src.domain import (
    CorrespondanceFournie,
    CoupleEmplois,
    ResultatAnalyseCouple,
)
from src.embeddings import EncodeurCompetences
from src.matching import analyser_correspondance, generer_correspondances_semantiques
from src.recommendations import determiner_besoin_formation


def analyser_couple(
    couple: CoupleEmplois,
    correspondances_fournies: Iterable[CorrespondanceFournie],
) -> ResultatAnalyseCouple:
    """Calcule C_s, G, D_n et B pour un seul couple actuel/cible."""

    _valider_couple(couple)
    donnees = tuple(correspondances_fournies)
    _valider_couverture_des_competences(couple, donnees)
    donnees_par_cible = {item.competence_cible.id: item for item in donnees}
    donnees = tuple(
        donnees_par_cible[competence.id] for competence in couple.cible.competences
    )

    correspondances = tuple(analyser_correspondance(item) for item in donnees)
    nombre_cibles = len(correspondances)
    couverture_semantique = sum(item.reconnue for item in correspondances) / nombre_cibles
    score_global = sum(item.niveau_suffisant for item in correspondances) / nombre_cibles
    ecart_moyen_normalise = sum(
        item.ecart_niveau / item.niveau_requis for item in correspondances
    ) / nombre_cibles
    besoin_collectif = couple.actuel.effectif * ecart_moyen_normalise

    besoins_formation = tuple(
        besoin
        for item in correspondances
        if (besoin := determiner_besoin_formation(item)) is not None
    )

    return ResultatAnalyseCouple(
        emploi_actuel=couple.actuel,
        emploi_cible=couple.cible,
        correspondances=correspondances,
        besoins_formation=besoins_formation,
        couverture_semantique=couverture_semantique,
        score_global=score_global,
        ecart_moyen_normalise=ecart_moyen_normalise,
        besoin_collectif=besoin_collectif,
    )


def analyser_couple_semantiquement(
    couple: CoupleEmplois,
    encodeur: EncodeurCompetences,
) -> ResultatAnalyseCouple:
    """Rapproche les compétences puis délègue tous les calculs à la phase 2."""

    correspondances = generer_correspondances_semantiques(couple, encodeur)
    return analyser_couple(couple, correspondances)


def _valider_couple(couple: CoupleEmplois) -> None:
    if couple.actuel.type != "actuel":
        raise ValueError("Le premier emploi du couple doit être de type 'actuel'.")
    if couple.cible.type != "cible":
        raise ValueError("Le second emploi du couple doit être de type 'cible'.")
    if couple.actuel.effectif is None or couple.actuel.effectif <= 0:
        raise ValueError("L'emploi actuel doit avoir un effectif strictement positif.")
    if not couple.cible.competences:
        raise ValueError("L'emploi cible doit contenir au moins une compétence.")


def _valider_couverture_des_competences(
    couple: CoupleEmplois,
    donnees: tuple[CorrespondanceFournie, ...],
) -> None:
    attendues = {competence.id: competence for competence in couple.cible.competences}
    actuelles = set(couple.actuel.competences)
    vues: set[str] = set()

    for item in donnees:
        cible = item.competence_cible
        if cible.id not in attendues or cible != attendues[cible.id]:
            raise ValueError(
                f"La compétence cible '{cible.id}' n'appartient pas à l'emploi cible."
            )
        if cible.id in vues:
            raise ValueError(
                f"La compétence cible '{cible.id}' possède plusieurs correspondances fournies."
            )
        vues.add(cible.id)

        if item.competence_actuelle is not None and item.competence_actuelle not in actuelles:
            raise ValueError(
                f"La compétence actuelle '{item.competence_actuelle.id}' "
                "n'appartient pas à l'emploi actuel."
            )

    manquantes = set(attendues) - vues
    if manquantes:
        ids = ", ".join(sorted(manquantes))
        raise ValueError(f"Correspondance fournie absente pour: {ids}.")
