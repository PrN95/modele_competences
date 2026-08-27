"""Calcul des indicateurs et sélection des emplois cibles."""

from collections.abc import Iterable

from src.config import GLOBAL_MATCH_ALERT_THRESHOLD
from src.domain import (
    CorrespondanceFournie,
    CoupleEmplois,
    ResultatAnalyseCouple,
    ResultatSelectionCibles,
)
from src.embeddings import EncodeurCompetences
from src.matching import analyser_correspondance, generer_correspondances_semantiques
from src.recommendations import determiner_besoin_formation


def analyser_couple(
    couple: CoupleEmplois,
    correspondances_fournies: Iterable[CorrespondanceFournie],
) -> ResultatAnalyseCouple:
    """Calcule la couverture, ``G_ef`` et ``Ecart_moyen_ef`` d'un couple."""

    _valider_couple(couple)
    donnees = tuple(correspondances_fournies)
    donnees = _ordonner_correspondances(couple, donnees)

    correspondances = tuple(analyser_correspondance(item) for item in donnees)
    nombre_cibles = len(correspondances)
    couverture_semantique = sum(item.reconnue for item in correspondances) / nombre_cibles
    score_global = sum(item.niveau_suffisant for item in correspondances) / nombre_cibles
    ecart_moyen = sum(item.ecart_niveau for item in correspondances) / nombre_cibles

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
        ecart_moyen=ecart_moyen,
    )


def analyser_couple_semantiquement(
    couple: CoupleEmplois,
    encodeur: EncodeurCompetences,
) -> ResultatAnalyseCouple:
    """Rapproche les compétences puis délègue tous les calculs à la phase 2."""

    correspondances = generer_correspondances_semantiques(couple, encodeur)
    return analyser_couple(couple, correspondances)


def selectionner_cibles(
    analyses: Iterable[ResultatAnalyseCouple],
) -> ResultatSelectionCibles:
    """Classe les cibles par ``G_ef`` puis ``Ecart_moyen_ef``.

    Toutes les cibles restant à égalité sur ces deux indicateurs sont
    conservées. L'ordre d'entrée est utilisé uniquement comme ordre stable
    d'affichage et ne départage jamais une égalité métier.
    """

    donnees = tuple(analyses)
    if not donnees:
        raise ValueError("Au moins une analyse de cible est obligatoire.")

    emploi_actuel = donnees[0].emploi_actuel
    if any(analyse.emploi_actuel is not emploi_actuel for analyse in donnees[1:]):
        raise ValueError("Toutes les analyses doivent concerner le même emploi actuel.")

    analyses_classees = tuple(
        sorted(donnees, key=lambda item: (-item.score_global, item.ecart_moyen))
    )
    meilleur_score = analyses_classees[0].score_global
    meilleur_ecart = min(
        item.ecart_moyen
        for item in analyses_classees
        if item.score_global == meilleur_score
    )
    meilleures = tuple(
        item
        for item in analyses_classees
        if item.score_global == meilleur_score and item.ecart_moyen == meilleur_ecart
    )
    alerte = (
        "correspondance globale faible"
        if meilleur_score < float(GLOBAL_MATCH_ALERT_THRESHOLD)
        else None
    )
    return ResultatSelectionCibles(
        analyses_classees=analyses_classees,
        meilleures_analyses=meilleures,
        alerte=alerte,
    )


def _valider_couple(couple: CoupleEmplois) -> None:
    if couple.actuel.type != "actuel":
        raise ValueError("Le premier emploi du couple doit être de type 'actuel'.")
    if couple.cible.type != "cible":
        raise ValueError("Le second emploi du couple doit être de type 'cible'.")
    if not couple.cible.competences:
        raise ValueError("L'emploi cible doit contenir au moins une compétence.")


def _ordonner_correspondances(
    couple: CoupleEmplois,
    donnees: tuple[CorrespondanceFournie, ...],
) -> tuple[CorrespondanceFournie, ...]:
    """Valide et ordonne sans utiliser les identifiants facultatifs."""

    restantes = list(donnees)
    ordonnees: list[CorrespondanceFournie] = []

    for cible in couple.cible.competences:
        index = next(
            (
                position
                for position, item in enumerate(restantes)
                if item.competence_cible is cible
            ),
            None,
        )
        if index is None:
            index = next(
                (
                    position
                    for position, item in enumerate(restantes)
                    if item.competence_cible == cible
                ),
                None,
            )
        if index is None:
            raise ValueError(
                f"Correspondance fournie absente pour: {cible.intitule}."
            )
        ordonnees.append(restantes.pop(index))

    if restantes:
        cible = restantes[0].competence_cible
        raise ValueError(
            f"La compétence cible '{cible.intitule}' n'appartient pas à l'emploi "
            "cible ou possède plusieurs correspondances fournies."
        )

    for item in ordonnees:
        actuelle = item.competence_actuelle
        if actuelle is not None and not any(
            actuelle is competence or actuelle == competence
            for competence in couple.actuel.competences
        ):
            raise ValueError(
                f"La compétence actuelle '{actuelle.intitule}' n'appartient pas "
                "à l'emploi actuel."
            )

    return tuple(ordonnees)
