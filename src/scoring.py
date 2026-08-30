"""Calcul des indicateurs et sélection des emplois cibles."""

from collections.abc import Iterable
from dataclasses import replace
from fractions import Fraction

from src.config import SEUIL_COUV
from src.domain import (
    CorrespondanceFournie,
    CoupleEmplois,
    ResultatAnalyseCouple,
    ResultatSelectionCibles,
)
from src.embeddings import EncodeurCompetences
from src.matching import analyser_correspondance, generer_correspondances_semantiques
from src.recommendations import determiner_besoin_formation


MESSAGE_AUCUNE_CIBLE = "Aucun métier cible ne correspond à cet emploi actuel"
MESSAGE_ARBITRAGE_RH = (
    "Plusieurs métiers cibles sont ex aequo ; le service RH devra trancher"
)


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
    nombre_reconnues = sum(item.reconnue for item in correspondances)
    nombre_satisfaites = sum(item.niveau_suffisant for item in correspondances)
    score_global_exact = Fraction(nombre_reconnues, nombre_cibles)
    score_strict_exact = Fraction(nombre_satisfaites, nombre_cibles)
    somme_niveaux_cibles = sum(item.niveau_requis for item in correspondances)
    ecart_moyen = (
        sum(
            item.ecart_niveau * item.niveau_requis
            for item in correspondances
        )
        / somme_niveaux_cibles
    )

    return ResultatAnalyseCouple(
        emploi_actuel=couple.actuel,
        emploi_cible=couple.cible,
        correspondances=correspondances,
        besoins_formation=(),
        couverture_semantique=float(score_global_exact),
        score_global=float(score_global_exact),
        score_strict=float(score_strict_exact),
        ecart_moyen=ecart_moyen,
        admissible=score_global_exact >= SEUIL_COUV,
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
    """Exclut les cibles non admissibles puis applique les trois départages.

    Les recommandations sont ajoutées uniquement aux analyses finalement
    retenues. L'ordre d'entrée reste un ordre stable d'affichage et ne rompt
    jamais une égalité métier persistante.
    """

    donnees = tuple(analyses)
    if not donnees:
        raise ValueError("Au moins une analyse de cible est obligatoire.")

    emploi_actuel = donnees[0].emploi_actuel
    if any(analyse.emploi_actuel is not emploi_actuel for analyse in donnees[1:]):
        raise ValueError("Toutes les analyses doivent concerner le même emploi actuel.")

    analyses_classees = tuple(
        sorted(
            donnees,
            key=lambda item: (
                -item.score_global,
                item.ecart_moyen,
                -item.score_strict,
            ),
        )
    )
    admissibles = tuple(item for item in analyses_classees if item.admissible)
    if not admissibles:
        return ResultatSelectionCibles(
            analyses_classees=analyses_classees,
            analyses_admissibles=(),
            meilleures_analyses=(),
            alerte=MESSAGE_AUCUNE_CIBLE,
        )

    meilleur_score = admissibles[0].score_global
    meilleur_ecart = min(
        item.ecart_moyen
        for item in admissibles
        if item.score_global == meilleur_score
    )
    meilleures_apres_ecart = tuple(
        item
        for item in admissibles
        if item.score_global == meilleur_score and item.ecart_moyen == meilleur_ecart
    )
    meilleur_score_strict = max(
        item.score_strict for item in meilleures_apres_ecart
    )
    meilleures_sans_recommandations = tuple(
        item
        for item in meilleures_apres_ecart
        if item.score_strict == meilleur_score_strict
    )
    meilleures = tuple(
        _ajouter_recommandations(item) for item in meilleures_sans_recommandations
    )
    alerte = MESSAGE_ARBITRAGE_RH if len(meilleures) > 1 else None
    return ResultatSelectionCibles(
        analyses_classees=analyses_classees,
        analyses_admissibles=admissibles,
        meilleures_analyses=meilleures,
        alerte=alerte,
    )


def _ajouter_recommandations(
    analyse: ResultatAnalyseCouple,
) -> ResultatAnalyseCouple:
    besoins = tuple(
        besoin
        for correspondance in analyse.correspondances
        if (besoin := determiner_besoin_formation(correspondance)) is not None
    )
    return replace(analyse, besoins_formation=besoins)


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
