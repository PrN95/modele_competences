"""Calcul des indicateurs et sélection des emplois cibles."""

from collections.abc import Iterable
from dataclasses import replace
from fractions import Fraction

from src.config import SEUIL_COUV
from src.domain import (
    CorrespondanceCompetence,
    CorrespondanceFournie,
    CoupleEmplois,
    ResultatAnalyseCouple,
    ResultatSelectionCibles,
)
from src.embeddings import EncodeurCompetences
from src.matching import analyser_correspondance, generer_correspondances_semantiques
from src.remote_models import CorrespondanceCategorielleGemma
from src.recommendations import determiner_besoin_formation


MESSAGE_AUCUNE_CIBLE = "Aucun emploi cible ne correspond à cet emploi actuel"
MESSAGE_ARBITRAGE_RH = (
    "Plusieurs emplois cibles sont ex aequo ; le service RH devra trancher"
)


def analyser_couple(
    couple: CoupleEmplois,
    correspondances_fournies: Iterable[CorrespondanceFournie],
    seuil_sim: float | Fraction | None = None,
    seuil_couv: float | Fraction | None = None,
    *,
    poids_dense: float | Fraction | None = None,
    poids_sparse: float | Fraction | None = None,
) -> ResultatAnalyseCouple:
    """Calcule la couverture, ``G_ef`` et ``Ecart_moyen_ef`` d'un couple."""

    _valider_couple(couple)
    donnees = tuple(correspondances_fournies)
    donnees = _ordonner_correspondances(couple, donnees)

    correspondances = tuple(
        analyser_correspondance(
            item,
            seuil_sim,
            poids_dense=poids_dense,
            poids_sparse=poids_sparse,
        )
        for item in donnees
    )
    nombre_cibles = len(correspondances)
    nombre_reconnues = sum(item.reconnue for item in correspondances)
    nombre_satisfaites = sum(item.niveau_suffisant for item in correspondances)
    score_global_exact = Fraction(nombre_reconnues, nombre_cibles)
    score_strict_exact = Fraction(nombre_satisfaites, nombre_cibles)
    ecart_moyen = _calculer_ecart_moyen(correspondances)

    seuil_couv_exact = (
        Fraction(str(seuil_couv)) if seuil_couv is not None else SEUIL_COUV
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
        admissible=score_global_exact >= seuil_couv_exact,
    )


def analyser_couple_semantiquement(
    couple: CoupleEmplois,
    encodeur: EncodeurCompetences,
    seuil_sim: float | Fraction | None = None,
    seuil_couv: float | Fraction | None = None,
    *,
    poids_dense: float | Fraction | None = None,
    poids_sparse: float | Fraction | None = None,
) -> ResultatAnalyseCouple:
    """Rapproche les compétences puis délègue tous les calculs à la phase 2."""

    correspondances = generer_correspondances_semantiques(
        couple,
        encodeur,
        poids_dense=poids_dense,
        poids_sparse=poids_sparse,
    )
    return analyser_couple(
        couple,
        correspondances,
        seuil_sim=seuil_sim,
        seuil_couv=seuil_couv,
        poids_dense=poids_dense,
        poids_sparse=poids_sparse,
    )


def analyser_cible_selectionnee_gemma(
    couple: CoupleEmplois,
    decisions: Iterable[CorrespondanceCategorielleGemma],
) -> ResultatAnalyseCouple:
    """Calcule des indicateurs informatifs après la sélection directe de Gemma."""

    _valider_couple(couple)
    restantes = list(decisions)
    correspondances: list[CorrespondanceCompetence] = []
    for competence_cible in couple.cible.competences:
        index = next(
            (
                position
                for position, decision in enumerate(restantes)
                if decision.competence_cible is competence_cible
            ),
            None,
        )
        if index is None:
            raise ValueError(
                f"Décision Gemma absente pour : {competence_cible.intitule}."
            )
        decision = restantes.pop(index)
        reconnue = decision.statut == "Reconnue"
        actuelle = decision.competence_actuelle
        if reconnue and actuelle is None:
            raise ValueError("Une décision Gemma Reconnue exige une compétence actuelle.")
        if not reconnue and actuelle is not None:
            raise ValueError("Une décision Gemma Absente ne doit pas référencer de compétence actuelle.")
        if actuelle is not None and not any(
            actuelle is competence for competence in couple.actuel.competences
        ):
            raise ValueError("La compétence actuelle choisie par Gemma est inconnue.")
        niveau_actuel = actuelle.niveau if actuelle is not None else 0
        niveau_requis = competence_cible.niveau
        if not reconnue:
            # Même convention historique que les embeddings : 0 est interne
            # pour une compétence absente, sans inventer de niveau cible.
            ecart = niveau_requis
            statut = "absente"
        elif niveau_actuel is None or niveau_requis is None:
            ecart = None
            statut = "niveau_non_renseigne"
        else:
            ecart = max(0, niveau_requis - niveau_actuel)
            statut = "niveau_insuffisant" if ecart else "niveau_suffisant"
        correspondances.append(
            CorrespondanceCompetence(
                competence_cible=competence_cible,
                competence_actuelle=actuelle,
                score_dense=None,
                score_sparse=None,
                score_hybride=None,
                score_llm=None,
                reconnue=reconnue,
                niveau_actuel=niveau_actuel,
                ecart_niveau=ecart,
                statut=statut,
            )
        )
    if restantes:
        raise ValueError("Gemma a fourni des décisions supplémentaires.")

    nombre_cibles = len(correspondances)
    g_epfq = Fraction(sum(item.reconnue for item in correspondances), nombre_cibles)
    gs_epfq = Fraction(sum(item.niveau_suffisant for item in correspondances), nombre_cibles)
    ecart_moyen = _calculer_ecart_moyen(correspondances)
    analyse = ResultatAnalyseCouple(
        emploi_actuel=couple.actuel,
        emploi_cible=couple.cible,
        correspondances=tuple(correspondances),
        besoins_formation=(),
        couverture_semantique=float(g_epfq),
        score_global=float(g_epfq),
        score_strict=float(gs_epfq),
        ecart_moyen=ecart_moyen,
        admissible=True,
    )
    return _ajouter_recommandations(analyse)


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
                _ecart_pour_departage(item.ecart_moyen),
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
        _ecart_pour_departage(item.ecart_moyen)
        for item in admissibles
        if item.score_global == meilleur_score
    )
    meilleures_apres_ecart = tuple(
        item
        for item in admissibles
        if item.score_global == meilleur_score
        and _ecart_pour_departage(item.ecart_moyen) == meilleur_ecart
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


def _calculer_ecart_moyen(
    correspondances: Iterable[CorrespondanceCompetence],
) -> float | None:
    """Calcule l'écart pondéré sur les seules comparaisons de niveau possibles."""

    comparables = tuple(
        item
        for item in correspondances
        if item.ecart_niveau is not None and item.niveau_requis is not None
    )
    if not comparables:
        return None
    somme_niveaux = sum(item.niveau_requis for item in comparables)
    return sum(
        item.ecart_niveau * item.niveau_requis for item in comparables
    ) / somme_niveaux


def _ecart_pour_departage(ecart_moyen: float | None) -> float:
    """Garde le départage déterministe si le niveau ne peut pas être comparé."""

    return ecart_moyen if ecart_moyen is not None else float("inf")


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
