"""Calcul d'une correspondance à partir de scores déjà fournis."""

from fractions import Fraction
from math import isfinite
from numbers import Real
from collections.abc import Mapping, Sequence

from src.config import DENSE_WEIGHT, SEMANTIC_MATCH_THRESHOLD, SPARSE_WEIGHT
from src.domain import (
    CorrespondanceCompetence,
    CorrespondanceFournie,
    CoupleEmplois,
    StatutCorrespondance,
)
from src.embeddings import EncodeurCompetences


def calculer_score_hybride(score_dense: float, score_sparse: float) -> float:
    """Calcule ``H = 2/3 × dense + 1/3 × sparse`` sans poids arrondis."""

    return float(_score_hybride_exact(score_dense, score_sparse))


def analyser_correspondance(
    correspondance: CorrespondanceFournie,
) -> CorrespondanceCompetence:
    """Applique le seuil sémantique puis compare séparément les niveaux."""

    score_hybride_exact = _score_hybride_exact(
        correspondance.score_dense,
        correspondance.score_sparse,
    )
    reconnue = score_hybride_exact >= SEMANTIC_MATCH_THRESHOLD

    if reconnue and correspondance.competence_actuelle is None:
        raise ValueError(
            "Une correspondance reconnue doit fournir une compétence actuelle "
            "afin de déterminer son niveau."
        )

    niveau_actuel = (
        correspondance.competence_actuelle.niveau
        if reconnue and correspondance.competence_actuelle is not None
        else 0
    )
    niveau_requis = correspondance.competence_cible.niveau
    ecart_niveau = max(0, niveau_requis - niveau_actuel)
    statut = _determiner_statut(reconnue, ecart_niveau)

    return CorrespondanceCompetence(
        competence_cible=correspondance.competence_cible,
        competence_actuelle=correspondance.competence_actuelle,
        score_dense=float(correspondance.score_dense),
        score_sparse=float(correspondance.score_sparse),
        score_hybride=float(score_hybride_exact),
        reconnue=reconnue,
        niveau_actuel=niveau_actuel,
        ecart_niveau=ecart_niveau,
        statut=statut,
    )


def generer_correspondances_semantiques(
    couple: CoupleEmplois,
    encodeur: EncodeurCompetences,
) -> tuple[CorrespondanceFournie, ...]:
    """Sélectionne la meilleure compétence actuelle pour chaque cible."""

    if not couple.actuel.competences:
        raise ValueError("L'emploi actuel doit contenir au moins une compétence.")
    if not couple.cible.competences:
        raise ValueError("L'emploi cible doit contenir au moins une compétence.")

    encodage_actuel = encodeur.encoder(couple.actuel.competences)
    encodage_cible = encodeur.encoder(couple.cible.competences)
    correspondances: list[CorrespondanceFournie] = []

    for cible_index, competence_cible in enumerate(couple.cible.competences):
        meilleure: CorrespondanceFournie | None = None
        meilleur_score_hybride: float | None = None

        for actuel_index, competence_actuelle in enumerate(couple.actuel.competences):
            score_dense = calculer_score_dense(
                encodage_actuel.vecteurs_dense[actuel_index],
                encodage_cible.vecteurs_dense[cible_index],
            )
            score_sparse = calculer_score_sparse(
                encodage_actuel.poids_sparse[actuel_index],
                encodage_cible.poids_sparse[cible_index],
            )
            score_hybride = calculer_score_hybride(score_dense, score_sparse)

            if meilleur_score_hybride is None or score_hybride > meilleur_score_hybride:
                meilleur_score_hybride = score_hybride
                meilleure = CorrespondanceFournie(
                    competence_cible=competence_cible,
                    competence_actuelle=competence_actuelle,
                    score_dense=score_dense,
                    score_sparse=score_sparse,
                )

        if meilleure is None:  # protégé par le contrôle des compétences actuelles
            raise RuntimeError("Aucune candidate actuelle n'a pu être sélectionnée.")
        correspondances.append(meilleure)

    return tuple(correspondances)


def calculer_score_dense(
    vecteur_actuel: Sequence[float],
    vecteur_cible: Sequence[float],
) -> float:
    """Produit scalaire des représentations dense normalisées par BGE-M3."""

    if not vecteur_actuel or len(vecteur_actuel) != len(vecteur_cible):
        raise ValueError("Les vecteurs dense doivent avoir la même dimension non nulle.")
    return float(sum(actuel * cible for actuel, cible in zip(vecteur_actuel, vecteur_cible)))


def calculer_score_sparse(
    poids_actuels: Mapping[str, float],
    poids_cibles: Mapping[str, float],
) -> float:
    """Somme des produits des poids lexicaux communs."""

    communs = poids_actuels.keys() & poids_cibles.keys()
    return float(sum(poids_actuels[token] * poids_cibles[token] for token in communs))


def _score_hybride_exact(score_dense: float, score_sparse: float) -> Fraction:
    dense = _as_fraction(score_dense, "score_dense")
    sparse = _as_fraction(score_sparse, "score_sparse")
    return (DENSE_WEIGHT * dense) + (SPARSE_WEIGHT * sparse)


def _as_fraction(value: float, field_name: str) -> Fraction:
    """Convertit la représentation décimale reçue, pas son approximation binaire.

    ``Fraction(str(value))`` préserve la valeur décimale exposée par l'entrée
    (par exemple ``0.9`` devient exactement ``9/10``). La comparaison au seuil
    reste ainsi rationnelle et ne dépend d'aucun arrondi arbitraire.
    """

    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        raise ValueError(f"{field_name} doit être un nombre réel fini.")
    return Fraction(str(value))


def _determiner_statut(reconnue: bool, ecart_niveau: int) -> StatutCorrespondance:
    if not reconnue:
        return "absente"
    if ecart_niveau > 0:
        return "niveau_insuffisant"
    return "niveau_suffisant"
