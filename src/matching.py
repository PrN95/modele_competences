"""Calcul d'une correspondance à partir de scores déjà fournis."""

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from numbers import Real
from collections.abc import Mapping, Sequence

from src.config import DENSE_WEIGHT, SEMANTIC_MATCH_THRESHOLD, SPARSE_WEIGHT
from src.domain import (
    Competence,
    CorrespondanceCompetence,
    CorrespondanceFournie,
    CoupleEmplois,
    Emploi,
    SignalementCompetenceActuelle,
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
        detail_egalite=correspondance.detail_egalite,
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
        candidates: list[_CandidateDirect] = []

        for actuel_index, competence_actuelle in enumerate(couple.actuel.competences):
            score_dense = calculer_score_dense(
                encodage_actuel.vecteurs_dense[actuel_index],
                encodage_cible.vecteurs_dense[cible_index],
            )
            score_sparse = calculer_score_sparse(
                encodage_actuel.poids_sparse[actuel_index],
                encodage_cible.poids_sparse[cible_index],
            )
            candidates.append(
                _CandidateDirect(
                    index_source=actuel_index,
                    competence_cible=competence_cible,
                    competence_actuelle=competence_actuelle,
                    score_dense=score_dense,
                    score_sparse=score_sparse,
                    score_hybride=_score_hybride_exact(score_dense, score_sparse),
                )
            )

        if not candidates:  # protégé par le contrôle des compétences actuelles
            raise RuntimeError("Aucune candidate actuelle n'a pu être sélectionnée.")
        correspondances.append(_selectionner_candidate_directe(candidates))

    return tuple(correspondances)


@dataclass(frozen=True, slots=True)
class _CandidateDirect:
    index_source: int
    competence_cible: Competence
    competence_actuelle: Competence
    score_dense: float
    score_sparse: float
    score_hybride: Fraction


def _selectionner_candidate_directe(
    candidates: Sequence[_CandidateDirect],
) -> CorrespondanceFournie:
    """Applique H_ac, niveau, L_ac, puis un choix textuel déterministe."""

    meilleur_hybride = max(item.score_hybride for item in candidates)
    ex_aequo_hybride = tuple(
        item for item in candidates if item.score_hybride == meilleur_hybride
    )
    meilleur_niveau = max(item.competence_actuelle.niveau for item in ex_aequo_hybride)
    ex_aequo_niveau = tuple(
        item
        for item in ex_aequo_hybride
        if item.competence_actuelle.niveau == meilleur_niveau
    )
    meilleur_sparse = max(
        _as_fraction(item.score_sparse, "score_sparse") for item in ex_aequo_niveau
    )
    ex_aequo_persistants = tuple(
        item
        for item in ex_aequo_niveau
        if _as_fraction(item.score_sparse, "score_sparse") == meilleur_sparse
    )
    choisie = min(ex_aequo_persistants, key=_cle_deterministe_candidate)
    detail_egalite = None
    if len(ex_aequo_persistants) > 1:
        noms = ", ".join(
            f"« {item.competence_actuelle.intitule} »"
            for item in sorted(ex_aequo_persistants, key=_cle_deterministe_candidate)
        )
        detail_egalite = (
            "Égalité exacte persistante de H_ac, du niveau actuel et de L_ac entre "
            f"{noms} ; choix déterministe : « {choisie.competence_actuelle.intitule} »."
        )

    return CorrespondanceFournie(
        competence_cible=choisie.competence_cible,
        competence_actuelle=choisie.competence_actuelle,
        score_dense=choisie.score_dense,
        score_sparse=choisie.score_sparse,
        detail_egalite=detail_egalite,
    )


def _cle_deterministe_candidate(candidate: _CandidateDirect) -> tuple[str, str, str, int]:
    competence = candidate.competence_actuelle
    return (
        competence.intitule.casefold(),
        (competence.description or "").casefold(),
        competence.id or "",
        candidate.index_source,
    )


@dataclass(frozen=True, slots=True)
class _CandidateInverse:
    emploi_cible: Emploi
    competence_cible: Competence
    score_dense: float
    score_sparse: float
    score_hybride: float


def controler_competences_actuelles_non_reprises(
    emploi_actuel: Emploi,
    emplois_cibles: Sequence[Emploi],
    emplois_cibles_selectionnes: Sequence[Emploi],
    encodeur: EncodeurCompetences,
) -> tuple[SignalementCompetenceActuelle, ...]:
    """Signale les compétences actuelles absentes des cibles sélectionnées.

    Ce calcul inverse produit uniquement des informations de restitution. Il
    ne reçoit ni ne modifie les résultats de scoring ou de sélection.
    """

    cibles = tuple(emplois_cibles)
    selectionnees = tuple(emplois_cibles_selectionnes)
    if emploi_actuel.type != "actuel":
        raise ValueError("Le contrôle inverse exige un emploi de type 'actuel'.")
    if not emploi_actuel.competences:
        return ()
    if not cibles or not selectionnees:
        raise ValueError("Les cibles analysées et sélectionnées sont obligatoires.")
    if any(cible.type != "cible" or not cible.competences for cible in cibles):
        raise ValueError("Chaque cible analysée doit contenir des compétences.")
    if any(
        not any(selectionnee is cible for cible in cibles)
        for selectionnee in selectionnees
    ):
        raise ValueError("Une cible sélectionnée n'a pas été analysée.")

    def est_selectionnee(cible: Emploi) -> bool:
        return any(cible is item for item in selectionnees)

    encodage_actuel = encodeur.encoder(emploi_actuel.competences)
    candidats_par_actuelle: list[list[_CandidateInverse]] = [
        [] for _ in emploi_actuel.competences
    ]

    for cible in cibles:
        encodage_cible = encodeur.encoder(cible.competences)
        for actuel_index in range(len(emploi_actuel.competences)):
            for cible_index, competence_cible in enumerate(cible.competences):
                score_dense = calculer_score_dense(
                    encodage_actuel.vecteurs_dense[actuel_index],
                    encodage_cible.vecteurs_dense[cible_index],
                )
                score_sparse = calculer_score_sparse(
                    encodage_actuel.poids_sparse[actuel_index],
                    encodage_cible.poids_sparse[cible_index],
                )
                candidats_par_actuelle[actuel_index].append(
                    _CandidateInverse(
                        emploi_cible=cible,
                        competence_cible=competence_cible,
                        score_dense=score_dense,
                        score_sparse=score_sparse,
                        score_hybride=calculer_score_hybride(score_dense, score_sparse),
                    )
                )

    signalements: list[SignalementCompetenceActuelle] = []
    for competence_actuelle, candidats in zip(
        emploi_actuel.competences, candidats_par_actuelle
    ):
        candidats_selectionnes = tuple(
            item for item in candidats if est_selectionnee(item.emploi_cible)
        )
        meilleure_selectionnee = max(
            candidats_selectionnes, key=lambda item: item.score_hybride
        )
        if _score_hybride_exact(
            meilleure_selectionnee.score_dense,
            meilleure_selectionnee.score_sparse,
        ) >= SEMANTIC_MATCH_THRESHOLD:
            continue

        candidats_autres = tuple(
            item for item in candidats if not est_selectionnee(item.emploi_cible)
        )
        meilleure_autre = (
            max(candidats_autres, key=lambda item: item.score_hybride)
            if candidats_autres
            else None
        )
        presente_autre = meilleure_autre is not None and _score_hybride_exact(
            meilleure_autre.score_dense,
            meilleure_autre.score_sparse,
        ) >= SEMANTIC_MATCH_THRESHOLD

        if presente_autre:
            meilleure = meilleure_autre
            type_non_reprise = "presente_dans_une_autre_cible"
            message = (
                "compétence actuelle absente de l'emploi cible sélectionné, "
                "mais présente dans une autre cible"
            )
        else:
            meilleure = max(candidats, key=lambda item: item.score_hybride)
            type_non_reprise = "absente_de_toutes_les_cibles"
            message = "compétence actuelle non reprise dans les emplois cibles analysés"

        signalements.append(
            SignalementCompetenceActuelle(
                competence_actuelle=competence_actuelle,
                type_non_reprise=type_non_reprise,
                meilleure_competence_cible=meilleure.competence_cible,
                meilleur_emploi_cible=meilleure.emploi_cible,
                score_dense=meilleure.score_dense,
                score_sparse=meilleure.score_sparse,
                score_hybride=meilleure.score_hybride,
                message=message,
            )
        )

    return tuple(signalements)


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
