"""Détermination des besoins de formation d'une correspondance."""

from src.domain import BesoinFormation, CorrespondanceCompetence


def determiner_besoin_formation(
    correspondance: CorrespondanceCompetence,
) -> BesoinFormation | None:
    """Retourne la progression nécessaire, ou ``None`` si le niveau suffit."""

    if correspondance.statut == "niveau_suffisant":
        return None

    motif = (
        "competence_absente"
        if correspondance.statut == "absente"
        else "niveau_insuffisant"
    )
    return BesoinFormation(
        competence_cible=correspondance.competence_cible,
        niveau_depart=correspondance.niveau_actuel,
        niveau_cible=correspondance.niveau_requis,
        motif=motif,
    )

