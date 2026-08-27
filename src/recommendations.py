"""Détermination des besoins de formation d'une correspondance."""

from src.domain import BesoinFormation, CorrespondanceCompetence


def determiner_besoin_formation(
    correspondance: CorrespondanceCompetence,
) -> BesoinFormation | None:
    """Retourne la progression nécessaire, ou ``None`` si le niveau suffit."""

    if correspondance.statut == "niveau_suffisant":
        return None

    if correspondance.statut == "absente":
        motif = "competence_absente"
        recommandation = "formation_complete"
    else:
        motif = "niveau_insuffisant"
        recommandation = (
            "progression_un_niveau"
            if correspondance.ecart_niveau == 1
            else "parcours_formation_important"
        )
    return BesoinFormation(
        competence_cible=correspondance.competence_cible,
        niveau_depart=correspondance.niveau_actuel,
        niveau_cible=correspondance.niveau_requis,
        motif=motif,
        recommandation=recommandation,
    )
