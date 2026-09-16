"""Détermination des besoins de formation d'une correspondance."""

from src.domain import BesoinFormation, CorrespondanceCompetence


FORMATION_COMPLETE = "Formation complète nécessaire pour acquérir la compétence"
FORMATION_LEGERE = "Formation légère pour progresser d’un niveau"
PARCOURS_CONSEQUENT = "Parcours de formation conséquent"


def determiner_besoin_formation(
    correspondance: CorrespondanceCompetence,
) -> BesoinFormation | None:
    """Retourne la progression nécessaire, ou ``None`` si le niveau suffit."""

    if correspondance.statut in ("niveau_suffisant", "niveau_non_renseigne"):
        return None

    if correspondance.statut == "absente":
        motif = "competence_absente"
        recommandation = "formation_complete"
        commentaire = FORMATION_COMPLETE
    else:
        motif = "niveau_insuffisant"
        if correspondance.ecart_niveau == 1:
            recommandation = "progression_un_niveau"
            commentaire = FORMATION_LEGERE
        elif correspondance.ecart_niveau in (2, 3):
            recommandation = "parcours_formation_consequent"
            commentaire = PARCOURS_CONSEQUENT
        else:
            raise ValueError(
                "Une compétence reconnue avec niveau insuffisant doit avoir "
                "un écart compris entre 1 et 3."
            )
    return BesoinFormation(
        competence_cible=correspondance.competence_cible,
        niveau_depart=correspondance.niveau_actuel,
        niveau_cible=correspondance.niveau_requis,
        motif=motif,
        recommandation=recommandation,
        commentaire=commentaire,
    )
