"""Objets métier indépendants de l'interface et du format d'entrée."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal

from src.config import COMPETENCE_LEVELS


EmploiType = Literal["actuel", "cible"]


@dataclass(frozen=True, slots=True)
class Competence:
    """Compétence et niveau requis ou détenu pour un emploi."""

    intitule: str
    description: str | None
    niveau: int | None
    id: str | None = field(default=None, compare=False)
    source_section: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.niveau is None:
            return
        if (
            isinstance(self.niveau, bool)
            or not isinstance(self.niveau, Integral)
            or self.niveau not in COMPETENCE_LEVELS
        ):
            raise ValueError(
                "Le niveau d'une compétence doit être absent ou compris entre 1 et 4."
            )

    @property
    def texte(self) -> str:
        """Texte qui sera ultérieurement envoyé au modèle d'encodage."""

        if self.description and self.description.strip():
            return f"{self.intitule}. {self.description}"
        return self.intitule

    @property
    def competence_intitule(self) -> str:
        """Nom du champ dans la structure normalisée commune aux extracteurs."""

        return self.intitule

    @property
    def competence_description(self) -> str | None:
        """Nom du champ dans la structure normalisée commune aux extracteurs."""

        return self.description

    @property
    def texte_competence(self) -> str:
        """Alias explicite du texte destiné ultérieurement à BGE-M3."""

        return self.texte


@dataclass(frozen=True, slots=True)
class Emploi:
    """Profil type chargé depuis une source structurée."""

    intitule: str
    type: EmploiType
    effectif: int | None
    competences: tuple[Competence, ...]
    fichier_source: str
    feuille_source: str | None
    id: str | None = field(default=None, compare=False)
    format_extraction: str | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class CoupleEmplois:
    """Un emploi actuel et un emploi cible à comparer."""

    actuel: Emploi
    cible: Emploi


StatutCorrespondance = Literal[
    "absente",
    "niveau_non_renseigne",
    "niveau_insuffisant",
    "niveau_suffisant",
]
MotifFormation = Literal["competence_absente", "niveau_insuffisant"]
TypeRecommandation = Literal[
    "formation_complete",
    "progression_un_niveau",
    "parcours_formation_consequent",
]
TypeNonReprise = Literal[
    "presente_dans_une_autre_cible",
    "absente_de_toutes_les_cibles",
]


@dataclass(frozen=True, slots=True)
class CorrespondanceFournie:
    """Meilleure candidate sémantique fournie au moteur de calcul."""

    competence_cible: Competence
    competence_actuelle: Competence | None
    score_dense: float | None
    score_sparse: float | None
    detail_egalite: str | None = None
    score_llm: float | None = None
    justification_courte: str | None = None


@dataclass(frozen=True, slots=True)
class CorrespondanceCompetence:
    """Résultat calculé pour une compétence cible."""

    competence_cible: Competence
    competence_actuelle: Competence | None
    score_dense: float | None
    score_sparse: float | None
    score_hybride: float | None
    score_llm: float | None
    reconnue: bool
    niveau_actuel: int | None
    ecart_niveau: int | None
    statut: StatutCorrespondance
    detail_egalite: str | None = None
    justification_courte: str | None = None
    # Champ renseigné uniquement par le contrat Gemma autonome. Il n'est jamais
    # dérivé par l'application.
    recommandation_llm: str | None = None

    @property
    def score_similarite(self) -> float:
        """Score utilisé par les règles métier, quelle que soit l'approche."""

        if self.score_llm is not None:
            return self.score_llm
        if self.score_hybride is None:
            raise ValueError("Aucun score de similarité n'est disponible.")
        return self.score_hybride

    @property
    def niveau_requis(self) -> int | None:
        return self.competence_cible.niveau

    @property
    def niveau_suffisant(self) -> bool:
        return self.statut == "niveau_suffisant"


@dataclass(frozen=True, slots=True)
class BesoinFormation:
    """Progression nécessaire sur une compétence cible."""

    competence_cible: Competence
    niveau_depart: int | None
    niveau_cible: int | None
    motif: MotifFormation
    recommandation: TypeRecommandation
    commentaire: str

    @property
    def ecart_niveau(self) -> int | None:
        if self.niveau_depart is None or self.niveau_cible is None:
            return None
        return self.niveau_cible - self.niveau_depart


@dataclass(frozen=True, slots=True)
class ResultatAnalyseCouple:
    """Résultat complet des indicateurs pour un unique couple d'emplois."""

    emploi_actuel: Emploi
    emploi_cible: Emploi
    correspondances: tuple[CorrespondanceCompetence, ...]
    besoins_formation: tuple[BesoinFormation, ...]
    couverture_semantique: float
    score_global: float
    score_strict: float
    ecart_moyen: float | None
    admissible: bool

    @property
    def g_ef(self) -> float:
        """Couverture des compétences cibles reconnues."""

        return self.score_global

    @property
    def gs_ef(self) -> float:
        """Part des compétences cibles entièrement satisfaites."""

        return self.score_strict


@dataclass(frozen=True, slots=True)
class ResultatSelectionCibles:
    """Classement et meilleure(s) cible(s) d'un emploi actuel."""

    analyses_classees: tuple[ResultatAnalyseCouple, ...]
    analyses_admissibles: tuple[ResultatAnalyseCouple, ...]
    meilleures_analyses: tuple[ResultatAnalyseCouple, ...]
    alerte: str | None

    @property
    def ex_aequo(self) -> bool:
        return len(self.meilleures_analyses) > 1

    @property
    def emplois_cibles_selectionnes(self) -> tuple[Emploi, ...]:
        return tuple(analyse.emploi_cible for analyse in self.meilleures_analyses)


@dataclass(frozen=True, slots=True)
class SignalementCompetenceActuelle:
    """Compétence actuelle absente de la ou des cibles sélectionnées."""

    competence_actuelle: Competence
    type_non_reprise: TypeNonReprise
    meilleure_competence_cible: Competence | None
    meilleur_emploi_cible: Emploi | None
    score_dense: float | None
    score_sparse: float | None
    score_hybride: float | None
    message: str
