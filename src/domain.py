"""Objets métier indépendants de l'interface et du format Excel."""

from dataclasses import dataclass
from typing import Literal


EmploiType = Literal["actuel", "cible"]


@dataclass(frozen=True, slots=True)
class Competence:
    """Compétence et niveau requis ou détenu pour un emploi."""

    id: str
    intitule: str
    description: str
    niveau: int

    @property
    def texte(self) -> str:
        """Texte qui sera ultérieurement envoyé au modèle d'encodage."""

        return f"{self.intitule}. {self.description}"


@dataclass(frozen=True, slots=True)
class Emploi:
    """Profil type chargé depuis un unique fichier Excel."""

    id: str
    intitule: str
    type: EmploiType
    effectif: int | None
    competences: tuple[Competence, ...]
    fichier_source: str
    feuille_source: str


@dataclass(frozen=True, slots=True)
class CoupleEmplois:
    """Premier incrément : un emploi actuel et un emploi cible."""

    actuel: Emploi
    cible: Emploi


StatutCorrespondance = Literal["absente", "niveau_insuffisant", "niveau_suffisant"]
MotifFormation = Literal["competence_absente", "niveau_insuffisant"]


@dataclass(frozen=True, slots=True)
class CorrespondanceFournie:
    """Meilleure candidate sémantique fournie au moteur de calcul."""

    competence_cible: Competence
    competence_actuelle: Competence | None
    score_dense: float
    score_sparse: float


@dataclass(frozen=True, slots=True)
class CorrespondanceCompetence:
    """Résultat calculé pour une compétence cible."""

    competence_cible: Competence
    competence_actuelle: Competence | None
    score_dense: float
    score_sparse: float
    score_hybride: float
    reconnue: bool
    niveau_actuel: int
    ecart_niveau: int
    statut: StatutCorrespondance

    @property
    def niveau_requis(self) -> int:
        return self.competence_cible.niveau

    @property
    def niveau_suffisant(self) -> bool:
        return self.reconnue and self.niveau_actuel >= self.niveau_requis


@dataclass(frozen=True, slots=True)
class BesoinFormation:
    """Progression nécessaire sur une compétence cible."""

    competence_cible: Competence
    niveau_depart: int
    niveau_cible: int
    motif: MotifFormation

    @property
    def ecart_niveau(self) -> int:
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
    ecart_moyen_normalise: float
    besoin_collectif: float

