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

