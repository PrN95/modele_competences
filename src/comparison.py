"""Exécution isolée de plusieurs moteurs pour une comparaison robuste."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from src.orchestration import ResultatOrchestration
from src.remote_models import (
    ErreurAPIDistante,
    ReponseLLMInvalide,
    ServiceDistantIndisponible,
)


StatutExecution = Literal[
    "succès",
    "indisponible",
    "erreur API",
    "réponse LLM invalide",
    "erreur",
]


@dataclass(frozen=True, slots=True)
class ResultatExecutionModele:
    modele: str
    statut: StatutExecution
    temps_execution_secondes: float
    resultat: ResultatOrchestration | None
    detail_erreur: str | None = None


def executer_comparaison(
    moteurs: Mapping[str, Callable[[], ResultatOrchestration]],
) -> tuple[ResultatExecutionModele, ...]:
    """Exécute chaque moteur indépendamment pour préserver les autres résultats."""

    executions: list[ResultatExecutionModele] = []
    for modele, lancer in moteurs.items():
        debut = perf_counter()
        try:
            resultat = lancer()
        except ServiceDistantIndisponible as exc:
            executions.append(_echec(modele, "indisponible", debut, exc))
        except ReponseLLMInvalide as exc:
            executions.append(_echec(modele, "réponse LLM invalide", debut, exc))
        except ErreurAPIDistante as exc:
            executions.append(_echec(modele, "erreur API", debut, exc))
        except Exception as exc:
            executions.append(_echec(modele, "erreur", debut, exc))
        else:
            executions.append(
                ResultatExecutionModele(
                    modele=modele,
                    statut="succès",
                    temps_execution_secondes=perf_counter() - debut,
                    resultat=resultat,
                )
            )
    return tuple(executions)


def _echec(
    modele: str,
    statut: StatutExecution,
    debut: float,
    erreur: Exception,
) -> ResultatExecutionModele:
    return ResultatExecutionModele(
        modele=modele,
        statut=statut,
        temps_execution_secondes=perf_counter() - debut,
        resultat=None,
        detail_erreur=str(erreur),
    )
