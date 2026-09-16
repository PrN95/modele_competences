"""Adaptateurs HTTP pour embeddings distants et jugement LLM Gemma."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Literal, Protocol

import requests

from src.config import (
    BGE_BASE_URL,
    BGE_MODEL,
    GEMMA_BASE_URL,
    GEMMA_MODEL,
    QWEN_BASE_URL,
    QWEN_MODEL,
    REMOTE_TIMEOUT_SECONDS,
)
from src.domain import Competence, Emploi
from src.embeddings import SortieEncodage


class ServiceDistantIndisponible(RuntimeError):
    """Le serveur distant n'a pas répondu avant le délai ou est inaccessible."""


class ErreurAPIDistante(RuntimeError):
    """Le serveur a répondu avec une erreur HTTP ou un contrat inattendu."""


class ReponseLLMInvalide(RuntimeError):
    """La réponse de Gemma ne respecte pas le JSON strict attendu."""


class ClientHTTP(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...
    def post(self, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class CorrespondanceCategorielleGemma:
    competence_cible: Competence
    competence_actuelle: Competence | None
    statut: Literal["Reconnue", "Absente"]
    niveau_actuel: int | None
    niveau_cible: int | None
    ecart_niveau: int | None
    statut_niveau: Literal["absente", "niveau_non_renseigne", "niveau_insuffisant", "niveau_suffisant"]
    recommandation_formation: str


@dataclass(frozen=True, slots=True)
class SelectionGemma:
    emploi_cible: Emploi
    correspondances: tuple[CorrespondanceCategorielleGemma, ...]
    g_epfq: float | None
    gs_epfq: float | None
    ecart_moyen: float | None
    r_epfq: float | None
    reponse_brute: str


PROMPT_SYSTEME_GEMMA = """Tu réalises une sélection directe d'emploi cible et les indicateurs métier associés.
Les compétences digitales et niveaux SAME 1 à 4 des emplois actuels et cibles te sont fournis.
Choisis directement exactement un emploi cible parmi ceux proposés, selon ton appréciation globale de la proximité des compétences. Cette sélection ne dépend d'aucun seuil de couverture et ne doit pas être modifiée par les indicateurs calculés ensuite.
Après ce choix, décide quelles compétences de cette cible sont reconnues comme similaires à une compétence actuelle. Calcule G_epfq (compétences cibles reconnues / total), Gs_epfq, R_epfq, les écarts de niveau et les recommandations de formation. Ces indicateurs sont exclusivement informatifs.
Ne fournis aucun score d'embedding ni seuil de similarité.
Réponds exclusivement par un objet JSON valide, sans Markdown ni texte autour, sous cette forme exacte :
{"emploi_cible_retenu":"[E1] ...","g_epfq":nombre,"gs_epfq":nombre,"ecart_moyen":nombre,"r_epfq":nombre,"correspondances":[{"competence_cible":"[C1] ...","competence_actuelle":"[A1] ... ou N/A","statut":"Reconnue ou Absente","niveau_actuel":1..4 ou null,"niveau_cible":1..4 ou null,"ecart_niveau":entier >= 0 ou null,"statut_niveau":"absente, niveau_non_renseigne, niveau_insuffisant ou niveau_suffisant","recommandation_formation":"texte"}]}
Chaque libellé et chaque niveau doivent être recopiés exactement depuis les données fournies. Une compétence Reconnue exige une compétence actuelle ; une compétence Absente exige "N/A". La liste couvre une et une seule fois les compétences de la cible retenue."""


class _AdaptateurHTTPBase:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = REMOTE_TIMEOUT_SECONDS,
        api_key: str | None = None,
        client: ClientHTTP | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("Le timeout distant doit être strictement positif.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self._api_key = api_key or None
        self._client = client or requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def tester_connexion(self) -> str:
        payload = self._get_json(f"{self.base_url}/models")
        modeles = payload.get("data") if isinstance(payload, Mapping) else None
        ids = [item.get("id") for item in modeles or () if isinstance(item, Mapping)]
        model_id = next((item for item in ids if isinstance(item, str) and item), None)
        if model_id is None:
            raise ErreurAPIDistante(
                f"Aucun modèle exploitable n'est annoncé par {self.base_url}/models."
            )
        # L'identifiant de la réponse du serveur est la source de vérité : les
        # valeurs de l'interface ne servent plus d'hypothèse sur le déploiement.
        self.model = model_id
        return model_id

    def _modele_annonce(self) -> str:
        return self.tester_connexion()

    def _get_json(self, url: str) -> Mapping[str, Any]:
        try:
            response = self._client.get(url, headers=self._headers(), timeout=self.timeout)
        except requests.Timeout as exc:
            raise ServiceDistantIndisponible(f"Timeout après {self.timeout:g} s pour {url}.") from exc
        except requests.RequestException as exc:
            raise ServiceDistantIndisponible(f"Serveur distant indisponible : {url}.") from exc
        return _decoder_reponse(response, url)

    def _post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            response = self._client.post(
                url,
                headers=self._headers(),
                json=dict(payload),
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise ServiceDistantIndisponible(f"Timeout après {self.timeout:g} s pour {url}.") from exc
        except requests.RequestException as exc:
            raise ServiceDistantIndisponible(f"Serveur distant indisponible : {url}.") from exc
        return _decoder_reponse(response, url)


class AdaptateurQwenEmbeddings(_AdaptateurHTTPBase):
    """Encodeur Qwen distant dense uniquement, compatible avec le moteur existant."""

    def __init__(
        self,
        base_url: str = QWEN_BASE_URL,
        model: str = QWEN_MODEL,
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url, model, **kwargs)
        self._cache: dict[tuple[str, ...], SortieEncodage] = {}

    def encoder(self, competences: Sequence[Competence]) -> SortieEncodage:
        if not competences:
            return SortieEncodage(vecteurs_dense=(), poids_sparse=())
        cle = tuple(competence.texte for competence in competences)
        if cle in self._cache:
            return self._cache[cle]
        model_id = self._modele_annonce()
        payload = self._post_json(
            f"{self.base_url}/embeddings",
            {
                "model": model_id,
                "input": [competence.texte for competence in competences],
                "encoding_format": "float",
            },
        )
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(competences):
            raise ErreurAPIDistante("Qwen a retourné un nombre d'embeddings inattendu.")
        try:
            ordonnes = sorted(data, key=lambda item: item["index"])
            vecteurs = tuple(_normaliser_dense(item["embedding"]) for item in ordonnes)
        except (KeyError, TypeError, ValueError) as exc:
            raise ErreurAPIDistante("La réponse d'embedding Qwen est inexploitable.") from exc
        sortie = SortieEncodage(
            vecteurs_dense=vecteurs,
            poids_sparse=tuple({} for _ in vecteurs),
        )
        self._cache[cle] = sortie
        return sortie


class AdaptateurBGEDistantEmbeddings(AdaptateurQwenEmbeddings):
    """Encodeur BGE distant OpenAI-compatible, exploité en dense uniquement.

    L'API distante ne fournit ici qu'un champ ``embedding``. Comme pour Qwen,
    aucun vecteur sparse n'est inféré ou inventé.
    """

    def __init__(
        self,
        base_url: str = BGE_BASE_URL,
        model: str = BGE_MODEL,
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url, model, **kwargs)


class AdaptateurGemma4(_AdaptateurHTTPBase):
    """Sélectionne une cible et ses correspondances catégorielles, sans score."""

    def __init__(
        self,
        base_url: str = GEMMA_BASE_URL,
        model: str = GEMMA_MODEL,
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url, model, **kwargs)
        self._cache: dict[tuple[Any, ...], SelectionGemma] = {}

    def selectionner(
        self,
        emploi_actuel: Emploi,
        emplois_cibles: Sequence[Emploi],
    ) -> SelectionGemma:
        cibles = tuple(emplois_cibles)
        if emploi_actuel.type != "actuel" or not emploi_actuel.competences:
            raise ValueError("Gemma exige un emploi actuel contenant des compétences.")
        if not cibles or any(cible.type != "cible" or not cible.competences for cible in cibles):
            raise ValueError("Gemma exige au moins un emploi cible contenant des compétences.")
        cle = (
            emploi_actuel.intitule,
            tuple((item.texte, item.niveau) for item in emploi_actuel.competences),
            tuple((cible.intitule, tuple((item.texte, item.niveau) for item in cible.competences)) for cible in cibles),
        )
        if cle in self._cache:
            return self._cache[cle]
        libelles_actuels = {
            f"[A{index}] {competence.intitule}": competence
            for index, competence in enumerate(emploi_actuel.competences, start=1)
        }
        libelles_cibles = {
            f"[E{index}] {emploi.intitule}": emploi
            for index, emploi in enumerate(cibles, start=1)
        }
        libelles_competences = {
            libelle_emploi: {
                f"[C{index}] {competence.intitule}": competence
                for index, competence in enumerate(emploi.competences, start=1)
            }
            for libelle_emploi, emploi in libelles_cibles.items()
        }
        contenu_utilisateur = _construire_entree_gemma(
            emploi_actuel,
            libelles_actuels,
            libelles_cibles,
            libelles_competences,
        )
        model_id = self._modele_annonce()
        payload = self._post_json(
            f"{self.base_url}/chat/completions",
            {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": PROMPT_SYSTEME_GEMMA},
                    {
                        "role": "user",
                        "content": contenu_utilisateur,
                    },
                ],
                "temperature": 0,
                "max_tokens": 4096,
                "response_format": {"type": "json_object"},
            },
        )
        try:
            contenu = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ReponseLLMInvalide("La réponse Gemma ne contient aucun contenu exploitable.") from exc
        selection = valider_reponse_gemma(
            contenu,
            libelles_actuels=libelles_actuels,
            libelles_cibles=libelles_cibles,
            libelles_competences=libelles_competences,
        )
        self._cache[cle] = selection
        return selection


def valider_reponse_gemma(
    contenu: Any,
    *,
    libelles_actuels: Mapping[str, Competence],
    libelles_cibles: Mapping[str, Emploi],
    libelles_competences: Mapping[str, Mapping[str, Competence]],
) -> SelectionGemma:
    if not isinstance(contenu, str):
        raise ReponseLLMInvalide("Le contenu Gemma doit être une chaîne JSON.")
    try:
        objet = json.loads(contenu)
    except json.JSONDecodeError as exc:
        raise ReponseLLMInvalide("Gemma n'a pas retourné un JSON valide.") from exc
    cles = {"emploi_cible_retenu", "g_epfq", "gs_epfq", "ecart_moyen", "r_epfq", "correspondances"}
    if not isinstance(objet, dict) or set(objet) != cles:
        raise ReponseLLMInvalide("Le JSON Gemma doit contenir exactement les deux clés attendues.")
    libelle_emploi = objet["emploi_cible_retenu"]
    if not isinstance(libelle_emploi, str) or libelle_emploi not in libelles_cibles:
        raise ReponseLLMInvalide("L'emploi cible retenu par Gemma ne fait pas partie des choix proposés.")
    mesures = {cle: objet[cle] for cle in ("g_epfq", "gs_epfq", "ecart_moyen", "r_epfq")}
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(float(value)) for value in mesures.values()):
        raise ReponseLLMInvalide("Les indicateurs Gemma d'une passerelle doivent être des nombres finis.")
    if not all(0 <= float(mesures[cle]) <= 1 for cle in ("g_epfq", "gs_epfq", "r_epfq")) or float(mesures["ecart_moyen"]) < 0:
        raise ReponseLLMInvalide("Les indicateurs Gemma sont hors de leurs bornes attendues.")
    donnees = objet["correspondances"]
    if not isinstance(donnees, list):
        raise ReponseLLMInvalide("Les correspondances Gemma doivent former une liste.")
    attendues = libelles_competences[libelle_emploi]
    vues: set[str] = set()
    correspondances: list[CorrespondanceCategorielleGemma] = []
    for item in donnees:
        if not isinstance(item, dict) or set(item) != {"competence_cible", "competence_actuelle", "statut", "niveau_actuel", "niveau_cible", "ecart_niveau", "statut_niveau", "recommandation_formation"}:
            raise ReponseLLMInvalide("Chaque correspondance Gemma doit contenir exactement les huit clés attendues.")
        libelle_cible = item["competence_cible"]
        libelle_actuelle = item["competence_actuelle"]
        statut = item["statut"]
        niveau_actuel, niveau_cible, ecart = item["niveau_actuel"], item["niveau_cible"], item["ecart_niveau"]
        statut_niveau, recommandation = item["statut_niveau"], item["recommandation_formation"]
        if (
            not isinstance(libelle_cible, str)
            or libelle_cible not in attendues
            or libelle_cible in vues
        ):
            raise ReponseLLMInvalide("Une compétence cible Gemma est inconnue ou dupliquée.")
        if not isinstance(statut, str) or statut not in {"Reconnue", "Absente"}:
            raise ReponseLLMInvalide("Le statut Gemma doit être exactement 'Reconnue' ou 'Absente'.")
        if statut == "Absente":
            if libelle_actuelle != "N/A":
                raise ReponseLLMInvalide("Une compétence Absente doit avoir 'N/A' comme compétence actuelle.")
            actuelle = None
        else:
            if not isinstance(libelle_actuelle, str) or libelle_actuelle not in libelles_actuels:
                raise ReponseLLMInvalide("Une compétence Reconnue doit référencer une compétence actuelle proposée.")
            actuelle = libelles_actuels[libelle_actuelle]
        if niveau_actuel != (actuelle.niveau if actuelle is not None else None) or niveau_cible != attendues[libelle_cible].niveau:
            raise ReponseLLMInvalide("Les niveaux Gemma doivent recopier exactement les niveaux SAME fournis.")
        if not (ecart is None or isinstance(ecart, int) and not isinstance(ecart, bool) and ecart >= 0):
            raise ReponseLLMInvalide("L'écart de niveau Gemma doit être un entier positif ou null.")
        if statut_niveau not in {"absente", "niveau_non_renseigne", "niveau_insuffisant", "niveau_suffisant"} or not isinstance(recommandation, str) or not recommandation.strip():
            raise ReponseLLMInvalide("Le statut de niveau ou la recommandation Gemma est invalide.")
        vues.add(libelle_cible)
        correspondances.append(
            CorrespondanceCategorielleGemma(attendues[libelle_cible], actuelle, statut, niveau_actuel, niveau_cible, ecart, statut_niveau, recommandation)
        )
    if vues != set(attendues):
        raise ReponseLLMInvalide("Gemma doit décider pour chaque compétence de la cible retenue.")
    par_competence = {id(item.competence_cible): item for item in correspondances}
    emploi_cible = libelles_cibles[libelle_emploi]
    return SelectionGemma(
        emploi_cible,
        tuple(par_competence[id(competence)] for competence in emploi_cible.competences),
        float(mesures["g_epfq"]), float(mesures["gs_epfq"]), float(mesures["ecart_moyen"]), float(mesures["r_epfq"]), contenu,
    )


def _construire_entree_gemma(
    emploi_actuel: Emploi,
    libelles_actuels: Mapping[str, Competence],
    libelles_cibles: Mapping[str, Emploi],
    libelles_competences: Mapping[str, Mapping[str, Competence]],
) -> str:
    lignes = [f"EMPLOI ACTUEL : {emploi_actuel.intitule}", "Compétences actuelles :"]
    lignes.extend(f"- {libelle} : {competence.texte} | Niveau SAME : {competence.niveau}" for libelle, competence in libelles_actuels.items())
    lignes.append("\nEMPLOIS CIBLES PROPOSÉS :")
    for libelle_emploi, emploi in libelles_cibles.items():
        lignes.append(f"\n{libelle_emploi}")
        lignes.extend(
            f"- {libelle} : {competence.texte} | Niveau SAME : {competence.niveau}"
            for libelle, competence in libelles_competences[libelle_emploi].items()
        )
    return "\n".join(lignes)


# Alias temporaire pour les imports historiques du PoC.
AdaptateurGemma3 = AdaptateurGemma4


def _decoder_reponse(response: Any, url: str) -> Mapping[str, Any]:
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        status = getattr(response, "status_code", "inconnu")
        raise ErreurAPIDistante(f"Erreur HTTP {status} pour {url}.") from exc
    try:
        payload = response.json()
    except (ValueError, TypeError) as exc:
        raise ErreurAPIDistante(f"La réponse de {url} n'est pas un JSON valide.") from exc
    if not isinstance(payload, Mapping):
        raise ErreurAPIDistante(f"La réponse JSON de {url} doit être un objet.")
    return payload


def _normaliser_dense(valeurs: Any) -> tuple[float, ...]:
    if not isinstance(valeurs, list) or not valeurs:
        raise ValueError("Vecteur dense absent.")
    vecteur = tuple(float(value) for value in valeurs)
    if not all(isfinite(value) for value in vecteur):
        raise ValueError("Vecteur dense non fini.")
    norme = sqrt(sum(value * value for value in vecteur))
    if norme == 0:
        raise ValueError("Vecteur dense nul.")
    return tuple(value / norme for value in vecteur)
