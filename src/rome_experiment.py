"""Campagne BGE-M3 dense sur le corpus ROME, isolée du matching métier."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import requests

from src.config import (
    BGE_API_KEY,
    BGE_BASE_URL,
    ROME_EXPERIMENT_BATCH_SIZE,
    ROME_EXPERIMENT_MAX_RETRIES,
    ROME_EXPERIMENTS_OUTPUT_ROOT,
    REMOTE_TIMEOUT_SECONDS,
)
from src.rome import RomeRelations


DEFAULT_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


@dataclass(frozen=True, slots=True)
class PreparedRomeCorpus:
    codes: tuple[str, ...]
    titles: dict[str, str]
    skill_texts: tuple[str, ...]
    skill_codes: tuple[str, ...]
    positions_by_code: dict[str, np.ndarray]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class RomeDenseExperiment:
    campaign_id: str
    summary: pd.DataFrame
    details: pd.DataFrame
    best_configuration: dict[str, object]


def rome_campaign_output_directory(campaign_name: str) -> Path:
    """Retourne le seul emplacement autorisé pour une nouvelle campagne ROME."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", campaign_name):
        raise ValueError("Le nom de campagne ROME ne peut contenir que lettres, chiffres, '_' et '-'.")
    return ROME_EXPERIMENTS_OUTPUT_ROOT / campaign_name


def prepare_rome_corpus(corpus: pd.DataFrame) -> PreparedRomeCorpus:
    """Construit l'index compétence → emploi à partir du corpus validé."""
    required = {"code_rome", "intitule", "competences", "statut_validation_pdf"}
    missing = required - set(corpus.columns)
    if missing:
        raise ValueError("Colonnes corpus ROME absentes: " + ", ".join(sorted(missing)))
    invalid = corpus[corpus["statut_validation_pdf"] != "valide"]
    if not invalid.empty:
        raise ValueError("Le corpus ROME doit être entièrement valide avant une campagne.")
    codes: list[str] = []
    titles: dict[str, str] = {}
    skill_texts: list[str] = []
    skill_codes: list[str] = []
    positions: dict[str, list[int]] = {}
    for row in corpus.sort_values("code_rome").itertuples(index=False):
        code, title = str(row.code_rome), str(row.intitule)
        if code in titles:
            raise ValueError(f"Le code ROME '{code}' est dupliqué dans le corpus.")
        skills = tuple(part.strip() for part in str(row.competences).splitlines() if part.strip())
        if not skills:
            raise ValueError(f"Le code ROME '{code}' ne contient aucune compétence.")
        codes.append(code)
        titles[code] = title
        positions[code] = []
        for skill in skills:
            positions[code].append(len(skill_texts))
            skill_texts.append(skill)
            skill_codes.append(code)
    payload = "\n".join(f"{code}\t{text}" for code, text in zip(skill_codes, skill_texts)).encode("utf-8")
    return PreparedRomeCorpus(tuple(codes), titles, tuple(skill_texts), tuple(skill_codes), {code: np.asarray(values, dtype=int) for code, values in positions.items()}, hashlib.sha256(payload).hexdigest())


class RomeDenseCache:
    """Cache de campagne pour les embeddings et la matrice cosinus correspondante."""

    def __init__(self, prepared: PreparedRomeCorpus, model_name: str, directory: str | Path) -> None:
        self.prepared = prepared
        safe_name = "".join(char if char.isalnum() or char in "_.-" else "_" for char in model_name)
        self.embedding_path = Path(directory) / f"{safe_name}-dense-{prepared.fingerprint[:16]}.npz"
        self.matrix_path = Path(directory) / f"{safe_name}-dense-similarites-{prepared.fingerprint[:16]}.npz"

    def load_embeddings(self) -> np.ndarray | None:
        if not self.embedding_path.exists():
            return None
        with np.load(self.embedding_path) as archive:
            vectors = archive["vectors"]
            texts = tuple(archive["texts"].tolist())
            codes = tuple(archive["codes"].tolist())
        if texts != self.prepared.skill_texts or codes != self.prepared.skill_codes:
            return None
        return vectors

    def save_embeddings(self, vectors: np.ndarray) -> Path:
        self.embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.embedding_path, vectors=vectors, texts=np.asarray(self.prepared.skill_texts), codes=np.asarray(self.prepared.skill_codes))
        return self.embedding_path

    def load_matrix(self) -> np.ndarray | None:
        if not self.matrix_path.exists():
            return None
        with np.load(self.matrix_path) as archive:
            matrix = archive["matrix"]
        expected = len(self.prepared.skill_texts)
        return matrix if matrix.shape == (expected, expected) else None

    def save_matrix(self, matrix: np.ndarray) -> Path:
        self.matrix_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.matrix_path, matrix=matrix)
        return self.matrix_path


class RemoteEmbeddingError(RuntimeError):
    """Erreur explicite d'un service d'embedding OpenAI-compatible."""


class RemoteOpenAIEmbeddings:
    """Client dense-only, avec découverte de modèle et lots adaptatifs."""

    def __init__(self, base_url: str = BGE_BASE_URL, *, api_key: str | None = BGE_API_KEY, timeout: float = REMOTE_TIMEOUT_SECONDS, max_retries: int = ROME_EXPERIMENT_MAX_RETRIES, session: Any | None = None) -> None:
        if timeout <= 0 or max_retries < 1:
            raise ValueError("Timeout positif et au moins une tentative sont requis.")
        self.base_url = base_url.rstrip("/")
        self.api_key, self.timeout, self.max_retries = api_key or None, float(timeout), max_retries
        self.session = session or requests.Session()
        self.model_id: str | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def discover_model(self) -> str:
        try:
            response = self.session.get(f"{self.base_url}/models", headers=self._headers(), timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("data", []) if isinstance(payload, dict) else []
            model_id = next((item.get("id") for item in models if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]), None)
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise RemoteEmbeddingError(f"Connexion impossible à {self.base_url}/models : {exc}") from exc
        if not model_id:
            raise RemoteEmbeddingError(f"{self.base_url}/models ne retourne aucun identifiant de modèle exploitable.")
        self.model_id = model_id
        return model_id

    def encode(self, texts: Sequence[str], *, batch_size: int = ROME_EXPERIMENT_BATCH_SIZE, progress: Callable[[str], None] = print) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("La taille de lot expérimentale doit être au moins 1.")
        model_id = self.model_id or self.discover_model()
        total, vectors, index, batches = len(texts), [], 0, 0
        started = perf_counter()
        while index < total:
            requested = min(batch_size, total - index)
            chunk, used = self._encode_adaptive(list(texts[index:index + requested]), model_id, requested, progress)
            vectors.extend(chunk)
            index += used
            batches += 1
            progress(f"Embeddings distants : {index}/{total} compétences, lot {batches}, durée {perf_counter() - started:.1f} s")
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def _encode_adaptive(self, texts: list[str], model_id: str, size: int, progress: Callable[[str], None]) -> tuple[list[list[float]], int]:
        current = texts[:size]
        while True:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = self.session.post(f"{self.base_url}/embeddings", headers=self._headers(), json={"model": model_id, "input": current, "encoding_format": "float"}, timeout=self.timeout)
                    response.raise_for_status()
                    payload = response.json()
                    data = payload.get("data") if isinstance(payload, dict) else None
                    if not isinstance(data, list) or len(data) != len(current):
                        raise RemoteEmbeddingError("Le serveur a retourné un nombre d'embeddings inattendu.")
                    ordered = sorted(data, key=lambda item: item["index"])
                    vectors = [item["embedding"] for item in ordered]
                    if any(not isinstance(vector, list) or not vector for vector in vectors):
                        raise RemoteEmbeddingError("Le serveur a retourné un embedding vide ou invalide.")
                    return vectors, len(current)
                except (requests.RequestException, ValueError, TypeError, KeyError, RemoteEmbeddingError) as exc:
                    if attempt < self.max_retries:
                        progress(f"Lot de {len(current)} en échec (tentative {attempt}/{self.max_retries}) : {exc}")
                        continue
                    if len(current) == 1:
                        raise RemoteEmbeddingError(f"Échec définitif sur un lot d'une compétence après {self.max_retries} tentatives : {exc}") from exc
                    smaller = max(1, len(current) // 2)
                    progress(f"Lot de {len(current)} refusé après {self.max_retries} tentatives ; nouvel essai par lots de {smaller}.")
                    current = current[:smaller]
                    break


def load_or_create_embeddings(cache: RomeDenseCache, encoder: Callable[[Sequence[str]], np.ndarray], progress: Callable[[str], None] = print) -> tuple[np.ndarray, bool]:
    cached = cache.load_embeddings()
    if cached is not None:
        progress(f"Embeddings lus depuis le cache : {cache.embedding_path}")
        return cached, True
    vectors = _l2_normalize(np.asarray(encoder(cache.prepared.skill_texts), dtype=np.float32))
    if vectors.ndim != 2 or len(vectors) != len(cache.prepared.skill_texts):
        raise ValueError("L'encodeur doit retourner un vecteur dense par compétence.")
    cache.save_embeddings(vectors)
    progress(f"Embeddings enregistrés : {cache.embedding_path}")
    return vectors, False


def load_or_create_similarity_matrix(cache: RomeDenseCache, vectors: np.ndarray, progress: Callable[[str], None] = print) -> tuple[np.ndarray, bool]:
    cached = cache.load_matrix()
    if cached is not None:
        progress(f"Matrice de similarités lue depuis le cache : {cache.matrix_path}")
        return cached, True
    progress("Calcul unique de la matrice des similarités cosinus…")
    matrix = _l2_normalize(vectors) @ _l2_normalize(vectors).T
    cache.save_matrix(matrix)
    progress(f"Matrice de similarités enregistrée : {cache.matrix_path}")
    return matrix, False


def run_dense_grid(prepared: PreparedRomeCorpus, relations: RomeRelations, matrix: np.ndarray, thresholds: Sequence[float] = DEFAULT_THRESHOLDS, campaign_id: str | None = None, progress: Callable[[str], None] = print, *, model_name: str = "bge-m3") -> RomeDenseExperiment:
    """Évalue toutes les configurations sans recalculer embeddings ni cosinus."""
    if matrix.shape != (len(prepared.skill_texts), len(prepared.skill_texts)):
        raise ValueError("La matrice de similarités ne correspond pas au corpus préparé.")
    thresholds = tuple(float(value) for value in thresholds)
    if not thresholds or any(value < 0 or value > 1 for value in thresholds):
        raise ValueError("Les seuils doivent être compris entre 0 et 1.")
    unknown = relations.codes - set(prepared.codes)
    if unknown:
        raise ValueError("Des relations ROME référencent des codes absents du corpus: " + ", ".join(sorted(unknown)))
    campaign_id = campaign_id or f"rome_{model_name.replace('-', '_')}_dense_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    all_pairs = tuple(frozenset((code_a, code_b)) for index, code_a in enumerate(prepared.codes) for code_b in prepared.codes[index + 1:])
    # Une table de vérité étiquetée évalue seulement ses paires explicites. Les
    # anciennes tables sans négatifs conservent leur comportement historique :
    # toute paire absente est une non-relation de référence.
    evaluated_pairs = relations.pairs | relations.negative_pairs if relations.negative_pairs else frozenset(all_pairs)
    pairs = tuple(tuple(sorted(pair)) for pair in sorted(evaluated_pairs, key=lambda pair: tuple(sorted(pair))))
    reference = relations.pairs
    summaries: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    total = len(thresholds) ** 2
    done = 0
    for seuil_sim in thresholds:
        coverage_rows = [(a, b, *_pair_coverages(prepared, matrix, a, b, seuil_sim)) for a, b in pairs]
        for seuil_couv in thresholds:
            start = perf_counter()
            tp = fp = fn = tn = 0
            configuration_details: list[dict[str, object]] = []
            for code_a, code_b, a_to_b, b_to_a, symmetric in coverage_rows:
                actual = frozenset((code_a, code_b)) in reference
                predicted = symmetric >= seuil_couv
                evaluation = "TP" if actual and predicted else "FP" if predicted else "FN" if actual else "TN"
                if evaluation == "TP": tp += 1
                elif evaluation == "FP": fp += 1
                elif evaluation == "FN": fn += 1
                else: tn += 1
                configuration_details.append({"identifiant_campagne": campaign_id, "code_rome_a": code_a, "intitule_a": prepared.titles[code_a], "code_rome_b": code_b, "intitule_b": prepared.titles[code_b], "relation_reference": actual, "relation_predite": predicted, "couverture_a_vers_b": a_to_b, "couverture_b_vers_a": b_to_a, "couverture_symetrique": symmetric, "seuil_sim": seuil_sim, "seuil_couv": seuil_couv, "classe_evaluation": evaluation})
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            accuracy = (tp + tn) / len(pairs) if pairs else 0.0
            summary = {"identifiant_campagne": campaign_id, "modele": model_name, "mode_scoring": "dense_uniquement_cosinus", "coefficient_dense": 1.0, "coefficient_sparse": 0.0, "seuil_sim": seuil_sim, "seuil_couv": seuil_couv, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": accuracy, "precision": precision, "rappel": recall, "f1": f1, "nombre_relations_predites": tp + fp, "duree_calcul_secondes": perf_counter() - start, "meilleure_configuration": False}
            summaries.append(summary)
            details.extend(configuration_details)
            done += 1
            progress(f"Évaluation {done}/{total} : seuil_sim={seuil_sim:.2f}, seuil_couv={seuil_couv:.2f}")
    summary_df = pd.DataFrame(summaries)
    best_index = min(summary_df.index, key=lambda index: (-float(summary_df.loc[index, "f1"]), -float(summary_df.loc[index, "rappel"]), -float(summary_df.loc[index, "precision"])))
    summary_df.loc[best_index, "meilleure_configuration"] = True
    best = summary_df.loc[best_index].to_dict()
    return RomeDenseExperiment(campaign_id, summary_df, pd.DataFrame(details), best)


def write_dense_experiment_outputs(experiment: RomeDenseExperiment, prepared: PreparedRomeCorpus, relations: RomeRelations, cache: RomeDenseCache, model_path: str | Path, output_directory: str | Path | None = None, *, endpoint: str | None = None, batch_size: int | None = None, model_name: str = "bge-m3", report_title: str = "BGE-M3", reference_table: str | Path | None = None) -> tuple[Path, Path, Path, Path]:
    """Écrit les exports standardisés et un rapport de synthèse Markdown."""
    directory = rome_campaign_output_directory(experiment.campaign_id) if output_directory is None else Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path, detail_path = directory / "resume_configurations.csv", directory / "detail_relations.csv"
    metadata_path, report_path = directory / "metadonnees.json", directory / "rapport_synthese.md"
    experiment.summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    experiment.details.to_csv(detail_path, index=False, encoding="utf-8-sig")
    metadata = {"date_heure_utc": datetime.now(timezone.utc).isoformat(), "identifiant_campagne": experiment.campaign_id, "modele": model_name, "identifiant_modele_annonce": str(model_path), "endpoint": endpoint, "table_reference": str(reference_table) if reference_table is not None else None, "taille_lot_experimentale": batch_size, "mode_scoring": "dense_uniquement_cosinus", "coefficient_dense": 1.0, "coefficient_sparse": 0.0, "grille_seuil_sim": list(DEFAULT_THRESHOLDS), "grille_seuil_couv": list(DEFAULT_THRESHOLDS), "empreinte_corpus": prepared.fingerprint, "nombre_emplois": len(prepared.codes), "nombre_competences": len(prepared.skill_texts), "nombre_relations_reference": len(relations.pairs), "nombre_non_relations_reference": len(relations.negative_pairs), "nombre_paires_reference_evaluees": len(relations.pairs | relations.negative_pairs) if relations.negative_pairs else None, "cache_embeddings": str(cache.embedding_path), "cache_matrice_similarites": str(cache.matrix_path), "note_reference": "Les paires absentes du graphe ROME sont classées non_relation_reference ; ce n'est pas une vérité humaine absolue."}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best = experiment.best_configuration
    best_details = experiment.details[(experiment.details["seuil_sim"] == best["seuil_sim"]) & (experiment.details["seuil_couv"] == best["seuil_couv"])]
    report_path.write_text(_markdown_report(best, best_details, report_title), encoding="utf-8")
    return summary_path, detail_path, metadata_path, report_path


def _pair_coverages(prepared: PreparedRomeCorpus, matrix: np.ndarray, code_a: str, code_b: str, threshold: float) -> tuple[float, float, float]:
    similarities = matrix[np.ix_(prepared.positions_by_code[code_a], prepared.positions_by_code[code_b])]
    a_to_b = float((similarities.max(axis=0) >= threshold).mean())
    b_to_a = float((similarities.max(axis=1) >= threshold).mean())
    return a_to_b, b_to_a, (a_to_b + b_to_a) / 2


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2: raise ValueError("Les embeddings denses doivent former une matrice à deux dimensions.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0): raise ValueError("Un embedding dense nul ne peut pas être normalisé.")
    return array / norms


def _markdown_report(best: dict[str, object], details: pd.DataFrame, report_title: str = "BGE-M3") -> str:
    def pairs(label: str) -> str:
        selected = details[details["classe_evaluation"] == label].sort_values("couverture_symetrique", ascending=False).head(10)
        if selected.empty: return "Aucune."
        return "\n".join(f"- {row.code_rome_a} — {row.code_rome_b} ({row.couverture_symetrique:.3f})" for row in selected.itertuples())
    return f"""# Campagne expérimentale ROME — {report_title} dense

Méthode : cosinus dense entre toutes les compétences ; couverture symétrique = moyenne des deux couvertures directionnelles. Une paire est prédite proche lorsque cette couverture atteint `seuil_couv`.

## Meilleure configuration

- seuil_sim : {best['seuil_sim']:.2f}
- seuil_couv : {best['seuil_couv']:.2f}
- F1 : {best['f1']:.4f}
- précision : {best['precision']:.4f}
- rappel : {best['rappel']:.4f}

Les paires absentes du graphe ROME sont `non_relation_reference`, et non une vérité humaine absolue.

## Principaux faux positifs

{pairs('FP')}

## Relations ROME non retrouvées

{pairs('FN')}
"""
