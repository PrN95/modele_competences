"""Campagne 4 ROME : décision binaire directe par Gemma4."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import GEMMA_API_KEY, GEMMA_BASE_URL, REMOTE_TIMEOUT_SECONDS
from src.rome import read_rome_relations
from src.rome_experiment import prepare_rome_corpus, rome_campaign_output_directory

SYSTEM_PROMPT = """Tu évalues directement la proximité de deux métiers à partir de leurs seules compétences.
Décide si les métiers sont similaires dans leur ensemble. N'utilise ni score, ni seuil, ni annotation externe.
Réponds exclusivement par un objet JSON valide : {\"decision\":\"Oui\"} ou {\"decision\":\"Non\"}."""


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if GEMMA_API_KEY:
        headers["Authorization"] = f"Bearer {GEMMA_API_KEY}"
    return headers


def _discover(session: requests.Session) -> str:
    response = session.get(f"{GEMMA_BASE_URL.rstrip('/')}/models", headers=_headers(), timeout=REMOTE_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json().get("data", [])
    model_id = next((item.get("id") for item in data if isinstance(item, dict) and item.get("id")), None)
    if not model_id:
        raise RuntimeError("Gemma ne retourne aucun modèle exploitable.")
    return model_id


def _prompt(code_a: str, job_a: pd.Series, code_b: str, job_b: pd.Series) -> str:
    def job(code: str, row: pd.Series) -> str:
        skills = "\n".join(f"- {item}" for item in str(row.competences).splitlines())
        return f"Métier {code} — {row.intitule}\nCompétences :\n{skills}"
    return f"{job(code_a, job_a)}\n\n{job(code_b, job_b)}"


def _metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for row in rows:
        actual, predicted = row["relation_reference"], row["decision_gemma"]
        if actual == "Oui" and predicted == "Oui": tp += 1
        elif actual == "Non" and predicted == "Oui": fp += 1
        elif actual == "Oui" and predicted == "Non": fn += 1
        elif actual == "Non" and predicted == "Non": tn += 1
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / total if total else 0.0, "precision": precision, "rappel": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "nombre_paires_evaluees": total}


def main() -> int:
    parser = argparse.ArgumentParser(description="Campagne ROME Gemma4.")
    parser.add_argument("--campaign-name", default=f"campagne_gemma4_{time.strftime('%Y%m%dT%H%M%S')}")
    args = parser.parse_args()
    output = rome_campaign_output_directory(args.campaign_name)
    if output.exists():
        raise FileExistsError(f"Le dossier existe déjà : {output}")
    corpus = pd.read_csv("outputs/rome/corpus_rome.csv")
    prepared = prepare_rome_corpus(corpus)
    relations, diagnostics = read_rome_relations("data/rome/Similarité Emplois ROME 6.xlsx")
    if any(item.severity == "erreur" for item in diagnostics):
        raise RuntimeError("La table ROME est invalide.")
    by_code = corpus.set_index("code_rome")
    unknown = relations.codes - set(by_code.index)
    if unknown:
        raise RuntimeError("Codes absents du corpus : " + ", ".join(sorted(unknown)))
    session = requests.Session()
    model_id = _discover(session)
    pairs = sorted(relations.pairs | relations.negative_pairs, key=lambda pair: tuple(sorted(pair)))
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, pair in enumerate(pairs, 1):
        code_a, code_b = sorted(pair)
        job_a, job_b = by_code.loc[code_a], by_code.loc[code_b]
        reference = "Oui" if pair in relations.pairs else "Non"
        payload = {"model": model_id, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": _prompt(code_a, job_a, code_b, job_b)}], "temperature": 0, "max_tokens": 64, "response_format": {"type": "json_object"}}
        row = {"code_rome_a": code_a, "intitule_a": job_a.intitule, "code_rome_b": code_b, "intitule_b": job_b.intitule, "relation_reference": reference, "decision_gemma": "", "statut": "succes", "erreur": "", "reponse_brute": ""}
        try:
            response = session.post(f"{GEMMA_BASE_URL.rstrip('/')}/chat/completions", headers=_headers(), json=payload, timeout=REMOTE_TIMEOUT_SECONDS)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            decision = json.loads(content)
            if not isinstance(decision, dict) or set(decision) != {"decision"} or decision["decision"] not in {"Oui", "Non"}:
                raise ValueError("JSON de décision invalide")
            row.update(decision_gemma=decision["decision"], reponse_brute=content)
        except (requests.RequestException, ValueError, TypeError, KeyError, IndexError) as exc:
            row.update(statut="erreur", erreur=str(exc))
        rows.append(row)
        print(f"Gemma : {index}/{len(pairs)} paires, durée {time.perf_counter() - started:.1f} s")
    decisions = pd.DataFrame(rows)
    valid = [row for row in rows if row["statut"] == "succes"]
    metrics = _metrics(valid)
    output.mkdir(parents=True)
    decisions.to_csv(output / "decisions_gemma.csv", index=False, encoding="utf-8-sig")
    metadata = {"date_heure_utc": datetime.now(timezone.utc).isoformat(), "identifiant_campagne": output.name, "modele": "gemma4", "identifiant_modele_annonce": model_id, "endpoint": GEMMA_BASE_URL, "temperature": 0, "mode_scoring": "decision_binaire_llm", "table_reference": "data/rome/Similarité Emplois ROME 6.xlsx", "empreinte_corpus": prepared.fingerprint, "nombre_paires_demandees": len(rows), "nombre_erreurs": int((decisions.statut == "erreur").sum()), **metrics}
    (output / "metadonnees.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = "# Campagne expérimentale ROME — Gemma4\n\nDécision binaire directe sans labels, scores, seuils ou embeddings dans les requêtes.\n\n## Métriques\n\n" + "\n".join(f"- {key} : {value:.4f}" if isinstance(value, float) else f"- {key} : {value}" for key, value in metrics.items()) + f"\n- erreurs de service : {metadata['nombre_erreurs']}\n"
    (output / "rapport_synthese.md").write_text(report, encoding="utf-8")
    print(f"Gemma terminé : F1={metrics['f1']:.4f}, accuracy={metrics['accuracy']:.4f}, erreurs={metadata['nombre_erreurs']}")
    return 0 if metadata["nombre_erreurs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
