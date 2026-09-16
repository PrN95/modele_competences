"""Diagnostique l'exposition sparse de Qwen sans afficher les embeddings."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import QWEN_API_KEY, QWEN_BASE_URL, REMOTE_TIMEOUT_SECONDS


TEST_TEXTS = ("analyse de données", "administration système")
REPORT_PATH = Path("outputs/rome/diagnostic_qwen_sparse.md")
_SPARSE_TERMS = ("sparse", "lexical", "token_weight", "tokenweights", "indices", "values")


class SparseDiagnosticError(RuntimeError):
    """L'API n'a pas fourni les éléments nécessaires à un diagnostic fiable."""


def _headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"} if api_key else {"Content-Type": "application/json"}


def _model_id(session: Any, base_url: str, headers: Mapping[str, str], timeout: float) -> str:
    response = session.get(f"{base_url.rstrip('/')}/models", headers=dict(headers), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") if isinstance(payload, Mapping) else None
    model_id = next((item.get("id") for item in data or () if isinstance(item, Mapping) and isinstance(item.get("id"), str) and item["id"]), None)
    if not model_id:
        raise SparseDiagnosticError("GET /models ne contient aucun identifiant de modèle exploitable.")
    return model_id


def _request(session: Any, base_url: str, headers: Mapping[str, str], timeout: float, model_id: str, *, return_sparse: bool) -> tuple[int, Mapping[str, Any] | None, str | None]:
    request = {"model": model_id, "input": list(TEST_TEXTS), "encoding_format": "float"}
    if return_sparse:
        request["return_sparse"] = True
    response = session.post(f"{base_url.rstrip('/')}/embeddings", headers=dict(headers), json=request, timeout=timeout)
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if response.status_code >= 400:
        return response.status_code, payload if isinstance(payload, Mapping) else None, "réponse HTTP d'erreur"
    if not isinstance(payload, Mapping):
        raise SparseDiagnosticError("POST /embeddings ne retourne pas un objet JSON.")
    return response.status_code, payload, None


def _type_name(value: Any) -> str:
    return type(value).__name__


def _summary(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"payload": "absent"}
    data = payload.get("data")
    entries = data if isinstance(data, list) else []
    first = entries[0] if entries and isinstance(entries[0], Mapping) else {}
    entry_fields = {key: _type_name(value) for key, value in first.items()}
    dense = first.get("embedding")
    sparse_fields = [key for key in first if any(term in key.lower().replace("-", "_") for term in _SPARSE_TERMS)]
    return {
        "cles_globales": {key: _type_name(value) for key, value in payload.items()},
        "nombre_entrees_data": len(entries),
        "cles_premiere_entree": entry_fields,
        "dimension_vecteur_dense": len(dense) if isinstance(dense, list) else None,
        "champs_sparse_ou_equivalents": sparse_fields,
    }


def _conclusion(standard: dict[str, Any], sparse: dict[str, Any], sparse_status: int) -> str:
    if standard.get("champs_sparse_ou_equivalents") or sparse.get("champs_sparse_ou_equivalents"):
        return "sparse_disponible"
    if sparse_status < 400:
        return "sparse_non_expose_par_api"
    if sparse_status in {400, 404, 422}:
        return "sparse_non_pris_en_charge_par_modele"
    return "diagnostic_indetermine"


def run_diagnostic(*, session: Any | None = None, base_url: str = QWEN_BASE_URL, api_key: str | None = QWEN_API_KEY, timeout: float = REMOTE_TIMEOUT_SECONDS, report_path: Path = REPORT_PATH) -> dict[str, Any]:
    """Exécute les deux requêtes et écrit un rapport sans sérialiser les vecteurs."""
    http = session or requests.Session()
    headers = _headers(api_key)
    model_id = _model_id(http, base_url, headers, timeout)
    standard_status, standard_payload, standard_error = _request(http, base_url, headers, timeout, model_id, return_sparse=False)
    sparse_status, sparse_payload, sparse_error = _request(http, base_url, headers, timeout, model_id, return_sparse=True)
    standard, sparse = _summary(standard_payload), _summary(sparse_payload)
    conclusion = _conclusion(standard, sparse, sparse_status)
    result = {"endpoint": base_url.rstrip("/"), "model_id": model_id, "standard_status": standard_status, "return_sparse_status": sparse_status, "standard": standard, "return_sparse": sparse, "standard_error": standard_error, "return_sparse_error": sparse_error, "conclusion": conclusion}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(result), encoding="utf-8")
    return result


def _report(result: Mapping[str, Any]) -> str:
    return f"""# Diagnostic sparse Qwen3-Embedding-8B

- Endpoint : `{result['endpoint']}`
- Modèle annoncé : `{result['model_id']}`
- Textes de test : 2 textes courts ; vecteurs non affichés.

## Requêtes testées

1. `POST /embeddings` avec `model`, `input` et `encoding_format=float`.
2. Même requête avec `return_sparse=true`.

## Structure résumée

### Requête standard — HTTP {result['standard_status']}

```json
{json.dumps(result['standard'], ensure_ascii=False, indent=2)}
```

### Requête avec `return_sparse=true` — HTTP {result['return_sparse_status']}

```json
{json.dumps(result['return_sparse'], ensure_ascii=False, indent=2)}
```

## Conclusion

`{result['conclusion']}`

Conséquence méthodologique : {'une campagne hybride dense+sparse est techniquement envisageable avec cette API.' if result['conclusion'] == 'sparse_disponible' else 'une campagne hybride dense+sparse Qwen n’est pas réalisable avec l’API actuelle ; rester en dense-only.'}
"""


def main() -> int:
    try:
        result = run_diagnostic()
    except (requests.RequestException, SparseDiagnosticError) as exc:
        print(f"Diagnostic Qwen sparse : échec — {exc}")
        return 1
    print(f"Diagnostic Qwen sparse : {result['conclusion']}")
    print(f"Rapport : {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
