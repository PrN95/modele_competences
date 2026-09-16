from pathlib import Path
import importlib.util


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "check_qwen_sparse_capability.py"
    spec = importlib.util.spec_from_file_location("check_qwen_sparse_capability", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload, self.status_code = payload, status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError("erreur")

    def json(self):
        return self.payload


class _Session:
    def get(self, url, **kwargs):
        return _Response({"data": [{"id": "qwen-reel"}]})

    def post(self, url, **kwargs):
        request = kwargs["json"]
        entry = {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
        if request.get("return_sparse"):
            entry["lexical_weights"] = {"42": 0.8}
        return _Response({"object": "list", "data": [entry, {**entry, "index": 1}], "model": request["model"]})


def test_sparse_diagnostic_summarizes_structure_without_embedding_values(tmp_path: Path) -> None:
    module = _load_module()
    report = tmp_path / "diagnostic.md"

    result = module.run_diagnostic(session=_Session(), base_url="http://qwen/v1", report_path=report)

    assert result["model_id"] == "qwen-reel"
    assert result["standard"]["dimension_vecteur_dense"] == 3
    assert result["return_sparse"]["champs_sparse_ou_equivalents"] == ["lexical_weights"]
    assert result["conclusion"] == "sparse_disponible"
    text = report.read_text(encoding="utf-8")
    assert "0.1" not in text
    assert "lexical_weights" in text
