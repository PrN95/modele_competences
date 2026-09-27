import os
from pathlib import Path

import pandas as pd
import pytest
import requests

from src.config import COMPETENCES_SHEET


@pytest.fixture(autouse=True)
def interdire_modele_bge_reel_pendant_pytest(monkeypatch):
    """Empêche tout chargement accidentel du vrai modèle dans les tests."""

    def _interdit(*args, **kwargs):
        raise AssertionError(
            "Le véritable modèle BGE-M3 ne doit jamais être chargé pendant pytest."
        )

    monkeypatch.setattr("src.embeddings._creer_modele_flagembedding", _interdit)


@pytest.fixture(autouse=True)
def interdire_serveurs_distants_reels_pendant_pytest(monkeypatch):
    """Empêche tout contact accidentel avec des services distants configurés."""

    methode_originale = requests.sessions.Session.request

    def _requete_filtree(self, method, url, *args, **kwargs):
        if any(
            endpoint and endpoint in str(url)
            for endpoint in (
                os.getenv("REMOTE_BGE_DISTANT_BASE_URL"),
                os.getenv("REMOTE_QWEN_DISTANT_BASE_URL"),
                os.getenv("REMOTE_GEMMA_DISTANT_BASE_URL"),
            )
        ):
            raise AssertionError(
                "Les serveurs distants configurés ne doivent jamais être contactés "
                "pendant pytest."
            )
        return methode_originale(self, method, url, *args, **kwargs)

    monkeypatch.setattr(requests.sessions.Session, "request", _requete_filtree)


@pytest.fixture
def valid_current_rows() -> list[dict[str, object]]:
    return [
        {
            "emploi_id": "EMP-ACT-01",
            "emploi_intitule": "Gestionnaire fictif",
            "emploi_type": "actuel",
            "emploi_effectif": 24,
            "competence_id": "COMP-01",
            "competence_intitule": "Analyser des données fictives",
            "competence_description": "Contrôler un jeu de données de démonstration",
            "competence_niveau": 2,
        },
        {
            "emploi_id": "EMP-ACT-01",
            "emploi_intitule": "Gestionnaire fictif",
            "emploi_type": "actuel",
            "emploi_effectif": 24,
            "competence_id": "COMP-02",
            "competence_intitule": "Présenter une synthèse fictive",
            "competence_description": "Restituer clairement des résultats de test",
            "competence_niveau": 1,
        },
    ]


@pytest.fixture
def valid_target_rows(valid_current_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [dict(row) for row in valid_current_rows]
    for row in rows:
        row["emploi_id"] = "EMP-CIB-01"
        row["emploi_intitule"] = "Responsable fictif"
        row["emploi_type"] = "cible"
        row["emploi_effectif"] = None
    return rows


@pytest.fixture
def write_workbook(tmp_path: Path):
    def _write(
        filename: str,
        rows: list[dict[str, object]],
        *,
        sheet_name: str = COMPETENCES_SHEET,
    ) -> Path:
        path = tmp_path / filename
        pd.DataFrame(rows).to_excel(path, sheet_name=sheet_name, index=False)
        return path

    return _write
