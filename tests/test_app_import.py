import ast
import os
import subprocess
import sys
from pathlib import Path

import src.config as config


RACINE_PROJET = Path(__file__).resolve().parents[1]


def test_imports_config_app_et_demarrage_minimal() -> None:
    """Détecte les constantes absentes avant tout démarrage Streamlit."""

    arbre = ast.parse((RACINE_PROJET / "app.py").read_text(encoding="utf-8"))
    noms_importes = {
        alias.name
        for noeud in ast.walk(arbre)
        if isinstance(noeud, ast.ImportFrom) and noeud.module == "src.config"
        for alias in noeud.names
    }
    noms_absents = sorted(nom for nom in noms_importes if not hasattr(config, nom))

    assert noms_absents == []
    assert config.QWEN_BASE_URL == os.getenv("REMOTE_QWEN_DISTANT_BASE_URL", "")
    assert config.BGE_BASE_URL == os.getenv("REMOTE_BGE_DISTANT_BASE_URL", "")
    assert config.GEMMA_BASE_URL == os.getenv("REMOTE_GEMMA_DISTANT_BASE_URL", "")
    assert config.QWEN_MODEL == "qwen3-embedding-8b"
    assert config.BGE_MODEL == "bge-m3"
    assert config.GEMMA_MODEL == "gemma4"
    assert float(config.DENSE_WEIGHT) == 2 / 3
    assert float(config.SPARSE_WEIGHT) == 1 / 3

    resultat = subprocess.run(
        [sys.executable, "-c", "import app"],
        cwd=RACINE_PROJET,
        capture_output=True,
        text=True,
        check=False,
    )

    assert resultat.returncode == 0, resultat.stderr


def test_uploader_transmet_le_nom_original_au_parseur_pdf() -> None:
    arbre = ast.parse((RACINE_PROJET / "app.py").read_text(encoding="utf-8"))
    appels = [
        noeud
        for noeud in ast.walk(arbre)
        if isinstance(noeud, ast.Call)
        and isinstance(noeud.func, ast.Name)
        and noeud.func.id == "read_emploi_pdf"
    ]

    assert len(appels) == 2
    for appel in appels:
        original = next(
            (mot for mot in appel.keywords if mot.arg == "original_filename"),
            None,
        )
        assert original is not None
        assert isinstance(original.value, ast.Attribute)
        assert isinstance(original.value.value, ast.Name)
        assert original.value.value.id == "f"
        assert original.value.attr == "name"


def test_comparaison_streamlit_exclut_bge_local_et_desactive_son_chemin() -> None:
    source = (RACINE_PROJET / "app.py").read_text(encoding="utf-8")

    assert 'MODE_COMPARAISON_DISTANTE = "Comparaison des trois modèles distant"' in source
    assert "MODELE_BGE_DISTANT: lancer_bge_distant" in source
    assert "MODELE_BGE: lancer_bge," not in source[source.index("selection_lanceurs ="):]
    assert 'disabled=(mode_execution == MODE_COMPARAISON_DISTANTE)' in source
