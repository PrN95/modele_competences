from pathlib import Path

import numpy as np
import pandas as pd

from src.rome import (
    RomeEmbeddingCache,
    RomePdfExtraction,
    build_rome_corpus,
    evaluate_rome_rankings,
    extract_rome_pdf_pages,
    read_rome_relations,
)


def _write_relations(path: Path) -> None:
    pd.DataFrame(
        [
            {"Intitulé emploi actuel": "Métier A", "Code ROME emploi actuel": "m 1000", "Intitulé emploi cible": "Métier B", "Code ROME emploi cible": "M2000"},
            {"Intitulé emploi actuel": "Métier B", "Code ROME emploi actuel": "M2000", "Intitulé emploi cible": "Métier A", "Code ROME emploi cible": "M1000"},
        ]
    ).to_excel(path, index=False)


def test_extract_rome_competences_over_multiple_pages_and_stops_at_contextes() -> None:
    extraction = extract_rome_pdf_pages((
        "M1000\nMETIER A\nCompétences\nSavoir-faire\n• Première compétence Transition numérique\n• Compétence coupée",
        "suite de la compétence\n• Deuxième compétence\nContextes de travail\n• Ne doit jamais être extraite\nSecteurs d'activité\n• Ni celle-ci",
    ))

    assert extraction.code == "M1000"
    assert extraction.title == "METIER A"
    assert extraction.competences == ("Première compétence", "Compétence coupée suite de la compétence", "Deuxième compétence")


def test_relations_are_symmetric_and_duplicates_reported(tmp_path: Path) -> None:
    excel = tmp_path / "rome.xlsx"
    _write_relations(excel)

    relations, diagnostics = read_rome_relations(excel)

    assert relations.neighbors("M1000") == frozenset({"M2000"})
    assert relations.neighbors("M2000") == frozenset({"M1000"})
    assert any(diagnostic.code == "doublon_relation" for diagnostic in diagnostics)


def test_build_corpus_reports_missing_pdf_without_hiding_valid_pdf(tmp_path: Path, monkeypatch) -> None:
    excel, pdfs = tmp_path / "rome.xlsx", tmp_path / "pdfs"
    pdfs.mkdir()
    _write_relations(excel)
    (pdfs / "M1000 - A.pdf").touch()

    monkeypatch.setattr("src.rome.extract_rome_pdf", lambda path: RomePdfExtraction("M1000", "Métier A", ("Compétence A",)))
    result = build_rome_corpus(excel, pdfs)

    assert result.corpus.set_index("code_rome").loc["M1000", "statut_validation_pdf"] == "valide"
    assert result.corpus.set_index("code_rome").loc["M2000", "statut_validation_pdf"] == "pdf_manquant"
    assert any(diagnostic.code == "pdf_manquant" and diagnostic.rome_code == "M2000" for diagnostic in result.diagnostics)


def test_build_corpus_marks_filename_header_code_inconsistency(tmp_path: Path, monkeypatch) -> None:
    excel, pdfs = tmp_path / "rome.xlsx", tmp_path / "pdfs"
    pdfs.mkdir()
    _write_relations(excel)
    (pdfs / "M9999 - mauvais code.pdf").touch()
    (pdfs / "M2000 - B.pdf").touch()
    responses = iter((RomePdfExtraction("M1000", "Métier A", ("A",)), RomePdfExtraction("M2000", "Métier B", ("B",))))
    monkeypatch.setattr("src.rome.extract_rome_pdf", lambda path: next(responses))

    result = build_rome_corpus(excel, pdfs)

    assert result.corpus.set_index("code_rome").loc["M1000", "statut_validation_pdf"] == "code_pdf_incoherent"
    assert any(diagnostic.code == "code_pdf_incoherent" for diagnostic in result.diagnostics)


def test_build_corpus_prefers_descriptive_filename_for_duplicate_rome_code(tmp_path: Path, monkeypatch) -> None:
    excel, pdfs = tmp_path / "rome.xlsx", tmp_path / "pdfs"
    pdfs.mkdir()
    _write_relations(excel)
    bare, descriptive = pdfs / "M1000.pdf", pdfs / "M1000 - Métier A.pdf"
    bare.touch()
    descriptive.touch()
    (pdfs / "M2000 - Métier B.pdf").touch()

    def extract(path: Path) -> RomePdfExtraction:
        if path.name == "M2000 - Métier B.pdf":
            return RomePdfExtraction("M2000", "Métier B", ("B",))
        return RomePdfExtraction("M1000", "Métier A", (path.name,))

    monkeypatch.setattr("src.rome.extract_rome_pdf", extract)
    result = build_rome_corpus(excel, pdfs)

    row = result.corpus.set_index("code_rome").loc["M1000"]
    assert row["fichier_pdf"] == "M1000 - Métier A.pdf"
    assert row["competences"] == "M1000 - Métier A.pdf"
    assert not result.has_errors
    assert any(diagnostic.code == "pdf_duplique_nom_descriptif_retenu" for diagnostic in result.diagnostics)


def test_cache_and_evaluation_are_model_independent(tmp_path: Path) -> None:
    corpus = pd.DataFrame({"code_rome": ["M1000", "M2000"], "competences": ["a", "b"]})
    cache = RomeEmbeddingCache(corpus, tmp_path / "cache")
    path = cache.save("bge-m3", np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert path.exists()
    assert np.array_equal(cache.load("bge-m3"), np.array([[1.0, 0.0], [0.0, 1.0]]))
    excel = tmp_path / "rome.xlsx"
    _write_relations(excel)
    relations, _ = read_rome_relations(excel)
    assert evaluate_rome_rankings(relations, {"M1000": ["M2000"], "M2000": ["M1000"]}, k=1) == {"queries_evaluees": 2.0, "recall@1": 1.0, "mrr": 1.0}
