"""Corpus expérimental ROME, indépendant du matching applicatif."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import unicodedata
from typing import Sequence

import numpy as np
import pandas as pd
import pdfplumber


ROME_REQUIRED_COLUMNS = (
    "Intitulé emploi actuel", "Code ROME emploi actuel",
    "Intitulé emploi cible", "Code ROME emploi cible",
)
ROME_TRUTH_TABLE_COLUMNS = (
    "code_emploi_a", "intitule_emploi_a", "code_emploi_b", "intitule_emploi_b", "similarite",
)
_ROME_CODE = re.compile(r"^[A-Z]\d{4}$")
_ROME_DESCRIPTIVE_FILENAME = re.compile(r"^[A-Z]\d{4}\s-\s.+\.pdf$", re.IGNORECASE)
_BULLET = re.compile(r"^[•\-–]\s*(.+)$")
_FOOTER = re.compile(r"^\d+\s*/\s*\d+\s*-\s*©", re.IGNORECASE)
_TRANSITION_SUFFIX = re.compile(r"\s+Transition\s+(?:numérique|écologique)\s*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RomeDiagnostic:
    severity: str
    code: str
    message: str
    rome_code: str | None = None
    source: str | None = None
    excel_row: int | None = None


@dataclass(frozen=True, slots=True)
class RomeRelations:
    titles_by_code: dict[str, tuple[str, ...]]
    pairs: frozenset[frozenset[str]]
    negative_pairs: frozenset[frozenset[str]] = frozenset()

    @property
    def codes(self) -> frozenset[str]:
        return frozenset(self.titles_by_code)

    def neighbors(self, code: str) -> frozenset[str]:
        normalized = normalize_rome_code(code)
        return frozenset(next(iter(pair - {normalized})) for pair in self.pairs if normalized in pair)


@dataclass(frozen=True, slots=True)
class RomePdfExtraction:
    code: str | None
    title: str | None
    competences: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RomeValidationResult:
    relations: RomeRelations
    corpus: pd.DataFrame
    diagnostics: tuple[RomeDiagnostic, ...]

    @property
    def has_errors(self) -> bool:
        return any(d.severity == "erreur" for d in self.diagnostics)

    def report_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{"gravite": d.severity, "code": d.code, "code_rome": d.rome_code, "source": d.source, "ligne_excel": d.excel_row, "message": d.message} for d in self.diagnostics])


def normalize_rome_code(value: object) -> str | None:
    """Normalise un code ROME sans jamais inventer une valeur manquante."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    normalized = re.sub(r"\s+", "", str(value).upper())
    return normalized if _ROME_CODE.fullmatch(normalized) else None


def read_rome_relations(excel_path: str | Path) -> tuple[RomeRelations, tuple[RomeDiagnostic, ...]]:
    """Lit l'Excel et représente les relations A-B et B-A par une seule arête."""
    source = Path(excel_path)
    dataframe = pd.read_excel(source, engine="openpyxl")
    if set(ROME_TRUTH_TABLE_COLUMNS).issubset(dataframe.columns):
        return _read_rome_truth_table(dataframe, source)
    missing = [column for column in ROME_REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Colonnes ROME obligatoires absentes: " + ", ".join(missing)
            + ". Colonnes attendues pour une table de vérité : " + ", ".join(ROME_TRUTH_TABLE_COLUMNS)
        )
    titles: dict[str, set[str]] = defaultdict(set)
    pairs: set[frozenset[str]] = set()
    diagnostics: list[RomeDiagnostic] = []
    seen: set[tuple[str, str]] = set()
    for index, row in dataframe.iterrows():
        row_number = int(index) + 2
        current = normalize_rome_code(row[ROME_REQUIRED_COLUMNS[1]])
        target = normalize_rome_code(row[ROME_REQUIRED_COLUMNS[3]])
        for code, title_column, role in ((current, ROME_REQUIRED_COLUMNS[0], "actuel"), (target, ROME_REQUIRED_COLUMNS[2], "cible")):
            title = row[title_column]
            if code is None:
                diagnostics.append(RomeDiagnostic("erreur", "code_manquant", f"Code ROME {role} absent ou invalide.", source=source.name, excel_row=row_number))
            elif isinstance(title, str) and title.strip():
                titles[code].add(_clean_text(title))
            else:
                diagnostics.append(RomeDiagnostic("erreur", "intitule_manquant", f"Intitulé emploi {role} absent.", code, source.name, row_number))
        if current is None or target is None:
            continue
        pair_key = tuple(sorted((current, target)))
        if current == target:
            diagnostics.append(RomeDiagnostic("avertissement", "relation_reflexive", "Relation vers le même code ROME ignorée.", current, source.name, row_number))
            continue
        if pair_key in seen:
            diagnostics.append(RomeDiagnostic("avertissement", "doublon_relation", "Relation dupliquée (sens A-B ou B-A).", current, source.name, row_number))
        seen.add(pair_key)
        pairs.add(frozenset(pair_key))
    for code, values in titles.items():
        if len({_normalize_title(value) for value in values}) > 1:
            diagnostics.append(RomeDiagnostic("avertissement", "incoherence_code_intitule", "Un même code ROME est associé à plusieurs intitulés dans Excel.", code, source.name))
    return RomeRelations({code: tuple(sorted(values)) for code, values in sorted(titles.items())}, frozenset(pairs)), tuple(diagnostics)


def _read_rome_truth_table(dataframe: pd.DataFrame, source: Path) -> tuple[RomeRelations, tuple[RomeDiagnostic, ...]]:
    """Lit la table de vérité ROME 6, dont les paires sont étiquetées Oui/Non."""
    titles: dict[str, set[str]] = defaultdict(set)
    pairs: set[frozenset[str]] = set()
    negative_pairs: set[frozenset[str]] = set()
    seen: set[tuple[str, str]] = set()
    diagnostics: list[RomeDiagnostic] = []
    for index, row in dataframe.iterrows():
        row_number = int(index) + 2
        current = normalize_rome_code(row["code_emploi_a"])
        target = normalize_rome_code(row["code_emploi_b"])
        similarity = str(row["similarite"]).strip()
        for code, title_column, role in ((current, "intitule_emploi_a", "a"), (target, "intitule_emploi_b", "b")):
            title = row[title_column]
            if code is None:
                diagnostics.append(RomeDiagnostic("erreur", "code_manquant", f"Code ROME {role} absent ou invalide.", source=source.name, excel_row=row_number))
            elif isinstance(title, str) and title.strip():
                titles[code].add(_clean_text(title))
            else:
                diagnostics.append(RomeDiagnostic("erreur", "intitule_manquant", f"Intitulé emploi {role} absent.", code, source.name, row_number))
        if similarity not in {"Oui", "Non"}:
            diagnostics.append(RomeDiagnostic("erreur", "similarite_invalide", "La colonne similarite doit valoir 'Oui' ou 'Non'.", source=source.name, excel_row=row_number))
        if current is None or target is None:
            continue
        pair_key = tuple(sorted((current, target)))
        if current == target:
            diagnostics.append(RomeDiagnostic("erreur", "relation_reflexive", "Une paire de vérité ne peut pas relier un code à lui-même.", current, source.name, row_number))
            continue
        if pair_key in seen:
            diagnostics.append(RomeDiagnostic("erreur", "doublon_relation", "Paire de vérité dupliquée (sens A-B ou B-A).", current, source.name, row_number))
            continue
        seen.add(pair_key)
        if similarity == "Oui":
            pairs.add(frozenset(pair_key))
        elif similarity == "Non":
            negative_pairs.add(frozenset(pair_key))
    for code, values in titles.items():
        if len({_normalize_title(value) for value in values}) > 1:
            diagnostics.append(RomeDiagnostic("avertissement", "incoherence_code_intitule", "Un même code ROME est associé à plusieurs intitulés dans Excel.", code, source.name))
    return RomeRelations(
        {code: tuple(sorted(values)) for code, values in sorted(titles.items())},
        frozenset(pairs),
        frozenset(negative_pairs),
    ), tuple(diagnostics)


def extract_rome_pdf(path: str | Path) -> RomePdfExtraction:
    """Extrait le code, l'intitulé et les puces de la rubrique Compétences."""
    with pdfplumber.open(Path(path), unicode_norm="NFC") as pdf:
        return extract_rome_pdf_pages(tuple(page.extract_text(x_tolerance=2, y_tolerance=3) or "" for page in pdf.pages))


def extract_rome_pdf_pages(pages: Sequence[str]) -> RomePdfExtraction:
    """Version purement textuelle, utilisable dans les tests sans fichier PDF."""
    code: str | None = None
    title: str | None = None
    collecting = False
    current: str | None = None
    competences: list[str] = []
    for page in pages:
        lines = [_clean_text(line) for line in page.splitlines()]
        for index, line in enumerate(lines):
            if not line or _FOOTER.match(line):
                continue
            found_code = normalize_rome_code(line)
            if found_code and code is None:
                code, title = found_code, _find_pdf_title(lines[index + 1:])
                continue
            marker = _normalize_title(line)
            if marker == "competences":
                collecting, current = True, None
                continue
            if collecting and marker == "contextes de travail":
                if current:
                    competences.append(current)
                collecting, current = False, None
                continue
            if not collecting or line.lower().startswith("fiche emploi "):
                continue
            bullet = _BULLET.match(line)
            if bullet:
                if current:
                    competences.append(current)
                current = _clean_competence(bullet.group(1))
            elif current and not _looks_like_heading(line):
                current = _clean_competence(f"{current} {line}")
    if current:
        competences.append(current)
    return RomePdfExtraction(code, title, tuple(dict.fromkeys(item for item in competences if item)))


def build_rome_corpus(excel_path: str | Path, pdf_directory: str | Path) -> RomeValidationResult:
    """Valide les sources et poursuit le traitement si une fiche est défectueuse."""
    relations, relation_diagnostics = read_rome_relations(excel_path)
    diagnostics = list(relation_diagnostics)
    pdf_candidates: dict[str, list[tuple[Path, RomePdfExtraction]]] = defaultdict(list)
    invalid_pdf_codes: set[str] = set()
    for pdf_path in sorted(Path(pdf_directory).glob("*.pdf")):
        try:
            extraction = extract_rome_pdf(pdf_path)
        except Exception as error:
            diagnostics.append(RomeDiagnostic("erreur", "pdf_illisible", str(error), source=pdf_path.name))
            continue
        filename_code = normalize_rome_code(pdf_path.name[:5])
        if extraction.code is None:
            diagnostics.append(RomeDiagnostic("erreur", "code_pdf_absent", "Code ROME introuvable dans l'en-tête du PDF.", source=pdf_path.name))
            continue
        if filename_code and filename_code != extraction.code:
            diagnostics.append(RomeDiagnostic("erreur", "code_pdf_incoherent", "Le code du nom de fichier diffère du code de l'en-tête PDF.", extraction.code, pdf_path.name))
            invalid_pdf_codes.add(extraction.code)
        pdf_candidates[extraction.code].append((pdf_path, extraction))
    pdfs: dict[str, tuple[Path, RomePdfExtraction]] = {}
    for code, candidates in pdf_candidates.items():
        if len(candidates) == 1:
            pdfs[code] = candidates[0]
            continue
        descriptive = [
            candidate
            for candidate in candidates
            if _ROME_DESCRIPTIVE_FILENAME.fullmatch(candidate[0].name)
        ]
        if len(descriptive) == 1:
            pdfs[code] = descriptive[0]
            diagnostics.append(
                RomeDiagnostic(
                    "avertissement",
                    "pdf_duplique_nom_descriptif_retenu",
                    "Doublon PDF : le fichier nommé 'CODE - intitulé.pdf' est retenu.",
                    code,
                    descriptive[0][0].name,
                )
            )
            continue
        diagnostics.append(
            RomeDiagnostic(
                "erreur",
                "pdf_duplique",
                "Plusieurs PDF portent le même code ROME sans unique fichier 'CODE - intitulé.pdf'.",
                code,
                ", ".join(candidate[0].name for candidate in candidates),
            )
        )
        # La campagne reste bloquée par le diagnostic ; garder le premier choix
        # permet néanmoins d'écrire un rapport complet pour investigation.
        pdfs[code] = candidates[0]
    rows: list[dict[str, object]] = []
    for code in sorted(relations.codes):
        excel_title = relations.titles_by_code[code][0]
        if code not in pdfs:
            diagnostics.append(RomeDiagnostic("erreur", "pdf_manquant", "Aucun PDF dont l'en-tête porte ce code ROME.", code))
            rows.append(_corpus_row(code, excel_title, (), "pdf_manquant", None))
            continue
        pdf_path, extraction = pdfs[code]
        status = "code_pdf_incoherent" if code in invalid_pdf_codes else "valide"
        if not extraction.competences:
            status = "competences_absentes"
            diagnostics.append(RomeDiagnostic("erreur", "competences_absentes", "Rubrique Compétences absente ou vide.", code, pdf_path.name))
        if extraction.title and _normalize_title(extraction.title) != _normalize_title(excel_title):
            diagnostics.append(RomeDiagnostic("avertissement", "intitule_pdf_different", "L'intitulé du PDF diffère de l'intitulé Excel; le code ROME reste la référence.", code, pdf_path.name))
        rows.append(_corpus_row(code, excel_title, extraction.competences, status, pdf_path.name))
    for code, (path, _) in pdfs.items():
        if code not in relations.codes:
            diagnostics.append(RomeDiagnostic("avertissement", "pdf_code_non_reference", "PDF présent mais code absent de l'Excel.", code, path.name))
    corpus = pd.DataFrame(rows, columns=("code_rome", "intitule", "competences", "nombre_competences", "statut_validation_pdf", "fichier_pdf"))
    return RomeValidationResult(relations, corpus, tuple(diagnostics))


def write_rome_outputs(result: RomeValidationResult, output_directory: str | Path = "outputs/rome") -> tuple[Path, Path]:
    """Écrit le corpus et le rapport de validation CSV."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    corpus_path, report_path = directory / "corpus_rome.csv", directory / "rapport_validation_rome.csv"
    result.corpus.to_csv(corpus_path, index=False, encoding="utf-8-sig")
    result.report_dataframe().to_csv(report_path, index=False, encoding="utf-8-sig")
    return corpus_path, report_path


class RomeEmbeddingCache:
    """Cache déterministe: une future campagne encode le corpus une seule fois par modèle."""
    def __init__(self, corpus: pd.DataFrame, cache_directory: str | Path = "outputs/rome/embeddings") -> None:
        self.corpus, self.cache_directory = corpus, Path(cache_directory)

    def path_for_model(self, model_name: str) -> Path:
        signature = hashlib.sha256(self.corpus[["code_rome", "competences"]].to_csv(index=False).encode()).hexdigest()[:16]
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
        return self.cache_directory / f"{name}-{signature}.npz"

    def load(self, model_name: str) -> np.ndarray | None:
        path = self.path_for_model(model_name)
        if not path.exists(): return None
        with np.load(path) as archive: vectors = archive["vectors"]
        return vectors if len(vectors) == len(self.corpus) else None

    def save(self, model_name: str, vectors: np.ndarray) -> Path:
        array = np.asarray(vectors, dtype=float)
        if array.ndim != 2 or len(array) != len(self.corpus):
            raise ValueError("Un vecteur d'embedding est requis pour chaque emploi du corpus.")
        path = self.path_for_model(model_name); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=array, codes=self.corpus["code_rome"].to_numpy())
        return path


def evaluate_rome_rankings(relations: RomeRelations, rankings: dict[str, Sequence[str]], k: int = 5) -> dict[str, float]:
    """Calcule Recall@k et MRR contre les voisins ROME non directionnels."""
    recalls, ranks = [], []
    for code in relations.codes:
        expected = relations.neighbors(code)
        if not expected or code not in rankings: continue
        ranked = [normalize_rome_code(item) for item in rankings[code][:k]]
        recalls.append(len(set(ranked) & expected) / len(expected))
        ranks.append(next((1 / (index + 1) for index, item in enumerate(ranked) if item in expected), 0.0))
    return {"queries_evaluees": float(len(recalls)), f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0, "mrr": float(np.mean(ranks)) if ranks else 0.0}


def _corpus_row(code: str, title: str, competences: Sequence[str], status: str, name: str | None) -> dict[str, object]:
    return {"code_rome": code, "intitule": title, "competences": "\n".join(competences), "nombre_competences": len(competences), "statut_validation_pdf": status, "fichier_pdf": name}

def _find_pdf_title(lines: Sequence[str]) -> str | None:
    for line in lines:
        if not line or _normalize_title(line) in {"definition", "competences", "acces a l emploi"}: break
        if not line.lower().startswith(("fiche emploi", "emploi transition", "cadre numérique")): return line
    return None

def _looks_like_heading(line: str) -> bool:
    return (len(line) < 70 and "," in line) or line in {"Savoir-faire", "Savoir-faire principaux", "Savoir-faire secondaires", "Savoir-être professionnels", "Savoirs", "Domaines d'expertise", "Normes et procédés", "Techniques professionnelles"}

def _clean_competence(text: str) -> str: return _TRANSITION_SUFFIX.sub("", _clean_text(text)).strip()
def _clean_text(value: object) -> str: return re.sub(r"\s+", " ", str(value)).strip()
def _normalize_title(value: object) -> str:
    text = unicodedata.normalize("NFKD", _clean_text(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
