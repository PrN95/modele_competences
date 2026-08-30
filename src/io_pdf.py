"""Extraction des deux formats de PDF natifs décrits dans le cadrage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Literal, Protocol, Sequence
import unicodedata

import pdfplumber

from src.config import COMPETENCE_LEVELS_BY_LABEL
from src.domain import Competence, Emploi


PDFDocumentType = Literal["emploi_actuel", "metier_cible"]

_CURRENT_SECTION = "COMPETENCES DIGITALES"
_TARGET_HEADERS = (
    "INTITULE DE LA COMPETENCE",
    "DETAILS DE LA COMPETENCE",
    "NIVEAU",
)
_TOP_LEVEL_SECTION = re.compile(r"^COMPETENCES\s+.+(?:\s+X)?$")
_CURRENT_HEADER_PREFIXES = ("TALENTSOFT ", "HTTP://", "HTTPS://")
_PAGE_FOOTER = re.compile(r"^\d+\s+SUR\s+\d+(?:\s|$)")
_MIN_EXPLOITABLE_CHARACTERS = 20


@dataclass(frozen=True, slots=True)
class PDFExtractionDiagnostic:
    """Diagnostic précis associé au rejet d'un PDF."""

    fichier: str
    code: str
    message: str
    page: int | None = None

    def __str__(self) -> str:
        emplacement = f", page={self.page}" if self.page is not None else ""
        return f"fichier='{self.fichier}'{emplacement}, code='{self.code}': {self.message}"


class PDFExtractionError(ValueError):
    """Erreur d'extraction regroupant un ou plusieurs diagnostics."""

    def __init__(
        self,
        diagnostics: PDFExtractionDiagnostic
        | Sequence[PDFExtractionDiagnostic],
    ) -> None:
        if isinstance(diagnostics, PDFExtractionDiagnostic):
            self.diagnostics = (diagnostics,)
        else:
            self.diagnostics = tuple(diagnostics)
        super().__init__(
            "PDF rejeté :\n" + "\n".join(f"- {diagnostic}" for diagnostic in self.diagnostics)
        )


class _PDFPage(Protocol):
    page_number: int
    width: float
    height: float

    def extract_text(self, **kwargs: Any) -> str | None: ...

    def extract_text_lines(self, **kwargs: Any) -> list[dict[str, Any]]: ...

    def find_tables(self, table_settings: dict[str, Any] | None = None) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class _VisualLine:
    text: str
    page_number: int
    top: float
    x0: float
    x1: float
    chars: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class _CurrentCompetenceBuilder:
    title_parts: list[str]
    niveau: int
    page_number: int
    description_parts: list[str] = field(default_factory=list)

    def build(self, source: Path) -> Competence:
        title = _join_fragments(self.title_parts)
        description = _join_fragments(self.description_parts) or None
        if not title:
            _reject(
                source,
                "association_ambigue",
                "un niveau a été trouvé sans intitulé de compétence associé",
                self.page_number,
            )
        return Competence(intitule=title, description=description, niveau=self.niveau)


def read_emploi_pdf(
    path: str | Path,
    *,
    document_type: PDFDocumentType,
) -> Emploi:
    """Sélectionne explicitement l'extracteur demandé, sans détection automatique."""

    if document_type == "emploi_actuel":
        return extract_emploi_actuel_pdf(path)
    if document_type == "metier_cible":
        return extract_metier_cible_pdf(path)
    raise ValueError(
        "document_type doit valoir exactement 'emploi_actuel' ou 'metier_cible'; "
        "le format n'est jamais deviné automatiquement"
    )


def extract_emploi_actuel_pdf(path: str | Path) -> Emploi:
    """Extrait l'en-tête et la seule section « COMPÉTENCES DIGITALES »."""

    source = _validate_pdf_path(path)
    with _open_pdf(source) as pdf:
        pages = tuple(pdf.pages)
        _ensure_native_text(pages, source)
        lines_by_page = tuple(_extract_visual_lines(page) for page in pages)
        title = _extract_current_title(lines_by_page[0], pages[0], source)
        section_position = _find_current_section(lines_by_page, pages)
        if section_position is None:
            _reject(
                source,
                "section_absente",
                "la section attendue « COMPÉTENCES DIGITALES » est absente",
            )

        competences = _parse_current_competences(
            lines_by_page,
            pages,
            section_position,
            source,
        )

    if not competences:
        _reject(
            source,
            "structure_inexploitable",
            "la section « COMPÉTENCES DIGITALES » ne contient aucune compétence exploitable",
        )
    return Emploi(
        intitule=title,
        type="actuel",
        effectif=None,
        competences=tuple(competences),
        fichier_source=source.name,
        feuille_source=None,
    )


def extract_metier_cible_pdf(path: str | Path) -> Emploi:
    """Extrait l'intitulé et les trois colonnes du tableau d'un métier cible."""

    source = _validate_pdf_path(path)
    with _open_pdf(source) as pdf:
        pages = tuple(pdf.pages)
        _ensure_native_text(pages, source)
        title: str | None = None
        competences: list[Competence] = []
        header_found = False

        for page in pages:
            tables = page.find_tables() or []
            continuation_candidates: list[tuple[Any, list[list[str | None]]]] = []
            header_candidates: list[tuple[Any, list[list[str | None]]]] = []
            for table in tables:
                rows = table.extract(x_tolerance=2, y_tolerance=3) or []
                if not rows:
                    continue
                if _is_target_header(rows[0]):
                    header_candidates.append((table, rows))
                elif header_found and _is_three_column_table(rows):
                    continuation_candidates.append((table, rows))

            if len(header_candidates) > 1:
                _reject(
                    source,
                    "structure_inexploitable",
                    "plusieurs tableaux portent l'en-tête cible attendu sur la même page",
                    page.page_number,
                )

            if header_candidates:
                if header_found:
                    selected_table, selected_rows = header_candidates[0]
                    rows_to_parse = selected_rows[1:]
                else:
                    selected_table, selected_rows = header_candidates[0]
                    header_found = True
                    title = _extract_target_title(page, selected_table, source)
                    rows_to_parse = selected_rows[1:]
                competences.extend(_parse_target_rows(rows_to_parse, source, page.page_number))
                continue

            if continuation_candidates:
                if len(continuation_candidates) > 1:
                    _reject(
                        source,
                        "structure_inexploitable",
                        "plusieurs tableaux peuvent être la continuation du tableau cible",
                        page.page_number,
                    )
                _, rows_to_parse = continuation_candidates[0]
                competences.extend(_parse_target_rows(rows_to_parse, source, page.page_number))

    if not header_found:
        _reject(
            source,
            "tableau_introuvable",
            "aucun tableau ne contient les colonnes « Intitulé de la compétence », "
            "« Détails de la compétence » et « Niveau »",
        )
    if not competences:
        _reject(
            source,
            "structure_inexploitable",
            "le tableau cible ne contient aucune ligne de compétence exploitable",
        )
    if title is None:
        _reject(
            source,
            "structure_inexploitable",
            "l'intitulé du métier cible placé au-dessus du tableau est introuvable",
        )
    return Emploi(
        intitule=title,
        type="cible",
        effectif=None,
        competences=tuple(competences),
        fichier_source=source.name,
        feuille_source=None,
    )


def _validate_pdf_path(path: str | Path) -> Path:
    source = Path(path)
    if source.suffix.lower() != ".pdf":
        _reject(source, "format_invalide", "le format obligatoire est PDF")
    return source


def _open_pdf(source: Path) -> Any:
    try:
        return pdfplumber.open(source, unicode_norm="NFC")
    except (OSError, ValueError) as error:
        _reject(source, "pdf_illisible", f"le fichier PDF est illisible: {error}")


def _ensure_native_text(pages: Sequence[_PDFPage], source: Path) -> None:
    if not pages:
        _reject(source, "structure_inexploitable", "le PDF ne contient aucune page")
    character_count = 0
    for page in pages:
        try:
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
        except (TypeError, ValueError) as error:
            _reject(
                source,
                "structure_inexploitable",
                f"le texte de la page ne peut pas être analysé: {error}",
                page.page_number,
            )
        character_count += sum(character.isalnum() for character in text)
    if character_count < _MIN_EXPLOITABLE_CHARACTERS:
        _reject(
            source,
            "pdf_sans_texte",
            "PDF scanné ou sans couche de texte native exploitable; aucun OCR n'est appliqué",
        )


def _extract_visual_lines(page: _PDFPage) -> tuple[_VisualLine, ...]:
    raw_lines = page.extract_text_lines(
        layout=False,
        strip=True,
        return_chars=True,
        x_tolerance=2,
        y_tolerance=3,
    )
    return tuple(
        _VisualLine(
            text=_clean_text(line.get("text")),
            page_number=page.page_number,
            top=float(line.get("top", 0.0)),
            x0=float(line.get("x0", 0.0)),
            x1=float(line.get("x1", 0.0)),
            chars=tuple(line.get("chars") or ()),
        )
        for line in raw_lines
        if _clean_text(line.get("text"))
    )


def _extract_current_title(
    lines: Sequence[_VisualLine],
    page: _PDFPage,
    source: Path,
) -> str:
    candidates: list[tuple[float, str]] = []
    for line in lines:
        if line.top > page.height * 0.25 or line.x0 > page.width * 0.65:
            continue
        black_bold_chars = _select_chars(line.chars, black=True, bold=True)
        if not black_bold_chars:
            continue
        max_size = max(float(character.get("size", 0.0)) for character in black_bold_chars)
        normalized = _normalize(line.text)
        if max_size >= 9 and not normalized.startswith(("RUBRIQUES", "TALENTSOFT")):
            candidates.append((max_size, line.text))
    if not candidates:
        _reject(
            source,
            "structure_inexploitable",
            "l'intitulé de l'emploi actuel est introuvable dans l'en-tête",
            1,
        )
    largest_size = max(size for size, _ in candidates)
    largest_titles = {text for size, text in candidates if abs(size - largest_size) < 0.1}
    if len(largest_titles) != 1:
        _reject(
            source,
            "association_ambigue",
            "plusieurs intitulés d'emploi sont possibles dans l'en-tête",
            1,
        )
    return largest_titles.pop()


def _find_current_section(
    lines_by_page: Sequence[Sequence[_VisualLine]],
    pages: Sequence[_PDFPage],
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for page_index, lines in enumerate(lines_by_page):
        for line_index, line in enumerate(lines):
            normalized = _normalize(line.text).removesuffix(" X")
            if (
                normalized == _CURRENT_SECTION
                and line.x0 <= pages[page_index].width * 0.25
                and _select_chars(line.chars, black=True, bold=True)
            ):
                candidates.append((page_index, line_index))
    return candidates[-1] if candidates else None


def _parse_current_competences(
    lines_by_page: Sequence[Sequence[_VisualLine]],
    pages: Sequence[_PDFPage],
    section_position: tuple[int, int],
    source: Path,
) -> list[Competence]:
    start_page, start_line = section_position
    builders: list[_CurrentCompetenceBuilder] = []
    current: _CurrentCompetenceBuilder | None = None

    for page_index in range(start_page, len(lines_by_page)):
        lines = lines_by_page[page_index]
        initial_index = start_line + 1 if page_index == start_page else 0
        for line in lines[initial_index:]:
            normalized = _normalize(line.text)
            if _is_repeated_page_chrome(normalized):
                continue
            if (
                normalized != _CURRENT_SECTION
                and _TOP_LEVEL_SECTION.fullmatch(normalized)
                and _select_chars(line.chars, black=True, bold=True)
                and line.x0 <= pages[page_index].width * 0.25
            ):
                return _finalize_builders(builders, current, source)

            black_bold_text = _chars_text(_select_chars(line.chars, black=True, bold=True))
            right_level_text = _chars_text(
                tuple(
                    character
                    for character in line.chars
                    if not _is_bold(character.get("fontname"))
                    and float(character.get("x0", 0.0)) >= pages[page_index].width * 0.72
                )
            )

            if _is_blue_category(line):
                continue

            level = _parse_level(right_level_text)
            if level is not None:
                if not black_bold_text:
                    _reject(
                        source,
                        "association_ambigue",
                        "un niveau reconnu n'est associé à aucun intitulé noir en gras",
                        line.page_number,
                    )
                if current is not None:
                    builders.append(current)
                current = _CurrentCompetenceBuilder(
                    title_parts=[black_bold_text],
                    niveau=level,
                    page_number=line.page_number,
                )
                continue

            if right_level_text and black_bold_text:
                _reject(
                    source,
                    "niveau_inconnu",
                    f"niveau non reconnu « {right_level_text} » "
                    f"pour la compétence « {black_bold_text} »",
                    line.page_number,
                )

            if black_bold_text:
                if current is None or current.description_parts:
                    _reject(
                        source,
                        "association_ambigue",
                        "intitulé en gras sans niveau reconnu ni association certaine: "
                        f"« {black_bold_text} »",
                        line.page_number,
                    )
                current.title_parts.append(black_bold_text)
                continue

            black_text = _chars_text(_select_chars(line.chars, black=True, bold=False))
            if black_text:
                if current is None:
                    _reject(
                        source,
                        "association_ambigue",
                        f"description sans compétence associée: « {black_text} »",
                        line.page_number,
                    )
                current.description_parts.append(black_text)

    return _finalize_builders(builders, current, source)


def _finalize_builders(
    builders: list[_CurrentCompetenceBuilder],
    current: _CurrentCompetenceBuilder | None,
    source: Path,
) -> list[Competence]:
    if current is not None:
        builders.append(current)
    return [builder.build(source) for builder in builders]


def _extract_target_title(page: _PDFPage, table: Any, source: Path) -> str:
    table_top = float(table.bbox[1])
    lines = _extract_visual_lines(page)
    candidates: list[str] = []
    for line in lines:
        if line.top >= table_top:
            continue
        match = re.match(r"^\s*Emploi\s+cible\s*:\s*(.+?)\s*$", line.text, flags=re.IGNORECASE)
        if match:
            candidates.append(_clean_text(match.group(1)))
    distinct = {candidate for candidate in candidates if candidate}
    if len(distinct) == 1:
        return distinct.pop()
    if len(distinct) > 1:
        _reject(
            source,
            "association_ambigue",
            "plusieurs intitulés de métier cible sont placés au-dessus du tableau",
            page.page_number,
        )
    _reject(
        source,
        "structure_inexploitable",
        "l'intitulé « Emploi cible : … » est absent au-dessus du tableau",
        page.page_number,
    )


def _parse_target_rows(
    rows: Sequence[Sequence[str | None]],
    source: Path,
    page_number: int,
) -> list[Competence]:
    competences: list[Competence] = []
    for row_number, row in enumerate(rows, start=1):
        if not any(_clean_text(cell) for cell in row):
            continue
        if len(row) != 3:
            _reject(
                source,
                "structure_inexploitable",
                f"ligne {row_number} du tableau: trois cellules sont attendues, "
                f"{len(row)} trouvées",
                page_number,
            )
        title, description, level_label = (_clean_text(cell) for cell in row)
        if not title:
            _reject(
                source,
                "association_ambigue",
                f"ligne {row_number} du tableau: détails ou niveau sans intitulé de compétence",
                page_number,
            )
        level = _parse_level(level_label)
        if level is None:
            shown_level = level_label or "valeur vide"
            _reject(
                source,
                "niveau_inconnu",
                f"ligne {row_number} du tableau, compétence « {title} »: "
                f"niveau non reconnu « {shown_level} »",
                page_number,
            )
        competences.append(
            Competence(
                intitule=title,
                description=description or None,
                niveau=level,
            )
        )
    return competences


def _is_target_header(row: Sequence[str | None]) -> bool:
    return len(row) == 3 and tuple(_normalize(cell) for cell in row) == _TARGET_HEADERS


def _is_three_column_table(rows: Sequence[Sequence[str | None]]) -> bool:
    return bool(rows) and all(len(row) == 3 for row in rows)


def _is_repeated_page_chrome(normalized_text: str) -> bool:
    return normalized_text.startswith(_CURRENT_HEADER_PREFIXES) or bool(
        _PAGE_FOOTER.match(normalized_text)
    )


def _is_blue_category(line: _VisualLine) -> bool:
    visible = tuple(
        character
        for character in line.chars
        if str(character.get("text", "")).strip()
    )
    return bool(visible) and all(
        not _is_black(character.get("non_stroking_color")) for character in visible
    )


def _select_chars(
    chars: Sequence[dict[str, Any]],
    *,
    black: bool,
    bold: bool,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        character
        for character in chars
        if _is_black(character.get("non_stroking_color")) is black
        and _is_bold(character.get("fontname")) is bold
    )


def _is_black(color: Any) -> bool:
    if color is None:
        return True
    if isinstance(color, (int, float)):
        return float(color) <= 0.05
    if isinstance(color, (tuple, list)) and color:
        numeric = [float(component) for component in color if isinstance(component, (int, float))]
        return bool(numeric) and max(numeric) <= 0.05
    return False


def _is_bold(fontname: Any) -> bool:
    return "BOLD" in str(fontname).upper()


def _chars_text(chars: Sequence[dict[str, Any]]) -> str:
    result = ""
    previous_x1: float | None = None
    for character in chars:
        text = str(character.get("text", ""))
        if not text:
            continue
        if text.isspace():
            if result and not result.endswith(" "):
                result += " "
            previous_x1 = float(character.get("x1", previous_x1 or 0.0))
            continue
        x0 = float(character.get("x0", 0.0))
        if previous_x1 is not None and x0 - previous_x1 > 0.8 and not result.endswith(" "):
            result += " "
        result += text
        previous_x1 = float(character.get("x1", x0))
    return _clean_text(result)


def _parse_level(value: str | None) -> int | None:
    return COMPETENCE_LEVELS_BY_LABEL.get(_normalize(value).lower())


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", _clean_text(value))
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    ).upper()


def _join_fragments(parts: Sequence[str]) -> str:
    result = ""
    for part in parts:
        cleaned = _clean_text(part)
        if not cleaned:
            continue
        if not result:
            result = cleaned
        elif result.endswith(("-", "/")):
            result += cleaned
        else:
            result += f" {cleaned}"
    return result


def _reject(
    source: Path,
    code: str,
    message: str,
    page: int | None = None,
) -> None:
    raise PDFExtractionError(
        PDFExtractionDiagnostic(
            fichier=source.name,
            code=code,
            message=message,
            page=page,
        )
    )
