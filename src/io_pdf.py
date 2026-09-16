"""Extraction des deux formats de PDF natifs décrits dans le cadrage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Literal, Protocol, Sequence
import unicodedata
import warnings

import pdfplumber

from src.config import COMPETENCE_LEVELS_BY_LABEL
from src.domain import Competence, Emploi


PDFDocumentType = Literal["emploi_actuel", "metier_cible"]

_CURRENT_SECTION = "COMPETENCES DIGITALES"
_TARGET_HEADER_ALIASES = {
    "intitule": (
        "Intitulé de la compétence",
        "Compétence digitale",
        "Compétence",
        "Intitulé compétence",
    ),
    "description": (
        "Détails de la compétence",
        "Détail de la compétence",
        "Détails compétence",
        "Détail compétence",
        "Description de la compétence",
    ),
    "niveau": (
        "Niveau",
        "Niveau de la compétence",
        "Niveau compétence",
    ),
}
_TOP_LEVEL_SECTION = re.compile(r"^COMPETENCES\s+.+(?:\s+X)?$")
_CURRENT_HEADER_PREFIXES = ("TALENTSOFT ", "HTTP://", "HTTPS://")
_PAGE_FOOTER = re.compile(r"^\d+\s+SUR\s+\d+(?:\s|$)")
_MIN_EXPLOITABLE_CHARACTERS = 20
_FILENAME_REFERENCE_SUFFIX = re.compile(r"\s*[-–—]\s*[A-Z]\d+\s*$", re.IGNORECASE)
_ROME_ALLOWED_SECTIONS = {
    "SAVOIR FAIRE PRINCIPAUX": "Savoir-faire principaux",
    "DOMAINES D EXPERTISE": "Domaines d’expertise",
}
_ROME_STOP_SECTIONS = {
    "SAVOIR FAIRE SECONDAIRES",
    "SAVOIRS",
    "CONTEXTES DE TRAVAIL",
    "SECTEURS D ACTIVITE",
}
_ROME_KNOWN_SECTION_HEADINGS = frozenset(
    set(_ROME_ALLOWED_SECTIONS) | _ROME_STOP_SECTIONS | {
        "COMPETENCES",
        "SAVOIR ETRE PROFESSIONNELS",
        "NORMES ET PROCEDES",
        "TECHNIQUES PROFESSIONNELLES",
    }
)
_ROME_CODE = re.compile(r"^[A-Z]\d{4}$")
_ROME_BULLET = re.compile(r"^[•●▪◦\-–—]\s*(.+)$")
_ROME_BADGE_SUFFIX = re.compile(
    r"\s*(?:[-–—]|\(|\[)?\s*Transition\s+(?:numérique|écologique)\s*(?:\)|\])?\s*$",
    re.IGNORECASE,
)


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


class PDFExtractionWarning(UserWarning):
    """Avertissement lorsqu'une structure PDF est interprétée par convention."""


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
    original_filename: str | None = None,
) -> Emploi:
    """Extrait le même profil PDF, puis applique le rôle choisi par l'utilisateur."""

    return extract_employment_profile_pdf(
        path,
        role=document_type,
        original_filename=original_filename,
    )


def extract_employment_profile_pdf(
    path: str | Path,
    *,
    role: PDFDocumentType,
    original_filename: str | None = None,
) -> Emploi:
    """Tente tableau puis Talentsoft, indépendamment du rôle dans le matching."""

    roles = {"emploi_actuel": "actuel", "metier_cible": "cible"}
    try:
        emploi_type = roles[role]
    except KeyError as error:
        raise ValueError(
            "role doit valoir exactement 'emploi_actuel' ou 'metier_cible'; "
            "le format n'est jamais deviné automatiquement"
        ) from error
    return _extract_profile_pdf_with_strategies(
        path,
        original_filename=original_filename,
        emploi_type=emploi_type,
    )


def extract_emploi_actuel_pdf(
    path: str | Path,
    *,
    original_filename: str | None = None,
) -> Emploi:
    """Compatibilité : extrait un profil générique avec le rôle actuel."""

    return extract_employment_profile_pdf(
        path, role="emploi_actuel", original_filename=original_filename
    )


def extract_metier_cible_pdf(
    path: str | Path,
    *,
    original_filename: str | None = None,
) -> Emploi:
    """Compatibilité : extrait un profil générique avec le rôle cible."""

    return extract_employment_profile_pdf(
        path, role="metier_cible", original_filename=original_filename
    )


def _extract_profile_pdf_with_strategies(
    path: str | Path,
    *,
    original_filename: str | None,
    emploi_type: Literal["actuel", "cible"],
) -> Emploi:
    """Implémente les deux stratégies d'extraction dans un ordre déterministe."""

    source = _validate_pdf_path(path)
    with _open_pdf(source) as pdf:
        pages = tuple(pdf.pages)
        _ensure_native_text(pages, source)
        title: str | None = None
        competences: list[Competence] = []
        header_found = False
        column_mapping: tuple[int, int, int] | None = None
        detected_headers: list[str] = []

        for page in pages:
            tables = page.find_tables() or []
            continuation_candidates: list[tuple[Any, list[list[str | None]]]] = []
            header_candidates: list[tuple[Any, list[list[str | None]], tuple[int, int, int]]] = []
            positional_candidates: list[tuple[Any, list[list[str | None]]]] = []
            for table in tables:
                rows = table.extract(x_tolerance=2, y_tolerance=3) or []
                if not rows:
                    continue
                detected_headers.append(_format_detected_header(page.page_number, rows[0]))
                mapping = _target_header_mapping(rows[0])
                if mapping is not None:
                    header_candidates.append((table, rows, mapping))
                elif not header_found and _is_positional_target_candidate(rows):
                    positional_candidates.append((table, rows))
                elif header_found and _is_three_column_table(rows):
                    continuation_candidates.append((table, rows))

            if len(header_candidates) > 1:
                _reject(
                    source,
                    "structure_inexploitable",
                    "plusieurs tableaux correspondent aux alias des colonnes de compétences ; "
                    f"{_target_header_help(detected_headers)}",
                    page.page_number,
                )

            if header_candidates:
                if header_found:
                    _, selected_rows, selected_mapping = header_candidates[0]
                    rows_to_parse = selected_rows[1:]
                else:
                    selected_table, selected_rows, selected_mapping = header_candidates[0]
                    header_found = True
                    column_mapping = selected_mapping
                    title = _extract_target_title(page, selected_table, source)
                    rows_to_parse = selected_rows[1:]
                competences.extend(
                    _parse_target_rows(
                        rows_to_parse,
                        source,
                        page.page_number,
                        column_mapping=selected_mapping,
                    )
                )
                continue

            if positional_candidates:
                if len(positional_candidates) > 1:
                    _reject(
                        source,
                        "structure_inexploitable",
                        "plusieurs tableaux à trois colonnes pourraient être interprétés par position ; "
                        f"{_target_header_help(detected_headers)}",
                        page.page_number,
                    )
                selected_table, selected_rows = positional_candidates[0]
                header_found = True
                column_mapping = (0, 1, 2)
                title = _extract_target_title(page, selected_table, source)
                detected = _format_header_row(selected_rows[0])
                warnings.warn(
                    "colonnes_interpretees_par_position : "
                    f"en-têtes détectés [{detected}] ; mapping utilisé "
                    "colonne 1=nom de compétence, colonne 2=détail, colonne 3=niveau.",
                    PDFExtractionWarning,
                    stacklevel=2,
                )
                competences.extend(
                    _parse_target_rows(
                        selected_rows[1:],
                        source,
                        page.page_number,
                        column_mapping=column_mapping,
                    )
                )
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
                competences.extend(
                    _parse_target_rows(
                        rows_to_parse,
                        source,
                        page.page_number,
                        column_mapping=column_mapping or (0, 1, 2),
                    )
                )

    if not header_found:
        try:
            return _extract_talentsoft_from_pages(
                pages,
                source,
                original_filename=original_filename,
                emploi_type=emploi_type,
                table_diagnostics=_target_header_help(detected_headers),
            )
        except PDFExtractionError as talentsoft_error:
            if not any(
                diagnostic.code in {"section_absente", "tableau_introuvable"}
                for diagnostic in talentsoft_error.diagnostics
            ):
                raise
            return _extract_rome_from_pages(
                pages,
                source,
                original_filename=original_filename,
                emploi_type=emploi_type,
                table_diagnostics=_target_header_help(detected_headers),
                talentsoft_diagnostics=talentsoft_error.diagnostics,
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
        type=emploi_type,
        effectif=None,
        competences=tuple(competences),
        fichier_source=original_filename or source.name,
        feuille_source=None,
        format_extraction="tableau",
    )


def _extract_talentsoft_from_pages(
    pages: Sequence[_PDFPage],
    source: Path,
    *,
    original_filename: str | None,
    emploi_type: Literal["actuel", "cible"],
    table_diagnostics: str | None = None,
) -> Emploi:
    """Extrait une fiche Talentsoft, quel que soit son rôle dans le matching."""

    lines_by_page = tuple(_extract_visual_lines(page) for page in pages)
    section_position = _find_current_section(lines_by_page, pages)
    if section_position is None:
        _reject(
            source,
            "section_absente" if emploi_type == "actuel" else "tableau_introuvable",
            (
                "la section attendue « COMPÉTENCES DIGITALES » est absente"
                if emploi_type == "actuel"
                else "aucun tableau de compétences n'a été identifié et la rubrique Talentsoft "
                "« COMPÉTENCES DIGITALES » est absente"
                + (f" ; {table_diagnostics}" if table_diagnostics else "")
            ),
        )
    # Les fiches Talentsoft peuvent fractionner ou répéter visuellement leur
    # en-tête. Dans l'interface, le nom original est fourni par l'utilisateur
    # et constitue donc la source de vérité explicite pour l'intitulé métier.
    title = _title_from_original_filename(original_filename) or _extract_current_title(
        lines_by_page[0],
        pages[0],
        source,
        original_filename=original_filename,
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
            "la rubrique « COMPÉTENCES DIGITALES » ne contient aucune compétence exploitable",
        )
    return Emploi(
        intitule=title,
        type=emploi_type,
        effectif=None,
        competences=tuple(competences),
        fichier_source=original_filename or source.name,
        feuille_source=None,
        format_extraction="talentsoft",
    )


def _extract_rome_from_pages(
    pages: Sequence[_PDFPage],
    source: Path,
    *,
    original_filename: str | None,
    emploi_type: Literal["actuel", "cible"],
    table_diagnostics: str | None,
    talentsoft_diagnostics: Sequence[PDFExtractionDiagnostic],
) -> Emploi:
    """Extrait les puces des deux rubriques ROME autorisées.

    Seuls « Savoir-faire principaux » et « Domaines d'expertise » sont
    retenus. Les badges Transition numérique/écologique sont retirés du
    libellé, sans influer sur la sélection. Une nouvelle rubrique arrête la
    collecte en cours afin qu'aucun texte extérieur ne soit rattaché à la
    dernière puce valide.
    """

    visual_lines = [
        (_clean_text(line["text"]), _line_is_bold(line))
        for page in pages
        for line in page.extract_text_lines(x_tolerance=2, y_tolerance=3)
        if _clean_text(line.get("text"))
    ]
    text_lines = [text for text, _ in visual_lines]
    active_section: str | None = None
    competences: list[Competence] = []
    saw_rome_structure = False
    current_bullet: list[str] | None = None

    def finalize_bullet() -> None:
        nonlocal current_bullet
        if current_bullet is None or active_section is None:
            current_bullet = None
            return
        raw = _join_fragments(current_bullet)
        label = _ROME_BADGE_SUFFIX.sub("", raw).strip(" -–—")
        if label:
            competences.append(
                Competence(
                    intitule=label,
                    description=None,
                    niveau=None,
                    source_section=active_section,
                )
            )
        current_bullet = None

    for text, is_bold in visual_lines:
        normalized = _normalize_rome_heading(text)
        section = _ROME_ALLOWED_SECTIONS.get(normalized)
        if section is not None:
            finalize_bullet()
            active_section = section
            saw_rome_structure = True
            continue
        if normalized in _ROME_STOP_SECTIONS:
            finalize_bullet()
            active_section = None
            saw_rome_structure = True
            continue
        if normalized in _ROME_KNOWN_SECTION_HEADINGS or (
            active_section is not None and is_bold and not _ROME_BULLET.match(text)
        ):
            finalize_bullet()
            active_section = None
            saw_rome_structure = True
            continue
        bullet = _ROME_BULLET.match(text)
        if bullet:
            finalize_bullet()
            current_bullet = [bullet.group(1)] if active_section is not None else None
        elif current_bullet is not None:
            current_bullet.append(text)
    finalize_bullet()

    if not saw_rome_structure or not competences:
        prior = talentsoft_diagnostics[0].message if talentsoft_diagnostics else "échec inconnu"
        _reject(
            source,
            "section_absente" if emploi_type == "actuel" else "tableau_introuvable",
            "formats tentés : tableau, Talentsoft, ROME ; aucun savoir-faire ou domaine "
            "d'expertise exploitable n'a été trouvé"
            + (f" ; tableau : {table_diagnostics}" if table_diagnostics else "")
            + f" ; Talentsoft : {prior}",
        )

    title = _extract_rome_title(text_lines, original_filename, source)
    return Emploi(
        intitule=title,
        type=emploi_type,
        effectif=None,
        competences=tuple(competences),
        fichier_source=original_filename or source.name,
        feuille_source=None,
        format_extraction="rome",
    )


def _extract_rome_title(
    lines: Sequence[str], original_filename: str | None, source: Path
) -> str:
    """Privilégie le titre suivant le code ROME, puis le nom déposé en repli."""

    for index, line in enumerate(lines[:-1]):
        if _ROME_CODE.fullmatch(_normalize(line).replace(" ", "")):
            candidate = _clean_text(lines[index + 1])
            if candidate and _normalize(candidate) not in _ROME_ALLOWED_SECTIONS:
                return candidate
    if original_filename:
        reference = _title_reference_from_filename(original_filename)
        if reference:
            return Path(original_filename).stem
    _reject(
        source,
        "structure_inexploitable",
        "l'intitulé de la fiche ROME est introuvable dans l'en-tête et le nom de fichier",
        1,
    )


def _line_is_bold(line: dict[str, Any]) -> bool:
    """Détecte une rubrique ROME par sa mise en forme, sans la confondre avec une puce."""

    chars = line.get("chars") or ()
    return bool(chars) and all(_is_bold(str(char.get("fontname", ""))) for char in chars)


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
    *,
    original_filename: str | None = None,
) -> str:
    candidates: list[tuple[float, _VisualLine]] = []
    for line in lines:
        if line.top > page.height * 0.25 or line.x0 > page.width * 0.65:
            continue
        black_bold_chars = _select_chars(line.chars, black=True, bold=True)
        if not black_bold_chars:
            continue
        max_size = max(float(character.get("size", 0.0)) for character in black_bold_chars)
        normalized = _normalize(line.text)
        if max_size >= 9 and not normalized.startswith(("RUBRIQUES", "TALENTSOFT")):
            candidates.append((max_size, line))
    if not candidates:
        _reject(
            source,
            "structure_inexploitable",
            "l'intitulé de l'emploi actuel est introuvable dans l'en-tête",
            1,
        )
    largest_size = max(size for size, _ in candidates)
    largest_lines = sorted(
        (
            line
            for size, line in candidates
            if abs(size - largest_size) < 0.1
        ),
        key=lambda line: line.top,
    )
    largest_titles = {_clean_header_title(line.text) for line in largest_lines}
    # Certaines fiches Talentsoft coupent un même intitulé sur plusieurs lignes
    # visuellement homogènes. Les lignes individuelles restent aussi candidates :
    # le nom de fichier doit toujours pouvoir départager une vraie ambiguïté.
    grouped: list[_VisualLine] = []
    for line in largest_lines:
        if grouped and (
            line.top - grouped[-1].top > 30
            or abs(line.x0 - grouped[-1].x0) > 4
        ):
            if len(grouped) > 1:
                largest_titles.add(_clean_header_title(_join_fragments([item.text for item in grouped])))
            grouped = []
        grouped.append(line)
    if len(grouped) > 1:
        largest_titles.add(_clean_header_title(_join_fragments([item.text for item in grouped])))
    if len(largest_titles) != 1:
        filename = original_filename or source.name
        resolved, reason = _resolve_title_from_filename(largest_titles, filename)
        if resolved is not None:
            warnings.warn(
                "intitule_resolu_par_nom_fichier : "
                f"nom original « {filename} » ; candidat retenu « {resolved} ».",
                PDFExtractionWarning,
                stacklevel=2,
            )
            return resolved
        _reject(
            source,
            "association_ambigue",
            "plusieurs intitulés d'emploi sont possibles dans l'en-tête ; "
            f"nom original « {filename} » ; intitulés candidats : "
            f"{', '.join(f'« {candidate} »' for candidate in sorted(largest_titles))}. "
            f"{reason}",
            1,
        )
    return largest_titles.pop()


def _resolve_title_from_filename(
    candidates: Sequence[str],
    original_filename: str,
) -> tuple[str | None, str]:
    """Résout uniquement une égalité normalisée univoque avec le nom du fichier."""

    reference = _title_reference_from_filename(original_filename)
    if not reference:
        return None, "le nom de fichier ne fournit pas d'intitulé exploitable."
    matches = [
        candidate
        for candidate in candidates
        if reference in _title_comparison_keys(candidate)
    ]
    if len(matches) == 1:
        return matches[0], "correspondance univoque trouvée."
    if not matches:
        return None, (
            f"aucun candidat ne correspond exactement à l'intitulé normalisé « {reference} » "
            "issu du nom de fichier."
        )
    return None, "plusieurs candidats correspondent au même intitulé normalisé du nom de fichier."


def _title_reference_from_filename(original_filename: str) -> str:
    name = Path(original_filename).name
    stem = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    stem = _FILENAME_REFERENCE_SUFFIX.sub("", stem)
    return _normalize_title_for_comparison(stem)


def _title_from_original_filename(original_filename: str | None) -> str | None:
    """Retourne le libellé métier explicitement fourni par l'import Streamlit."""

    if not original_filename:
        return None
    title = _clean_text(Path(original_filename).stem)
    return title or None


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
    *,
    column_mapping: tuple[int, int, int] = (0, 1, 2),
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
        title_index, description_index, level_index = column_mapping
        title = _clean_text(row[title_index])
        description = _clean_text(row[description_index])
        level_label = _clean_text(row[level_index])
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


def _target_header_mapping(row: Sequence[str | None]) -> tuple[int, int, int] | None:
    """Associe des en-têtes PDF aux rôles métier, quel que soit leur libellé."""

    if len(row) != 3:
        return None
    aliases = {
        role: {_normalize_header(alias) for alias in values}
        for role, values in _TARGET_HEADER_ALIASES.items()
    }
    roles_by_index: dict[int, str] = {}
    for index, value in enumerate(row):
        normalized = _normalize_header(value)
        matches = [role for role, values in aliases.items() if normalized in values]
        if len(matches) != 1:
            return None
        roles_by_index[index] = matches[0]
    if set(roles_by_index.values()) != set(aliases):
        return None
    return (
        next(index for index, role in roles_by_index.items() if role == "intitule"),
        next(index for index, role in roles_by_index.items() if role == "description"),
        next(index for index, role in roles_by_index.items() if role == "niveau"),
    )


def _is_positional_target_candidate(rows: Sequence[Sequence[str |None]]) -> bool:
    """Évite le repli par position pour un tableau non assimilable à des compétences."""

    if len(rows) < 2 or not _is_three_column_table(rows):
        return False
    header = rows[0]
    if not all(_clean_text(cell) for cell in header):
        return False
    data_rows = [row for row in rows[1:] if any(_clean_text(cell) for cell in row)]
    return bool(data_rows) and all(
        _clean_text(row[0]) and _parse_level(_clean_text(row[2])) is not None
        for row in data_rows
    )


def _format_header_row(row: Sequence[str | None]) -> str:
    return " | ".join(_clean_text(cell) or "∅" for cell in row)


def _format_detected_header(page_number: int, row: Sequence[str | None]) -> str:
    return f"page {page_number} : [{_format_header_row(row)}]"


def _target_header_help(detected_headers: Sequence[str]) -> str:
    detected = "; ".join(detected_headers) if detected_headers else "aucun tableau détecté"
    aliases = "; ".join(
        f"{role} = " + ", ".join(f"« {value} »" for value in values)
        for role, values in _TARGET_HEADER_ALIASES.items()
    )
    return f"en-têtes détectés : {detected}. Alias acceptés : {aliases}."


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


def _normalize_rome_heading(value: Any) -> str:
    """Normalise aussi les tirets et apostrophes des rubriques ROME."""

    return " ".join(re.findall(r"[A-Z0-9]+", _normalize(value)))


def _normalize_header(value: Any) -> str:
    """Normalise la casse, les accents, la ponctuation et les pluriels des titres."""

    tokens = re.findall(r"[A-Z0-9]+", _normalize(value))
    singulars = {"NIVEAUX": "NIVEAU"}
    return " ".join(
        singulars.get(token, token[:-1] if len(token) > 3 and token.endswith("S") else token)
        for token in tokens
    )


def _normalize_title_for_comparison(value: Any) -> str:
    """Clé explicable commune aux intitulés PDF et aux noms de fichiers."""

    return " ".join(re.findall(r"[A-Z0-9]+", _normalize(value)))


def _clean_header_title(value: Any) -> str:
    """Retire le marqueur graphique isolé vu après certains codes de fiche."""

    return re.sub(r"(\b[A-Z]\d{2,4})\s+v$", r"\1", _clean_text(value), flags=re.IGNORECASE)


def _title_comparison_keys(value: Any) -> frozenset[str]:
    """Compare un titre complet et sa variante privée du suffixe code de fiche."""

    title = _clean_header_title(value)
    without_code = _FILENAME_REFERENCE_SUFFIX.sub("", title)
    return frozenset(
        _normalize_title_for_comparison(item)
        for item in (title, without_code)
    )


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
