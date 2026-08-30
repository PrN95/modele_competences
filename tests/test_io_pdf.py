from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

import pytest

from src.domain import Competence
from src.io_pdf import (
    PDFExtractionError,
    extract_emploi_actuel_pdf,
    extract_metier_cible_pdf,
    read_emploi_pdf,
)


BLACK = (0.0, 0.0, 0.0)
BLUE = (0.12, 0.27, 0.57)


def _chars(
    text: str,
    *,
    x0: float,
    top: float,
    fontname: str,
    size: float,
    color: tuple[float, float, float],
) -> list[dict[str, Any]]:
    chars: list[dict[str, Any]] = []
    position = x0
    for character in text:
        width = size * (0.3 if character == " " else 0.5)
        chars.append(
            {
                "text": character,
                "x0": position,
                "x1": position + width,
                "top": top,
                "bottom": top + size,
                "fontname": fontname,
                "size": size,
                "non_stroking_color": color,
            }
        )
        position += width
    return chars


def _line(
    text: str,
    *,
    top: float,
    kind: str = "description",
    level: str | None = None,
) -> dict[str, Any]:
    if kind == "title":
        chars = _chars(
            text,
            x0=50,
            top=top,
            fontname="Test-Bold",
            size=14,
            color=BLACK,
        )
    elif kind == "section":
        chars = _chars(
            text,
            x0=50,
            top=top,
            fontname="Test-Bold",
            size=7,
            color=BLACK,
        )
    elif kind == "category":
        chars = _chars(
            text,
            x0=50,
            top=top,
            fontname="Test-Bold",
            size=7,
            color=BLUE,
        )
    elif kind == "competence":
        assert level is not None
        chars = _chars(
            text,
            x0=50,
            top=top,
            fontname="Test-Bold",
            size=7,
            color=BLACK,
        )
        chars.extend(
            _chars(
                level,
                x0=500,
                top=top,
                fontname="Test-Regular",
                size=7,
                color=BLUE,
            )
        )
        text = f"{text} {level}"
    else:
        chars = _chars(
            text,
            x0=50,
            top=top,
            fontname="Test-Regular",
            size=7,
            color=BLACK,
        )
    return {
        "text": text,
        "top": top,
        "x0": min(character["x0"] for character in chars),
        "x1": max(character["x1"] for character in chars),
        "chars": chars,
    }


@dataclass
class FakeTable:
    rows: list[list[str | None]]
    bbox: tuple[float, float, float, float] = (50.0, 100.0, 790.0, 500.0)

    def extract(self, **kwargs: Any) -> list[list[str | None]]:
        return self.rows


class FakePage:
    width = 600.0
    height = 800.0

    def __init__(
        self,
        page_number: int,
        lines: list[dict[str, Any]],
        tables: list[FakeTable] | None = None,
        *,
        extracted_text: str | None = None,
    ) -> None:
        self.page_number = page_number
        self._lines = lines
        self._tables = tables or []
        self._extracted_text = extracted_text

    def extract_text(self, **kwargs: Any) -> str:
        if self._extracted_text is not None:
            return self._extracted_text
        return "\n".join(line["text"] for line in self._lines)

    def extract_text_lines(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._lines

    def find_tables(self, table_settings: dict[str, Any] | None = None) -> list[FakeTable]:
        return self._tables


class FakePDF:
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages

    def __enter__(self) -> FakePDF:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


@pytest.fixture
def fake_pdf(monkeypatch):
    def _install(pages: list[FakePage]) -> None:
        monkeypatch.setattr("src.io_pdf.pdfplumber.open", lambda *args, **kwargs: FakePDF(pages))

    return _install


def _current_pages() -> list[FakePage]:
    return [
        FakePage(
            1,
            [
                _line("EMPLOI ACTUEL FICTIF", top=40, kind="title"),
                _line("RESPONSABILITÉS", top=100, kind="section"),
                _line("Contenu à exclure de l'extraction", top=120),
                _line("COMPÉTENCES DIGITALES X", top=200, kind="section"),
                _line("ACTIVITÉS DIGITALES / TEST", top=220, kind="category"),
                _line("Niveau SAME cible", top=230, kind="category"),
                _line("Compétence sur deux pages", top=250, kind="competence", level="Maîtrise"),
                _line("Description commencée", top=265),
            ],
        ),
        FakePage(
            2,
            [
                _line("Talentsoft https://exemple.invalid", top=1),
                _line("et terminée à la page suivante", top=40),
                _line(
                    "Compétence sans identifiant",
                    top=80,
                    kind="competence",
                    level="Application",
                ),
                _line("COMPÉTENCES FONCTIONNELLES X", top=140, kind="section"),
                _line("Compétence à exclure", top=160, kind="competence", level="Expertise"),
            ],
        ),
    ]


def _target_pages() -> list[FakePage]:
    headers = ["Intitulé de la compétence", "Détails de la compétence", "Niveau"]
    return [
        FakePage(
            1,
            [_line("Emploi cible : Métier cible fictif", top=50)],
            [
                FakeTable(
                    [
                        headers,
                        ["Compétence un", "Description\nsur plusieurs lignes", "Sensibilisation"],
                        ["Compétence deux", None, "Application"],
                    ]
                )
            ],
        ),
        FakePage(
            2,
            [_line("Suite du tableau cible fictif", top=20)],
            [
                FakeTable(
                    [
                        ["Compétence trois", "Description trois", "Maîtrise"],
                        ["Compétence quatre", "Description quatre", "Expertise"],
                    ],
                    bbox=(50.0, 50.0, 790.0, 120.0),
                )
            ],
        ),
    ]


def test_extracteur_emploi_actuel_lit_uniquement_la_section_digitale(fake_pdf) -> None:
    fake_pdf(_current_pages())

    emploi = extract_emploi_actuel_pdf("emploi_actuel.pdf")

    assert emploi.intitule == "EMPLOI ACTUEL FICTIF"
    assert emploi.type == "actuel"
    assert [competence.intitule for competence in emploi.competences] == [
        "Compétence sur deux pages",
        "Compétence sans identifiant",
    ]
    assert emploi.competences[0].description == (
        "Description commencée et terminée à la page suivante"
    )
    assert all("Niveau SAME cible" not in competence.intitule for competence in emploi.competences)


def test_extracteur_metier_cible_lit_un_tableau_multipage_sans_entete_repetee(fake_pdf) -> None:
    fake_pdf(_target_pages())

    emploi = extract_metier_cible_pdf("metier_cible.pdf")

    assert emploi.intitule == "Métier cible fictif"
    assert emploi.type == "cible"
    assert len(emploi.competences) == 4
    assert [competence.niveau for competence in emploi.competences] == [1, 2, 3, 4]
    assert emploi.competences[0].description == "Description sur plusieurs lignes"


def test_structure_normalisee_commune_et_texte_sans_niveau(fake_pdf) -> None:
    fake_pdf(_target_pages())

    emploi = read_emploi_pdf("cible.pdf", document_type="metier_cible")
    competence = emploi.competences[0]

    assert isinstance(competence, Competence)
    assert competence.competence_intitule == "Compétence un"
    assert competence.competence_description == "Description sur plusieurs lignes"
    assert competence.texte_competence == "Compétence un. Description sur plusieurs lignes"
    assert "Sensibilisation" not in competence.texte_competence
    assert competence.id is None
    assert emploi.id is None


def test_description_absente_utilise_seulement_intitule(fake_pdf) -> None:
    fake_pdf(_target_pages())

    competence = extract_metier_cible_pdf("cible.pdf").competences[1]

    assert competence.competence_description is None
    assert competence.texte_competence == "Compétence deux"


def test_type_extracteur_est_obligatoirement_explicite() -> None:
    with pytest.raises(ValueError, match="n'est jamais deviné"):
        read_emploi_pdf("document.pdf", document_type="automatique")  # type: ignore[arg-type]


def test_pdf_scanned_ou_sans_texte_est_rejete(fake_pdf) -> None:
    fake_pdf([FakePage(1, [], extracted_text="")])

    with pytest.raises(PDFExtractionError) as caught:
        extract_emploi_actuel_pdf("scan.pdf")

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "pdf_sans_texte"
    assert "aucun OCR" in diagnostic.message


def test_section_digitale_absente_est_rejetee(fake_pdf) -> None:
    fake_pdf(
        [
            FakePage(
                1,
                [
                    _line("EMPLOI ACTUEL FICTIF", top=40, kind="title"),
                    _line("COMPÉTENCES FONCTIONNELLES X", top=100, kind="section"),
                    _line("Texte natif suffisamment long pour le diagnostic", top=120),
                ],
            )
        ]
    )

    with pytest.raises(PDFExtractionError) as caught:
        extract_emploi_actuel_pdf("sans_section.pdf")

    assert caught.value.diagnostics[0].code == "section_absente"


def test_tableau_cible_introuvable_est_rejete(fake_pdf) -> None:
    fake_pdf(
        [
            FakePage(
                1,
                [
                    _line("Emploi cible : Métier fictif", top=40),
                    _line("Texte natif suffisamment long sans tableau cible", top=80),
                ],
            )
        ]
    )

    with pytest.raises(PDFExtractionError) as caught:
        extract_metier_cible_pdf("sans_tableau.pdf")

    assert caught.value.diagnostics[0].code == "tableau_introuvable"


@pytest.mark.parametrize("extractor_kind", ["actuel", "cible"])
def test_niveau_non_reconnu_est_rejete(fake_pdf, extractor_kind: str) -> None:
    if extractor_kind == "actuel":
        pages = [
            FakePage(
                1,
                [
                    _line("EMPLOI ACTUEL FICTIF", top=40, kind="title"),
                    _line("COMPÉTENCES DIGITALES X", top=100, kind="section"),
                    _line(
                        "Compétence ambiguë",
                        top=120,
                        kind="competence",
                        level="Intermédiaire",
                    ),
                ],
            )
        ]
        fake_pdf(pages)
        call = lambda: extract_emploi_actuel_pdf("niveau_inconnu.pdf")
    else:
        pages = _target_pages()
        pages[0]._tables[0].rows[1][2] = "Intermédiaire"
        fake_pdf(pages)
        call = lambda: extract_metier_cible_pdf("niveau_inconnu.pdf")

    with pytest.raises(PDFExtractionError) as caught:
        call()

    assert caught.value.diagnostics[0].code == "niveau_inconnu"
    assert "Intermédiaire" in caught.value.diagnostics[0].message


def test_association_description_competence_ambigue_est_rejetee(fake_pdf) -> None:
    fake_pdf(
        [
            FakePage(
                1,
                [
                    _line("EMPLOI ACTUEL FICTIF", top=40, kind="title"),
                    _line("COMPÉTENCES DIGITALES X", top=100, kind="section"),
                    _line("Description orpheline suffisamment longue", top=120),
                ],
            )
        ]
    )

    with pytest.raises(PDFExtractionError) as caught:
        extract_emploi_actuel_pdf("association_ambigue.pdf")

    assert caught.value.diagnostics[0].code == "association_ambigue"


def test_structure_de_tableau_inexploitable_est_rejetee(fake_pdf) -> None:
    pages = _target_pages()
    pages[0]._tables[0].rows.append(["ligne", "à deux cellules"])
    fake_pdf(pages)

    with pytest.raises(PDFExtractionError) as caught:
        extract_metier_cible_pdf("structure_inexploitable.pdf")

    assert caught.value.diagnostics[0].code == "structure_inexploitable"
    assert "trois cellules" in caught.value.diagnostics[0].message


def test_les_extracteurs_ne_chargent_pas_bge_m3(fake_pdf) -> None:
    fake_pdf(_current_pages())

    emploi = extract_emploi_actuel_pdf("sans_modele.pdf")

    assert emploi.competences


def test_les_pdf_confidentiels_ne_sont_pas_suivis_par_git() -> None:
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "--", "*.pdf"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_names = {Path(line).name for line in result.stdout.splitlines()}

    assert "Fiche d'emploi - Concepteur développeur logiciel embarqué.pdf" not in tracked_names
    assert "Emploi cible - Développeur logiciel - DevSecOps.pdf" not in tracked_names
