from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any
import warnings

import pytest

from src.domain import Competence, CorrespondanceFournie, CoupleEmplois, Emploi
from src.io_pdf import (
    PDFExtractionError,
    PDFExtractionWarning,
    extract_emploi_actuel_pdf,
    extract_employment_profile_pdf,
    extract_metier_cible_pdf,
    read_emploi_pdf,
)
from src.scoring import analyser_couple


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


def _ambiguous_current_title_pages() -> list[FakePage]:
    return [
        FakePage(
            1,
            [
                _line("Concepteur développeur logiciel SI 1", top=40, kind="title"),
                _line("Analyste développeur logiciel SI 1", top=62, kind="title"),
                _line("COMPÉTENCES DIGITALES X", top=120, kind="section"),
                _line("Développement sécurisé", top=145, kind="competence", level="Application"),
            ],
        )
    ]


def _split_current_title_pages() -> list[FakePage]:
    return [
        FakePage(
            1,
            [
                _line("RESPONSABLE DE GESTION DE", top=40, kind="title"),
                _line("CONFIGURATION LOGICIELLE 4 - G13 v", top=59, kind="title"),
                _line("COMPÉTENCES DIGITALES X", top=120, kind="section"),
                _line("Gestion de configuration", top=145, kind="competence", level="Application"),
            ],
        )
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
    assert emploi.format_extraction == "talentsoft"
    assert [competence.intitule for competence in emploi.competences] == [
        "Compétence sur deux pages",
        "Compétence sans identifiant",
    ]
    assert emploi.competences[0].description == (
        "Description commencée et terminée à la page suivante"
    )
    assert all("Niveau SAME cible" not in competence.intitule for competence in emploi.competences)


def test_meme_fiche_talentsoft_est_acceptee_comme_emploi_cible(fake_pdf) -> None:
    fake_pdf(_current_pages())

    emploi = read_emploi_pdf(
        "fiche_talentsoft.pdf",
        document_type="metier_cible",
        original_filename="EMPLOI ACTUEL FICTIF.pdf",
    )

    assert emploi.type == "cible"
    assert emploi.format_extraction == "talentsoft"
    assert emploi.intitule == "EMPLOI ACTUEL FICTIF"
    assert [competence.intitule for competence in emploi.competences] == [
        "Compétence sur deux pages",
        "Compétence sans identifiant",
    ]


@pytest.mark.parametrize(
    ("role", "type_attendu"),
    [("emploi_actuel", "actuel"), ("metier_cible", "cible")],
)
def test_meme_pdf_tabulaire_est_accepte_dans_les_deux_zones(role, type_attendu, fake_pdf) -> None:
    fake_pdf(_target_pages())

    emploi = extract_employment_profile_pdf(
        "Emploi_cible_1_Developpeur_logiciel_DevSecOps_D07.pdf",
        role=role,
        original_filename="Emploi_cible_1_Developpeur_logiciel_DevSecOps_D07.pdf",
    )

    assert emploi.type == type_attendu
    assert emploi.format_extraction == "tableau"
    assert len(emploi.competences) == 4


def test_fiche_talentsoft_privilegie_le_nom_original_du_document(fake_pdf) -> None:
    fake_pdf(_current_pages())

    emploi = extract_emploi_actuel_pdf(
        "/tmp/tmpkxc22q45.pdf",
        original_filename="Nom sans rapport - E10.pdf",
    )

    assert emploi.intitule == "Nom sans rapport - E10"
    assert emploi.fichier_source == "Nom sans rapport - E10.pdf"


def test_intitule_ambigu_est_resolu_par_le_nom_original_de_fichier(fake_pdf) -> None:
    fake_pdf(_ambiguous_current_title_pages())

    emploi = read_emploi_pdf(
        "/tmp/tmpkxc22q45.pdf",
        document_type="emploi_actuel",
        original_filename="Concepteur développeur logiciel SI 1 - E10.pdf",
    )

    assert emploi.intitule == "Concepteur développeur logiciel SI 1 - E10"
    assert emploi.fichier_source == "Concepteur développeur logiciel SI 1 - E10.pdf"


def test_nom_fichier_normalise_resout_casse_accents_et_tirets(fake_pdf) -> None:
    fake_pdf(_ambiguous_current_title_pages())

    emploi = extract_emploi_actuel_pdf(
        "/tmp/tmpkxc22q45.pdf",
        original_filename="CONCEPTEUR-déVELOPPEUR logiciel si 1 — E10.PDF",
    )

    assert emploi.intitule == "CONCEPTEUR-déVELOPPEUR logiciel si 1 — E10"


def test_nom_fichier_ne_departage_pas_un_intitule_ambigu(fake_pdf) -> None:
    fake_pdf(_ambiguous_current_title_pages())

    with pytest.raises(PDFExtractionError) as caught:
        extract_emploi_actuel_pdf("/tmp/tmpkxc22q45.pdf")

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "association_ambigue"
    assert "tmpkxc22q45.pdf" in diagnostic.message
    assert "Concepteur développeur logiciel SI 1" in diagnostic.message
    assert "aucun candidat ne correspond exactement" in diagnostic.message


def test_intitule_talentsoft_sur_deux_lignes_est_reconcilie_avec_le_nom(fake_pdf) -> None:
    fake_pdf(_split_current_title_pages())

    emploi = extract_emploi_actuel_pdf(
        "fiche.pdf",
        original_filename="Responsable de gestion de configuration logicielle 4 - G13.pdf",
    )

    assert emploi.intitule == "Responsable de gestion de configuration logicielle 4 - G13"


def test_extracteur_metier_cible_lit_un_tableau_multipage_sans_entete_repetee(fake_pdf) -> None:
    fake_pdf(_target_pages())

    emploi = extract_metier_cible_pdf("metier_cible.pdf")

    assert emploi.intitule == "Métier cible fictif"
    assert emploi.type == "cible"
    assert emploi.format_extraction == "tableau"
    assert len(emploi.competences) == 4
    assert [competence.niveau for competence in emploi.competences] == [1, 2, 3, 4]
    assert emploi.competences[0].description == "Description sur plusieurs lignes"


def test_extracteur_cible_accepte_les_alias_de_colonnes(fake_pdf) -> None:
    pages = _target_pages()
    pages[0]._tables[0].rows[0] = [
        "Compétence digitale",
        "Détail de la compétence",
        "Niveau de la compétence",
    ]
    fake_pdf(pages)

    emploi = extract_metier_cible_pdf("alias.pdf")

    assert [competence.intitule for competence in emploi.competences] == [
        "Compétence un",
        "Compétence deux",
        "Compétence trois",
        "Compétence quatre",
    ]


def test_extracteur_cible_normalise_casse_accents_pluriels_et_retours_ligne(fake_pdf) -> None:
    pages = _target_pages()
    pages[0]._tables[0].rows[0] = [
        "  COMPÉTENCES\nDIGITALES ",
        "dÉTAILS---COMPÉTENCES",
        "Niveaux\ncompétences",
    ]
    fake_pdf(pages)

    with warnings.catch_warnings():
        warnings.simplefilter("error", PDFExtractionWarning)
        emploi = extract_metier_cible_pdf("alias_normalise.pdf")

    assert len(emploi.competences) == 4
    assert emploi.competences[0].niveau == 1


def test_extracteur_cible_interprete_un_tableau_par_position_en_dernier_recours(fake_pdf) -> None:
    pages = _target_pages()
    pages[0]._tables[0].rows[0] = ["Savoir-faire", "Explication", "Attendu"]
    fake_pdf(pages)

    with pytest.warns(PDFExtractionWarning, match="colonnes_interpretees_par_position") as caught:
        emploi = extract_metier_cible_pdf("position.pdf")

    assert len(emploi.competences) == 4
    assert "Savoir-faire | Explication | Attendu" in str(caught[0].message)


def test_tableaux_positionnels_ambigus_sont_rejetes_avec_les_alias(fake_pdf) -> None:
    first = FakeTable(
        [["Savoir-faire", "Explication", "Attendu"], ["Python", "Code", "Application"]]
    )
    second = FakeTable(
        [["Domaine", "Précisions", "Palier"], ["SQL", "Requêtes", "Maîtrise"]]
    )
    fake_pdf(
        [
            FakePage(
                1,
                [_line("Emploi cible : Métier fictif", top=40)],
                [first, second],
            )
        ]
    )

    with pytest.raises(PDFExtractionError) as caught:
        extract_metier_cible_pdf("ambigu.pdf")

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "structure_inexploitable"
    assert "en-têtes détectés" in diagnostic.message
    assert "Alias acceptés" in diagnostic.message


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

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "tableau_introuvable"
    assert "en-têtes détectés" in diagnostic.message
    assert "Alias acceptés" in diagnostic.message


@pytest.mark.parametrize(
    ("role", "expected_type"),
    [("emploi_actuel", "actuel"), ("metier_cible", "cible")],
)
def test_fiche_rome_ne_conserve_que_savoir_faire_principaux_et_domaines_expertise(
    fake_pdf, role: str, expected_type: str
) -> None:
    fake_pdf(
        [
            FakePage(
                1,
                [
                    _line("D1401", top=20),
                    _line("Développeur logiciel", top=40),
                    _line("COMPÉTENCES", top=80, kind="section"),
                    _line("Savoir-faire principaux", top=100, kind="section"),
                    _line("• Développer une application - Transition numérique", top=120),
                    _line("• Concevoir une API - Transition écologique", top=140),
                    _line("• Documenter une solution", top=160),
                    _line("Domaines d’expertise", top=180, kind="section"),
                    _line("• Cybersécurité Transition numérique", top=200),
                    _line("Savoir-faire secondaires", top=220, kind="section"),
                    _line("• Administrer un serveur Transition numérique", top=240),
                    _line("Savoirs", top=250, kind="section"),
                    _line("• Connaître les normes de sécurité", top=255),
                    _line("Contextes de travail", top=260, kind="section"),
                    _line("• Texte à exclure Transition numérique", top=280),
                    _line("Secteurs d’activité", top=290, kind="section"),
                    _line("• Secteur à exclure", top=300),
                ],
            )
        ]
    )

    emploi = read_emploi_pdf(
        "rome.pdf", document_type=role, original_filename="Developpeur logiciel - D1401.pdf"
    )

    assert emploi.type == expected_type
    assert emploi.format_extraction == "rome"
    assert emploi.intitule == "Développeur logiciel"
    assert [item.intitule for item in emploi.competences] == [
        "Développer une application",
        "Concevoir une API",
        "Documenter une solution",
        "Cybersécurité",
    ]
    assert [item.source_section for item in emploi.competences] == [
        "Savoir-faire principaux",
        "Savoir-faire principaux",
        "Savoir-faire principaux",
        "Domaines d’expertise",
    ]
    assert all(item.description is None and item.niveau is None for item in emploi.competences)


def test_emploi_rome_sans_niveau_reste_comparable_semantiquement(fake_pdf) -> None:
    fake_pdf(
        [
            FakePage(
                1,
                [
                    _line("D1401", top=20),
                    _line("Développeur logiciel", top=40),
                    _line("Savoir-faire principaux", top=80, kind="section"),
                    _line("• Développer une application Transition numérique", top=100),
                ],
            )
        ]
    )
    actuel = read_emploi_pdf("rome.pdf", document_type="emploi_actuel")
    cible_competence = Competence("Concevoir une application", None, 2)
    cible = Emploi("Cible", "cible", None, (cible_competence,), "cible.pdf", None)

    analyse = analyser_couple(
        CoupleEmplois(actuel, cible),
        (CorrespondanceFournie(cible_competence, actuel.competences[0], 0.8, 0.8),),
    )

    assert analyse.couverture_semantique == 1.0
    assert analyse.correspondances[0].ecart_niveau is None
    assert analyse.ecart_moyen is None


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
