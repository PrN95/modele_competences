from src.domain import Competence


def test_competence_text_uses_only_title_and_description() -> None:
    competence = Competence(
        id="COMP-01",
        intitule="Intitulé fictif",
        description="Description fictive",
        niveau=2,
    )

    assert competence.texte == "Intitulé fictif. Description fictive"

