import json

import pytest
import requests

from src.comparison import executer_comparaison
from src.domain import Competence, Emploi
from src.orchestration import ResultatOrchestration, orchestrer_emplois_llm
from src.remote_models import (
    AdaptateurBGEDistantEmbeddings,
    AdaptateurGemma4,
    AdaptateurQwenEmbeddings,
    CorrespondanceCategorielleGemma,
    ReponseLLMInvalide,
    SelectionGemma,
    ServiceDistantIndisponible,
)


class FauxReponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FauxClient:
    def __init__(self, *, get_payload=None, post_payload=None, post_error=None) -> None:
        self.get_payload = get_payload
        self.post_payload = post_payload
        self.post_error = post_error
        self.posts = []
        self.gets = []

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        return FauxReponse(self.get_payload)

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if self.post_error is not None:
            raise self.post_error
        return FauxReponse(self.post_payload)


def competence(nom: str, niveau: int = 2) -> Competence:
    return Competence(nom, "Description professionnelle", niveau)


def emploi(nom: str, type_emploi: str, competences) -> Emploi:
    return Emploi(nom, type_emploi, None, tuple(competences), f"{nom}.pdf", None)


def reponse_gemma(contenu: str) -> dict:
    return {"choices": [{"message": {"content": contenu}}]}


def test_qwen_dense_only_normalizes_vectors_and_never_invents_sparse() -> None:
    client = FauxClient(
        get_payload={"data": [{"id": "modele-qwen-annonce"}]},
        post_payload={
            "object": "list",
            "model": "qwen3-embedding-8b",
            "data": [
                {"index": 1, "embedding": [0.0, 2.0]},
                {"index": 0, "embedding": [3.0, 4.0]},
            ],
        }
    )
    adaptateur = AdaptateurQwenEmbeddings(
        "http://service/qwen/v1",
        "qwen3-embedding-8b",
        client=client,
    )

    sortie = adaptateur.encoder((competence("A"), competence("B")))

    assert sortie.vecteurs_dense == ((0.6, 0.8), (0.0, 1.0))
    assert sortie.poids_sparse == ({}, {})
    url, appel = client.posts[0]
    assert url == "http://service/qwen/v1/embeddings"
    assert appel["json"] == {
        "model": "modele-qwen-annonce",
        "input": [
            "A. Description professionnelle",
            "B. Description professionnelle",
        ],
        "encoding_format": "float",
    }


def test_bge_distant_uses_the_openai_embedding_contract_in_dense_only() -> None:
    client = FauxClient(
        get_payload={"data": [{"id": "bge-m3-deploye"}]},
        post_payload={"data": [{"index": 0, "embedding": [3.0, 4.0]}]},
    )
    adaptateur = AdaptateurBGEDistantEmbeddings("http://service/bge/v1", "bge-m3", client=client)

    sortie = adaptateur.encoder((competence("A"),))

    assert sortie.vecteurs_dense == ((0.6, 0.8),)
    assert sortie.poids_sparse == ({},)
    assert client.posts[0][0] == "http://service/bge/v1/embeddings"
    assert client.posts[0][1]["json"]["model"] == "bge-m3-deploye"


def test_gemma_receives_levels_and_returns_its_complete_business_reasoning() -> None:
    contenu = json.dumps(
        {
            "emploi_cible_retenu": "[E2] Cible B",
            "g_epfq": 1.0,
            "gs_epfq": 0.5,
            "ecart_moyen": 1.5,
            "r_epfq": 1.0,
            "correspondances": [
                {
                    "competence_cible": "[C1] Cible reconnue",
                    "competence_actuelle": "[A1] Actuelle",
                    "statut": "Reconnue",
                    "niveau_actuel": 4,
                    "niveau_cible": 1,
                    "ecart_niveau": 0,
                    "statut_niveau": "niveau_suffisant",
                    "recommandation_formation": "Aucune formation nécessaire",
                },
                {
                    "competence_cible": "[C2] Cible absente",
                    "competence_actuelle": "N/A",
                    "statut": "Absente",
                    "niveau_actuel": None,
                    "niveau_cible": 3,
                    "ecart_niveau": 3,
                    "statut_niveau": "absente",
                    "recommandation_formation": "Formation complète nécessaire pour acquérir la compétence",
                },
            ],
        }
    )
    client = FauxClient(post_payload=reponse_gemma(contenu))
    client.get_payload = {"data": [{"id": "modele-gemma-annonce"}]}
    adaptateur = AdaptateurGemma4(
        "http://service/v1",
        "gemma4",
        client=client,
    )
    actuelle = competence("Actuelle", 4)
    cible_a = emploi("Cible A", "cible", (competence("Autre", 1),))
    cible_b = emploi(
        "Cible B",
        "cible",
        (competence("Cible reconnue", 1), competence("Cible absente", 3)),
    )
    actuel = emploi("Actuel", "actuel", (actuelle,))

    selection = adaptateur.selectionner(actuel, (cible_a, cible_b))

    assert selection.emploi_cible is cible_b
    assert selection.correspondances[0].competence_actuelle is actuelle
    assert selection.correspondances[0].statut == "Reconnue"
    assert selection.correspondances[1].competence_actuelle is None
    assert selection.correspondances[1].statut == "Absente"
    requete = client.posts[0][1]["json"]
    assert requete["response_format"] == {"type": "json_object"}
    assert "niveau same : 4" in requete["messages"][1]["content"].lower()
    assert "score_similarite" not in requete["messages"][0]["content"]
    assert selection.g_epfq == 1.0
    assert selection.reponse_brute == contenu


@pytest.mark.parametrize(
    "contenu",
    [
        "pas du json",
        '{"emploi_cible_retenu":"[E1] Cible","correspondances":[]}',
        '{"emploi_cible_retenu":"[E1] Cible","correspondances":['
        '{"competence_cible":"[C1] Cible","competence_actuelle":"N/A",'
        '"statut":"Reconnue"}]}',
    ],
)
def test_gemma_rejects_invalid_or_unusable_json(contenu: str) -> None:
    adaptateur = AdaptateurGemma4(
        client=FauxClient(get_payload={"data": [{"id": "gemma"}]}, post_payload=reponse_gemma(contenu))
    )
    actuel = emploi("Actuel", "actuel", (competence("Actuelle"),))
    cible = emploi("Cible", "cible", (competence("Cible"),))

    with pytest.raises(ReponseLLMInvalide):
        adaptateur.selectionner(actuel, (cible,))


def test_gemma_accepts_a_selected_target_below_70_percent_coverage() -> None:
    contenu = json.dumps(
        {
            "emploi_cible_retenu": "[E1] Cible",
            "g_epfq": 0.0,
            "gs_epfq": 0.0,
            "ecart_moyen": 4.0,
            "r_epfq": 0.0,
            "correspondances": [{
                "competence_cible": "[C1] Cible",
                "competence_actuelle": "N/A",
                "statut": "Absente",
                "niveau_actuel": None,
                "niveau_cible": 2,
                "ecart_niveau": 2,
                "statut_niveau": "absente",
                "recommandation_formation": "Formation complète nécessaire",
            }],
        }
    )
    adaptateur = AdaptateurGemma4(
        client=FauxClient(get_payload={"data": [{"id": "gemma"}]}, post_payload=reponse_gemma(contenu))
    )
    actuel = emploi("Actuel", "actuel", (competence("Actuelle"),))
    cible = emploi("Cible", "cible", (competence("Cible"),))

    selection = adaptateur.selectionner(actuel, (cible,))

    assert selection.emploi_cible is cible
    assert selection.g_epfq == 0.0
    assert selection.reponse_brute == contenu


def test_gemma_timeout_is_reported_as_unavailable() -> None:
    adaptateur = AdaptateurGemma4(
        timeout=3,
        client=FauxClient(get_payload={"data": [{"id": "gemma"}]}, post_error=requests.Timeout()),
    )

    with pytest.raises(ServiceDistantIndisponible, match="Timeout après 3 s"):
        adaptateur.selectionner(
            emploi("Actuel", "actuel", (competence("A"),)),
            (emploi("Cible", "cible", (competence("C"),)),),
        )


def test_gemma_network_error_is_reported_as_unavailable() -> None:
    adaptateur = AdaptateurGemma4(
        client=FauxClient(get_payload={"data": [{"id": "gemma"}]}, post_error=requests.ConnectionError()),
    )

    with pytest.raises(ServiceDistantIndisponible, match="indisponible"):
        adaptateur.selectionner(
            emploi("Actuel", "actuel", (competence("A"),)),
            (emploi("Cible", "cible", (competence("C"),)),),
        )


def test_comparison_keeps_two_results_when_one_model_fails() -> None:
    resultat = ResultatOrchestration((), 0.7, 0.7)

    def echec():
        raise ServiceDistantIndisponible("hors ligne")

    executions = executer_comparaison(
        {
            "BGE": lambda: resultat,
            "Qwen": lambda: resultat,
            "Gemma": echec,
        }
    )

    assert [item.statut for item in executions] == ["succès", "succès", "indisponible"]
    assert executions[0].resultat is resultat
    assert executions[1].resultat is resultat
    assert executions[2].resultat is None


def test_gemma_business_indicators_are_transferred_without_recalculation() -> None:
    actuelle_a = competence("Gestion de projet", 2)
    actuelle_b = competence("Architecture", 4)
    emploi_actuel = emploi("Actuel", "actuel", (actuelle_a, actuelle_b))
    cible_non_retenue = emploi("Cible facile", "cible", (competence("Architecture", 1),))
    cible_retenue = emploi(
        "Cible choisie",
        "cible",
        (competence("Pilotage de projet", 3), competence("Sécurité", 2)),
    )

    class FauxSelecteur:
        def selectionner(self, emploi_actuel, emplois_cibles):
            return SelectionGemma(
                cible_retenue,
                (
                    CorrespondanceCategorielleGemma(
                        cible_retenue.competences[0], actuelle_a, "Reconnue", 2, 3, 42,
                        "niveau_insuffisant", "Recommandation formulée par Gemma"
                    ),
                    CorrespondanceCategorielleGemma(
                        cible_retenue.competences[1], None, "Absente", None, 2, 99,
                        "absente", "Formation complète formulée par Gemma"
                    ),
                ),
                0.75, 0.25, 9.75, 0.25, '{"brut":true}',
            )

    resultat = orchestrer_emplois_llm(
        (emploi_actuel,),
        (cible_non_retenue, cible_retenue),
        FauxSelecteur(),
    )
    resultat_emploi = resultat.resultats_emplois[0]
    retenue = resultat_emploi.cibles_retenues[0]
    correspondance = retenue.analyse.correspondances[0]

    assert retenue.analyse.emploi_cible is cible_retenue
    assert len(resultat_emploi.selection.analyses_classees) == 1
    assert correspondance.score_dense is None
    assert correspondance.score_sparse is None
    assert correspondance.score_hybride is None
    assert correspondance.score_llm is None
    assert correspondance.reconnue is True
    assert correspondance.ecart_niveau == 42
    assert correspondance.recommandation_llm == "Recommandation formulée par Gemma"
    assert retenue.analyse.g_ef == 0.75
    assert retenue.analyse.gs_ef == 0.25
    assert retenue.analyse.ecart_moyen == pytest.approx(9.75)
    assert retenue.r_epfq == 0.25
    assert resultat_emploi.reponse_brute_gemma == '{"brut":true}'
    assert resultat.seuil_sim is None and resultat.seuil_couv is None
    assert resultat.poids_dense is None and resultat.poids_sparse is None
    assert resultat.type_score == "décision catégorielle LLM"
