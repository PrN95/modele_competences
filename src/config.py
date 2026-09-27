"""Constantes centralisées pour l'import et les calculs métier."""

from fractions import Fraction
import os
from pathlib import Path

COMPETENCES_SHEET = "Competences"

REQUIRED_COLUMNS = (
    "emploi_intitule",
    "emploi_type",
    "emploi_effectif",
    "competence_intitule",
    "competence_niveau",
)

EMPLOI_TYPES = frozenset({"actuel", "cible"})
COMPETENCE_LEVELS = frozenset({1, 2, 3, 4})
COMPETENCE_LEVELS_BY_LABEL = {
    "sensibilisation": 1,
    "application": 2,
    "maitrise": 3,
    "expertise": 4,
}

# Les fractions exactes évitent d'utiliser les approximations 0,667 et 0,333.
DENSE_WEIGHT = Fraction(2, 3)
SPARSE_WEIGHT = Fraction(1, 3)
HYBRID_WEIGHTS_SUM_TOLERANCE = 1e-9
SEUIL_SIM = Fraction(7, 10)
SEMANTIC_MATCH_THRESHOLD = SEUIL_SIM
SEUIL_COUV = Fraction(7, 10)

MODEL_PATH = Path("models/bge-m3")
MODEL_DEVICE = "cpu"
MODEL_USE_FP16 = False
MODEL_NORMALIZE_EMBEDDINGS = True
MODEL_BATCH_SIZE = 1
MODEL_MAX_LENGTH = 256

# Les endpoints distants des campagnes expérimentales sont optionnels. Ils ne
# sont jamais requis pour les modèles locaux ni pour les campagnes purement
# locales ; configurez-les explicitement dans l'environnement si nécessaire.
BGE_BASE_URL = os.getenv("REMOTE_BGE_DISTANT_BASE_URL", "")
QWEN_BASE_URL = os.getenv("REMOTE_QWEN_DISTANT_BASE_URL", "")
GEMMA_BASE_URL = os.getenv("REMOTE_GEMMA_DISTANT_BASE_URL", "")
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY") or None
BGE_API_KEY = os.getenv("REMOTE_BGE_API_KEY", REMOTE_API_KEY) or None
QWEN_API_KEY = os.getenv("REMOTE_QWEN_API_KEY", REMOTE_API_KEY) or None
GEMMA_API_KEY = os.getenv("REMOTE_GEMMA_API_KEY", REMOTE_API_KEY) or None
REMOTE_TIMEOUT_SECONDS = float(os.getenv("REMOTE_TIMEOUT_SECONDS", "60"))

# Identifiants par défaut de l'interface. Les scripts expérimentaux découvrent
# systématiquement l'identifiant réellement annoncé par GET /models.
BGE_MODEL = "bge-m3"
QWEN_MODEL = "qwen3-embedding-8b"
GEMMA_MODEL = "gemma4"

# Campagnes ROME : isolé du batch conservateur du matching de production.
ROME_EXPERIMENT_BATCH_SIZE = int(os.getenv("ROME_EXPERIMENT_BATCH_SIZE", "32"))
ROME_EXPERIMENT_MAX_RETRIES = int(os.getenv("ROME_EXPERIMENT_MAX_RETRIES", "3"))

# Référence et sorties des futures campagnes ROME. Les anciennes tables restent
# consultables uniquement lorsqu'elles sont fournies explicitement en argument.
ROME_REFERENCE_TABLE = Path("data/rome/Similarité Emplois ROME 6.xlsx")
ROME_EXPERIMENTS_OUTPUT_ROOT = Path("outputs/rome/experiments_2")
