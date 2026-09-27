# Passation IA — PoC de rapprochement des compétences

## 1. Objectif du projet

Ce dépôt contient un PoC local d’aide à la décision RH : pour un ou plusieurs **emplois actuels** (profils types, jamais des personnes), il compare les compétences avec plusieurs **emplois cibles**, écarte les cibles non admissibles, sélectionne les meilleures, explique les écarts de niveau et recommande les formations.

Le corpus **ROME** est une source de calibration expérimentale, distincte des données internes de l’entreprise. Il sert à mesurer des proximités entre emplois à partir de relations **non directionnelles** ; il ne représente ni des parcours, ni des évolutions de carrière, ni une vérité métier exhaustive.

Les sources de vérité fonctionnelles sont `AGENTS.md` et `docs/Feuille_cadrage_IA.docx`.

## 2. Architecture et arborescence utile

- `app.py` : interface Streamlit de l’application RH. Ne pas la modifier pour les expérimentations ROME sans décision explicite.
- `src/` : logique métier indépendante de Streamlit.
  - `io_pdf.py` : extracteurs PDF distincts emploi actuel / emploi cible.
  - `embeddings.py`, `matching.py`, `scoring.py`, `orchestration.py`, `recommendations.py` : moteur de production.
  - `remote_models.py` : adaptateurs Qwen/Gemma pour l’application.
  - `rome.py` : lecture Excel ROME, extraction/validation des PDF et corpus.
  - `rome_experiment.py` : client OpenAI-compatible dense, cache et grille de calibration génériques.
  - `config.py` : constantes, endpoints et variables d’environnement.
- `scripts/` : commandes explicites de validation, connectivité, campagnes et diagnostics.
- `tests/` : tests pytest sans modèle réel ni appel à un endpoint distant configuré.
- `data/rome/` : Excel de référence et PDF ROME locaux (ignorés par Git).
- `outputs/rome/` : corpus généré, diagnostics, résultats et caches expérimentaux (ignorés par Git).
- `models/` : modèle BGE-M3 local éventuel, ignoré par Git ; réservé au moteur de production local, pas aux campagnes expérimentales distantes.

Fichiers générés importants :

- `outputs/rome/corpus_rome.csv`, `outputs/rome/rapport_validation_rome.csv` ;
- par campagne : `resume_configurations.csv`, `detail_relations.csv`, `metadonnees.json`, `rapport_synthese.md`, `cache/` ;
- `outputs/rome/diagnostic_qwen_sparse.md`.

État Git constaté : branche `main`, arbre de travail non propre avec modifications et fichiers non suivis issus du développement courant. Les résultats sous `outputs/` sont ignorés : ne pas les supprimer ni les régénérer sans nécessité.

## 3. Modèles et endpoints distants

Les campagnes utilisent des services OpenAI-compatibles et découvrent toujours l’identifiant réellement servi par `GET /models` avant l’encodage.

| Service | Endpoint par défaut | Variable d’URL | Variable de clé dédiée |
|---|---|---|---|
| BGE-M3 | `REMOTE_BGE_DISTANT_BASE_URL` | `REMOTE_BGE_DISTANT_BASE_URL` | `REMOTE_BGE_API_KEY` |
| Qwen3-Embedding-8B | `REMOTE_QWEN_DISTANT_BASE_URL` | `REMOTE_QWEN_DISTANT_BASE_URL` | `REMOTE_QWEN_API_KEY` |
| Gemma 4 | `REMOTE_GEMMA_DISTANT_BASE_URL` | `REMOTE_GEMMA_DISTANT_BASE_URL` | `REMOTE_GEMMA_API_KEY` |

Variables communes : `REMOTE_API_KEY` (repli sans clé spécifique), `REMOTE_TIMEOUT_SECONDS` et `ROME_EXPERIMENT_BATCH_SIZE` (défaut `32`). Les serveurs sans authentification sont supportés. Ne jamais écrire une clé dans le dépôt.

Pour les expérimentations, **ne jamais charger de modèle localement** : utiliser l’API distante, par lots, avec cache séparé par modèle/endpoint/corpus.

## 4. Données ROME

- Excel de référence : `data/rome/Similarité Emplois ROME.xlsx`.
- PDF : `data/rome/Fiches emploi ROME/`.
- La clé de rapprochement est le **code ROME** de l’en-tête PDF ; les écarts d’intitulé avec l’Excel sont des avertissements.
- `src/rome.py` extrait uniquement les puces de la rubrique `Compétences`, et s’arrête à `Contextes de travail`.
- Les relations de l’Excel sont dédoublonnées et traitées comme des paires symétriques/non directionnelles.
- Les PDF et les données confidentielles sont ignorés par Git. Ne jamais ajouter les PDF au dépôt ; conserver leur version locale et leur correspondance avec l’Excel avant de régénérer le corpus.

## 5. Campagnes déjà réalisées

Protocole identique : dense-only, cosinus dense entre compétences, coefficient dense `1,0`, sparse `0,0`, couverture A→B et B→A, couverture symétrique = moyenne ; prédiction si couverture symétrique ≥ `seuil_couv`. Les 64 couples de `seuil_sim` et `seuil_couv` couvrent `0,20` à `0,90`, pas `0,10`. Meilleure configuration : F1, puis rappel, puis précision.

| Campagne | Commande | Résultat optimal |
|---|---|---|
| BGE-M3 dense-only | `.venv/bin/python scripts/run_rome_bge_dense_experiment.py --batch-size 32` | seuils `0,60 / 0,80` ; TP 22, FP 496, FN 0, TN 43 ; précision 0,04247 ; rappel 1,00000 ; F1 0,08148 |
| Qwen3-Embedding-8B dense-only | `.venv/bin/python scripts/run_rome_qwen_dense_experiment.py --batch-size 32` | seuils `0,80 / 0,30` ; TP 17, FP 356, FN 5, TN 183 ; précision 0,04558 ; rappel 0,77273 ; F1 0,08608 |

Résultats et caches :

- BGE : `outputs/rome/experiments/bge_m3_dense/` et `.../bge_m3_dense/cache/` ; 34 emplois, 1 726 compétences, 22 relations de référence.
- Qwen : `outputs/rome/experiments/qwen3_embedding_8b_dense/` et `.../qwen3_embedding_8b_dense/cache/` ; même corpus et même empreinte.

Les faux positifs sont une limite méthodologique : une paire absente du graphe est une `non_relation_reference`, **pas** une incompatibilité réelle des emplois.

## 6. Décisions méthodologiques

- BGE et Qwen sont actuellement évalués en **dense-only** dans ROME.
- La similarité est le cosinus des vecteurs denses normalisés L2.
- Grille de seuils : `0,20` à `0,90`, pas `0,10`.
- BGE-M3 : le serveur actuel n’expose pas encore de représentation sparse exploitable (test `return_sparse=true` : réponse dense seule).
- Qwen : dense-only ; le diagnostic réel accepte `return_sparse=true` mais ne retourne que `embedding`, `index`, `object` par entrée, sans sparse/lexical/token weights. Ne pas chercher de sparse natif dans cette API actuelle.
- Ne jamais modifier le moteur de production à partir du corpus ROME sans décision explicite.

## 7. État actuel et prochaines étapes

À traiter dans cet ordre :

1. Demander/exposer les poids sparse BGE-M3 côté serveur.
2. Adapter le client seulement si l’API retourne réellement, par exemple, `sparse_embedding` ou une structure équivalente exploitable.
3. Lancer une campagne BGE hybride dense+sparse, avec paramètres et cache séparés.
4. Comparer BGE dense, BGE hybride et Qwen dense.
5. Produire une analyse qualitative et les métriques de classement complémentaires.
6. Appliquer les paramètres retenus aux emplois actuels et cibles de l’entreprise.
7. Tester Gemma 4 séparément avec les prompts RH, sans seuils ni score hybride.

## 8. Règles fonctionnelles du matching interne

- Considérer uniquement les **compétences digitales** des PDF d’emplois actuels.
- Niveaux : Sensibilisation = 1, Application = 2, Maîtrise = 3, Expertise = 4. Le niveau 0 est interne et signifie non-reconnu.
- Le niveau ne modifie jamais la similarité sémantique ; il sert ensuite à calculer les écarts et recommandations.
- Orchestrer tous les emplois actuels contre tous les emplois cibles.
- Sorties attendues : cible(s) recommandée(s), compétences mobilisables/réutilisées, compétences absentes ou insuffisantes, niveaux à atteindre et recommandations de formation.

Le moteur de production local BGE combine, lorsque les deux représentations existent : `H_ac = poids_dense × D_ac + poids_sparse × L_ac`. Les poids par défaut sont `2/3` et `1/3`, doivent être dans `[0,1]` et leur somme doit valoir 1 ; ne jamais les normaliser automatiquement.

## 9. Commandes utiles

```bash
# Validation et génération du corpus ROME
.venv/bin/python scripts/validate_rome_corpus.py

# Connectivité BGE, Qwen et Gemma (découverte GET /models)
.venv/bin/python scripts/test_remote_models.py

# Campagnes denses distantes
.venv/bin/python scripts/run_rome_bge_dense_experiment.py --batch-size 32
.venv/bin/python scripts/run_rome_qwen_dense_experiment.py --batch-size 32

# Diagnostic sparse Qwen
.venv/bin/python scripts/check_qwen_sparse_capability.py

# Tests automatiques, sans serveur réel
.venv/bin/python -m pytest -q
```

## 10. Points de vigilance pour Antigravity

- Lire `AGENTS.md` avant toute intervention.
- Ne jamais supprimer ou écraser des résultats, caches ou rapports existants.
- Préserver strictement les formats CSV de campagne pour garder la comparaison BGE/Qwen possible.
- Conserver la séparation entre expérimentation ROME et application RH de l’entreprise.
- Ne jamais inscrire de secret ou clé API dans le dépôt, les exports ou les logs.
- Ne pas faire charger BGE localement aux campagnes expérimentales ; ne pas faire appeler les serveurs distants par pytest.

## Instruction de reprise pour une IA

> Lis d’abord `AGENTS.md`, puis ce document et `README.md`. Inspecte `git status` avant toute modification, préserve les sorties ROME existantes et ne touche ni à Streamlit ni au moteur de production sans demande explicite. Reprends les expérimentations à partir des caches et scripts existants : demander d’abord une exposition sparse BGE côté serveur, adapter uniquement le client expérimental si la réponse fournit réellement une structure sparse, puis lancer une campagne hybride dans un nouveau répertoire et comparer ses CSV aux campagnes BGE/Qwen dense. N’ajoute jamais de secret ni de PDF/données internes au Git.
