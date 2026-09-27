# Mise à jour projet du 06 Septembre 2026

Ce document est une passation autonome, établie par inspection locale. Il distingue les faits vérifiés pendant cette inspection, les éléments attestés par les sources du projet mais non réexécutés, et les travaux prévus. En cas de conflit, `AGENTS.md` et `docs/Feuille_cadrage_IA.docx` restent les sources de vérité fonctionnelles.

## 1. Objectif et règles métier actives

Le dépôt porte un PoC local d'aide à la décision RH. Il compare les compétences d'un ou plusieurs **emplois actuels** (profils types, jamais des personnes) à plusieurs **emplois cibles**, écarte les cibles non admissibles, sélectionne la ou les meilleures et produit les écarts de niveau ainsi que des recommandations de formation. Plusieurs emplois actuels peuvent mener à la même cible.

Règles structurantes :

- La logique métier est indépendante de Streamlit ; les PDF sont la source principale et Excel est une compatibilité secondaire/support de test.
- Les niveaux valides sont 1 Sensibilisation, 2 Application, 3 Maîtrise et 4 Expertise. Le niveau 0 est strictement interne : compétence cible non reconnue. Il ne figure jamais dans les PDF.
- Le niveau n'est jamais encodé et ne change jamais les scores sémantiques. Il intervient après le matching dans les écarts, `Gs` et les recommandations.
- Les recommandations ne sont calculées que pour les cibles sélectionnées. Le contrôle des compétences actuelles non reprises et `R_epfq` sont informatifs : ils ne changent ni les scores, ni l'admissibilité, ni la sélection.
- OCR, ColBERT, reranker, cohortes automatiques, génération automatique de cibles et risque d'obsolescence sont hors périmètre courant.

## 2. Architecture actuelle

| Emplacement | Rôle constaté |
| --- | --- |
| `app.py` | Interface Streamlit : chargements PDF séparés, paramètres, exécution mono-modèle ou comparative, restitution et exports CSV. |
| `src/domain.py` | Objets métier (`Emploi`, `Competence`, correspondances, résultats). |
| `src/io_pdf.py` | Extraction `pdfplumber` des PDF Talentsoft actuels/cibles et des fiches ROME. |
| `src/io_excel.py`, `src/validation.py` | Compatibilité Excel et validations. |
| `src/embeddings.py` | Adaptateur BGE-M3 local (`FlagEmbedding.BGEM3FlagModel`), dense + sparse, hors ligne. |
| `src/matching.py` | Scores dense/sparse/hybride, choix de la meilleure compétence et contrôle inverse. |
| `src/scoring.py`, `src/recommendations.py`, `src/orchestration.py` | Indicateurs, sélection, besoins de formation, orchestration multi-emplois et calcul de `R_epfq`. |
| `src/remote_models.py`, `src/comparison.py` | Clients Qwen/Gemma, validation des réponses et isolation des résultats/erreurs par moteur. |
| `src/rome.py`, `src/rome_experiment.py` | Corpus ROME et campagnes distantes isolées de l'application RH. |
| `scripts/` | Validation ROME, test des services distants, diagnostics sparse et campagnes. |
| `tests/` | Tests pytest avec faux encodeurs, PDF et HTTP simulés. |
| `outputs/rome/` | Corpus, diagnostics, caches et résultats de campagnes ; ignorés par Git, à préserver. |

La configuration est principalement dans `src/config.py` : seuils, poids, chemin BGE, batch de production et variables d'environnement des services distants. `requirements.txt` épingle notamment `FlagEmbedding`, `pdfplumber`, `pandas`, `openpyxl`, `requests`, `streamlit` et PyTorch CPU. `sentence-transformers` apparaît comme dépendance transitive/épinglée, mais le code n'importe pas directement cette bibliothèque, conformément au cadrage.

## 3. Entrées acceptées et extraction PDF

L'interface accepte plusieurs **PDF natifs structurés** dans deux zones séparées. La zone choisie impose le type ; il n'est jamais deviné.

| Dépôt | Extraction attendue | Champs normalisés |
| --- | --- | --- |
| Emploi actuel Talentsoft | Intitulé d'en-tête ; seulement `Compétences digitales` ; intitulé noir, description sous l'intitulé, niveau à droite. Les catégories bleues, responsabilités, compétences fonctionnelles et `Niveau SAME cible` sont ignorés. Les coupures inter-pages sont gérées. | `competence_intitule`, `competence_description`, `niveau` |
| Emploi cible | Intitulé au-dessus du tableau ; colonnes `Intitulé de la compétence`, `Détails de la compétence`, `Niveau`. Les tableaux multipages, même sans nouvel en-tête, sont pris en charge. | mêmes champs |
| Fiche ROME | Voir section suivante. L'extracteur ROME est tenté par le parseur commun en complément des formats précédents. | intitulé, description absente, niveau absent |

Un PDF scanné, ambigu, sans texte natif, structure exploitable ou section attendue est rejeté avec diagnostic ; l'OCR est hors périmètre. Le texte encodé est exactement `intitulé + ". " + description` si une description est disponible, sinon le seul intitulé. Les métadonnées et niveaux restent séparés.

**Statut de vérification.** Les extracteurs sont couverts par des PDF simulés dans pytest. Ils sont implémentés, mais ne sont pas intégrés à une campagne de tests sur vrais PDF Talentsoft : **à vérifier sur les fichiers réels représentatifs**.

## 4. Décisions ROME : extraction, niveaux et comportement

ROME est un corpus de calibration expérimental, distinct des emplois internes ; il ne décrit ni parcours de mobilité ni vérité métier exhaustive. Les relations de la table Excel sont dédoublonnées et utilisées comme paires non directionnelles. Le code ROME de l'en-tête PDF est la clé de rapprochement ; un écart de libellé Excel/PDF n'est qu'un avertissement.

Deux usages doivent rester distincts :

1. `src/rome.py` construit le corpus de campagne en extrayant les puces de la rubrique générale `Compétences`, jusqu'à `Contextes de travail`.
2. Dans le chargeur commun Streamlit (`src/io_pdf.py`), une fiche ROME devient un emploi à partir de toutes les puces de `Savoir-faire principaux` et de `Domaines d’expertise`. Les badges « Transition numérique » et « Transition écologique » sont ignorés et retirés du libellé. `Savoirs`, `Savoir-faire secondaires`, `Contextes de travail`, `Secteurs d’activité` et tout texte après une nouvelle rubrique sont exclus. La rubrique source est conservée pour chaque compétence.

Les compétences ROME n'ont pas de niveau SAME : leur `niveau` reste `None`. Les calculs nécessitant des niveaux produisent donc un écart non calculable et n'inventent aucun niveau. Cette prise en charge est implémentée et testée sur données simulées ; son rendu avec une vraie fiche chargée dans Streamlit est **à vérifier**.

## 5. Matching effectivement implémenté

### Embeddings BGE-M3 et Qwen

Pour une compétence actuelle `a` et cible `c` :

```text
D_ac = produit scalaire de vecteurs denses normalisés
L_ac = cosinus L2 des dictionnaires sparse (tokens communs)
H_ac = poids_dense × D_ac + poids_sparse × L_ac
```

- BGE-M3 local utilise dense + sparse. Valeurs par défaut exactes : `2/3` et `1/3`. Les deux poids sont réglables séparément dans Streamlit, doivent être dans `[0,1]` et totaliser 1 (tolérance numérique). Une somme invalide bloque l'analyse ; aucune normalisation implicite n'est faite.
- Qwen est dense-only : vecteurs normalisés L2 localement, `H = D`, dense `1,00`, sparse `0,00`, score sparse `N/A`.
- Pour chaque compétence cible, le meilleur `H_ac` est retenu. À égalité exacte : niveau actuel le plus élevé, puis `L_ac` le plus élevé, puis choix déterministe avec détail d'égalité. Une compétence actuelle peut couvrir plusieurs cibles.
- Une compétence est reconnue si `H_ac >= seuil_sim`; sinon le niveau actuel interne utilisé est 0. Les valeurs de référence de `seuil_sim` et `seuil_couv` sont 0,70, pas 0,05 ; elles sont affichées et exportées pour les embeddings.

Pour chaque couple emploi actuel `e` / cible `f` :

```text
Ecart_c = max(0, niveau_cible - niveau_actuel)
Ecart_moyen_ef = somme(Ecart_c × niveau_cible) / somme(niveau_cible)
G_ef = compétences cibles reconnues / compétences cibles totales
Gs_ef = compétences reconnues avec niveau atteint / compétences cibles totales
```

Une cible est admissible si `G_ef >= seuil_couv`. Les admissibles sont classées par `G_ef` décroissant, écart moyen croissant, puis `Gs_ef` décroissant. Après triple égalité, toutes sont conservées et l'arbitrage RH est signalé. Sans admissible, le message est exactement : « Aucun emploi cible ne correspond à cet emploi actuel ».

Après sélection, `R_epfq` est calculé séparément par cible retenue : part des compétences actuelles dont le meilleur score vers cette seule cible atteint `seuil_sim`. Le détail Streamlit présente les correspondances, scores disponibles, niveaux, écarts, recommandations, compétences actuelles non reprises et exports consolidés/matrice/détails sélectionnés.

### Gemma4

Gemma ne produit pas de score sémantique. Il sélectionne directement une unique cible proposée, puis doit retourner pour **chaque** compétence de cette cible une compétence actuelle proposée ou `N/A`, avec exactement `Reconnue` ou `Absente`. Toute sélection inconnue, décision incomplète/supplémentaire ou incohérente est rejetée. Après ce choix seulement, le code calcule `G_epfq`, `Gs_epfq`, écart moyen, recommandations et `R_epfq` à partir des décisions catégorielles et niveaux locaux. Ces indicateurs sont informatifs et ne peuvent pas modifier le choix Gemma. Aucun seuil ni poids n'est applicable, affiché ou exporté pour Gemma ; les scores dense/sparse/hybride sont `N/A`.

## 6. Modèles et endpoints distants

| Moteur | Endpoint par défaut configurable | Identifiant / comportement | État établi |
| --- | --- | --- | --- |
| BGE-M3 local | chemin `models/bge-m3` (`MODEL_PATH`) | `BGEM3FlagModel`, dense+sparse ; ColBERT et reranker désactivés | Implémenté. `models/bge-m3` est présent lors de l'inspection. Exécution réelle hors ligne attestée par le cadrage : dense 1024, ~22 s chargement+première inférence, ~0,86 s pour 4 compétences ensuite, ~1,97 Gio sans swap. Non rejouée ici. |
| BGE-M3 distant (campagnes ROME) | `REMOTE_BGE_DISTANT_BASE_URL` | découverte `GET /models`, embeddings en campagne dense-only | Fonctionnement historique attesté par les sorties ROME ; disponibilité actuelle **à vérifier**. Sparse distant non exploité dans les campagnes. |
| Qwen3-Embedding-8B | `REMOTE_QWEN_DISTANT_BASE_URL` | `qwen3-embedding-8b`, `POST /embeddings`, dense 4096 | Le diagnostic archivé a reçu HTTP 200, avec et sans `return_sparse=true`, mais aucune donnée sparse. L'application est donc dense-only. Connectivité actuelle **à vérifier**. |
| Gemma4 | `REMOTE_GEMMA_DISTANT_BASE_URL` | `gemma4`, `POST /chat/completions`, JSON strict | Adaptateur, validation et interface implémentés/testés avec HTTP simulé. Aucune sortie de campagne/résultat Gemma réel n'est présente : fonctionnement réel de bout en bout **à vérifier**. |

Variables communes : `REMOTE_API_KEY` (repli), clés dédiées `REMOTE_BGE_API_KEY`, `REMOTE_QWEN_API_KEY`, `REMOTE_GEMMA_API_KEY`, et `REMOTE_TIMEOUT_SECONDS` (60 s). Les clés ne doivent être ni codées, ni exportées. Streamlit rend URL, identifiant, timeout et clé optionnelle configurables, avec boutons de test Qwen/Gemma.

## 7. Campagnes ROME réalisées et leurs limites

Corpus des campagnes : 34 emplois, 1 726 compétences, 22 relations de référence, empreinte `5b3fd77f41c007ef…`. Les deux modèles ont été évalués en dense-only, cosinus de compétences, couverture A→B et B→A moyennée, grille 8 × 8 des seuils 0,20 à 0,90. La meilleure configuration maximise F1, puis rappel, puis précision.

| Modèle | Version/référence | Meilleurs seuils sim/couv | TP / FP / FN / TN | Précision | Rappel | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BGE-M3 dense-only | v1 | 0,60 / 0,80 | 22 / 496 / 0 / 43 | 0,04247 | 1,00000 | 0,08148 |
| Qwen3-Embedding-8B dense-only | v1 | 0,80 / 0,30 | 17 / 356 / 5 / 183 | 0,04558 | 0,77273 | 0,08608 |
| BGE-M3 dense-only | v2 | 0,70 / 0,70 | 6 / 19 / 16 / 520 | 0,24000 | 0,27273 | 0,25532 |
| Qwen3-Embedding-8B dense-only | v2 | 0,90 / 0,30 | 6 / 18 / 16 / 521 | 0,25000 | 0,27273 | 0,26087 |

Les versions v2 ont relu les caches v1 : aucun serveur n'a été appelé. Elles emploient une table de référence corrigée (`Similarité Emplois ROME 2.xlsx`) qui conserve 2 paires, en ajoute 20 et en retire 20, tout en gardant 22 relations. La hausse de F1/précision et la baisse de rappel mesurent donc le changement de référence, **pas une amélioration intrinsèque des embeddings**. Les paires absentes du graphe sont des `non_relation_reference`, non des incompatibilités humaines : les faux positifs ne doivent pas être interprétés comme des erreurs métiers certaines. Aucune campagne hybride BGE, campagne sparse Qwen, métrique de classement complémentaire ou campagne Gemma n'est présente.

Les artefacts sont dans `outputs/rome/experiments/{bge_m3_dense,qwen3_embedding_8b_dense}` (v1) et les dossiers suffixés `_rome_v2` (v2), avec rapports, CSV et métadonnées. `outputs/rome/diagnostic_qwen_sparse.md` établit l'absence de champ sparse exposé par l'API Qwen testée.

## 8. Changements depuis `docs/PASSATION_IA_PROJET.md`

La passation existe ; les points suivants sont donc les changements constatables **depuis son contenu**, pas seulement depuis le dernier commit Git.

- La campagne ROME v2 et les comparaisons v1/v2 sont maintenant présentes dans `outputs/rome/` ; la référence v2 corrigée, ses résultats et leur interprétation sont documentés ci-dessus.
- Le jeu local a évolué au-delà des 115 tests annoncés dans la passation : 151 tests sont collectés et passent lors de cette inspection. Sont notamment présents des tests et modules non suivis pour ROME, clients distants, comparaison, import Streamlit et restitution.
- Le chargeur PDF inclut désormais une stratégie ROME limitée à `Savoir-faire principaux` et `Domaines d’expertise`, avec niveaux absents, en plus des formats Talentsoft/tableau décrits par la passation.
- Le statut Git est **non propre** : modifications suivies de `app.py`, `src/`, tests, README, cadrage et `.gitignore`, et nouveaux fichiers non suivis (`docs/PASSATION_IA_PROJET.md`, `scripts/`, `src/comparison.py`, `src/remote_models.py`, `src/rome.py`, `src/rome_experiment.py`, tests associés). Ces éléments préexistaient à cette mise à jour et ne sont pas des changements effectués par cette tâche.
- Les commits récents s'arrêtent au 30 août 2026 (`e42f36b`, « Finalise le PoC de rapprochement des compétences »). Les évolutions ROME/distantes actuellement visibles sont dans l'arbre de travail, pas dans un commit inspecté.

## 9. Tests et validation

**Commande exécutée pendant cette inspection :**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider
```

Résultat vérifié : **151 passed** en 3,07 s, 151 tests collectés. Les tests utilisent des faux encodeurs, PDF simulés et réponses HTTP simulées ; ils ne chargent pas le vrai BGE-M3 et ne contactent pas d'endpoint distant configuré.

La formulation « 115 tests réussis » dans `AGENTS.md`, README, cadrage et passation est donc périmée par rapport à l'état local inspecté. Les 127 fonctions nommées `test_*` ne correspondent pas au décompte pytest, car la suite contient aussi des tests paramétrés.

Tests encore nécessaires :

1. tests sur PDF réels anonymisés/reproductibles pour les deux formats Talentsoft et une fiche ROME chargée dans Streamlit ;
2. test automatisé de l'interface Streamlit en interaction réelle (les tests actuels couvrent l'import et les fonctions de restitution, pas un parcours navigateur complet) ;
3. intégration manuelle, explicitement hors pytest, du BGE local réel ;
4. contrôles manuels des endpoints actuels, puis scénario RH complet Qwen et Gemma ;
5. si une API sparse BGE devient disponible, tests de contrat, puis campagne BGE hybride distincte.

## 10. Problèmes connus, décisions à prendre et prochaines étapes

1. **Décider si les changements non validés doivent être relus et commités.** L'arbre est largement modifié ; ne pas mélanger une nouvelle évolution avec ce lot sans revue. Les fichiers de sortie et caches ROME doivent être préservés.
2. **Valider les PDF réels.** L'extraction est seulement attestée par simulés dans pytest ; la couverture d'exemples réels est à établir sans ajouter les PDF au Git.
3. **Tester réellement Qwen et Gemma.** Le code et les mocks existent ; la disponibilité actuelle du réseau/services et le comportement Gemma de bout en bout sont à vérifier. Un échec d'un moteur doit rester isolé dans la comparaison.
4. **Ne pas déduire de paramètres de production à partir de ROME seul.** Les résultats v1/v2 dépendent fortement de la table de référence et les non-relations ne sont pas des négatifs métiers certains.
5. **Pour BGE hybride ROME : obtenir d'abord une sortie sparse réellement exploitable côté API.** Ensuite seulement adapter le client expérimental, créer cache/répertoire séparés, lancer la campagne et comparer BGE dense/hybride/Qwen dense.
6. **Compléter l'évaluation méthodologique** (analyse qualitative, métriques de classement) avant toute calibration RH ; Gemma doit être évalué séparément sans seuil ni score hybride.

## Informations indispensables pour reprendre le projet

Lire d'abord `AGENTS.md`, puis ce document, `docs/Feuille_cadrage_IA.docx`, README et `docs/PASSATION_IA_PROJET.md`. Le PoC compare des **profils d'emploi**, non des personnes. Ne changez jamais silencieusement l'extraction, les formules, seuils, poids ou dépendances.

Le moteur interne BGE local est dense+sparse, avec `H = 2/3 D + 1/3 L` par défaut, niveaux séparés et sélection par `G`, écart moyen puis `Gs`. Qwen est dense-only (`H=D`) ; Gemma est un LLM catégoriel, sans score ni seuil. Garder cette séparation dans le code, l'affichage et les exports.

Les deux zones Streamlit imposent le type PDF. Les formats Talentsoft/tableau sont testés avec simulés ; valider des fichiers réels avant de déclarer l'extraction achevée. Les fiches ROME chargées dans l'application ne gardent que `Savoir-faire principaux` et `Domaines d’expertise`, sans niveau.

La suite actuelle vérifiée est `151 passed` avec la commande de section 9, sans modèle ni serveur réels. Les validations réelles BGE/Qwen/Gemma restent manuelles et hors pytest. Les endpoints par défaut sont dans `src/config.py`; aucune clé ne doit être écrite ni exportée.

ROME est une calibration isolée : v2 modifie la référence, pas les embeddings. Ne pas appliquer ses seuils à la production sans décision. Préserver `outputs/rome/` et les caches. Avant tout travail, inspecter `git status` : l'arbre est déjà non propre et les changements ROME/distants visibles ne sont pas tous commités. Traiter d'abord leur revue/commit, les PDF réels et les contrôles distants.
