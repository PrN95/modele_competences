# PoC local de rapprochement des emplois

L'outil développé dans ce projet est un **Proof of Concept**, restreint dans son application mais qui a pour objectif de démontrer l’intérêt de l'utilisation des modèles d’embedding pour soutenir l’analyse des compétences en entreprise et guider les décisions RH.

Pour chaque emploi actuel du référentiel de l’entreprise, le PoC compare les compétences numériques et les niveaux de maîtrise associés à ceux de plusieurs emplois cibles représentant des évolutions professionnelles possibles. Il identifie le ou les emplois cibles les plus proches, met en évidence les compétences absentes ou insuffisamment maîtrisées et fournit une première indication des besoins de formation correspondants.

Un emploi représente toujours un profil type, jamais une personne. 

Plusieurs emplois actuels peuvent être associés au même emploi cible. 

L'outils a été développé à avec l'utilisation d'assistant IA (Codex avec GPT - 5.6 Terra).

Les sources d'instructions pour l’assistance IA sont dans `AGENTS.md` et `docs/Feuille_cadrage_IA.docx`.

## Expérimentation ROME

Cette expérimentation évalue les résultats du modèle à partir d’une table de vérité constituée manuellement à partir de paires de fiches emploi issues du référentiel ROME de France Travail.
Le module `src.rome` prépare un corpus pour les campagnes BGE-M3 et Qwen Embedding, sans modifier le matching de production qui n'obéit pas à la même logique que l'évaluation sur la table de vérité. La table de vérité de référence par défaut est `data/rome/Similarité Emplois ROME 6.xlsx` ; Les relations sont non directionnelles et le code ROME extrait de l'en-tête PDF est la clé de rapprochement.

```bash
.venv/bin/python scripts/validate_rome_corpus.py
```

La commande crée `outputs/rome/corpus_rome.csv` et `outputs/rome/rapport_validation_rome.csv`. Elle retourne un code 1 en cas de PDF manquant, de code PDF absent/incohérent ou de rubrique `Compétences` vide, tout en générant les résultats des autres fiches. Les écarts d'intitulé Excel/PDF sont des avertissements : le code ROME prévaut. En cas de doublon de code ROME, le PDF nommé `CODE - intitulé.pdf` est systématiquement retenu ; plusieurs candidats portant cette forme restent une erreur à résoudre.

### Première campagne BGE-M3 dense

Après validation du corpus, lancer la campagne expérimentale séparée :

```bash
.venv/bin/python scripts/run_rome_bge_dense_experiment.py
```

Elle encode chaque compétence une seule fois via BGE-M3 distant, en dense-only (cosinus, coefficient dense `1,0`, sparse `0,0`), met en cache les embeddings et la matrice cosinus, puis teste exactement 64 combinaisons des seuils `0,20` à `0,90`. Chaque exécution crée exclusivement son propre dossier `outputs/rome/experiments_2/<nom_de_campagne>/`, avec le résumé des configurations, le détail par paire, les métadonnées, le cache et `rapport_synthese.md`. Le nom est horodaté par défaut et peut être fixé avec `--campaign-name`; un dossier existant est refusé. La sélection maximise F1, puis le rappel, puis la précision. Les paires absentes du graphe ROME sont des `non_relation_reference`, pas une vérité humaine absolue.

Les endpoints distants sont configurables avec `REMOTE_BGE_DISTANT_BASE_URL`, `REMOTE_QWEN_DISTANT_BASE_URL` et `REMOTE_GEMMA_DISTANT_BASE_URL`. Ils sont facultatifs et ne sont pas nécessaires aux campagnes purement locales. Une clé facultative peut être fournie par `REMOTE_API_KEY`, ou précisément par `REMOTE_BGE_API_KEY`, `REMOTE_QWEN_API_KEY` et `REMOTE_GEMMA_API_KEY`. Le client fonctionne sans clé. Le timeout se règle avec `REMOTE_TIMEOUT_SECONDS`.

Vérifier les trois services (découverte par `GET /models`) :

```bash
.venv/bin/python scripts/test_remote_models.py
```

La campagne vérifie BGE avant le moindre encodage. La taille de lot expérimentale est `32` par défaut, indépendante du batch de production (`1`) ; elle se règle avec `ROME_EXPERIMENT_BATCH_SIZE` ou `--batch-size` :

```bash
.venv/bin/python scripts/run_rome_bge_dense_experiment.py --batch-size 32
```

La campagne Qwen3-Embedding-8B applique exactement le même protocole dense-only et écrit dans un cache et un répertoire de résultats séparés :

```bash
.venv/bin/python scripts/run_rome_qwen_dense_experiment.py --batch-size 32
```

## Entrées et modèle métier

L'entrée principale est constituée de deux formats de PDF natifs structurés, traités par deux extracteurs `pdfplumber` distincts. Le type n'est jamais deviné : la zone de dépôt Streamlit impose l'extracteur d'emploi actuel ou l'extracteur d'emploi cible.

- L'extracteur d'emploi actuel lit l'intitulé et uniquement la section « Compétences digitales ».
- L'extracteur d'emploi cible lit l'intitulé et le tableau composé des colonnes « Intitulé de la compétence », « Détails de la compétence » et « Niveau ».
- Les deux extracteurs produisent la même structure normalisée : `competence_intitule`, `competence_description` et `niveau`.
- L'entrée Excel reste disponible comme compatibilité secondaire et comme support de test.

Les niveaux autorisés sont :

- 1 : Sensibilisation ;
- 2 : Application ;
- 3 : Maîtrise ;
- 4 : Expertise.

Le niveau 0 est exclusivement interne et représente une compétence cible non reconnue. Le texte envoyé au modèle contient l'intitulé et, lorsqu'elle existe, la description ; il ne contient jamais le niveau.

## BGE-M3 local et scores sémantiques

Le modèle local `models/bge-m3` est chargé hors ligne via `FlagEmbedding.BGEM3FlagModel`. Le premier PoC active les représentations dense et sparse et n'utilise pas ColBERT ni de reranker.

Les scores officiels sont :

```text
D_ac = produit scalaire des vecteurs denses normalisés

L_ac = somme_tokens_communs(x_t × y_t)
       / (sqrt(somme_t x_t²) × sqrt(somme_t y_t²))

H_ac = poids_dense × D_ac + poids_sparse × L_ac
```

`L_ac` est une similarité cosinus avec normalisation L2 des vecteurs sparse. Pour chaque compétence cible, le moteur conserve la compétence actuelle ayant le `H_ac` maximal. Une même compétence actuelle peut couvrir plusieurs compétences cibles.

Les valeurs de référence par défaut sont initialement `poids_dense = 2/3` et `poids_sparse = 1/3`. L'interface Streamlit permet de modifier les deux poids manuellement et indépendamment pour expérimenter (par exemple avec `0,50 / 0,50` ou `0,80 / 0,20`). Chaque valeur doit être comprise entre `0` et `1` et leur somme doit valoir `1` avec une tolérance numérique adaptée. Une somme invalide affiche une erreur et bloque l'analyse ; aucune normalisation automatique n'est appliquée. Les valeurs utilisées sont affichées et enregistrées dans tous les exports CSV.


## Comparaison expérimentale de trois approches

Le PoC peut exécuter un seul moteur ou comparer les trois moteurs sur les mêmes PDF et les mêmes seuils :

- **BGE-M3 local** reste l'approche d'embedding dense + sparse avec score hybride paramétrable.
- **Qwen3-Embedding-8B distant** utilise l'API `REMOTE_QWEN_DISTANT_BASE_URL/embeddings` et l'identifiant `qwen3-embedding-8b`. Le service fournit uniquement des vecteurs denses de dimension 4096. Ils sont normalisés L2 localement, puis comparés par cosinus avec `H = D`, `poids_dense = 1,00` et `poids_sparse = 0,00`. Le sparse reste `N/A`.
- **Gemma distant**, présenté dans l'interface comme l'approche « Gemma 4 », est un LLM classique et non un modèle d'embedding. Le service `REMOTE_GEMMA_DISTANT_BASE_URL/chat/completions` annonce l'identifiant exact `gemma4`. Pour chaque emploi actuel, il sélectionne directement un emploi cible parmi ceux proposés, puis retourne pour chaque compétence de cette cible la compétence actuelle correspondante ou `N/A`, avec le statut catégoriel `Reconnue` ou `Absente`.

En mode Gemma, le LLM reçoit les compétences digitales et les niveaux SAME. Il sélectionne directement une cible, puis calcule lui-même les correspondances, `G_epfq`, `Gs_epfq`, écart moyen, `R_epfq` et les recommandations. Ces indicateurs sont uniquement informatifs et ne peuvent ni exclure ni reclasser la cible choisie. L'application ne recalcule ni ne corrige ces éléments : elle valide le contrat JSON, conserve la réponse brute et la restitue. Aucun score d'embedding, seuil de similarité, seuil de couverture, poids dense ou poids sparse n'est appliqué pour Gemma. Chaque moteur reste isolé : une indisponibilité, une erreur API ou une réponse LLM invalide n'empêche pas l'affichage des autres résultats.

## Seuils, scoring et sélection

Streamlit expose deux curseurs distincts :

- `seuil_sim`, utilisé pour reconnaître une compétence cible ;
- `seuil_couv`, utilisé pour déterminer l'admissibilité d'un emploi cible.

Leurs valeurs de référence par défaut sont intialement `0,70`, avec un pas de `0,05`. Les valeurs choisies sont transmises au moteur, affichées dans les résultats et conservées dans les exports. Leur calibration sert notamment à étudier le compromis entre faux positifs et faux négatifs.

Pour un emploi actuel `e` et un emploi cible `f` :

```text
Ecart_c = max(0, niveau_cible_c - niveau_actuel_c)

Ecart_moyen_epfq = somme_c(Ecart_c × niveau_cible_c)
                    / somme_c(niveau_cible_c)

G_epfq = nombre_competences_cibles_reconnues
         / nombre_total_competences_cibles

Gs_epfq = nombre_competences_entierement_satisfaites
          / nombre_total_competences_cibles
```

Une compétence est entièrement satisfaite si `H_ac >= seuil_sim` et si son niveau actuel atteint le niveau cible. Dans le moteur actuel, `G_epfq` et `Gs_epfq` sont exposés sous les propriétés historiques `g_ef` et `gs_ef`.

Les emplois cibles dont `G_epfq < seuil_couv` sont exclus. Les cibles admissibles sont départagées dans cet ordre :

1. `G_epfq` décroissant ;
2. `Ecart_moyen_epfq` croissant ;
3. `Gs_epfq` décroissant.

Après triple égalité, toutes les cibles ex aequo sont conservées pour arbitrage RH. Les recommandations sont calculées uniquement après cette sélection et doivent rester limitées aux emplois cibles sélectionnés.

`R_epfq` est un indicateur informatif calculé après la sélection, séparément pour chaque cible retenue. Le taux est le nombre de compétences actuelles réutilisées divisé par leur nombre total. Il est affiché en pourcentage et ne participe ni à l'admissibilité, ni au classement, ni au choix d'un emploi cible.

## Interface Streamlit et exports

L'interface Streamlit permet :

- le dépôt séparé de plusieurs PDF d'emplois actuels et de plusieurs PDF d'emplois cibles ;
- l'analyse globale de tous les emplois actuels contre toutes les cibles ;
- l'appel au modèle BGE-M3 local par défaut, et aux modèles distants via l'url des endpoints
- le réglage des deux seuils ;
- le réglage indépendant de `poids_dense` et `poids_sparse`, avec validation de leur somme avant l'analyse ;
- le choix entre un seul modèle et la comparaison BGE-M3 / Qwen / Gemma ;
- la configuration des URL, identifiants et timeouts distants, une clé API optionnelle non persistée et deux boutons de test de connexion ;
- l'affichage de `G_epfq`, `Gs_epfq`, de l'écart moyen pondéré, de `R_epfq` et des correspondances de compétences ;
- l'affichage des niveaux insuffisants, des recommandations et du contrôle informatif des compétences actuelles non reprises ;
- les exports CSV consolidés de la comparaison, de la matrice complète et des détails des seules cibles retenues, avec la colonne `modele`, les statuts, les temps et les scores réellement disponibles.

Lancement local :

```bash
.venv/bin/streamlit run app.py
```

## Limites actuelles

Les tests avec ColBERT activé et avec ajout d'un reranker n'ont finalement pas été effectués.
