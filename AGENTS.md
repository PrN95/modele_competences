# Cadrage du projet

## Source de vérité et objectif

Construire un PoC local d'aide à la décision RH qui compare un ou plusieurs emplois actuels à plusieurs emplois cibles, retient les emplois cibles admissibles, explique les écarts de compétences et de niveaux et recommande les formations nécessaires.

La référence fonctionnelle et technique détaillée est `docs/Feuille_cadrage_IA.docx`. Le présent fichier contient les règles impératives pour le développement. Un emploi décrit un profil type, jamais un collaborateur individuel. Plusieurs emplois actuels peuvent être associés au même emploi cible.

## Entrées PDF et niveaux

- Utiliser deux extracteurs `pdfplumber` distincts. Le type est imposé par la zone de chargement Streamlit, « emploi actuel » ou « emploi cible » ; ne jamais le deviner.
- Emploi actuel : PDF natif structuré ; extraire l'intitulé dans l'en-tête et uniquement les compétences de la section « Compétences digitales » (intitulé noir, description dessous, niveau à droite). Ignorer responsabilités, compétences fonctionnelles, autres sections, titres thématiques bleus et l'en-tête « Niveau SAME cible ». Gérer les contenus coupés entre pages.
- Emploi cible : PDF natif structuré en tableau ; extraire l'intitulé au-dessus du tableau et les colonnes « Intitulé de la compétence », « Détails de la compétence » et « Niveau ». Gérer les tableaux multipages, même sans répétition de l'en-tête des colonnes.
- Les deux extracteurs produisent `competence_intitule`, `competence_description`, `niveau`. Aucun identifiant d'emploi ou de compétence n'est obligatoire. Ne jamais ajouter les PDF d'exemple au dépôt Git.
- Construire le texte encodé avec `competence_intitule + ". " + competence_description` si la description existe, sinon avec le seul intitulé. Le niveau et les métadonnées restent séparés.
- Niveaux autorisés : 1 = Sensibilisation, 2 = Application, 3 = Maîtrise, 4 = Expertise. Le niveau 0 est exclusivement interne lorsqu'une compétence cible n'est pas reconnue.

## BGE-M3 local et rapprochement sémantique

- Utiliser le modèle local `models/bge-m3` via `FlagEmbedding.BGEM3FlagModel`, avec dense et sparse activés, ColBERT et reranker désactivés. Ne jamais importer ni utiliser directement `sentence-transformers`.
- L'exécution réelle hors ligne est validée : vecteurs denses de dimension 1024 ; premier chargement à froid et première inférence d'environ 22 secondes ; inférence sur quatre compétences après chargement d'environ 0,86 seconde ; mémoire maximale observée d'environ 1,97 Gio, sans swap.
- `D_ac` est le produit scalaire des vecteurs denses normalisés.
- `L_ac` est la similarité cosinus L2 entre les vecteurs sparse : `L_ac = somme_tokens_communs(x_t * y_t) / (sqrt(somme_t x_t^2) * sqrt(somme_t y_t^2))`.
- Appliquer exactement `H_ac = poids_dense * D_ac + poids_sparse * L_ac`. Les valeurs de référence par défaut sont exactement `poids_dense = 2/3` et `poids_sparse = 1/3`.
- `poids_dense` et `poids_sparse` sont réglables manuellement et indépendamment dans Streamlit à des fins d'expérimentation. Chaque poids doit appartenir à `[0, 1]` et leur somme doit être égale à 1 avec une tolérance numérique adaptée. Ne jamais normaliser automatiquement les valeurs saisies ; une somme invalide bloque l'analyse avec une erreur claire.
- Les poids effectivement choisis doivent être transmis à tous les calculs utilisant `H_ac`, affichés dans les résultats et conservés dans tous les exports CSV afin de rendre chaque expérience reproductible.
- Pour chaque compétence cible, retenir la compétence actuelle ayant le `H_ac` maximal. En cas d'égalité exacte : niveau actuel le plus élevé, puis `L_ac` le plus élevé, puis choix déterministe avec signalement de l'égalité dans le détail.
- Une compétence cible est reconnue si `H_ac >= seuil_sim`; sinon son niveau actuel interne vaut 0. Le niveau ne modifie jamais `D_ac`, `L_ac` ou `H_ac`. Une compétence actuelle peut couvrir plusieurs compétences cibles et cette réutilisation reste visible.

## Expérimentation comparative Qwen et Gemma

- L'interface propose « Un seul modèle » ou « Comparaison des trois modèles ». La comparaison exécute BGE-M3 local, Qwen distant et Gemma distant avec les mêmes PDF. `seuil_sim` et `seuil_couv` ne s'appliquent qu'aux moteurs d'embedding. L'échec d'un moteur ne masque jamais les résultats des autres.
- Le service Qwen réel est OpenAI-compatible à l'URL de base `REMOTE_QWEN_DISTANT_BASE_URL` en minuscules. Il annonce `qwen3-embedding-8b` sur `GET /models` et accepte `POST /embeddings`. Les vecteurs denses observés ont une dimension de 4096. Aucun score sparse n'est fourni ni inventé : les vecteurs sont normalisés L2 localement, `H = D`, `poids_dense = 1,00`, `poids_sparse = 0,00`, et sparse est restitué comme « N/A ».
- Le service de jugement LLM est OpenAI-compatible à l'URL de base `REMOTE_GEMMA_DISTANT_BASE_URL` et annonce l'identifiant exact `gemma4` sur `GET /models`. Il accepte `POST /chat/completions` avec `response_format={"type": "json_object"}`. Il ne doit jamais être présenté comme un modèle d'embedding.
- En mode Gemma, envoyer l'emploi actuel et tous les emplois cibles avec leurs compétences digitales et niveaux SAME. Gemma sélectionne directement une cible, puis réalise lui-même les correspondances, calcule `G_epfq`, `Gs_epfq`, l'écart moyen et `R_epfq`, et produit les recommandations. Ces indicateurs sont uniquement informatifs : ils ne modifient jamais la cible sélectionnée. L'application valide uniquement le JSON, conserve la réponse brute et la restitue ; elle ne recalcule ni ne modifie ce raisonnement.
- Aucun score d'embedding, seuil de similarité, seuil de couverture ou poids dense/sparse n'est appliqué à Gemma.
- Les URL, identifiants, timeouts et une éventuelle clé API sont configurables dans Streamlit. Aucune clé n'est codée en dur ou exportée. Des boutons testent les connexions Qwen et Gemma.
- Les restitutions et CSV distinguent explicitement les scores d'embedding des décisions catégorielles Gemma et conservent le modèle, les seuls paramètres applicables, le temps, le statut et les erreurs.

## Seuils, scoring et sélection

- `seuil_sim` et `seuil_couv` sont configurables dans Streamlit par deux curseurs. Leurs valeurs de référence par défaut sont 0,70, avec un pas de 0,05.
- Pour BGE-M3 et Qwen, les valeurs effectivement choisies doivent être transmises au moteur, affichées dans les résultats et conservées dans les exports. Leur calibration reste expérimentale afin d'étudier le compromis entre faux positifs et faux négatifs. Elles sont sans objet pour Gemma.
- Pour chaque emploi actuel `e`, emploi cible `f` et compétence cible `c`, appliquer `Ecart_c = max(0, niveau_cible_c - niveau_actuel_c)`.
- Appliquer `Ecart_moyen_ef = somme_c(Ecart_c * niveau_cible_c) / somme_c(niveau_cible_c)`.
- Appliquer `G_ef = nombre_competences_cibles_reconnues / nombre_total_competences_cibles` ; le niveau n'intervient pas dans `G_ef`.
- Appliquer `Gs_ef = nombre_competences_entierement_satisfaites / nombre_total_competences_cibles`, avec satisfaction entière si `H_ac >= seuil_sim` et `niveau_actuel_c >= niveau_cible_c`.
- Exclure toute cible avec `G_ef < seuil_couv`. S'il n'en reste aucune, afficher exactement « Aucun emploi cible ne correspond à cet emploi actuel ».
- Parmi les cibles admissibles, départager par `G_ef` décroissant, puis `Ecart_moyen_ef` croissant, puis `Gs_ef` décroissant. Après triple égalité, conserver toutes les cibles ex aequo et indiquer que le service RH devra trancher.
- Plusieurs emplois actuels peuvent être associés au même emploi cible.

## Résultats, recommandations et contrôle informatif

- Produire les recommandations uniquement après sélection et uniquement pour les emplois cibles sélectionnés : pour les embeddings, `H_ac < seuil_sim`, et pour Gemma, statut `Absente` → « Formation complète nécessaire pour acquérir la compétence » ; `Ecart_c = 0` → aucun commentaire particulier ; `Ecart_c = 1` → « Formation légère pour progresser d'un niveau » ; `Ecart_c = 2` ou `3` → « Parcours de formation conséquent ».
- Afficher le détail des correspondances, scores, compétences absentes, niveaux insuffisants, niveaux à atteindre et recommandations ; permettre l'export CSV.
- Signaler les compétences actuelles non reprises dans les emplois cibles analysés, sans conclure à leur obsolescence. Ce contrôle informatif ne modifie jamais `H_ac`, `G_ef`, `Gs_ef`, `Ecart_moyen_ef` ni la sélection.
- Calculer `R_epfq` uniquement après la sélection et séparément pour chaque emploi cible retenu `f_q`. Pour chaque compétence actuelle `a_i` de l'emploi actuel `e_p`, calculer `Meilleur_score_aiepfq = max_cj_appartenant_a_fq(H_aicj)` parmi les seules compétences de `f_q` ; la compétence actuelle est réutilisée si `Meilleur_score_aiepfq >= seuil_sim`.
- Appliquer `R_epfq = nombre_competences_actuelles_de_ep_reutilisees_dans_fq / nombre_total_competences_actuelles_de_ep`. Pour Gemma, une compétence actuelle est réutilisée si elle apparaît dans au moins une correspondance catégorielle `Reconnue` de la cible déjà sélectionnée. `R_epfq` est un réel dans `[0, 1]`, affiché en pourcentage. Il ne dépend pas des niveaux et ne modifie jamais `G_ef`, `Gs_ef`, `Ecart_moyen_ef`, l'admissibilité ou la sélection.

## État implémenté, tests et limites

- Sont implémentés : les deux extracteurs PDF distincts ; BGE-M3 local dense+sparse ; le rapprochement ; les niveaux et recommandations ; le scoring avec admissibilité, `G_ef`, `Gs_ef`, écart moyen pondéré et départage ; l'interface Streamlit avec import PDF, résultats et export CSV.
- Les 115 tests automatisés réussissent. Ils utilisent des faux encodeurs, des réponses HTTP simulées et des PDF simulés. Ils ne doivent jamais charger le véritable modèle BGE-M3 ni contacter un endpoint distant configuré.
- Le test d'intégration du véritable BGE-M3 reste manuel, explicite, hors ligne et séparé de pytest.
- L'interface orchestre automatiquement tous les emplois actuels chargés contre tous les emplois cibles chargés, sélectionne séparément pour chaque emploi actuel et calcule `R_epfq` après sélection pour chaque cible retenue.
- Il n'existe pas encore de test automatisé Streamlit, de test BGE-M3 intégré à pytest ni de test sur PDF réels.
- Les recommandations et leur export restent limités aux emplois cibles sélectionnés. Les seuils et poids hybrides utilisés, le statut de sélection et le détail des égalités sont restitués et exportés.

## Périmètre technique et règles de travail

- Premier PoC : Python, `pdfplumber`, `pandas`, `openpyxl`, `FlagEmbedding`, PyTorch, BGE-M3 local dense+sparse, rapprochement, niveaux, recommandations et interface Streamlit. Fonctionnement hors ligne après installation ; chemin local du modèle configurable ; chargement unique à la première utilisation par instance.
- Améliorations ultérieures : ColBERT, reranker, cohortes automatiques de 12 personnes, création automatique d'emplois cibles et évaluation du risque d'obsolescence.
- Séparer la logique métier de Streamlit et centraliser chemins, poids et valeurs par défaut dans la configuration.
- Ajouter des tests à chaque incrément, préserver la traçabilité des sources et calculs, et ne jamais modifier silencieusement une formule, un seuil, une règle d'extraction ou une dépendance.
- En cas d'ambiguïté ou de contradiction, arrêter l'implémentation concernée et demander une décision.
