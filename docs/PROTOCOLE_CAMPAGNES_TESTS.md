# Protocole des campagnes de test ROME

## Règles communes

* Les campagnes sont réalisées **une par une**. Une campagne suivante ne doit être lancée qu’après analyse et validation des résultats de la précédente.
* La table de vérité de référence est `data/rome/Similarité Emplois ROME 6.xlsx`, onglet `table-verite`.
* Elle contient 150 paires d’emplois ROME uniques, annotées « Oui » ou « Non ».
* Les annotations de la table de vérité ne doivent jamais être communiquées au modèle évalué, notamment à Gemma4.
* Les campagnes 1 à 3 utilisent une similarité non orientée : la couverture de A vers B et celle de B vers A sont calculées puis moyennées.
* Cette moyenne bidirectionnelle est utilisée uniquement pour l’évaluation sur la table de vérité. Dans l’interface Streamlit, la recommandation reste orientée : elle mesure la couverture de l’emploi actuel par un emploi cible.
* Chaque campagne crée ses sorties dans `outputs/rome/experiments_2/<nom_de_la_campagne>/`, sans modifier les sorties existantes dans `outputs/rome/experiments/`.
* Chaque dossier de sortie doit contenir au minimum un `rapport_synthese.md`, les résultats détaillés en CSV, les métadonnées de configuration et les graphiques éventuels.

## Campagne 1 : `campagne_dense_only`

**Objectif :** établir une référence de performance en utilisant uniquement la similarité dense.

Les emplois ne sont pas comparés à partir de leurs intitulés, mais à partir des compétences extraites de leurs fiches. Pour chaque paire d’emplois de la table de vérité, le modèle calcule les similarités denses entre les compétences de l’emploi A et celles de l’emploi B.

Les paramètres sont fixés ainsi :

* `poids_dense = 1`
* `poids_sparse = 0`

Deux seuils sont ensuite testés :

* `seuil_sim`, qui détermine si deux compétences sont considérées comme correspondantes ;
* `seuil_couv`, qui détermine si la couverture globale entre les deux emplois est suffisante pour conclure qu’ils sont similaires.

Pour chaque combinaison de seuils, le modèle applique la logique suivante :

1. Il calcule le score dense entre les compétences.
2. Il considère qu’une compétence est reconnue si son score dense dépasse `seuil_sim`.
3. Il calcule la couverture de A vers B et de B vers A.
4. Il moyenne ces deux couvertures.
5. Il considère les emplois comme similaires si la couverture moyenne dépasse `seuil_couv`.

Pour chaque combinaison de seuils, le système prédit « Oui » ou « Non » pour chaque paire d’emplois. Cette prédiction est comparée au label humain de la table de vérité.

Les métriques calculées sont :

* accuracy ;
* précision ;
* rappel ;
* F1 ;
* TP, FP, FN et TN.

La combinaison `seuil_sim` et `seuil_couv` qui obtient le meilleur F1 est retenue comme référence dense seule.

Cette campagne répond à la question suivante : *quelle est la performance du modèle lorsqu’il utilise uniquement la représentation dense ?*

La campagne est réalisée séparément avec :

* BGE-M3 dense ;
* Qwen3-Embedding dense.

Qwen3-Embedding ne participe pas aux campagnes 2 et 3, car il ne fournit pas de représentation sparse native dans la configuration retenue.

## Campagne 2 : `campagne_hybride_auc_roc`

**Objectif :** déterminer la combinaison de poids dense/sparse qui distingue le mieux les paires d’emplois similaires des paires non similaires, sans dépendre d’un seuil de décision.

Cette campagne concerne uniquement BGE-M3, qui fournit les scores denses et sparse.

Un balayage est réalisé avec un pas de `0,10` pour le poids dense. Le poids sparse est déduit automatiquement :

* dense = 1,0 ; sparse = 0,0 ;
* dense = 0,9 ; sparse = 0,1 ;
* dense = 0,8 ; sparse = 0,2 ;
* ... ;
* dense = 0,0 ; sparse = 1,0.

Pour chaque combinaison de poids, le score hybride entre deux compétences est calculé :

$$
H_{a_i,c_j} =
(poids_{dense} \times D_{a_i,c_j}) +
(poids_{sparse} \times L_{a_i,c_j})
$$

avec :

* $D_{a_i,c_j}$ : similarité dense entre une compétence de l’emploi A et une compétence de l’emploi B ;
* $L_{a_i,c_j}$ : similarité sparse entre ces mêmes compétences.

Ensuite, un **score continu hybride d’emploi** est calculé pour chaque paire d’emplois :

1. Pour chaque compétence de l’emploi B, le système identifie la compétence de l’emploi A ayant le meilleur score hybride.
2. Il calcule la moyenne de ces meilleurs scores. Cela donne le score de proximité de A vers B.
3. Il réalise le même calcul de B vers A.
4. Il fait la moyenne des deux scores.

Cette moyenne bidirectionnelle constitue le **score continu hybride** de la paire d’emplois.

À ce stade, aucun `seuil_sim` ni `seuil_couv` n’est utilisé. Le système ne prédit pas encore « Oui » ou « Non ».

Pour chaque combinaison de poids, l’AUC-ROC compare les scores continus hybrides aux labels humains de la table de vérité.

Elle répond à la question suivante : *avec quels poids les paires réellement similaires reçoivent-elles globalement de meilleurs scores que les paires non similaires ?*

La combinaison de poids ayant l’AUC-ROC la plus élevée est retenue :

$$
(poids_{dense}^{*}, poids_{sparse}^{*})
$$

Cette campagne permet d’isoler le choix des poids, sans qu’un seuil de décision favorable influence artificiellement le résultat.

## Campagne 3 : `campagne_hybride_final`

**Objectif :** déterminer les seuils de décision les plus adaptés une fois les poids dense/sparse fixés.

Les paramètres fixés sont ceux retenus en campagne 2 :

* `poids_dense = poids_dense optimal retenu`
* `poids_sparse = 1 - poids_dense optimal retenu`

Le système fait ensuite varier :

* `seuil_sim` ;
* `seuil_couv`.

Pour chaque combinaison de seuils, le modèle applique la logique suivante :

1. Il calcule le score hybride entre les compétences.
2. Il considère qu’une compétence est reconnue si son score hybride dépasse `seuil_sim`.
3. Il calcule la couverture de A vers B et de B vers A.
4. Il moyenne ces deux couvertures.
5. Il considère les emplois comme similaires si la couverture moyenne dépasse `seuil_couv`.

Les prédictions sont comparées aux labels humains. Les métriques calculées sont :

* accuracy ;
* précision ;
* rappel ;
* F1 ;
* TP, FP, FN et TN.

La configuration finale retenue est :

$$
(poids_{dense}^{*}, poids_{sparse}^{*}, seuil_{sim}^{*}, seuil_{couv}^{*})
$$

Elle est comparée à la configuration dense seule de la campagne 1 afin de mesurer l’apport réel du sparse.

## Campagne 4 : `campagne_gemma4`

**Objectif :** évaluer la capacité de Gemma4 à apprécier directement la proximité entre deux métiers à partir de leurs compétences extraites, sans règles de calcul explicites, contrairement au modèle d’embedding.

Cette campagne utilise le même jeu de test que les campagnes précédentes. Gemma4 reçoit les paires d’emplois ROME ainsi que les compétences associées à chaque emploi. Les annotations « Oui » ou « Non » de la table de vérité ne lui sont pas communiquées.

Pour chaque paire, Gemma4 doit décider directement si les deux métiers sont similaires au regard de leurs compétences. Il ne reçoit ni score de similarité, ni seuil de décision, ni consigne de calcul de couverture.

Il produit uniquement une décision binaire :

* « Oui » lorsqu’il estime qu’une proximité suffisante existe ;
* « Non » dans le cas contraire.

Les résultats sont enregistrés dans un fichier CSV comportant les deux emplois et la décision produite par le modèle.

Les décisions de Gemma4 sont ensuite comparées, séparément, aux annotations de la table de vérité. Le programme calcule :

* accuracy ;
* précision ;
* rappel ;
* F1 ;
* TP, FP, FN et TN.

La température doit être fixée à `0` afin de rendre les réponses aussi reproductibles que possible.

Cette campagne compare donc un modèle d’embedding fondé sur des scores et des règles contrôlables avec un LLM qui produit une décision à partir de son interprétation globale des compétences.

L’AUC-ROC n’est pas calculée pour Gemma4, car le modèle retourne uniquement une réponse binaire « Oui » ou « Non », sans score continu.
