# Fiche de préparation — campagnes ROME hybrides dense + sparse

## Objet et périmètre

Ces campagnes évaluent l'apport d'un score hybride sur le corpus ROME. Elles restent séparées du moteur RH de production et de Streamlit : elles ne modifient ni les paramètres proposés aux utilisateurs, ni les règles de sélection des emplois internes.

Les campagnes 2 et 3 ne pourront commencer pour un modèle que si son serveur distant expose effectivement une représentation sparse exploitable, en plus du vecteur dense. Une réponse qui ne contient que `embedding` conserve le protocole dense-only actuel ; elle ne permet pas de déclarer une campagne hybride. À ce jour, Qwen ne fournit pas cette représentation sparse.

Pour deux compétences `a` et `c`, le score testé est :

`H_ac = poids_dense × D_ac + poids_sparse × L_ac`

où `D_ac` est le cosinus dense des vecteurs L2-normalisés et `L_ac` le cosinus sparse. Les niveaux des compétences ne participent pas à cette calibration ROME.

## Paramètres communs aux trois campagnes

- Le corpus ROME, sa version et son empreinte doivent être identiques à ceux de la comparaison dense-only.
- Les relations Excel ROME sont des références non directionnelles. Une paire absente est classée `non_relation_reference` pour calculer les métriques ; ce n'est pas une preuve d'incompatibilité métier.
- Pour une paire d'emplois A/B, une compétence est couverte lorsqu'elle trouve une compétence de l'autre emploi avec `H_ac >= seuil_sim`.
- On calcule les couvertures A vers B et B vers A, puis leur moyenne : la couverture symétrique.
- La paire est prédite liée lorsque sa couverture symétrique est au moins égale à `seuil_couv`.
- Les embeddings dense et sparse, ainsi que les matrices de similarité nécessaires, sont calculés une fois puis mis en cache dans un répertoire propre à la campagne et à l'empreinte du corpus. Chaque configuration réutilise ce cache.
- Une configuration est évaluée avec TP, FP, FN, TN, précision, rappel et F1. Le classement reste : F1 décroissant, puis rappel décroissant, puis précision décroissante.
- Chaque export conserve les poids, seuils, version/identifiant du modèle, endpoint, empreinte du corpus, durée, résultats par configuration et détail par paire.

## Campagne 1 — calibration dense-only des seuils

### But

Établir d'abord une référence dense-only et ses seuils optimaux. Les poids sont fixes : `poids_dense = 1,00` et `poids_sparse = 0,00`.

### Paramètres variables et volume

`seuil_sim` et `seuil_couv` sont testés indépendamment aux valeurs `0,20`, `0,30`, `0,40`, `0,50`, `0,60`, `0,70`, `0,80`, `0,90`, soit un pas de `0,10` et `8 × 8 = 64` configurations.

Les seuils de la meilleure configuration, notés `sim*` et `couv*`, sont retenus pour la campagne 2.

Les campagnes dense-only BGE et Qwen déjà réalisées peuvent servir de campagne 1 si, et seulement si, le corpus, l'endpoint, le modèle annoncé et le mode de calcul sont inchangés. Sinon la calibration est rejouée.

## Campagne 2 — exploration des pondérations hybrides

### But

Mesurer l'apport du sparse à seuils constants, afin d'isoler l'effet des pondérations de celui des seuils.

### Paramètres variables

Le poids sparse varie avec un pas de `0,10` :

| poids_sparse | poids_dense |
|---:|---:|
| 0,00 | 1,00 |
| 0,10 | 0,90 |
| 0,20 | 0,80 |
| 0,30 | 0,70 |
| 0,40 | 0,60 |
| 0,50 | 0,50 |
| 0,60 | 0,40 |
| 0,70 | 0,30 |
| 0,80 | 0,20 |
| 0,90 | 0,10 |
| 1,00 | 0,00 |

Le poids dense **n'est jamais saisi ou balayé indépendamment** : `poids_dense = 1 - poids_sparse`. Chaque paire somme ainsi exactement à 1, sans normalisation automatique.

Les bornes servent de contrôles : `sparse = 0,00` doit reproduire le dense-only ; `sparse = 1,00` mesure le sparse seul.

Les seuils sont fixes et égaux à `seuil_sim = sim*` et `seuil_couv = couv*`, issus de la campagne 1.

### Volume

`11 pondérations`.

### Décision après exploration

Une poursuite vers le raffinement est justifiée si la meilleure configuration réellement hybride (`0 < poids_sparse < 1`) améliore le F1 par rapport au contrôle dense-only, sans régression jugée rédhibitoire lors de la revue des faux positifs et faux négatifs. La comparaison porte sur le même corpus et les mêmes relations de référence.

Si le meilleur résultat est `poids_sparse = 0,00`, le sparse n'a pas démontré d'apport dans ce protocole : il n'y a pas de campagne de raffinement. Si le meilleur poids est à une borne, la décision de raffiner doit être confirmée à partir de la revue qualitative ; le balayage ne permet pas d'extrapoler au-delà de `[0,1]`.

## Campagne 3 — raffinement des seuils hybrides

### But

Préciser les seuils du score hybride, avec la pondération optimale déjà fixée. Cette campagne mesure l'effet des seuils sans le confondre avec celui des poids.

### Précondition et paramètres fixes

Cette campagne est lancée uniquement après la décision ci-dessus. Elle utilise le meilleur `poids_sparse` de la campagne 2, noté `w*`.

Les poids sont fixes : `poids_sparse = w*` et `poids_dense = 1 - w*`.

### Paramètres variables

Les deux seuils sont raffinés autour des valeurs denses retenues à la campagne 1 :

`max(0, sim* - 0,10) ≤ seuil_sim ≤ min(1, sim* + 0,10)`

`max(0, couv* - 0,10) ≤ seuil_couv ≤ min(1, couv* + 0,10)`

Le pas est `0,025` pour chacun.

Exemple : si la campagne 1 retient `sim* = 0,60` et `couv* = 0,80`, les seuils testés sont :

- `seuil_sim` : `0,50`, `0,525`, `0,55`, `0,575`, `0,60`, `0,625`, `0,65`, `0,675`, `0,70` ;
- `seuil_couv` : `0,70`, `0,725`, `0,75`, `0,775`, `0,80`, `0,825`, `0,85`, `0,875`, `0,90`.

### Volume

Dans le cas courant, `9 valeurs de seuil_sim × 9 valeurs de seuil_couv = 81 configurations`.

Si le meilleur résultat est sur le bord de la fenêtre, celle-ci est déplacée ou étendue avant toute conclusion : le protocole n'interprète pas une borne comme un optimum. Si le raffinement déplace fortement les seuils, on vérifie de nouveau les pondérations à ces nouveaux seuils avant de figer une recommandation ; cela protège contre une interaction poids/seuils non observée dans le protocole séquentiel.

## Restitution attendue

La conclusion compare explicitement : dense-only calibré, meilleur résultat de pondération hybride, puis meilleur résultat de seuils hybrides. Elle indique les paramètres complets, les métriques, les principaux TP/FP/FN, les limites de la référence ROME et une recommandation : conserver dense-only, retenir une pondération hybride pour une évaluation RH ultérieure, ou poursuivre l'investigation.

Gemma ne fait pas partie de ces campagnes : c'est un modèle de jugement catégoriel, sans score hybride ni grille de seuils.
