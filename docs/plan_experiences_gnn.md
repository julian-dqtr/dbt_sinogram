# Plan des expériences : limites des GNN locaux (GCN vs SNN, 6 / 12 / 18 couches)

Statut (27/09) : **plan validé** (D1 = option A1, D2, D3). **Phase 0 faite** pour P1 à P4. P5 et
P6 (smoke test) sont reportés. Prochaine étape : la phase 1, que tu lances toi-même.

**Mise à jour du 03/10.**
- **GCN.** Les phases 1 à 3 sont faites. Les entraînements finaux L6, L12 et L18 ont tourné le
  03/10 (200 epochs, 8 GPU), avec un meilleur epoch à 197, 197 et 193. Sur le test,
  `mse_wedge` vaut 9,13e-3, 2,70e-3 et 2,40e-3, contre 8,24e-3 pour LinearInterp et 8,15e-4 pour
  UNet2D. Le théorème de portée est vérifié sur GCN_L6. Avec P4, le gain est de −71 %, −32 % et
  −23 %, contre −11 % pour UNet2D.
- **SNN.** Il utilise désormais le **transport par décalage** (`harmonic_sheaves.md` §6 bis), et
  non plus $SO(2)$.
  - Les études `SNN_L{6,12}_parallel_optimizer_only` portent sur l'ancien SNN $SO(2)$ : elles
    sont obsolètes, et `launch_best_training.py` ne doit plus les utiliser.
  - Le SNN est entraîné **avec la configuration Optuna du GCN** de même profondeur
    (`launch_best_training.py GCN_L{L}_parallel_optimizer_only --model SNN`) : seul le transport
    diffère.
  - **L6 est sauté** : il est limité par la portée pour les deux modèles (D3).
  - Les runs SNN L12 puis L18 sont lancés le 03/10 au soir (tmux `snn_final`).
- **Résultats SNN-shift (test, 04/10).** 200 epochs, meilleur epoch à 197 pour les deux runs ;
  3 h 41 pour L12 et 5 h 26 pour L18.

  | | GCN | SNN | écart | SNN meilleur | Wilcoxon p |
  |---|---|---|---|---|---|
  | `mse_wedge` L12 | 2,70e-3 | 2,61e-3 | −3,4 % | 108/200 | 0,24 |
  | `mse_wedge` L18 | 2,40e-3 | 1,93e-3 | **−19,7 %** | 166/200 | 1e-21 |
  | + P4, L12 | 1,84e-3 | 1,89e-3 | +2,7 % | 88/200 | 0,03 |
  | + P4, L18 | 1,84e-3 | 1,47e-3 | **−20,0 %** | 169/200 | 5e-22 |

  - À L18, le SNN est meilleur à toutes les distances (−15 à −30 %).
  - À L12, il est moins bon près de la fenêtre (+43 % pour d ≤ 12°) mais meilleur au bout du
    wedge (−24 % pour d = 60–65°, ρ 0,76 contre 0,65). Il porte l'information plus loin que la
    portée effective du GCN.
  - **Le critère du §4.2 n'est pas rempli** : à L12, `mse_wedge` n'est pas significatif ; à L18,
    le rapport angulaire ne l'est pas (p = 0,54). Le gain de L18 est donc à présenter comme
    exploratoire. Il repose aussi sur une seule seed : une seconde seed de la paire L18 dirait si
    −20 % dépasse le bruit d'un entraînement à l'autre.
  - Le rapport le long du détecteur reste loin de celui du U-Net (médianes 0,44 à L18 et 0,33 à
    L12, contre 0,78) : le décalage n'a pas réglé le flou le long du détecteur.
- **Critère (a)/(b).** Il reste celui du §4.2, fixé à l'avance. Le rapport le long du détecteur
  (GCN_L18 0,40 contre UNet2D 0,78) est rapporté comme observation exploratoire.

Règle d'or respectée : aucun nouveau modèle. On n'utilise que `SinoGCN` (transport identité) et
`SinoSheafNet` (rotations $SO(2)$ codées en dur), déjà présents dans `src_2D/models/SinoSheavesNN/`.
Le code ajouté en phase 0 sert uniquement à l'optimisation, à l'évaluation et aux figures.

---

## Les chiffres qui structurent tout

| Quantité | Valeur | Source |
|---|---|---|
| Vues du sinogramme complet | 180, pas de 1°, $[-90°, 90°)$ | `DBTGeometryConfig` |
| Fenêtre acquise | $[-25°, 25°]$, 51 vues | idem |
| Vues manquantes (wedge) | 129 : distance à la fenêtre $d = 1..65°$ à gauche, $1..64°$ à droite | idem |
| Portée d'une couche (kNN, $k = 12$) | $k/2 = 6$ vues $= 6°$ | `graph_data.angular_reach_deg` |
| Portée à $L$ couches | $6L$ : **36° (L6), 72° (L12), 108° (L18)** | idem |
| Profondeur minimale pour couvrir le wedge | 11 couches | $\lceil 65/6 \rceil$ |
| Vues hors de portée à L6 | $d \ge 37°$ : 29 + 28 = **57 vues, soit 44 % du wedge** | calcul |
| Vues hors de portée à L12 et L18 | 0 | calcul |
| Paramètres, identiques pour GCN et SNN | 207 585 (L6), 406 881 (L12), 606 177 (L18), contre 2 141 249 pour le U-Net | `build_model`, 32 stalks |

**Pourquoi la limite est exacte.** L'encodeur, le décodeur (convolutions `(1, 3)` le long du
détecteur) et `ViewNorm` agissent vue par vue. Seule la `ViewGraphConv` mélange les vues, et
chaque couche ne le fait qu'à $k/2$ vues de distance. Les vues manquantes valent zéro en entrée,
et le drapeau « acquise » y vaut aussi zéro. Au-delà de la portée, la sortie ne dépend donc
d'aucune mesure : **elle est identique pour tous les échantillons**. Son erreur ne peut pas
descendre sous la variance de la vérité terrain, calculée vue par vue sur le jeu de test.
C'est déjà prouvé structurellement par
`tests/test_graph_models.py::test_angular_receptive_field_is_exactly_layers_times_half_k`.
Les expériences doivent maintenant le **montrer sur les modèles entraînés**.

---

## Décisions (validées le 27/09)

**D1. Option A1.** La topologie et la capacité sont fixées aux valeurs par défaut de
`train.py` : $k = 12$, σ = 5°, `num_stalks` = 32. On ne cherche que `lr` et `weight_decay`, dans
une petite étude par (modèle, profondeur), soit 6 études. GCN et SNN ont alors le même graphe et
le même nombre de paramètres à chaque profondeur ; seul le transport diffère.

Pourquoi : `sigma_deg` change la **portée effective**. Le poids d'une arête entre deux vues
séparées de 6° vaut $e^{-36/(2\sigma^2)}$ : 0.01 pour σ = 2°, 0.23 pour 3.5°, 0.49 pour 5°, 0.84
pour 10°. Et `num_stalks` change le nombre de paramètres. Les laisser libres aurait mélangé
« transport », « portée effective » et « capacité ».

**D2. Même budget que les U-Net** : 200 epochs, 2000 échantillons, batch 4, même nombre de GPU.
Après chaque run, on vérifie `best_epoch` : s'il vaut la dernière epoch, le modèle n'a pas
convergé (c'est le cas de UNet2dHLCC), et on le dit dans le mémoire ou on prolonge.

**D3. L'oversmoothing est isolé à L12 et L18.** À 6 couches, la « bande grise » au-delà de 36°
existe pour **les deux** modèles : c'est l'effet de portée, pas l'oversmoothing. À 12 et 18
couches, tout le wedge est à portée, donc un wedge délavé ne peut plus venir de la portée.

---

## Phase 0 — Préparer le code (fait, sans aucun entraînement)

| # | Tâche | Ce qui a été fait | Vérification |
|---|---|---|---|
| P1 | Recherche option A1 | `--optimizer-only` dans `src_2D/optuna_search.py` (cherche `lr` et `weight_decay` seulement ; nom d'étude suffixé `_optimizer_only`) ; `--create-only` ; file d'attente des 6 études `scripts/launch_gnn_optuna_queue.sh` | espace de recherche contrôlé avec `study.ask()` ; file testée à blanc (enchaînement, reprise, arrêt si tous les workers échouent) |
| P2 | Reproduire l'architecture au run final | chaque essai enregistre sa `model_config` complète ; `launch_best_training.py` la retransmet, `--grad_checkpoint` compris. L'option `--num_layers` prévue pour A2 est devenue inutile avec A1 | commande affichée pour un faux `best_params.json` |
| P3 | Plancher et variance inter-échantillons | `evaluate_all.py` écrit `per_view_var.csv` (colonne `GroundTruth` = plancher, puis une colonne par modèle) et deux colonnes par échantillon, `dirichlet_angle_ratio` et `dirichlet_detector_ratio` | `tests/test_evaluation_statistics.py` (6 tests) ; évaluation relancée sur les U-Net, métriques identiques aux précédentes |
| P4 | Notebook | sections 3 (portée) et 4 (oversmoothing) de `results/model_analysis.ipynb`, figures exportées en PDF dans `outputs/figures/` | exécuté sur une copie du dépôt avec des GCN/SNN **non entraînés**, puis sur les vraies données |
| P5 | *(reporté)* Énergie de Dirichlet des features par couche (hooks) | — | — |
| P6 | *(reporté)* Smoke test : temps par epoch et mémoire GPU à L18 | — | — |

---

## Phase 1 — Recherche d'hyperparamètres (toi)

Les 6 études s'enchaînent dans une seule session tmux. Chacune utilise les 8 GPU avec 4 essais
par GPU, soit 32 essais, de 30 epochs sur 500 échantillons.

```bash
bash scripts/launch_gnn_optuna_queue.sh           # GCN:6 SNN:6 GCN:12 SNN:12 GCN:18 SNN:18
tmux attach -t optuna_gnn_queue                   # suivre la file (Ctrl-b d pour détacher)
```

- Suivi : `outputs/2d/optuna/<étude>/best_params.json` est réécrit après chaque essai ; les logs
  sont dans `outputs/2d/optuna/<étude>/logs/gpu<i>.log`.
- Plus d'essais : `N_TRIALS_PER_GPU=6 bash scripts/launch_gnn_optuna_queue.sh`.
- Relancer après une interruption : la même commande. Les études terminées (fichier `DONE`) sont
  sautées.
- Mémoire à L18 : non mesurée (pas de smoke test). Avec 32 stalks, l'estimation est de quelques
  Go, sous les 12 Go d'un K80. Si tous les workers d'une étude échouent, la file s'arrête. On
  relance alors les études L18 avec
  `EXTRA_ARGS="--grad_checkpoint" bash scripts/launch_gnn_optuna_queue.sh GCN:18 SNN:18`.
  L'option est retransmise au run final, et le temps par epoch à L18 inclura le recalcul :
  à signaler dans le tableau.

---

## Phase 2 — Entraînements finaux (toi)

Un run par étude, avec la meilleure configuration (torchrun sur 8 GPU, 200 epochs, 2000
échantillons). Pour les enchaîner dans une session tmux :

```bash
tmux new -s gnn_final
for s in GCN_L6 SNN_L6 GCN_L12 SNN_L12 GCN_L18 SNN_L18; do
    .venv/bin/python scripts/launch_best_training.py ${s}_parallel_optimizer_only --run
done
```

Sans `--run`, la commande est seulement affichée. Chaque run écrit dans
`outputs/2d/checkpoints/{GCN,SNN}_L{6,12,18}/`, exactement les noms qu'attend
`evaluate_all.py`. `training_stats.json` enregistre `world_size`, donc les temps affichés dans le
tableau sont comparables.

À vérifier après chaque run :

1. la ligne `angular reach = … deg` affichée au démarrage ;
2. `best_epoch` par rapport à `epochs_run` ;
3. les courbes wandb ;
4. la cohérence : dans sa portée, un modèle doit au moins battre `ZeroFilling`.

---

## Phase 3 — Évaluation (inférence seule)

```bash
.venv/bin/python src_2D/evaluate_all.py --num_samples 200
```

Cette commande produit, dans `outputs/evaluation/`, `model_comparison_metrics.csv` (avec les
rapports de Dirichlet), `per_view_mse.csv` et `per_view_var.csv`. On exécute ensuite le notebook
en entier : les sections 1 à 4 intègrent automatiquement les 6 nouveaux modèles.

---

## Phase 4 — Extraction des preuves (dans le notebook)

### 4.1 La limite de portée angulaire (section 3 du notebook, `angular_reach.pdf`)

- **Axe x** : distance à la fenêtre acquise, $d(\theta) = |\theta| - 25°$. Les deux côtés sont
  repliés (moyenne gauche/droite à $d$ égal ; $d = 65°$ n'existe qu'à gauche). Cet axe, et non
  l'angle absolu, correspond à la portée.
- **Panneau gauche** : MSE par vue (échelle log) et **plancher**, la variance de la vérité
  terrain par vue, c'est-à-dire l'erreur du meilleur prédicteur qui ignore les mesures.
- **Panneau droit** : $\rho(d) = \mathrm{std}_{\text{échantillons}}(\text{prédiction}) / \mathrm{std}_{\text{échantillons}}(\text{vérité terrain})$,
  la part de la variabilité réelle que le modèle reproduit. Elle vaut exactement 0 au-delà de la
  portée.
- **Style** : couleur = famille, trait = profondeur (plein L6, tirets L12, pointillés L18).
  UNet2D et LinearInterp servent de références. Une verticale marque la portée de L6 (36°).
- **Tableau de vérification** : pour chaque modèle de graphe, il donne :
  - l'écart-type inter-échantillons maximal au-delà de la portée (théorie : exactement 0) ;
  - le rapport minimal MSE / plancher (théorie : au moins 1) ;
  - la MSE dans la portée et hors portée.

  Les références sont coupées à 36°, comme contrôle. Le notebook imprime alors « theorem
  verified » ou une alerte.
- **Premières observations (U-Net et baselines, 27/09)** :
  - le plancher vaut environ $1.2 \cdot 10^{-2}$ par vue et reste plat sur tout le wedge ;
  - le U-Net monte de $8 \cdot 10^{-5}$ près de la fenêtre à environ $1.4 \cdot 10^{-3}$ vers
    50–65°, soit environ 10× sous le plancher, avec $\rho \approx 0.95$ : il n'est pas limité en
    portée ;
  - l'interpolation linéaire passe au-dessus du plancher vers 50° : à cette distance, elle fait
    pire que prédire la moyenne.
- **Formulation juste pour le mémoire.** Pas « l'erreur explose après 36° », mais : « au-delà de
  36°, la sortie du modèle à 6 couches est rigoureusement indépendante des mesures ; son erreur
  est bornée inférieurement par la variance de la vérité terrain ». La montée peut commencer
  **avant** 36°. Les poids gaussiens et la diffusion lente (trou spectral d'environ $2 \cdot 10^{-3}$,
  `harmonic_sheaves.md` §6, point 4) atténuent l'information : la portée effective est plus courte
  que la portée nominale. Si l'erreur de L12 remonte en bout de wedge (60–65°) alors que celle de
  L18 ne remonte pas, c'est la signature de cette portée effective. C'est un résultat de plus, à
  montrer.

### 4.2 L'oversmoothing, GCN vs SNN à L12 et L18 (section 4 du notebook)

- **Images** (`oversmoothing_sinograms_L{12,18}.pdf`, `oversmoothing_fbp_L{12,18}.pdf`) : les
  échantillons de test 0 à 3, fixés **avant** de regarder (`OVERSMOOTHING_SAMPLES`). Les panneaux
  sont :
  - entrée, GCN, SNN et vérité terrain, sur la même échelle de gris ;
  - les cartes d'erreur $|\text{prédiction} - \text{vérité terrain}|$, sur une échelle commune par
    ligne ;
  - les FBP.
- **Profils angulaires** (`oversmoothing_profiles.pdf`) : $p(\theta, s_0)$ pour trois positions
  du détecteur. La trace de la vérité terrain est-elle prolongée, ou aplatie vers une constante ?
- **Mesures par échantillon** (colonnes du CSV) :
  - `mse_wedge` ;
  - `dirichlet_angle_ratio` $= E_\theta(\text{prédiction}) / E_\theta(\text{vérité terrain})$, avec
    $E_\theta = \sum \lVert p_{v+1} - p_v \rVert^2$ sur les paires de vues manquantes consécutives ;
  - `dirichlet_detector_ratio`, son équivalent le long du détecteur.

  Pour les deux rapports, < 1 signifie plus lisse que la vérité terrain.
- **Tendance en profondeur** (`oversmoothing_depth_trend.pdf`) : `mse_wedge` à 6, 12 et 18
  couches. Le passage de L12 à L18 est le contraste propre, puisque la portée n'y est plus
  limitante.
- **Critère fixé à l'avance, et codé dans le notebook** : le SNN bat le GCN à 12 **et** 18
  couches, sur `mse_wedge` **et** sur $|\log \text{dirichlet\_angle\_ratio}|$, avec à chaque fois un
  Wilcoxon apparié p < 0.05 sur les mêmes échantillons de test. Le notebook imprime la conclusion
  (a) ou (b) :
  - **(a) critère rempli** : le transport $SO(2)$ limite la dilution de l'information le long de
    l'axe angulaire ;
  - **(b) critère non rempli** : le faisceau plat appliqué à des canaux bruts n'apporte pas de
    propagation harmonique exploitable, et la limite dominante des GNN locaux est la portée.

**Honnêteté : le résultat n'est pas garanti.** `harmonic_sheaves.md` §6 le dit lui-même :

- la connexion est plate (jauge pure), donc l'opérateur d'agrégation du SNN est celui du GCN à
  un changement de jauge près ; le SNN équivaut à un GCN dont les poids sont modulés par
  l'angle absolu ;
- le lien physique entre vues voisines est un décalage le long du détecteur, proportionnel à
  la profondeur (inconnue), et aucune rotation de canaux ne le représente (§6, point 5) ;
- les rotations agissent sur des canaux bruts, dont le contenu est dominé par $m = 0$, que
  l'identité transporte mieux ;
- les connexions résiduelles de chaque bloc freinent déjà l'oversmoothing.

La conclusion (b) reste un résultat valide pour une preuve de concept sur les limites des GNN
locaux.

---

## Phase 5 — Intégration dans le mémoire

- Tableau final : métriques de test, paramètres, temps d'entraînement, GPU (section 1 du notebook).
- Figures PDF de `outputs/figures/`.
- Texte : reprendre les formulations de `harmonic_sheaves.md` §8, et ajouter les chiffres mesurés
  (44 % du wedge hors de portée à L6, plancher atteint, résultat des Wilcoxon).

## Risques

| Risque | Parade |
|---|---|
| Mémoire GPU insuffisante à L18 (non mesurée) | la file s'arrête d'elle-même ; relancer GCN:18 et SNN:18 avec `EXTRA_ARGS="--grad_checkpoint"` |
| Des essais Optuna de 30 epochs favorisent les modèles qui convergent vite, donc désavantagent L18 | même budget pour tous ; contrôler `best_epoch` en phase 2 |
| Budget GPU total inconnu (pas de smoke test) | les études L6 passent en premier et donnent la durée réelle d'un essai ; ajuster `N_TRIALS_PER_GPU` pour les suivantes si besoin |
| Les runs HLCC (λ faible, 400 epochs) concurrencent les GNN sur les 8 GPU | les GNN sont le cœur du mémoire : les lancer en priorité |

## Ordre et dépendances

1. ~~Validation des décisions D1, D2, D3~~ (fait).
2. ~~Phase 0, P1 à P4~~ (fait).
3. Phase 1.
4. Phase 2.
5. Phase 3.
6. Phase 4 : exécuter le notebook.
7. Phase 5.

P5 et P6 restent possibles à tout moment, sur ta demande.
