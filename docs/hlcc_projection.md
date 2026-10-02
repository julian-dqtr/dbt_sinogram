# HLCC en pratique : perte, régression des moments, data consistency dure

*Note de travail (30/09/2026). Elle explique ce qui a été ajouté au code ce jour-là, pourquoi, et
comment lire les résultats. La théorie générale (faisceaux, énergie de Dirichlet) est dans
`docs/harmonic_sheaves.md` ; ici on reste sur le U-Net et les conditions de Helgason-Ludwig.*

Tous les chiffres viennent de mesures en inférence seule sur le jeu de test (100 échantillons, bruit
$I_0 = 10^5$), avec les checkpoints du 20/09 : `UNet2D`, et `UNet2dHLCC` entraîné avec $\lambda = 0{,}1$.
Ils sont reproductibles avec les commandes de la section 9.

---

## 0. En bref

| Quoi | Où | Pourquoi |
| --- | --- | --- |
| Data consistency **dure** | `get_data_consistency_mask` (`src_2D/utils/evaluation.py`) | les vues mesurées ressortent intactes ; la version douce dégradait les 8 vues de bord |
| $\lambda$ par défaut $10^{-3}$, plage Optuna $[10^{-5}, 10^{-2}]$ | `train.py`, `optuna_search.py` | avec $0{,}1$ la physique écrasait la MSE |
| **Régression des moments** (Huang et al. 2017) | `src_2D/utils/hlcc.py`, modèles `..._P4` | corriger les moments des vues manquantes sans entraînement |
| Résidus HLCC dans l'évaluation | colonnes `hlcc_residual_<n>` de `evaluate_all.py` | mesurer la cohérence physique d'une sortie |
| « Pourquoi $K = 4$ » | `scripts/hlcc_order_study.py` | preuve empirique + figure |
| Peu / beaucoup de données | `scripts/launch_unet_hlcc_data_study.sh`, `--train_repeats`, `scripts/data_study_report.py` | voir quand la pénalité HLCC est utile |

---

## 1. Les moments d'une vue, en forme de Tchebychev

Une vue est une ligne de 128 valeurs $p(\theta, s)$. On note $x = s / S \in (-1, 1)$ la position sur
le détecteur, normalisée par sa demi-largeur $S = 102{,}4$ mm. Le **moment d'ordre $n$** de la vue est

$$a_n(\theta) = \sum_s p(\theta, s)\; U_n(x_s)\; \Delta x ,$$

où $U_n$ est le polynôme de Tchebychev de deuxième espèce : $U_0 = 1$, $U_1 = 2x$,
$U_2 = 4x^2 - 1$, $U_3 = 8x^3 - 4x$, puis $U_{n+1} = 2x\,U_n - U_{n-1}$.

| Ordre | Ce qu'il mesure | Lien avec les moments « physiques » |
| --- | --- | --- |
| 0 | la quantité totale de matière vue | $a_0 = M_0 / S$ |
| 1 | où tombe le centre de l'ombre | $a_1 = 2 M_1 / S^2$ |
| 2 | la largeur de l'ombre | combinaison de $M_2$ et $M_0$ |
| 3 | son asymétrie | combinaison de $M_3$ et $M_1$ |

**Pourquoi Tchebychev plutôt que $s^n$ ?** C'est la même information, mais $s^3$ monte à
$10^6$ mm³ au bord du détecteur alors que $U_n$ reste entre $-(n+1)$ et $n+1$. Et c'est la base de
Huang et al. : elle donne aussi la formule pour reconstruire une vue à partir de ses moments
(section 3).

**Les HLCC.** Quand on fait varier l'angle, la courbe $a_n(\theta)$ n'est pas quelconque :

$$a_n(\theta) \in \mathcal H_n = \operatorname{vect}\{\cos m\theta,\ \sin m\theta \;:\; m \le n,\ m \equiv n \pmod 2\},
\qquad \dim \mathcal H_n = n + 1 .$$

- $a_0$ est constant (1 inconnue : la masse) ;
- $a_1 = A\cos\theta + B\sin\theta$ (2 inconnues : le centre de masse) ;
- $a_2$ contient les fréquences 0 et 2 (3 inconnues), $a_3$ les fréquences 1 et 3 (4 inconnues).

Vérifié sur la vérité terrain du repo : pour $n \le 4$, la part de $a_n$ hors de $\mathcal H_n$
vaut entre $5 \cdot 10^{-7}$ et $4 \cdot 10^{-5}$ de son énergie.

---

## 2. Deux façons d'utiliser les HLCC

Les HLCC sont une propriété du sinogramme **entier**. Mais elles sont si contraignantes (seulement
$n + 1$ inconnues par ordre) qu'on peut s'en servir de deux façons différentes.

| | Pénalité (la perte) | Régression (la projection) |
| --- | --- | --- |
| Question posée | « ce sinogramme vient-il d'**un** objet ? » | « quels sont les moments de **cet** objet dans les vues non mesurées ? » |
| Données utilisées | les 180 vues de la sortie du réseau | les 51 vues mesurées |
| Calcul | distance de $a_n$ à $\mathcal H_n$ | ajuster les $n+1$ coefficients sur la fenêtre, prolonger la courbe |
| Quand | pendant l'entraînement | après le réseau, sans entraînement |
| Limite | un sinogramme cohérent mais faux la satisfait | instable au-delà de l'ordre 4 (section 4) |

L'image à garder : dire « ces points sont alignés » est une propriété de tous les points, mais deux
points suffisent à connaître toute la droite. La pénalité vérifie l'alignement ; la régression trace
la droite à partir des points connus.

---

## 3. La régression des moments, pas à pas

Référence : Y. Huang et al., *Restoration of missing data in limited angle tomography based on
Helgason-Ludwig consistency conditions*, 2017. Code : `HLCCMomentProjection` dans
`src_2D/utils/hlcc.py`.

On part d'un sinogramme complété par un modèle (U-Net, interpolation...) et de l'entrée mesurée.
Pour chaque ordre $n = 0, \dots, K$ :

**Étape 1 — mesurer.** On calcule $a_n(\theta)$ sur les 51 vues acquises (bruitées).

**Étape 2 — ajuster puis prolonger** (éq. 13–14 de l'article). On cherche les $n+1$ coefficients
$\beta_n$ tels que $X_n(\theta_{\text{acq}})\,\beta_n \approx a_n(\theta_{\text{acq}})$ au sens des
moindres carrés, où les colonnes de $X_n$ sont les $\cos m\theta$ et $\sin m\theta$ autorisés. Puis
on évalue la courbe partout : $\hat a_n(\theta) = X_n(\theta)\,\beta_n$. Dans le code, ces deux
opérations sont une seule matrice par ordre (`extrapolation`, de taille 180 × 51).

**Étape 3 — corriger.** Dans chaque vue manquante, le modèle a produit un moment
$a_n^{\text{modèle}}(\theta)$ qui diffère de $\hat a_n(\theta)$. On ajoute à la vue

$$\delta(\theta, s) = \sum_{n=0}^{K} c_n(\theta)\; W(s)\, U_n(s), \qquad W(s) = \sqrt{1 - x_s^2},$$

avec les coefficients $c_n$ choisis pour que les moments d'ordre 0 à $K$ de la vue corrigée soient
exactement $\hat a_n(\theta)$. $W U_n$ est la brique de la transformée de Tchebychev inverse (éq. 7) :
un dôme ($n=0$), un dôme penché ($n=1$), etc., nuls au bord du détecteur. Sur un détecteur discret
ces briques ne sont qu'approximativement orthogonales ; on résout donc un petit système
$(K+1) \times (K+1)$, une fois pour toutes (`synthesis`).

**Étape 4 — rester dans l'ombre de l'objet.** On ne garde la correction que là où la vue prédite
dépasse 3 % de son maximum. En dehors de l'ombre, un sinogramme vaut zéro et doit le rester.

Les vues mesurées ne sont jamais modifiées, et il n'y a aucun paramètre appris.

**Exemple avec l'ordre 0.** Les 51 vues mesurées donnent toutes la même somme, disons $m$ (au bruit
près). Si une vue prédite par le U-Net somme à $0{,}96\,m$, il lui manque 4 % de matière : on lui
ajoute un dôme $W(s)$ de la bonne hauteur, limité à l'ombre.

### Pourquoi l'étape 4

Sans elle, la correction déborde dans le fond noir du sinogramme. La MSE s'améliore quand même,
mais le SSIM baisse, car il est très sensible à un petit décalage là où la vérité vaut exactement 0.

| UNet2D, ordres 0–4 | MSE wedge | SSIM wedge |
| --- | --- | --- |
| sortie brute | 7,34e-4 | 0,894 |
| correction sur tout le détecteur (`support_threshold=None`) | 6,62e-4 | 0,866 |
| correction dans l'ombre seulement (défaut) | **6,56e-4** | **0,897** |

Le résultat ne dépend presque pas du seuil (1 %, 3 % ou 10 % donnent la même chose). Contrepartie :
les moments ne sont plus imposés exactement, seulement approchés. J'ai aussi essayé de recalculer
les coefficients à l'intérieur de l'ombre : c'est instable (l'erreur explose), donc abandonné.

### Les noms de modèles `_P4`

`UNet2D_P4` = le modèle `UNet2D` suivi de la régression des ordres 0 à 4. Ça marche avec tout modèle
et toute valeur : `LinearInterp_P4`, `UNet2dHLCC_N200_P2`... Le checkpoint est celui du nom sans `_P4`.

---

## 4. Pourquoi on s'arrête à l'ordre 4

`scripts/hlcc_order_study.py` le montre sur 100 images de test (figure
`outputs/figures/hlcc_order_selection.pdf`, tableau `outputs/evaluation/hlcc_order_study.csv`).

| Ordre $n$ | Conditionnement | Amplification du bruit | Erreur des moments : régression | UNet2D | UNet2dHLCC | MSE wedge après projection de 0..$n$ : UNet2D | UNet2dHLCC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0,02 | **0,06 %** | 3,9 % | 2,4 % | −3,2 % | −0,8 % |
| 1 | 3,8 | 0,2 | **0,18 %** | 6,5 % | 3,7 % | −4,1 % | −1,0 % |
| 2 | 16 | 2,8 | **0,26 %** | 5,7 % | 5,4 % | −6,1 % | −2,2 % |
| 3 | 69 | 44 | **1,5 %** | 8,8 % | 8,9 % | −8,0 % | −3,8 % |
| 4 | 294 | 742 | **9,2 %** | 14 % | 16 % | **−10,7 %** | −6,1 % |
| 5 | 1 270 | 13 000 | 34 % | **17 %** | **25 %** | −9,4 % | −8,3 % |
| 6 | 5 440 | 233 000 | 153 % | **19 %** | **25 %** | +98 % | +69 % |
| 7 | 23 400 | 4,3 millions | 914 % | **24 %** | **37 %** | +2 180 % | +1 600 % |

Lecture des trois panneaux de la figure :

- **(a) Le bruit est amplifié.** Pour l'ordre $n$ on ajuste $n+1$ inconnues sur une fenêtre de 50°
  et on prolonge sur 130°. Sur une petite fenêtre, les courbes de haute fréquence se ressemblent
  toutes : le conditionnement est multiplié par environ 4,3 à chaque ordre, et la variance du bruit
  par 11 à 18. Jusqu'à l'ordre 1 la régression **réduit** le bruit (elle moyenne 51 vues) ; à
  l'ordre 4 elle l'amplifie 742 fois ; à l'ordre 5, 13 000 fois.
- **(b) La régression n'est utile que tant qu'elle bat le réseau.** Jusqu'à l'ordre 4, les moments
  prolongés depuis les mesures sont plus justes que ceux des U-Nets. À l'ordre 5, c'est l'inverse.
  C'est le critère : on remplace les moments du réseau seulement là où on sait mieux que lui.
- **(c) L'effet sur le sinogramme le confirme.** L'erreur du UNet2D baisse jusqu'à $K = 4$, remonte
  à 5, et explose à 6.

Pour l'ancien UNet2dHLCC, le minimum est à $K = 5$ : ce modèle est sous-entraîné, ses moments
d'ordre 5 sont mauvais. On garde $K = 4$ pour tout le monde, c'est le critère du panneau (b).

Le lien avec l'article : chez Huang et al. il manque 20° et les ordres jusqu'à 40 environ sont
récupérés. Ici il manque 130°, et la limite tombe à 4. C'est la version « moments » du caractère mal
posé de l'angle limité, et c'est pour ça qu'il faut un réseau : lui seul fournit les ordres élevés.

---

## 5. Ce que la projection apporte aux modèles actuels

| Modèle | MSE wedge | PSNR wedge | SSIM wedge | PSNR image (FBP) | SSIM image |
| --- | --- | --- | --- | --- | --- |
| LinearInterp | 8,1e-3 | 22,81 | 0,691 | 16,15 | 0,247 |
| LinearInterp_P4 | 5,3e-3 | 24,19 | 0,703 | 16,61 | 0,274 |
| UNet2D | 7,3e-4 | 33,83 | 0,895 | 23,66 | 0,540 |
| UNet2D_P4 | 6,6e-4 | 34,15 | 0,897 | 23,70 | 0,542 |
| UNet2dHLCC ($\lambda = 0{,}1$) | 1,10e-3 | 31,02 | 0,833 | 21,74 | 0,465 |
| UNet2dHLCC_P4 | 1,03e-3 | 31,23 | 0,837 | 21,79 | 0,467 |

Ce qu'il faut en retenir, honnêtement :

- **Le gain est réel mais modeste** : +0,3 dB sur le wedge du UNet2D, et presque rien sur l'image
  reconstruite (+0,05 dB).
- **C'est attendu.** Les ordres 0 à 4 ne décrivent que les très basses fréquences de l'objet
  (éq. 8–9 de l'article : $W U_n$ ne contient que des fréquences au-dessus de $n$ environ, donc
  restaurer les ordres $\le n$ revient à restaurer un disque de basses fréquences de rayon $n$). Or
  l'erreur du U-Net est surtout dans les bords. Et la FBP (filtre rampe) atténue justement les basses
  fréquences.
- **La cohérence physique, elle, s'améliore nettement** (résidus HLCC, plus petit = plus cohérent) :

| Résidu HLCC | ordre 0 | ordre 1 | ordre 2 | ordre 3 | ordre 4 |
| --- | --- | --- | --- | --- | --- |
| Vérité terrain (plancher) | 4,6e-7 | 4,1e-5 | 6,5e-7 | 2,0e-5 | 6,2e-6 |
| UNet2D | 1,5e-3 | 6,6e-3 | 1,7e-3 | 5,0e-3 | 3,0e-3 |
| UNet2D_P4 | 3,2e-4 | 1,4e-3 | 3,2e-4 | 1,4e-3 | 7,9e-4 |
| UNet2dHLCC ($\lambda = 0{,}1$) | 4,7e-4 | 1,5e-3 | 1,2e-3 | 4,6e-3 | 3,9e-3 |

On voit aussi que l'ancienne pénalité (ordres 0–1) a bien réduit les résidus d'ordres 0 et 1, mais
presque pas ceux d'ordres 2 à 4 : elle ne corrige que ce qu'elle vise.

---

## 6. La perte HLCC jusqu'à l'ordre K

`HelgasonLudwigLoss(geom, max_order=K)` calcule, pour chaque ordre, la part de la courbe de moment
hors de son espace autorisé :

$$L_n = \frac{1}{V}\,\big\|(I - P_n)\,a_n\big\|^2 ,\qquad
\text{total} = \text{MSE} + \alpha(\text{epoch}) \sum_{n=0}^{K} \lambda_n \,\frac{L_n}{L_n(\text{entrée zero-filled})} .$$

- $P_n$ est le projecteur sur $\mathcal H_n$. Chaque terme est normalisé par sa valeur sur l'entrée
  zero-filled : il vaut environ 1 pour l'entrée naïve et environ 0 pour la vérité.
- **$K = 1$ redonne exactement l'ancienne perte** (variance de $M_0$, résidu de $M_1$ hors de
  $\{\cos, \sin\}$). C'est vérifié par `test_orders_0_and_1_are_the_historical_loss`.
- $\lambda_0$ = `--lambda_m0`, $\lambda_1$ = `--lambda_m1`, et tous les ordres $\ge 2$ partagent
  `--lambda_high`. L'ordre maximal se choisit avec `--hlcc_max_order` (1 par défaut).
- $\alpha$ n'est que la rampe de 0 à 1 sur `--anneal_epochs` epochs. **Ce qui pèse, c'est $\lambda$.**

**Le problème de l'ancien run.** Il utilisait $\lambda = 0{,}1$ (l'ancien défaut). Mesuré : le
gradient de la physique valait 52 fois celui de la MSE au départ, et encore 26 fois à l'optimum du
UNet2D. La physique a piloté l'entraînement : MSE d'entraînement 11 fois plus haute que celle du
UNet2D à l'epoch 200, run encore en progrès à la dernière epoch. Ce rapport est proportionnel à
$\lambda$ : environ 0,3 à $10^{-3}$ (le nouveau défaut), 3 à $10^{-2}$.

**Piège d'Optuna.** Les essais durent 30 epochs sur 500 échantillons. Dans ce régime l'effet
régulariseur est plus fort qu'au run final : un $\lambda$ qui gagne en essai doit être revérifié à
pleine échelle.

---

## 7. L'expérience « peu / beaucoup de données »

**L'idée.** La vérité terrain satisfait déjà les HLCC, donc la pénalité ne déplace pas l'optimum :
elle ne peut aider que comme régulariseur, contre le surapprentissage. Elle devrait donc compter
davantage quand les données sont rares.

**Les quatre runs** (`scripts/launch_unet_hlcc_data_study.sh`, lancés le 30/09 à 17h29) :

| Run | Fantômes d'entraînement | Perte |
| --- | --- | --- |
| `UNet2D_N200` | 200, vus 10 fois par epoch | MSE |
| `UNet2dHLCC_N200` | 200, vus 10 fois par epoch | MSE + HLCC ordres 0–1 |
| `UNet2D` | 2000 | MSE |
| `UNet2dHLCC` | 2000 | MSE + HLCC ordres 0–1 |

- **Tout le reste est identique** : la configuration du run UNet2D de référence (lr 3e-4, weight
  decay 1e-4, 32 filtres, batch 4 par GPU sur 8 GPU, 200 epochs, graine 0), $\lambda = 10^{-3}$ pour
  tous les ordres, annealing sur 20 epochs, data consistency dure.
- **Pourquoi « vus 10 fois par epoch ».** Avec 200 fantômes et le même nombre d'epochs, le réseau
  ferait 10 fois moins de pas d'optimisation : on comparerait « moins de données » et « moins
  d'entraînement » à la fois. Avec `--train_repeats 10`, les quatre runs font exactement le même nombre
  de pas, de validations, et suivent les mêmes rampes. Seul le nombre de fantômes **différents**
  change.
- **Hyperparamètres.** Il n'existe pas d'étude Optuna du UNet2D dans le protocole actuel (celle du
  31/08 date du fan-beam, avec un autre objectif). J'ai donc repris la configuration du meilleur
  UNet2D existant.
- **Durée.** Environ 50 min par run, un peu plus de 3 h au total.

**Suivre la file.**

```bash
tmux attach -t unet_hlcc_data_study            # Ctrl-b d pour détacher
ls outputs/2d/logs/unet_hlcc_data_study/       # un log par run (les lignes "Epoch ..." y arrivent par paquets)
```

Les courbes sont en direct sur wandb (projet `dbt-sinogram-completion-2D`). Un run terminé a un
fichier `DONE` dans son dossier de checkpoint ; relancer le script saute les runs finis.

**Les anciens checkpoints** `UNet2D` et `UNet2dHLCC` du 20/09 ne sont pas écrasés : le script les
déplace dans `outputs/backups/checkpoints_<date>/` juste avant de réentraîner sous le même nom.

**Évaluer, une fois la file terminée.**

```bash
MODELS=$(.venv/bin/python scripts/data_study_report.py --print_models)   # les 6 runs, avec et sans _P4, + les 2 baselines
.venv/bin/python src_2D/evaluate_all.py --num_samples 200 --models $MODELS
.venv/bin/python scripts/data_study_report.py                            # tableaux appariés + figures
```

`scripts/data_study_report.py` compare, pour chaque taille du jeu d'entraînement, chaque variante
HLCC au U-Net simple **entraîné sur les mêmes fantômes**. Il écrit
`outputs/evaluation/data_study_summary.csv`, `outputs/figures/hlcc_data_study.pdf` (l'effet de chaque
variante) et `outputs/figures/hlcc_data_study_curves.pdf` (les courbes de validation).

Les colonnes de la comparaison :

| Colonne | Ce qu'elle dit |
| --- | --- |
| `change` | variation de la MSE moyenne du wedge par rapport au U-Net simple (négatif = mieux) |
| `95 % interval` | incertitude sur cette variation, due au fait qu'on n'a que 200 images de test (bootstrap apparié) |
| `better on` | part des images de test où la variante fait mieux que le U-Net simple |
| `p` | test de Wilcoxon apparié, image par image (non corrigé pour le nombre de comparaisons) |
| `img_psnr gain (dB)` | la même comparaison sur l'image reconstruite par FBP |

La comparaison est **appariée** : les deux modèles sont notés sur les mêmes images, donc la
difficulté propre à chaque image s'annule. C'est pour ça qu'un écart de 2 % peut être net alors que
l'écart-type entre images vaut 50 % de la moyenne.

**Comment lire le résultat.**

- *HLCC utile quand les données sont rares* : `UNet2dHLCC_N200` nettement meilleur que `UNet2D_N200`
  sur `mse_wedge`, alors que l'écart est petit ou nul à 2000.
- *HLCC sans effet sur la MSE* : écarts de l'ordre du pour-cent dans les deux régimes. Regarder alors
  les colonnes `hlcc_residual_<n>` : la pénalité peut rendre les sorties plus cohérentes sans changer
  la MSE, ce qui reste un résultat.
- **Attention à une seule graine.** L'intervalle à 95 % ne couvre que le hasard du jeu de test, pas
  celui de l'entraînement : deux runs identiques à la graine près diffèrent eux aussi
  « significativement » sur un jeu de test fixe. Un écart de 2 ou 3 % ne compte que s'il garde son
  signe sur une réplique : `SEED=1 bash scripts/launch_unet_hlcc_data_study.sh` (runs `..._S1`), puis
  `scripts/data_study_report.py --seed 1`.

### Résultats (file du 30/09, graine 0, 200 images de test)

Variation de la MSE du wedge par rapport au U-Net simple entraîné sur les mêmes fantômes
(intervalle à 95 % entre crochets ; figure `outputs/figures/hlcc_data_study.pdf`) :

| | 200 fantômes | 2000 fantômes |
| --- | --- | --- |
| U-Net simple (référence) | 2,75e-3 | 8,15e-4 |
| + perte HLCC ordres 0–1 | −1,4 % [−3,8 ; +0,7] | +0,2 % [−2,9 ; +3,2] |
| + régression des moments (`_P4`) | **−26,8 %** [−31 ; −23], 100 % des images | **−11,0 %** [−12,5 ; −9,4], 94 % des images |
| + perte 0–1 et régression | −27,7 % | −9,7 % |

Sur l'image reconstruite (FBP), la régression apporte +0,24 dB à 200 fantômes et +0,06 dB à 2000.

- **La régression des moments est l'usage des HLCC qui marche**, et d'autant mieux que les données
  sont rares : avec peu de données le réseau se trompe davantage sur les basses fréquences, alors que
  la régression ne dépend que des mesures.
- **La pénalité dans la perte ne change presque pas l'erreur.** −1,4 % à 200 fantômes et +0,2 % à
  2000, deux écarts compatibles avec zéro. Elle rend en revanche les sorties plus cohérentes (résidus HLCC −20 à −40 %). Avec
  $\lambda = 10^{-3}$ elle n'est plus nuisible : l'ancien run à $\lambda = 0{,}1$ était 51 % moins bon
  que le U-Net.
- **Pourquoi.** Les deux régimes surapprennent (erreur de validation / erreur d'entraînement : 100 à
  120 à 200 fantômes, 10 à 2000). La pénalité n'est calculée que sur les fantômes d'entraînement, que
  le réseau reproduit déjà presque parfaitement : il n'y a plus rien à y corriger. Piste possible :
  appliquer la pénalité à des sinogrammes limités **sans vérité terrain** (elle n'en a pas besoin).
- **Ordre de la régression.** `scripts/hlcc_order_study.py` relancé sur les nouveaux modèles : $K = 4$
  reste le meilleur choix pour les modèles à 2000 fantômes (−10,8 %). Pour ceux à 200 fantômes, dont
  les moments sont moins bons, $K = 5$ fait encore mieux (−36 % contre −31 % à $K = 4$ sur ces 100
  images) ; $K = 6$ dégrade tout le monde.

---

## 8. La data consistency dure

**Avant.** `sortie = w · mesure + (1 − w) · prédiction`, avec une rampe **à l'intérieur** de la fenêtre :

| Vue | ±25° | ±24° | ±23° | ±22° | ±21° à 0° |
| --- | --- | --- | --- | --- | --- |
| Poids $w$ de la mesure | 0,095 | 0,345 | 0,655 | 0,905 | 1 |

**Le problème.** La perte est calculée après cette étape. Le réseau ne reçoit donc aucun gradient là
où $w = 1$, et un gradient atténué sur les bords : il n'apprend jamais à reproduire les vues
mesurées. Résultat mesuré sur les 8 vues de bord : une erreur 15 fois (UNet2D) à 19 fois
(UNet2dHLCC) plus grande que celle de la mesure brute qu'elles remplaçaient.

**Maintenant.** $w = 1$ sur les 51 vues acquises, 0 ailleurs (`blend_width_deg = 0`, le défaut).

- Les 51 vues mesurées ressortent bit à bit.
- Les vues manquantes ne changent pas : le masque ne les touche pas. Les `mse_wedge` déjà mesurés
  restent donc valables, y compris pour les anciens checkpoints.
- Le raccord au bord de la fenêtre ne se dégrade pas : saut moyen entre ±25° et ±26° de 7,6e-5 en
  dur contre 8,7 à 9,9e-5 en doux (vérité terrain : 2,4 à 3,3e-5).
- L'ancien comportement reste disponible (`blend_width_deg=5.0`) pour reproduire les runs d'avant
  le 30/09.

Les études Optuna des GNN déjà faites (lr et weight decay) l'ont été avec la version douce. Leurs
résultats restent utilisables : seules 8 vues sur 180 étaient concernées, et pas celles du wedge.

---

## 9. Commandes

```bash
# Pourquoi K = 4 : figure + tableau, 100 images de test
.venv/bin/python scripts/hlcc_order_study.py

# Évaluation avec la régression des moments en post-traitement
.venv/bin/python src_2D/evaluate_all.py --models UNet2D UNet2D_P4 UNet2dHLCC UNet2dHLCC_P4

# Optuna sur les lambda seuls (plage [1e-5, 1e-2]), ordres 0-1
bash scripts/launch_optuna_8gpus.sh UNet2dHLCC 6 auto --physics-only

# Peu / beaucoup de données (file de quatre runs, puis répliques avec une autre graine)
bash scripts/launch_unet_hlcc_data_study.sh
SEED=1 bash scripts/launch_unet_hlcc_data_study.sh

# ... puis la comparaison appariée sur le jeu de test (tableaux + figures)
.venv/bin/python src_2D/evaluate_all.py --models $(.venv/bin/python scripts/data_study_report.py --print_models)
.venv/bin/python scripts/data_study_report.py
```

---

## 10. Formulations pour le mémoire

**Régression des moments.** « Suivant Huang et al. (2017), les coefficients harmoniques de chaque
courbe de moment de Tchebychev sont ajustés sur les vues acquises puis évalués sur les vues
manquantes. Depuis une fenêtre de ±25°, cette régression reste plus précise que le réseau jusqu'à
l'ordre 4 ; au-delà, l'amplification du bruit (× 13 000 à l'ordre 5) la rend inutilisable. »

**Rôle respectif.** « Les conditions de Helgason-Ludwig fixent les basses fréquences de l'objet
(ordres 0 à 4) ; le réseau fournit le reste. Imposer ces moments en post-traitement réduit l'erreur
quadratique du wedge d'environ 10 % et rapproche la sortie d'un sinogramme physiquement cohérent,
sans effet notable sur l'image reconstruite par FBP. »

**Data consistency.** « Les vues acquises sont recopiées telles quelles dans la sortie de chaque
modèle. Une version progressive, avec une rampe à l'intérieur de la fenêtre, a été écartée : elle
remplaçait des mesures par des prédictions que le réseau n'est jamais entraîné à produire. »

---

## 11. Références

*(Détails bibliographiques à vérifier avant citation.)*

- Y. Huang, X. Huang, O. Taubmann, Y. Xia, V. Haase, J. Hornegger, G. Lauritsch, A. Maier,
  *Restoration of missing data in limited angle tomography based on Helgason-Ludwig consistency
  conditions*, Biomed. Phys. Eng. Express, 2017.
- D. Ludwig, *The Radon transform on Euclidean space*, Comm. Pure Appl. Math., 1966.
- S. Helgason, *The Radon transform*, Birkhäuser, 1980.
- A. K. Louis, *Incomplete data problems in X-ray computerized tomography I*, Numer. Math., 1986.
- M. E. Davison, *The ill-conditioned nature of the limited angle tomography problem*, SIAM J. Appl.
  Math., 1983.
- F. Natterer, *The Mathematics of Computerized Tomography*, 1986.
