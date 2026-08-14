# Détails de l'Implémentation 2D

Ce document fournit une explication détaillée de l'architecture du modèle U-Net 2D implémenté ainsi que de la façon dont le dataset de complétion de sinogrammes a été construit.

---

## 1. Modèle : SinogramUNet (U-Net 2D)

L'architecture est définie dans la classe `SinogramUNet` (`src_2D/models/Unet2D/unet_2d.py`). Il s'agit d'un réseau de type U-Net, spécialisé pour traiter des données 2D comme les sinogrammes.

### Composants clés du modèle :
1. **Base MONAI (DynUNet)** : Le modèle utilise la classe `DynUNet` de la bibliothèque médicale MONAI. Cette classe fournit une implémentation robuste et dynamique du U-Net.
2. **Profondeur et Filtres** : Le réseau comporte 4 niveaux (un encodeur de 4 blocs et un décodeur symétrique). Le nombre de filtres croît de manière exponentielle : `[filters, filters * 2, filters * 4, filters * 8]`. (Par défaut, avec `filters=32`, on a 32, 64, 128 et 256 canaux par niveau respectif).
3. **Strides (Sous-échantillonnage)** : Les *strides* configurés sont `[1, 1], [2, 2], [2, 2], [2, 2]`. 
   - Le premier bloc maintient la résolution spatiale.
   - Les 3 blocs suivants réduisent la taille de l'image d'un facteur 2 sur chaque axe (soit un sous-échantillonnage total de $2 \times 2 \times 2 = 8$).
4. **Padding Dynamique** : Puisque le modèle effectue un sous-échantillonnage d'un facteur global de 8, la taille de l'image en entrée doit impérativement être un multiple de 8. La méthode `_pad_to_divisor` vient dynamiquement ajouter du *padding* à l'image si ses dimensions (ex. taille du sinogramme) ne sont pas divisibles par 8. À la sortie, `_crop_to_shape` rogne l'image pour lui rendre ses dimensions initiales.
5. **Normalisation** : Le réseau utilise la `Instance Normalization` au lieu de la `Batch Normalization`, ce qui est souvent plus performant pour des tâches de génération/restauration image-à-image avec de très petits *batch sizes* (ex: 2).
6. **Pas de Deep Supervision** : L'option `deep_supervision` est désactivée, ce qui allège le modèle en s'assurant que la fonction de coût n'est calculée que sur la sortie finale du modèle.

---

## 2. Dataset : SinogramCompletionDataset

Le pipeline de données est défini dans la classe `SinogramCompletionDataset` (`src_2D/data/dataset_2d.py`). Il génère de manière dynamique des paires `(sinogramme_incomplet, sinogramme_complet, fantôme)`.

### Processus de Génération (Étape par Étape) :

1. **Création d'un Fantôme 2D (PhantomGenerator)** :
   - Au moment du chargement de la donnée, un "fantôme" (image 2D) est créé avec des valeurs comprises entre 0 et 1.
   - Le type de fantôme peut être aléatoire ou spécifique : `shepp_logan`, `ellipses` (plusieurs ellipses superposées), `blobs` (des tâches gaussiennes floues), `rectangles` ou un mix de tout ça (`mixed`).
   - Pour la robustesse, on applique au fantôme généré une **transformation rigide aléatoire** (rotation, redimensionnement et légère translation).

2. **Génération du Sinogramme Complet (Forward Projection)** :
   - On fait passer le fantôme au travers d'un projecteur implémenté avec l'API CUDA de la bibliothèque **ASTRA-Toolbox**.
   - ASTRA simule le processus d'acquisition par rayons X selon la géométrie DBT (Digital Breast Tomosynthesis) définie dans le fichier de configuration `geometry_conf_2d.py` (comme le rayon de la source, la taille des détecteurs, les angles, etc.).
   - Le résultat est le `full_sinogram` (le sinogramme contenant toutes les vues/projections simulées sur un arc complet).

3. **Ajout de Bruit (et Normalisation)** :
   - Bien qu'une méthode `_add_poisson_noise` soit implémentée pour simuler du bruit de type Poisson, elle agit actuellement comme un *placeholder* avec le bruit à 0.
   - Ensuite, le sinogramme complet est **divisé par une constante de normalisation** (`global_sino_norm = 100.0`) afin de ramener de façon approximative ses valeurs dans l'intervalle `[0, 1]`.

4. **Génération du Sinogramme Incomplet (Masquage)** :
   - Enfin, le système identifie quelles projections tombent **en dehors de la fenêtre d'acquisition simulée** de la DBT (qui est un arc partiel, défini entre `angle_min_deg` et `angle_max_deg`).
   - La méthode `_crop_to_acquired_views` crée le `incomplete_sinogram` en mettant tout simplement à 0 toutes les projections qui se trouvent en dehors de cet arc d'acquisition. Le réseau devra donc "deviner" ou reconstruire les informations masquées.

### En résumé pour l'entraînement :
À chaque itération, le Dataloader fournit :
- `incomplete` : L'entrée du réseau (le sinogramme masqué / partiel).
- `target` : La vérité terrain du réseau (le sinogramme complet non masqué).
- `phantom` : L'image d'origine simulée (optionnelle, surtout utilisée pour valider l'impact sur la reconstruction finale).
