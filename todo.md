# Todo List & Optuna Runs

## 1. Lancer les Optimisations Optuna (Multi-GPU)
Ces commandes sont prêtes à être exécutées. Elles distribueront automatiquement 1 job par GPU sur les 8 GPUs.

- [ ] **UNet 2D Classique**
  ```bash
  python src_2D/models/Unet2D/run_optuna.py --n-trials 50 --n-epochs 15 --n-jobs 8 --use-wandb
  ```
- [x] **UNet2D + HLCC (Helgason-Ludwig)** (Terminé : 400 trials, best SSIM 0.5924)
  ```bash
  python src_2D/models/Unet2dHLCC/run_optuna.py --n-trials 50 --n-epochs 15 --n-jobs 8 --use-wandb
  ```
- [/] **SinoSheavesNN (GNN Vues)** (En cours : 400 trials sur 8 GPU dans tmux `optuna_sinosheaves`, W&B connecté)
  ```bash
  python src_2D/models/SinoSheavesNN/run_optuna.py --n-trials 50 --n-epochs 15 --n-jobs 8 --batch-size 2 --use-wandb
  ```

## 2. Entraînement Final & Évaluation
Une fois les best params trouvés :
- [ ] Entraîner le **UNet2D** avec ses hyperparamètres optimisés.
- [/] Entraîner le **UNet2dHLCC** avec ses hyperparamètres optimisés (En cours : 300 epochs sur 8 GPU DDP).
- [ ] Entraîner le **SinoSheavesNN** avec ses hyperparamètres optimisés.
- [ ] Entraîner le **GLM (Graph Neural Network for Line Manifolds)** :
  ```bash
  python -m src_2D.models.GLM.run_training --use-wandb
  ```
- [ ] Lancer `evaluate_all.py` pour comparer les performances finales (SSIM, PSNR, MSE).

## 3. P4: Data Consistency
- [x] Résoudre le problème des discontinuités (ringing artifacts) dans HLCC.
