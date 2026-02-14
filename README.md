# Ensemble Learning : Stacking 2 Niveaux vs Modèles Individuels en R

## Description du Projet

Ce projet implémente et compare une architecture de **Stacking à 2 niveaux** avec des modèles individuels sur **trois datasets de domaines différents** pour tirer des conclusions généralisables sur l'efficacité du stacking.

L'objectif est de démontrer que la combinaison intelligente de modèles diversifiés (ensemble learning) surpasse les modèles utilisés individuellement, et d'analyser comment cette amélioration varie selon la **diversité des modèles** (corrélation des prédictions), la **taille du dataset**, et la **dimensionnalité des features**.

---

## Architecture du Stacking

```
┌─────────────────────────────────────────────────────────────┐
│                   DONNÉES D'ENTRÉE (3 Datasets)                        │
│  Ames Housing | Pima Diabetes | Bank Marketing (Financial/Commercial)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          │     NIVEAU 0 (Base)     │
          │   5 Modèles Diversifiés │
          ├─────────────────────────┤
          │  🌲 Random Forest       │
          │  📊 SVM (Radial)        │
          │  📈 Régression Logist.  │
          │  🎯 KNN                 │
          │  📉 Naive Bayes         │
          └────────────┬────────────┘
                       │
              Out-of-Fold Predictions
              (Validation Croisée 5-Fold)
                       │
          ┌────────────┼────────────┐
          │    NIVEAU 1 (Meta)      │
          │   2 Meta-Modèles        │
          ├─────────────────────────┤
          │  🔷 Ridge Regression    │
          │  🚀 XGBoost             │
          └────────────┬────────────┘
                       │
              Prédictions Finales
```

### Pourquoi ces choix architecturaux ?

#### Modèles de Niveau 0 (Base Learners)

| Modèle | Type | Justification |
|--------|------|---------------|
| **Random Forest** | Bagging, Non-linéaire | Robuste aux outliers, gère bien les interactions entre features, faible variance grâce au bagging de multiples arbres |
| **SVM (Radial)** | Kernel method | Excellent en haute dimension, frontières de décision complexes via le noyau RBF, approche par marge maximale |
| **Régression Logistique** | Linéaire | Modèle linéaire simple → apporte de la diversité face aux modèles non-linéaires, interprétable |
| **KNN** | Instance-based | Approche non-paramétrique, capture les patterns locaux dans l'espace des features, complément des méthodes globales |
| **Naive Bayes** | Probabiliste | Hypothèse d'indépendance conditionnelle → perspective très différente des autres modèles |

**La clé : LA DIVERSITÉ** - Des modèles aux hypothèses différentes capturent des patterns complémentaires.

#### Modèles de Niveau 1 (Meta-Learners)

| Meta-Modèle | Avantages | Inconvénients | Quand l'utiliser |
|-------------|-----------|---------------|------------------|
| **Ridge (L2)** | Régularisation empêche l'overfitting, interprétable (montre les poids de chaque modèle), stable | Ne capture pas les interactions non-linéaires entre prédictions | Bon choix par défaut, surtout avec peu de meta-features |
| **XGBoost** | Capture les interactions non-linéaires entre prédictions de base, excellente performance | Plus complexe, risque d'overfitting avec peu de features | Quand les patterns sont complexes et qu'on a suffisamment de données |

### Stacking (OOF) vs Blending

| Aspect | Stacking (OOF) | Blending (Holdout) |
|--------|----------------|-------------------|
| **Méthode** | Validation croisée K-fold | Split train/blend/test fixe |
| **Données utilisées** | 100% pour entraînement | ~75% seulement |
| **Variance** | Plus faible (moyenne sur K folds) | Plus élevée (1 seul split) |
| **Complexité** | Plus élevée (K × N modèles) | Plus simple |
| **Risque overfitting** | Plus faible | Plus élevé |
| **Recommandation** | ✅ À privilégier | Acceptable pour prototypage rapide |

---

## Datasets utilisés

### Dataset 1 : Ames Housing (Immobilier)

- **Source** : [AmesHousing R package](https://cran.r-project.org/package=AmesHousing) - Dean De Cock (2011)
- **Taille** : 2,930 observations × 82 variables
- **Type de features** : Mixte (numériques + catégorielles)
- **Cible** : `Sale_Price` → transformé en classification binaire (High/Low par rapport à la médiane ~$160,000)
- **Domaine** : Immobilier, prix des maisons à Ames, Iowa
- **Baseline accuracy** : ~90% (modèles individuels)

**Pourquoi ce dataset ?**
- Riche en features hétérogènes → teste la robustesse des modèles face à la complexité
- Taille suffisante pour le stacking sans surapprentissage
- Problème réaliste et bien documenté dans la littérature

### Dataset 2 : Pima Indians Diabetes (Médical)

- **Source** : [mlbench R package](https://cran.r-project.org/package=mlbench) - National Institute of Diabetes and Digestive and Kidney Diseases
- **Taille** : 768 observations × 8 variables
- **Type de features** : Uniquement numériques (glucose, pression artérielle, IMC, âge, etc.)
- **Cible** : `diabetes` (pos/neg - présence de diabète)
- **Domaine** : Médical, dépistage du diabète
- **Baseline accuracy** : ~75% (modèles individuels)

**Pourquoi ce dataset ?**
- **Domaine différent** : médical vs immobilier → teste la généralisation du stacking
- **Taille réduite** : 768 vs 2,930 observations → évalue la robustesse du stacking avec moins de données
- **Features uniquement numériques** : pas de variables catégorielles → simplifie le preprocessing
- **Bruit et valeurs manquantes** : valeurs impossibles (0 pour glucose, pression) → teste la robustesse
- **Problème plus difficile** : baseline ~75% vs ~90% pour Ames → teste si le stacking aide davantage sur un problème complexe

### Dataset 3 : Bank Marketing (Financial/Commercial)

- **Source** : [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) - Portuguese Banking Institution
- **Taille** : 41,188 observations × 20 variables
- **Type de features** : Mixte (numériques + catégorielles)
- **Cible** : `y` (yes/no - client a souscrit un dépôt à terme)
- **Domaine** : Finance / Marketing, campagnes de marketing direct par téléphone
- **Baseline accuracy** : ~90% (Random Forest)

**Pourquoi ce dataset ?**
- **Taille adéquate** : 41,188 obs >> Ionosphere (351) → prévient le surapprenti ssage du meta-modèle
- **Real-world data** : données financières/commerciales réelles, non contrôlées
- **Preprocessing complexe** : variables catégorielles (nombreuses) requièrent encoding (comme Ames)
- **Classes déséquilibrées** : 88.7% no, 11.3% yes → scenario réaliste et problématique
- **Baseline 90%** → laisse de la place pour que le stacking apporte des gains
- **Complément multi-domaines** : Immobilier (Ames) + Médical (Pima) + Finance (Bank Marketing)

### Pourquoi comparer trois datasets ?

La comparaison multi-datasets permet de tirer des **conclusions généralisables** :
1. **Impact de la taille** : Le stacking profite-t-il davantage d'un dataset plus grand ?
2. **Impact de la complexité** : Sur quel type de problème (facile vs difficile) le stacking apporte-t-il le plus ?
3. **Impact de la diversité des features** : Les features mixtes vs purement numériques vs haute dimension affectent-elles le gain ?
4. **Impact de la diversité des modèles** : La corrélation entre prédictions de base est-elle le facteur clé du succès du stacking ?
5. **Généralisation inter-domaines** : Le stacking est-il universel ou spécifique au domaine ?

---

## Structure du Projet

```
ensemble_ml_Stacking/
├── README.md                          # Ce fichier
├── data/
│   └── README.md                      # Instructions pour les datasets
├── notebooks/
│   └── stacking_dual_dataset.ipynb    # 🎯 NOTEBOOK PRINCIPAL (à exécuter)
├── diagrams/
│   └── stacking_architecture.drawio   # Architecture visuelle (draw.io)
├── output/                            # 📊 Résultats générés après exécution
│   ├── results_ames_housing.csv
│   ├── results_pima_diabetes.csv
│   ├── results_bank_marketing.csv
│   ├── correlation_matrix_*.csv
│   ├── cross_dataset_comparison.csv
│   ├── corrplot_*.png
│   ├── accuracy_comparison_*.png
│   ├── roc_curves_*.png
│   ├── training_times_*.png
│   ├── correlation_vs_stacking_gain.png
│   ├── dataset_profile_comparison.png
│   └── ...
└── images/                            # (vide - pour exports supplémentaires)
```

---

## Installation & Exécution

### Prérequis

- **R** (≥ 4.0)
- **IRkernel** (pour exécuter R dans Jupyter)
- **Jupyter Notebook** / **VS Code** avec extension Jupyter

### Installation des packages R

Le notebook installe automatiquement les packages manquants, mais vous pouvez les installer manuellement :

```r
install.packages(c(
  "caret", "randomForest", "e1071", "class", "naivebayes",
  "glmnet", "xgboost", "ggplot2", "corrplot", "reshape2",
  "dplyr", "tidyr", "pROC", "scales", "gridExtra",
  "data.table", "AmesHousing", "mlbench"
))
```

### Installation IRkernel (si pas déjà fait)

```r
install.packages('IRkernel')
IRkernel::installspec()
```

### Exécution du projet

1. **Cloner le dépôt** :
```bash
git clone https://github.com/votre-username/ensemble_ml_Stacking.git
cd ensemble_ml_Stacking
```

2. **Ouvrir le notebook principal** :
```bash
jupyter notebook notebooks/stacking_dual_dataset.ipynb
```
Ou ouvrir dans VS Code avec l'extension Jupyter.

3. **Exécuter toutes les cellules** (`Run All`) :
   - Le notebook charge automatiquement les trois datasets
   - Génère tous les modèles, prédictions et visualisations
   - Sauvegarde tous les résultats dans `output/`

4. **Consulter les résultats** :
   - Tableaux de performance dans le notebook
   - Fichiers CSV et graphiques dans le dossier `output/`

---

## Méthodologie détaillée

### 1. Prétraitement

**Pour Ames Housing :**
- Transformation de `Sale_Price` en classification binaire (High/Low basé sur la médiane)
- Sélection de 19 features numériques + 5 catégorielles (basé sur la corrélation)
- Imputation des valeurs manquantes (médiane pour numériques)
- One-hot encoding des variables catégorielles
- Suppression des features à variance quasi-nulle
- Résultat : **21 features finales** après preprocessing

**Pour Pima Indians Diabetes :**
- Correction des valeurs impossibles : 0 pour glucose, pression, etc. → remplacés par NA
- Imputation par la médiane
- Pas d'encoding nécessaire (déjà purement numérique)
- Résultat : **8 features** (toutes numériques)

**Pour Bank Marketing :**
- Suppression de la colonne `duration` (fuite d'information - contient des infos post-appel)
- Identification des 10 colonnes catégorielles et 8 numériques
- One-hot encoding des variables catégorielles
- Suppression des features à variance quasi-nulle
- Résultat : **52 features** (après encoding des catégorielles)

**Commun aux trois :**
- Split **80% train / 20% test** (stratifié pour conserver les proportions de classes)
- **Normalisation** (centrage-réduction) : indispensable pour SVM et KNN sensibles à l'échelle

### 2. Validation Croisée Out-of-Fold (OOF)

```
Pour chaque modèle de base :
  Pour chaque fold k (k=1..5) :
    1. Entraîner le modèle sur les 4 autres folds
    2. Prédire les probabilités sur le fold k → stockées dans la matrice OOF (train)
    3. Prédire les probabilités sur le test set → moyennées sur les 5 folds

Résultat : Matrice OOF train (N_train × 5) et test (N_test × 5)
```

**Pourquoi l'OOF ?**
- ✅ **Pas de data leakage** : Chaque prédiction OOF est faite sur des données non vues pendant l'entraînement de ce fold
- ✅ **100% des données** sont utilisées pour générer les méta-features (vs blending qui "perd" des données)
- ✅ **Estimations plus stables** : Moyenne sur 5 folds réduit la variance des prédictions

### 3. Entraînement du Meta-Modèle

**Données d'entrée du meta-modèle :**
- **Features** : Les 5 colonnes de prédictions OOF (une par modèle de base)
- **Target** : Les vraies classes du training set

**Ridge Regression (L2) :**
- Cross-validation pour trouver le `lambda` optimal (paramètre de régularisation)
- Retourne les **poids** de chaque modèle de base → interprétabilité
- Combine linéairement les prédictions de base

**XGBoost :**
- Paramètres conservateurs (`max_depth=2`) pour éviter l'overfitting avec seulement 5 features
- Cross-validation pour trouver le nombre optimal d'itérations (early stopping)
- Peut capturer les **interactions non-linéaires** entre les prédictions de base

### 4. Blending (Comparaison)

- Split : **75% train / 25% blend** (du training set initial)
- Modèles de base entraînés sur les 75%
- Prédictions sur le blend set (25%) → méta-features
- Meta-modèle entraîné sur ces prédictions
- **Résultat attendu** : Performance légèrement inférieure à l'OOF (moins de données, plus de variance)

---

## Résultats attendus et analyses

### Métriques de comparaison

| Métrique | Description | Pourquoi l'utiliser |
|----------|-------------|---------------------|
| **Accuracy** | Taux de classification correcte | Mesure globale de performance |
| **AUC-ROC** | Aire sous la courbe ROC | Capacité de discrimination entre classes |
| **Precision** | Proportion de vrais positifs parmi les prédictions positives | Important si le coût des faux positifs est élevé |
| **Recall** | Proportion de vrais positifs parmi les positifs réels | Important si le coût des faux négatifs est élevé |
| **F1-Score** | Moyenne harmonique de Precision et Recall | Équilibre entre les deux |
| **Temps d'entraînement** | Coût computationnel | Compromis performance/temps |

### Visualisations produites

Le notebook génère automatiquement **14+ visualisations par dataset** :

**Analyse de la diversité (Niveau 0) :**
1. **Matrice de corrélation** des prédictions OOF (corrplot + heatmap)
   - Interprétation : Corrélation < 0.7 = bonne diversité
2. **Distribution des probabilités prédites** par modèle (densité par classe)
   - Montre si les modèles ont des biais différents

**Comparaison de performance :**
3. **Barplot des Accuracy** (Stacking vs Individuels vs Blending)
4. **Courbes ROC superposées** avec AUC
5. **Comparaison multi-métriques** (Accuracy/AUC/F1)

**Analyse temporelle :**
6. **Temps d'entraînement** par modèle (barplot)
7. **Compromis Performance vs Temps** (scatter plot)

**Analyse cross-dataset (3 datasets) :**
8. **Comparaison side-by-side** des accuracy (Ames vs Pima vs Bank Marketing)
9. **Gains du stacking** (points de pourcentage) par métrique et dataset
10. **Corrélations Niveau 0** comparées entre datasets
11. **Corrélation vs Gain du Stacking** : Diagramme montrant la relation entre diversité des modèles et gain du stacking
12. **Profil comparatif des 3 datasets** : Taille, features, corrélation, gain normalisés

### Conclusions attendues

**Hypothèses à vérifier :**

1. **Le stacking améliore-t-il toujours les performances ?**
   - Dépend de la diversité des modèles de base (corrélation des prédictions)
   - Si corrélation > 0.9 : gain marginal ou négatif
   - Si corrélation < 0.8 : gain significatif possible

2. **Impact de la taille du dataset**
   - Bank Marketing (41,188 obs) devrait montrer un gain de stacking très stable
   - Ames (2,930 obs) devrait avoir des gains stables
   - Pima (768 obs) risque un surapprentissage du meta-modèle

3. **Impact de la diversité des modèles (FACTEUR CLÉ)**
   - La corrélation entre prédictions de base détermine le succès du stacking
   - Bank Marketing (52 features mixtes) devrait produire des patterns intéressants
   - Ames (haute corrélation ~0.94) → gain minimal

4. **OOF vs Blending**
   - OOF devrait systématiquement surpasser le blending (+0.5% à +2% d'accuracy)
   - Écart plus grand sur petit dataset (Pima) où "perdre" 25% des données a plus d'impact

5. **Choix du meta-modèle (Ridge vs XGBoost)**
   - Ridge : plus stable, surtout sur Pima avec peu de données
   - XGBoost : peut surpasser Ridge sur Ames avec plus de données et patterns complexes

---

## Concepts clés expliqués

### Pourquoi le Stacking fonctionne-t-il ?

1. **Diversité = Complémentarité**
   - Des modèles aux hypothèses différentes font des erreurs sur des exemples différents
   - Le meta-modèle apprend à exploiter leurs forces respectives

2. **Correction d'erreurs**
   - Si un modèle est systématiquement trop confiant ou pas assez, le meta-modèle peut corriger ce biais
   - Exemple : Si RF prédit toujours "High" avec 0.9 de probabilité mais se trompe 20% du temps, le meta-modèle apprendra à downweighter ces prédictions

3. **Réduction de variance**
   - Théorème de la "sagesse des foules" : La moyenne de prédicteurs indépendants réduit la variance
   - Le stacking va plus loin qu'une simple moyenne : il apprend la **pondération optimale**

4. **Non-linéarité (avec XGBoost)**
   - XGBoost peut apprendre des règles comme "Si RF dit High ET SVM dit Low, alors Low"
   - Capture les interactions entre modèles de base

### Pourquoi la validation OOF évite le data leakage ?

**Mauvaise approche (avec leakage) :**
```
1. Entraîner RF sur tout le train set
2. Prédire sur tout le train set → méta-features
3. Entraîner Ridge sur ces méta-features
❌ Problème : RF a déjà vu ces données, prédictions trop optimistes
```

**Bonne approche (OOF) :**
```
1. Pour le fold 1 : Entraîner RF sur folds 2-5, prédire sur fold 1
2. Pour le fold 2 : Entraîner RF sur folds 1,3-5, prédire sur fold 2
...
✅ Résultat : Chaque prédiction OOF est out-of-sample
```

### Le stacking est-il toujours la meilleure approche ?

**Non ! Le stacking ne vaut la peine que si :**
- ✅ Les modèles de base sont **vraiment diversifiés** (corrélations < 0.8)
- ✅ Vous avez **suffisamment de données** (règle empirique : N > 500 pour le train)
- ✅ Le **coût computationnel** est acceptable (5 modèles × K folds = 25 entraînements)
- ✅ Le problème est **suffisamment complexe** (si un modèle simple suffit, le stacking n'apportera rien)

**Quand utiliser des alternatives :**
- **Vote majoritaire** : Si vous voulez quelque chose de simple et interprétable
- **Simple averaging** : Si vos modèles ont des performances similaires
- **Un seul modèle bien tunné** : Si vous manquez de données ou de temps

---

## Références académiques

### Stacking et Ensemble Learning
- **Wolpert, D.H. (1992)**. *Stacked Generalization*. Neural Networks, 5(2), 241-259.
  - 📄 Article fondateur du stacking
- **Breiman, L. (1996)**. *Stacked Regressions*. Machine Learning, 24(1), 49-64.
  - 📄 Extension du stacking aux problèmes de régression

### Datasets
- **De Cock, D. (2011)**. *Ames, Iowa: Alternative to the Boston Housing Data*. Journal of Statistics Education, 19(3).
  - 📄 Description du dataset Ames Housing
- **Smith, J.W., et al. (1988)**. *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*. Proceedings of the Symposium on Computer Applications and Medical Care, 261-265.
  - 📄 Dataset Pima Indians Diabetes original
- **Moro, S., Cortez, P., & Rita, P. (2014)**. *A data-driven approach to predict the success of bank telemarketing*. Decision Support Systems, 62, 22-31.
  - 📄 Dataset Bank Marketing original

### Théorie de l'Ensemble Learning
- **Hastie, T., Tibshirani, R., & Friedman, J. (2009)**. *The Elements of Statistical Learning*. Springer.
  - 📖 Chapitre 8 : Model Inference and Averaging
- **Zhou, Z.-H. (2012)**. *Ensemble Methods: Foundations and Algorithms*. CRC Press.
  - 📖 Référence complète sur les méthodes d'ensemble

---

## Auteur

- **Nom**: Bellatreche Mohamed Amine
- **GitHub**: [aminedubs](https://github.com/amine-dubs)
- **Contact**: aminedubs@gmail.com

---

## Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## FAQ

**Q : Combien de temps prend l'exécution complète ?**
R : ~5-15 minutes selon votre machine (principalement le SVM avec validation croisée).

**Q : Puis-je utiliser mes propres datasets ?**
R : Oui ! Le code est modulaire. Utilisez la fonction `run_stacking_pipeline()` avec vos données preprocessées (X_train, y_train, X_test, y_test).

**Q : Pourquoi 5 folds et pas 10 ?**
R : Compromis variance/bias. 5 folds est standard pour des datasets de taille moyenne. Avec Pima (768 obs), 10 folds donnerait des folds trop petits (68 obs/fold).

**Q : Le stacking marche-t-il pour la régression ?**
R : Oui ! Même principe, remplacez juste les métriques de classification par MAE/RMSE/R².

**Q : Dois-je toujours utiliser Ridge ET XGBoost comme meta-modèles ?**
R : Non, c'est pour comparer. En production, choisissez-en un seul (souvent Ridge pour la simplicité).

---

## Prochaines étapes possibles

1. **Feature engineering avancé** : Créer des interactions, polynômes, etc.
2. **Hyperparameter tuning** : Grid search sur les modèles de base
3. **Stacking multi-niveaux** : Ajouter un 3ème niveau (attention à l'overfitting !)
4. **Autres datasets** : Tester sur d'autres domaines (finance, NLP, vision...)
5. **Deployment** : Packager le meilleur modèle avec FastAPI/Plumber
