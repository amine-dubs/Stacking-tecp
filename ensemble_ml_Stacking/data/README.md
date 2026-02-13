# Datasets pour l'analyse de Stacking

Ce projet utilise **trois datasets de domaines différents** pour évaluer la généralisation du stacking et l'impact de la diversité des modèles.

---

## Dataset 1 : Ames Housing (Immobilier)

### Source
Le notebook charge les données directement depuis le package R `AmesHousing`.

```r
install.packages("AmesHousing")
library(AmesHousing)
data <- make_ames()
```

### Description
- **Observations** : 2,930 maisons
- **Variables** : 82 features (numériques + catégorielles)
- **Cible** : `Sale_Price` (prix de vente en dollars)
- **Transformation** : Converti en classification binaire (High/Low par rapport à la médiane ~$160,000)
- **Domaine** : Vente de maisons résidentielles à Ames, Iowa (2006-2010)

### Caractéristiques principales
- **Features numériques** : Surface habitable, nombre de chambres, année de construction, qualité globale, etc.
- **Features catégorielles** : Type de maison, quartier, état de la cuisine, présence de garage, etc.
- **Valeurs manquantes** : Présentes dans certaines colonnes (garage, sous-sol) → imputées par la médiane
- **Équilibre des classes** : 50/50 (par construction, car seuil sur la médiane)

### Référence
**De Cock, D. (2011)**. *Ames, Iowa: Alternative to the Boston Housing Data*. Journal of Statistics Education, 19(3).

**Lien original** : https://www.kaggle.com/datasets/prevek18/ames-housing-dataset

---

## Dataset 2 : Pima Indians Diabetes (Médical)

### Source
Le notebook charge les données directement depuis le package R `mlbench`.

```r
install.packages("mlbench")
library(mlbench)
data("PimaIndiansDiabetes")
```

### Description
- **Observations** : 768 femmes d'origine Pima (Arizona, USA)
- **Variables** : 8 features numériques
- **Cible** : `diabetes` (pos/neg - présence ou absence de diabète)
- **Domaine** : Dépistage médical du diabète de type 2

### Features (toutes numériques)
1. **pregnant** : Nombre de grossesses
2. **glucose** : Concentration de glucose plasmatique (test de tolérance au glucose oral)
3. **pressure** : Pression artérielle diastolique (mm Hg)
4. **triceps** : Épaisseur du pli cutané du triceps (mm)
5. **insulin** : Insuline sérique (mu U/ml)
6. **mass** : Indice de masse corporelle (poids en kg / (taille en m)²)
7. **pedigree** : Fonction de pedigree du diabète (historique familial)
8. **age** : Âge (années)

### Particularités et nettoyage
⚠️ **Valeurs impossibles** : Certaines variables ont des valeurs 0 qui sont physiquement impossibles (ex: glucose=0, pression=0, IMC=0). Ces valeurs représentent en réalité des **données manquantes**.

**Preprocessing appliqué :**
```r
# Remplacer les 0 impossibles par NA
zero_cols <- c("glucose", "pressure", "triceps", "insulin", "mass")
for (col in zero_cols) {
  data[[col]][data[[col]] == 0] <- NA
}

# Imputation par la médiane
for (col in zero_cols) {
  data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
}
```

### Équilibre des classes
- **Négative (pas de diabète)** : 500 observations (65%)
- **Positive (diabète)** : 268 observations (35%)
- ⚠️ Dataset légèrement déséquilibré → justifie l'utilisation de métriques comme AUC, Precision, Recall

### Référence
**Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988)**. *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*. Proceedings of the Annual Symposium on Computer Application in Medical Care, 261-265.

**Lien UCI** : https://archive.ics.uci.edu/ml/datasets/diabetes

---

## Dataset 3 : Bank Marketing (Finance / Marketing)

### Source
Le notebook télécharge les données directement depuis l'UCI Machine Learning Repository.

```r
# Download and extract from UCI
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
temp_zip <- tempfile(fileext = ".zip")
download.file(url, temp_zip, mode = "wb")
unzip(temp_zip, exdir = tempdir())
bank <- read.csv(file.path(tempdir(), "bank-additional/bank-additional-full.csv"), sep=";")
```

### Description
- **Observations** : 41,188 appels téléphoniques de campagnes de marketing
- **Variables** : 20 features (numériques + catégorielles)
- **Cible** : `y` (yes/no - client a souscrit un dépôt à terme)
- **Domaine** : Finance / Marketing direct, institution bancaire portugaise (2008-2010)

### Features
**Numériques:**
1. **age** : Âge du client
2. **duration** : Durée du dernier appel (en secondes) ⚠️ SUPPRIMÉE (leakage)
3. **campaign** : Nombre d'appels pour ce client dans cette campagne
4. **pdays** : Nombre de jours depuis dernier contact
5. **previous** : Nombre de contacts pré-campagne
6. **emp.var.rate** : Taux de variation de l'emploi
7. **cons.price.idx** : Indice des prix à la consommation
8. **cons.conf.idx** : Indice de confiance du consommateur
9. **euribor3m** : Taux EURIBOR 3 mois

**Catégorielles (one-hot encodées) :**
- **job** : Profession (12 catégories)
- **marital** : État civil (divorced, married, single)
- **education** : Éducation (4 catégories)
- **default** : Crédit par défaut (yes/no)
- **housing** : Crédit immobilier (yes/no)
- **loan** : Crédit personnel (yes/no)
- **contact** : Type de contact (cellular/telephone)
- **month** : Mois du dernier contact
- **day_of_week** : Jour de la semaine
- **poutcome** : Résultat de campagne précédente

### Particularités et nettoyage
⚠️ **Fuite d'information (Leakage)** : La colonne `duration` (durée de l'appel) n'est connu qu'APRÈS l'appel. Elle ne peut pas être utilisée en prédiction réelle.

**Preprocessing appliqué :**
```r
# Supprimer duration (leakage)
bank <- bank %>%
  select(-duration) %>%
  rename(target = y)

# One-hot encoding des variables catégorielles
bank_encoded <- bank
for (col in cat_cols) {
  dummies <- model.matrix(~ . - 1, data.frame(bank_encoded[[col]]))
  colnames(dummies) <- paste0(col, "_", colnames(dummies))
  bank_encoded[[col]] <- NULL
  bank_encoded <- cbind(bank_encoded, dummies[, -ncol(dummies)])
}

# Suppression des features à variance quasi-nulle
nzv_bank <- nearZeroVar(bank_encoded)
if (length(nzv_bank) > 0) {
  bank_encoded <- bank_encoded[, -nzv_bank]
}
```

### Équilibre des classes
- **No (pas de souscription)** : 36,548 observations (88.7%)
- **Yes (souscription)** : 4,640 observations (11.3%)
- ⚠️ Dataset **très déséquilibré** → justifie l'utilisation de AUC et F1-score plutôt que l'accuracy

### Pourquoi ce dataset pour le stacking ?
- **Taille adéquate** : 41,188 observations >> Pima (768) >> Ionosphere (351) → limite le surapprenti ssage du meta-modèle
- **Real-world complexity** : Données réelles de campagne marketing, non contrôlées
- **Features mixtes** : Nécessite un preprocessing significatif (comme Ames)
  - Numériques simples (age, duration, etc.)
  - Variables catégorielles (job, education, contact type, etc.)
  - Après encoding : **~52 features** au total
- **Classe minoritaire clair** : Cas de classification déséquilibrée (11.3% yes) → scenario réaliste
- **Baseline performance** : RF ~90% accuracy → laisse de la place pour les gains du stacking

### Référence
**Moro, S., Cortez, P., & Rita, P. (2014)**. *A data-driven approach to predict the success of bank telemarketing*. Decision Support Systems, 62, 22-31.

**Lien UCI** : https://archive.ics.uci.edu/ml/datasets/bank+marketing

---

## Comparaison des données

| Caractéristique | Ames Housing | Pima Diabetes | Bank Marketing |
|----------------|--------------|---------------|----------------|
| **Domaine** | Immobilier | Médical | Finance / Marketing |
| **Observations** | 2,930 | 768 | 41,188 |
| **Features** | 82 (mixtes) | 8 (numériques) | 20 (mixtes) |
| **Features après preprocessing** | ~40 | 8 | ~52 |
| **Type de problème** | Classification binaire | Classification binaire | Classification binaire |
| **Équilibre des classes** | 50/50 (équilibré) | 65/35 (léger déséquilibre) | 89/11 (très déséquilibré) |
| **Baseline accuracy** | ~90% | ~75% | ~90% |
| **Difficulté** | Modérée | Élevée | Modérée-Élevée |
| **Valeurs manquantes** | Oui (garage, sous-sol) | Oui (masquées en 0) | Non (sauf by design) |
| **Complexité features** | Élevée (catégorielles + numériques) | Faible (numériques simples) | Élevée (catégorielles + macro  indicateurs) |
| **Corrélation modèles attendue** | Haute (~0.94) | Moyenne-haute (~0.88) | Moyenne |

---

## Pourquoi ces trois datasets ?

### Complémentarité
1. **Taille très différente** : Très grand (Bank Marketing, 41,188) vs Grand (Ames, 2,930) vs Petit (Pima, 768) → teste la robustesse du stacking à différentes échelles
2. **Domaines différents** : Immobilier vs Médical vs Finance/Marketing → évalue la généralisation inter-domaines
3. **Complexité features différente** : Features mixtes vs purement numériques simples vs catégorielles + macro
4. **Équilibre des classes très différent** : 50/50 vs 65/35 vs 89/11 → impact de l'imbalance sur le stacking
5. **Diversité des modèles** : Haute corrélation (Ames) vs moyenne (Pima/Bank) → démontre l'impact du `diversity on gains

### Questions de recherche
- La diversité des modèles (corrélation faible) est-elle le facteur clé du succès du stacking ?
- Le stacking apporte-t-il plus sur un problème difficile (Pima) ou avec plus de données (Bank Marketing) ?
- L'impact de la taille du dataset sur les gains du stacking ?
- Comment l'imbalance des classes affecte-t-elle le stacking ?
- Le stacking est-il universel ou spécifique au domaine ?

---

## Notes sur l'utilisation

### Pas de fichiers CSV nécessaires
Les trois datasets sont **chargés automatiquement** par le notebook R :
- `AmesHousing::make_ames()` pour Ames Housing
- `data("PimaIndiansDiabetes", package="mlbench")` pour Pima
- `download.file(url) + unzip()` pour Bank Marketing (téléchargé depuis UCI)

✅ **Aucun téléchargement manuel requis** - le notebook gère tout.

### Pour exporter manuellement (optionnel)

Si vous souhaitez sauvegarder les données brutes en CSV :

```r
# Ames Housing
library(AmesHousing)
ames <- make_ames()
write.csv(ames, "data/ames_housing.csv", row.names = FALSE)

# Pima Indians Diabetes
library(mlbench)
data("PimaIndiansDiabetes")
write.csv(PimaIndiansDiabetes, "data/pima_diabetes.csv", row.names = FALSE)

# Bank Marketing
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
temp_zip <- tempfile(fileext = ".zip")
download.file(url, temp_zip, mode = "wb")
unzip(temp_zip, exdir = "data/")
# Ou simplement utiliser: bank <- read.csv("data/bank-additional/bank-additional-full.csv", sep=";")
```

Mais ce n'est **pas nécessaire** pour l'exécution du projet.

---

## Licences et citations

### Ames Housing
- **Licence** : Public Domain / CC0
- **Citation** : De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data. *Journal of Statistics Education*, 19(3).

### Pima Indians Diabetes
- **Licence** : Domaine public (National Institute of Diabetes and Digestive and Kidney Diseases)
- **Citation** : Smith, J.W., et al. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Symposium on Computer Applications and Medical Care*, 261-265.

### Bank Marketing
- **Licence** : Creative Commons Attribution 4.0
- **Citation** : Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22-31.
