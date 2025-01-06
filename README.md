# Classification multi-étiquette pour les synopsis de films

Ce dépôt contient un flux de travail complet pour la classification multi-étiquette des synopsis de films. Le projet exploite Apache Spark et les bibliothèques de machine learning de PySpark pour prétraiter les données, entraîner plusieurs modèles de machine learning et évaluer leurs performances.

## Aperçu
L'objectif de ce projet est de classifier les synopsis de films en plusieurs genres ou étiquettes (par exemple, `murder`, `romantic`, `comedy`) en utilisant une **méthode de transformation de problème** pour la classification multi-étiquette. Chaque étiquette est traitée comme une tâche de classification binaire indépendante.

## Méthodologie
La méthode **Binary Relevance** est utilisée, transformant le problème multi-étiquette en plusieurs problèmes de classification binaire. Pour chaque étiquette, un classificateur binaire séparé est entraîné. Les modèles de machine learning suivants sont implémentés et évalués :

- **Régression logistique**
- **Forêt aléatoire**
- **Arbres de Gradient Boosting (GBT)**
- **Machines à vecteurs de support (SVM)**
- **Naive Bayes**

Chaque modèle est entraîné indépendamment pour toutes les étiquettes sélectionnées, et leurs performances sont mesurées à l'aide de métriques d'évaluation standard.

## Pipeline de données
1. **Chargement et prétraitement des données**
   - Charger un fichier CSV contenant les synopsis de films et les étiquettes associées.
   - Nettoyer les données textuelles en les convertissant en minuscules, en supprimant la ponctuation et en filtrant les IDs IMDb.
   - Fractionner la colonne `tags` en lignes individuelles.

2. **Ingénierie des caractéristiques**
   - Tokenisation du texte.
   - Suppression des mots vides, ponctuation et des caractéres spéciaux
   - Transformation du texte en vecteurs de caractéristiques à l'aide de **HashingTF**.
   - Exploration de word2vec et TF-IDF

3. **Division des données**
   - Diviser les données en ensembles d'entraînement, de test et de validation.

4. **Entraînement des modèles**
   - Entraîner des classificateurs binaires séparés pour chaque étiquette à l'aide des modèles sélectionnés.

5. **Évaluation**
   - Mesurer les performances de chaque modèle à l'aide des métriques suivantes :
     - **Précision**
     - **Score F1**

## Structure du projet
- **data_preprocessing.py**
  - Contient des fonctions pour initialiser Spark, charger et prétraiter les données, et sauvegarder les ensembles de données.

- **models/**
  - `train_logistic_regression.py` : Entraîner les modèles de régression logistique.
  - `train_random_forest.py` : Entraîner les modèles de forêt aléatoire.
  - `train_gbt.py` : Entraîner les modèles d'arbres de Gradient Boosting.
  - `train_naive_bayes.py` : Entraîner les modèles de Naive Bayes.
  - `train_svm.py` : Entraîner les modèles de machines à vecteurs de support.

- **evaluation.py**
  - Fonctions utilitaires pour afficher et comparer les résultats des modèles.

- **main.py**
  - Orchestration complète du pipeline, du prétraitement des données à l'entraînement et l'évaluation des modèles.

## Installation et exécution

1. Installez les dépendances :
   ```bash
   pip install pyspark matplotlib pandas
   ```

2. Placez le fichier de données `mpst_full_data.csv` sur kaggle:
```bash
   https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags/data
```
4. Exécutez le script principal :
   ```bash
   python main.py
   ```

## Résultats
Les performances des modèles sont comparées en utilisant la **précision moyenne** et le **score F1 moyen** pour toutes les étiquettes :

| Métrique           | Régression Logistique | Forêt Aléatoire | Gradient Boosted Trees | SVM   | Naive Bayes |
|--------------------|-----------------------|-----------------|-------------------------|-------|-------------|
| **Précision Moyenne** | 0.75186               | 0.81810         | -                 | 0.7102| 95561      |
| **Score F1 Moyen**  | 0.73529               | 0.74310         | -                       | -     | -           |

## Détails de l'approche
- **Méthode de transformation de problème** :
  - La classification multi-étiquette est transformée en plusieurs tâches de classification binaire.
  - Chaque modèle prédit indépendamment la présence ou l'absence de chaque étiquette.

## Travaux futurs
- Implémenter des techniques avancées de multi-étiquettes comme **Classifier Chains** ou **Label Powerset**.
- Intégrer des modèles d'apprentissage profond pour l'extraction des caractéristiques et la classification.
