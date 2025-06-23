# 📉 Prédiction du Churn & Rétention Optimisée par Intelligence Artificielle (Ooredoo)

## 🎯 Objectif du projet
Ce projet vise à **prédire les clients à risque de churn** (désabonnement) dans le secteur des télécommunications et à proposer des **actions de rétention automatisées** grâce à l’intelligence artificielle, en particulier les modèles de Machine Learning et les grands modèles de langage (LLM).

---

## 📁 Étapes du pipeline

### 1. 🔧 Traitement des données
- Suppression des colonnes avec trop de valeurs manquantes
- Imputation des valeurs manquantes par méthodes déterministes et stochastiques
- Encodage des variables catégorielles
- Normalisation et transformation des distributions (Box-Cox, Quantile, etc.)

### 2. 📊 Exploration des données
- Analyse univariée et bivariée
- Étude des corrélations (Pearson, Cramer V)
- Profilage des clients churners vs non-churners

### 3. ⏱️ Analyse de survie
- Utilisation de Kaplan-Meier pour la courbe de survie globale
- Modèle de Cox pour quantifier l’effet de l’ancienneté, de l’âge, etc.
- Identification des tranches critiques (3–12 mois)

### 4. 🤖 Modélisation du churn
- Équilibrage des classes avec **SMOTETomek** et **ADASYN**
- Entraînement de plusieurs modèles :
  - Régression Logistique (baseline)
  - Random Forest, CatBoost, XGBoost, LightGBM
  - Réseau de neurones profond (MLP)
  - **TabNet** (réseau avec attention pour données tabulaires)
- Évaluation via **F1-score**, **AUC**, courbes **ROC/PR**

### 5. 🧠 Interprétabilité & IA Générative
- Explication locale et globale des modèles via **SHAP** et **LIME**
- Intégration d’un LLM (**LLaMA 3 70B via Groq API**) pour :
  - Générer une **explication textuelle personnalisée**
  - Rédiger un **email de rétention marketing** dynamique

---

## 🧰 Technologies utilisées

| Catégorie | Librairies / Frameworks |
|----------|--------------------------|
| Traitement des données | `pandas`, `numpy`, `scikit-learn`, `PowerTransformer`, `SMOTETomek`, `ADASYN` |
| Modélisation | `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `keras`, `tabnet` |
| Visualisation | `matplotlib`, `seaborn`, `shap`, `lime` |
| Analyse de survie | `lifelines` |
| IA Générative | `LLaMA 3`, `Groq API`, `OpenAI-compatible API` |
| Application | `Streamlit` |

---

## 📈 Résultats clés

- **F1-score pour les churners** :
  - Réseau de neurones profond : **0.86**
  - TabNet : **0.85**
  - XGBoost/CatBoost : **0.81**
- **Application web complète** pour :
  - Prédire le churn d’un client
  - Afficher l’importance des variables
  - Générer automatiquement une explication et un email

---

## 🚀 Lancer l'application

```bash
streamlit run app.py
