# === app_churn_final.py ===

import streamlit as st
st.set_page_config(page_title="Prédiction Churn Ooredoo", layout="wide")  # <--- ICI EN PREMIER !

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix
)
from tensorflow.keras.models import load_model
from pytorch_tabnet.tab_model import TabNetClassifier
from openai import OpenAI

warnings.filterwarnings("ignore")

# ==== OpenAI Client ====
client = OpenAI(
    api_key="XXXXXXXXXXXXXXXXXXXXXXXXXX",
    base_url="https://api.groq.com/openai/v1"
)

# ==== Data Loading ====
@st.cache_data
def load_data():
    df_logreg = pd.read_csv("final_df_transformed_per_period.csv")
    df_other  = pd.read_csv("final_df_encoded.csv")
    return df_logreg, df_other

df_logreg, df_other = load_data()

features_base = [c for c in df_logreg.columns if c not in ["subscriber_id", "churn"]]

# ==== Model Loading ====
logreg_model     = joblib.load("logreg_model_optimal.pkl")
threshold_logreg = joblib.load("logreg_threshold.pkl")["best_threshold"]
rf_model         = joblib.load("best_random_forest_model.pkl")
dt_model         = joblib.load("best_decision_tree.pkl")
catboost_model   = joblib.load("catboost_best_model.pkl")
xgb_model        = joblib.load("xgb_model.pkl")
lgb_model        = joblib.load("best_model_lgb.pkl")
scaler_logreg    = joblib.load("scaler_logreg.pkl")
keras_model      = load_model("modele_churn_prediction.keras")
keras_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
tabnet_model     = TabNetClassifier()
tabnet_model.load_model("tabnet_churn_model.zip")

# ==== Features for each model ====
expected_features_logreg  = [c for c in df_logreg.columns if c not in ["subscriber_id", "churn"]]
expected_features_rf      = rf_model.feature_names_in_
expected_features_dt      = dt_model.feature_names_in_
expected_features_cb      = catboost_model.feature_names_
expected_features_xgb     = xgb_model.get_booster().feature_names
expected_features_lgb     = lgb_model.feature_name_
expected_features_tabnet  = expected_features_xgb
expected_features_nn      = expected_features_logreg

# ==== Split Train/Test ====
train_df_logreg, test_df_logreg = train_test_split(df_logreg, test_size=0.2, stratify=df_logreg["churn"], random_state=42)
train_df_other,  test_df_other  = train_test_split(df_other,  test_size=0.2, stratify=df_other["churn"], random_state=42)
y_test = test_df_logreg["churn"].values

# ==== Harmonisation des colonnes ====
def harmonize_features(df, expected_features):
    dfc = df.copy()
    for feat in expected_features:
        if feat not in dfc.columns:
            dfc[feat] = 0
    return dfc[expected_features]

# ==== Préparation des X_test pour chaque modèle ====
X_test_logreg      = harmonize_features(test_df_logreg, expected_features_logreg)
X_test_logreg_scaled = scaler_logreg.transform(X_test_logreg)
X_test_rf          = harmonize_features(test_df_other, expected_features_rf)
X_test_dt          = harmonize_features(test_df_other, expected_features_dt)
X_test_cb          = harmonize_features(test_df_other, expected_features_cb)
X_test_xgb         = harmonize_features(test_df_other, expected_features_xgb)
X_test_lgb         = harmonize_features(test_df_other, expected_features_lgb)
X_test_tabnet      = harmonize_features(test_df_other, expected_features_tabnet).to_numpy().astype(np.float32)
X_test_nn          = X_test_logreg_scaled

# ==== Seuils de décision ====
thresholds = {
    "LogReg":        threshold_logreg,
    "Random Forest": 0.5,
    "Decision Tree": 0.5,
    "CatBoost":      0.5,
    "XGBoost":       0.519,
    "LightGBM":      0.519,
    "NeuralNet":     0.5,
    "TabNet":        0.5
}

# ==== Probas test ====
probas_test = {
    "LogReg":        logreg_model.predict_proba(X_test_logreg_scaled)[:, 1],
    "Random Forest": rf_model.predict_proba(X_test_rf)[:, 1],
    "Decision Tree": dt_model.predict_proba(X_test_dt)[:, 1],
    "CatBoost":      catboost_model.predict_proba(X_test_cb)[:, 1],
    "XGBoost":       xgb_model.predict_proba(X_test_xgb)[:, 1],
    "LightGBM":      lgb_model.predict_proba(X_test_lgb)[:, 1],
    "NeuralNet":     keras_model.predict(X_test_nn).reshape(-1),
    "TabNet":        tabnet_model.predict_proba(X_test_tabnet)[:, 1]
}

# ==== Streamlit UI ====
st.title(":bar_chart: Prédiction et IA personnalisée du churn")

subscriber_id = st.selectbox(":pushpin: Choisir un abonné :", train_df_logreg["subscriber_id"].unique())
client_row_logreg = train_df_logreg[train_df_logreg["subscriber_id"] == subscriber_id]
client_row_other  = train_df_other[train_df_other["subscriber_id"] == subscriber_id]

X_client_logreg = harmonize_features(client_row_logreg, expected_features_logreg)
X_client_logreg_scaled = scaler_logreg.transform(X_client_logreg)
X_rf_client     = harmonize_features(client_row_other, expected_features_rf)
X_dt_client     = harmonize_features(client_row_other, expected_features_dt)
X_cb_client     = harmonize_features(client_row_other, expected_features_cb)
X_xgb_client    = harmonize_features(client_row_other, expected_features_xgb)
X_lgb_client    = harmonize_features(client_row_other, expected_features_lgb)
X_tabnet_client = harmonize_features(client_row_other, expected_features_tabnet).to_numpy().astype(np.float32)
X_nn_client     = X_client_logreg_scaled

probas_client = {
    "LogReg":        logreg_model.predict_proba(X_client_logreg_scaled)[:, 1][0],
    "Random Forest": rf_model.predict_proba(X_rf_client)[:, 1][0],
    "Decision Tree": dt_model.predict_proba(X_dt_client)[:, 1][0],
    "CatBoost":      catboost_model.predict_proba(X_cb_client)[:, 1][0],
    "XGBoost":       xgb_model.predict_proba(X_xgb_client)[:, 1][0],
    "LightGBM":      lgb_model.predict_proba(X_lgb_client)[:, 1][0],
    "NeuralNet":     keras_model.predict(X_nn_client)[0][0],
    "TabNet":        tabnet_model.predict_proba(X_tabnet_client)[0][1]
}
preds_client = {m: int(probas_client[m] >= thresholds[m]) for m in probas_client}

# ==== Calcul des métriques globales ====
metrics_list = []
for model_name, y_score in probas_test.items():
    y_pred = (y_score >= thresholds[model_name]).astype(int)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc     = auc(fpr, tpr)
    precisions, recalls, _ = precision_recall_curve(y_test, y_score)
    cm = confusion_matrix(y_test, y_pred)
    metrics_list.append({
        "Modèle":         model_name,
        "Précision":      precision,
        "Rappel":         recall,
        "F1-Score":       f1,
        "ROC AUC":        roc_auc,
        "FPR":            fpr,
        "TPR":            tpr,
        "Precisions":     precisions,
        "Recalls":        recalls,
        "ConfusionMatrix": cm
    })

best_model_row = max(metrics_list, key=lambda x: x["F1-Score"])
selected_model = best_model_row["Modèle"]
best_proba     = probas_client[selected_model]

# ==== Mapping pour l'explication IA : DataFrame/ndarray compatible ====
feature_map = {
    "LogReg":      (X_client_logreg, expected_features_logreg),
    "NeuralNet":   (X_client_logreg, expected_features_nn),
    "Random Forest": (X_rf_client, expected_features_rf),
    "Decision Tree": (X_dt_client, expected_features_dt),
    "CatBoost":      (X_cb_client, expected_features_cb),
    "XGBoost":       (X_xgb_client, expected_features_xgb),
    "LightGBM":      (X_lgb_client, expected_features_lgb),
    "TabNet":        (X_tabnet_client, expected_features_tabnet)
}
X_cur, cur_features = feature_map[selected_model]
if isinstance(X_cur, np.ndarray):
    input_dict = dict(zip(cur_features, X_cur[0]))
else:
    input_dict = X_cur.iloc[0].to_dict()

st.success(f"\U0001f3c6 Modèle le plus performant : **{selected_model}**")

@st.cache_data(show_spinner=False)
def explain_and_email(prob, input_dict, sub_id):
    prompt_exp = f"""
Tu es un data scientist télécom. Client ID : {sub_id}
Voici ses infos : {input_dict}
Explique en 3 phrases si ce client risque de partir, sans chiffres ni variables.
"""
    exp = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt_exp}]
    ).choices[0].message.content.strip()

    prompt_email = f"""
Tu es un expert marketing Ooredoo.
Client ID : {sub_id}, Infos : {input_dict}, Résumé : {exp}
Rédige un email convaincant avec 3 incitations à rester.
"""
    email = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt_email}]
    ).choices[0].message.content.strip()

    return exp, email

explanation, email = explain_and_email(best_proba, input_dict, subscriber_id)

st.subheader(":robot_face: Résultat pour l'abonné")
cols = st.columns(len(probas_client))
for i, model_name in enumerate(probas_client):
    with cols[i]:
        st.markdown(f"**{model_name}**")
        st.metric(
            label="Prédiction",
            value="Churn" if preds_client[model_name] else "Non churn",
            delta=f"{probas_client[model_name]:.3f}"
        )

# Jauge du meilleur modèle (optionnel)
try:
    import utils as ut
    st.plotly_chart(ut.create_gauge_chart(best_proba), use_container_width=True)
except:
    pass

st.subheader("\U0001f9e0 Explication IA")
st.markdown(explanation)
st.subheader("\U0001f4e7 Email personnalisé")
st.markdown(email)

st.subheader("\U0001f4ca Évaluation des modèles")
metrics_df = pd.DataFrame(metrics_list)[["Modèle", "Précision", "Rappel", "F1-Score", "ROC AUC"]]
st.dataframe(metrics_df.style.format({col: "{:.3f}" for col in ["Précision", "Rappel", "F1-Score", "ROC AUC"]}))

def plot_metrics(metrics):
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(metrics["FPR"], metrics["TPR"], label=f"ROC (AUC = {metrics['ROC AUC']:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curve - {metrics['Modèle']}")
    ax_roc.legend()

    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(metrics["Recalls"], metrics["Precisions"], label="Precision-Recall")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-Recall Curve - {metrics['Modèle']}")
    ax_pr.legend()

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(metrics["ConfusionMatrix"], annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_title(f"Confusion Matrix - {metrics['Modèle']}")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")

    return fig_roc, fig_pr, fig_cm

for metrics in metrics_list:
    st.markdown("---")
    st.subheader(f"\U0001f4c8 Graphiques - {metrics['Modèle']}")
    fig_roc, fig_pr, fig_cm = plot_metrics(metrics)
    st.pyplot(fig_roc)
    plt.close(fig_roc)
    st.pyplot(fig_pr)
    plt.close(fig_pr)
    st.pyplot(fig_cm)
    plt.close(fig_cm)
