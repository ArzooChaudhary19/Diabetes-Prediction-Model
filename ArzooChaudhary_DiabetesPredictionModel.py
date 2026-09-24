"""
Diabetes Prediction App
=======================
A single-file full-stack application that:
  - Trains a Random Forest classifier on the Pima-style diabetes dataset
  - Serves an interactive Streamlit frontend with:
      • Dataset Explorer
      • Data Visualisation
      • Model Performance Dashboard
      • Live Prediction Tool
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import warnings
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score,
)
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH = "diabetes_prediction_dataset.csv"
MODEL_PATH   = "diabetes_model.pkl"
SCALER_PATH  = "diabetes_scaler.pkl"
ENCODER_PATH = "diabetes_encoders.pkl"

FEATURE_COLUMNS = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level",
]
TARGET_COLUMN = "diabetes"

SMOKING_OPTIONS  = ["never", "No Info", "current", "former", "ever", "not current"]
GENDER_OPTIONS   = ["Female", "Male", "Other"]

# Colour palette
CLR_PRIMARY   = "#3B82F6"
CLR_DANGER    = "#EF4444"
CLR_SUCCESS   = "#22C55E"
CLR_WARNING   = "#F59E0B"
CLR_MUTED     = "#6B7280"
CLR_POSITIVE  = "#DC2626"   # diabetic
CLR_NEGATIVE  = "#16A34A"   # non-diabetic

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration (MUST be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ── global font ── */
        html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

        /* ── metric cards ── */
        div[data-testid="metric-container"] {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,.06);
        }
        div[data-testid="metric-container"] label {
            font-size: 0.8rem;
            color: #64748B;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: .05em;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
        }

        /* ── section headers ── */
        .section-header {
            font-size: 1.3rem;
            font-weight: 700;
            color: #1E293B;
            border-left: 4px solid #3B82F6;
            padding-left: 10px;
            margin: 20px 0 12px 0;
        }

        /* ── result banner ── */
        .result-positive {
            background: #FEF2F2;
            border: 2px solid #F87171;
            border-radius: 12px;
            padding: 20px 24px;
            color: #991B1B;
            font-size: 1.15rem;
            font-weight: 600;
        }
        .result-negative {
            background: #F0FDF4;
            border: 2px solid #4ADE80;
            border-radius: 12px;
            padding: 20px 24px;
            color: #166534;
            font-size: 1.15rem;
            font-weight: 600;
        }

        /* ── sidebar ── */
        [data-testid="stSidebar"] { background: #1E293B; }
        [data-testid="stSidebar"] * { color: #E2E8F0 !important; }

        /* ── dividers ── */
        hr { border: none; border-top: 1px solid #E2E8F0; margin: 18px 0; }

        /* ── tab styling ── */
        button[data-baseweb="tab"] {
            font-weight: 600;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# ─── BACKEND: Data Loading & Preprocessing ───────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the CSV dataset and return a cleaned DataFrame."""
    df = pd.read_csv(path)

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Encode gender
    df["gender_encoded"] = df["gender"].map({"Female": 0, "Male": 1, "Other": 2}).fillna(0)

    # Encode smoking_history (ordinal-ish order)
    smoking_map = {
        "No Info": 0, "never": 1, "ever": 2,
        "former": 3, "not current": 4, "current": 5,
    }
    df["smoking_encoded"] = df["smoking_history"].map(smoking_map).fillna(0)

    return df


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    """
    Train four classifiers and return the best model, scaler, and metrics.
    Returns
    -------
    best_model, scaler, results_dict, X_test, y_test
    """
    feature_cols = [
        "gender_encoded", "age", "hypertension", "heart_disease",
        "smoking_encoded", "bmi", "HbA1c_level", "blood_glucose_level",
    ]

    X = df[feature_cols].values
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Random Forest":         RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting":     GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
        "Logistic Regression":   LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred  = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

        results[name] = {
            "model":       model,
            "accuracy":    accuracy_score(y_test, y_pred),
            "f1":          f1_score(y_test, y_pred),
            "precision":   precision_score(y_test, y_pred),
            "recall":      recall_score(y_test, y_pred),
            "auc_roc":     roc_auc_score(y_test, y_proba),
            "cv_mean":     cv_scores.mean(),
            "cv_std":      cv_scores.std(),
            "y_pred":      y_pred,
            "y_proba":     y_proba,
            "report":      classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]),
        }

    # Best model by AUC-ROC
    best_name = max(results, key=lambda k: results[k]["auc_roc"])
    best_model = results[best_name]["model"]

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return best_model, best_name, scaler, results, X_test_s, y_test, feature_cols


def predict_diabetes(model, scaler, input_dict: dict) -> tuple[int, float]:
    """Run inference for a single patient. Returns (label, probability)."""
    smoking_map = {
        "No Info": 0, "never": 1, "ever": 2,
        "former": 3, "not current": 4, "current": 5,
    }
    gender_map = {"Female": 0, "Male": 1, "Other": 2}

    features = np.array([[
        gender_map.get(input_dict["gender"], 0),
        input_dict["age"],
        input_dict["hypertension"],
        input_dict["heart_disease"],
        smoking_map.get(input_dict["smoking_history"], 0),
        input_dict["bmi"],
        input_dict["HbA1c_level"],
        input_dict["blood_glucose_level"],
    ]])
    features_s = scaler.transform(features)
    label  = model.predict(features_s)[0]
    proba  = model.predict_proba(features_s)[0][1]
    return int(label), float(proba)


# ─────────────────────────────────────────────────────────────────────────────
# ─── FRONTEND: Plotting helpers ──────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _style_fig(fig, ax_list=None):
    """Apply a clean, publication-quality style to a figure."""
    fig.patch.set_facecolor("white")
    axes = ax_list or fig.get_axes()
    for ax in axes:
        ax.set_facecolor("#F8FAFC")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(colors="#475569", labelsize=9)
        ax.title.set_color("#1E293B")
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_color("#475569")
        ax.yaxis.label.set_color("#475569")
    return fig


def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Non-Diabetic", "Diabetic"]
    colours = [CLR_NEGATIVE, CLR_POSITIVE]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart
    bars = axes[0].bar(labels, counts.values, color=colours, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1E293B")
    axes[0].set_title("Class Distribution (Count)")
    axes[0].set_ylabel("Number of Patients")

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=labels, colors=colours,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    axes[1].set_title("Class Distribution (Proportion)")

    fig.suptitle("Target Variable — Diabetes Class Balance", fontsize=13, fontweight="bold", color="#1E293B")
    plt.tight_layout()
    _style_fig(fig, [axes[0]])
    return fig


def plot_feature_distributions(df: pd.DataFrame) -> plt.Figure:
    num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for i, col in enumerate(num_cols):
        for j, label in enumerate([0, 1]):
            colour = CLR_NEGATIVE if label == 0 else CLR_POSITIVE
            subset = df[df[TARGET_COLUMN] == label][col]
            axes[0][i].hist(subset, bins=30, alpha=0.6, color=colour,
                            label="Non-Diabetic" if label == 0 else "Diabetic",
                            edgecolor="white", linewidth=0.5)
        axes[0][i].set_title(col.replace("_", " ").title())
        axes[0][i].legend(fontsize=8)
        axes[0][i].set_ylabel("Frequency")

    # Box plots
    for i, col in enumerate(num_cols):
        data_nd = df[df[TARGET_COLUMN] == 0][col]
        data_d  = df[df[TARGET_COLUMN] == 1][col]
        bp = axes[1][i].boxplot(
            [data_nd, data_d],
            labels=["Non-Diabetic", "Diabetic"],
            patch_artist=True,
            boxprops={"linewidth": 1.5},
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            flierprops={"markersize": 3, "alpha": 0.4},
        )
        bp["boxes"][0].set_facecolor(CLR_NEGATIVE + "99")
        bp["boxes"][1].set_facecolor(CLR_POSITIVE + "99")
        axes[1][i].set_title(col.replace("_", " ").title() + " (Box)")
        axes[1][i].set_ylabel(col.replace("_", " ").title())

    fig.suptitle("Feature Distributions by Diabetes Status", fontsize=13, fontweight="bold", color="#1E293B")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _style_fig(fig)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level",
            "hypertension", "heart_disease", "gender_encoded", "smoking_encoded", "diabetes"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.patch.set_facecolor("white")
    return fig


def plot_categorical_features(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Gender
    gender_counts = df.groupby(["gender", TARGET_COLUMN]).size().unstack(fill_value=0)
    gender_counts.columns = ["Non-Diabetic", "Diabetic"]
    gender_counts.plot(kind="bar", ax=axes[0], color=[CLR_NEGATIVE, CLR_POSITIVE],
                       edgecolor="white", linewidth=1, width=0.65)
    axes[0].set_title("Diabetes by Gender")
    axes[0].set_xlabel("Gender")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(fontsize=9)

    # Hypertension
    hyp_counts = df.groupby(["hypertension", TARGET_COLUMN]).size().unstack(fill_value=0)
    hyp_counts.columns = ["Non-Diabetic", "Diabetic"]
    hyp_counts.index = ["No Hypertension", "Hypertension"]
    hyp_counts.plot(kind="bar", ax=axes[1], color=[CLR_NEGATIVE, CLR_POSITIVE],
                    edgecolor="white", linewidth=1, width=0.55)
    axes[1].set_title("Diabetes by Hypertension")
    axes[1].set_xlabel("Hypertension Status")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(fontsize=9)

    # Smoking history
    smoke_counts = df.groupby(["smoking_history", TARGET_COLUMN]).size().unstack(fill_value=0)
    smoke_counts.columns = ["Non-Diabetic", "Diabetic"]
    smoke_counts.plot(kind="bar", ax=axes[2], color=[CLR_NEGATIVE, CLR_POSITIVE],
                      edgecolor="white", linewidth=1, width=0.65)
    axes[2].set_title("Diabetes by Smoking History")
    axes[2].set_xlabel("Smoking History")
    axes[2].set_ylabel("Count")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend(fontsize=9)

    fig.suptitle("Categorical Feature Analysis", fontsize=13, fontweight="bold", color="#1E293B")
    plt.tight_layout()
    _style_fig(fig)
    return fig


def plot_model_comparison(results: dict) -> plt.Figure:
    names    = list(results.keys())
    metrics  = ["accuracy", "f1", "precision", "recall", "auc_roc"]
    labels   = ["Accuracy", "F1 Score", "Precision", "Recall", "AUC-ROC"]
    colours  = ["#3B82F6", "#8B5CF6", "#10B981", "#F59E0B", "#EF4444"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))

    for i, (metric, label, colour) in enumerate(zip(metrics, labels, colours)):
        vals = [results[n][metric] for n in names]
        short_names = [n.replace(" ", "\n") for n in names]
        bars = axes[i].bar(short_names, vals, color=colour, alpha=0.85, edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                         f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        axes[i].set_title(label)
        axes[i].set_ylim(0, 1.12)
        axes[i].set_ylabel("Score" if i == 0 else "")
        axes[i].tick_params(axis="x", labelsize=8)

    fig.suptitle("Model Performance Comparison", fontsize=13, fontweight="bold", color="#1E293B")
    plt.tight_layout()
    _style_fig(fig)
    return fig


def plot_roc_curves(results: dict, y_test: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    colours = [CLR_PRIMARY, "#8B5CF6", CLR_SUCCESS, CLR_WARNING]

    for (name, res), colour in zip(results.items(), colours):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, lw=2, color=colour, label=f"{name} (AUC={res['auc_roc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random (AUC=0.500)")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="grey")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves — All Models", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    _style_fig(fig, [ax])
    return fig


def plot_confusion_matrix(results: dict, best_name: str, y_test: np.ndarray) -> plt.Figure:
    y_pred = results[best_name]["y_pred"]
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Non-Diabetic", "Diabetic"],
        yticklabels=["Non-Diabetic", "Diabetic"],
        linewidths=0.5, linecolor="#E2E8F0",
        annot_kws={"size": 14, "weight": "bold"},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.patch.set_facecolor("white")
    return fig


def plot_feature_importance(model, feature_cols: list) -> plt.Figure:
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    nice_names   = [c.replace("_encoded", "").replace("_", " ").title() for c in feature_cols]
    sorted_idx   = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 5))
    colours  = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importances)))
    bars = ax.barh(
        [nice_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color=[colours[i] for i in range(len(sorted_idx))],
        edgecolor="white", linewidth=0.8,
    )
    for bar, val in zip(bars, importances[sorted_idx]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=9, fontweight="bold", color="#1E293B")
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title("Feature Importance (Best Model)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, importances.max() * 1.2)
    plt.tight_layout()
    _style_fig(fig, [ax])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ─── FRONTEND: Sidebar ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("## 🩺 Diabetes Prediction")
        st.markdown("---")

        st.markdown("### 📊 Dataset Summary")
        st.metric("Total Records",   f"{len(df):,}")
        st.metric("Diabetic Cases",  f"{df[TARGET_COLUMN].sum():,}")
        st.metric("Features",        str(len(FEATURE_COLUMNS)))

        pct = df[TARGET_COLUMN].mean() * 100
        st.markdown(f"**Diabetes Rate:** `{pct:.1f}%`")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "This system uses machine learning to predict diabetes risk "
            "from clinical and demographic features."
        )
        st.markdown("---")
        st.caption("Built with Python · Streamlit · scikit-learn")


# ─────────────────────────────────────────────────────────────────────────────
# ─── FRONTEND: Tab — Dataset Explorer ────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def tab_dataset(df: pd.DataFrame):
    st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",     f"{len(df):,}")
    col2.metric("Diabetic Cases",    f"{df[TARGET_COLUMN].sum():,}")
    col3.metric("Non-Diabetic Cases",f"{(df[TARGET_COLUMN] == 0).sum():,}")
    col4.metric("Diabetes Rate",     f"{df[TARGET_COLUMN].mean() * 100:.1f}%")

    st.markdown("---")

    st.markdown('<div class="section-header">🔍 Data Sample</div>', unsafe_allow_html=True)
    display_cols = [c for c in df.columns if not c.endswith("_encoded")]

    filter_col, n_col = st.columns([3, 1])
    with filter_col:
        filter_by = st.selectbox("Filter by Diabetes Status", ["All", "Diabetic", "Non-Diabetic"])
    with n_col:
        n_rows = st.number_input("Rows to display", min_value=5, max_value=500, value=20, step=5)

    filtered = df.copy()
    if filter_by == "Diabetic":
        filtered = filtered[filtered[TARGET_COLUMN] == 1]
    elif filter_by == "Non-Diabetic":
        filtered = filtered[filtered[TARGET_COLUMN] == 0]

    st.dataframe(
        filtered[display_cols].head(n_rows).reset_index(drop=True),
        use_container_width=True,
        height=340,
    )

    st.markdown("---")
    st.markdown('<div class="section-header">📈 Descriptive Statistics</div>', unsafe_allow_html=True)
    num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    stats = df[num_cols].describe().T.round(2)
    stats.index = [c.replace("_", " ").title() for c in stats.index]
    st.dataframe(stats, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">❓ Missing Value Report</div>', unsafe_allow_html=True)
    missing = df[display_cols].isnull().sum().reset_index()
    missing.columns = ["Feature", "Missing Count"]
    missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
    missing["Status"] = missing["Missing Count"].apply(lambda x: "✅ Complete" if x == 0 else "⚠️ Has Nulls")
    st.dataframe(missing, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# ─── FRONTEND: Tab — Visualisations ──────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def tab_visualisations(df: pd.DataFrame):
    st.markdown('<div class="section-header">📊 Data Visualisations</div>', unsafe_allow_html=True)

    viz_options = [
        "Class Distribution",
        "Feature Distributions",
        "Correlation Heatmap",
        "Categorical Features",
    ]
    selected = st.multiselect(
        "Select charts to display",
        viz_options,
        default=viz_options,
    )

    if "Class Distribution" in selected:
        st.markdown("#### Target Variable Distribution")
        st.pyplot(plot_class_distribution(df), use_container_width=True)

    if "Feature Distributions" in selected:
        st.markdown("#### Numerical Feature Distributions")
        st.pyplot(plot_feature_distributions(df), use_container_width=True)

    if "Correlation Heatmap" in selected:
        st.markdown("#### Correlation Heatmap")
        st.pyplot(plot_correlation_heatmap(df), use_container_width=True)

    if "Categorical Features" in selected:
        st.markdown("#### Categorical Feature Analysis")
        st.pyplot(plot_categorical_features(df), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ─── FRONTEND: Tab — Model Performance ───────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def tab_model_performance(results: dict, best_name: str, y_test: np.ndarray, feature_cols: list, best_model):
    st.markdown('<div class="section-header">🤖 Model Training & Evaluation</div>', unsafe_allow_html=True)

    # Best model banner
    best = results[best_name]
    st.success(f"🏆 **Best Model:** {best_name}  |  AUC-ROC: **{best['auc_roc']:.4f}**  |  Accuracy: **{best['accuracy']:.4f}**")

    st.markdown("---")
    st.markdown("#### 📋 All-Model Metrics Summary")

    summary_rows = []
    for name, res in results.items():
        summary_rows.append({
            "Model":         name,
            "Accuracy":      f"{res['accuracy']:.4f}",
            "F1 Score":      f"{res['f1']:.4f}",
            "Precision":     f"{res['precision']:.4f}",
            "Recall":        f"{res['recall']:.4f}",
            "AUC-ROC":       f"{res['auc_roc']:.4f}",
            "CV Mean ± Std": f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}",
        })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Model Comparison")
        st.pyplot(plot_model_comparison(results), use_container_width=True)
    with col2:
        st.markdown("#### 📈 ROC Curves")
        st.pyplot(plot_roc_curves(results, y_test), use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"#### 🔲 Confusion Matrix — {best_name}")
        st.pyplot(plot_confusion_matrix(results, best_name, y_test), use_container_width=True)
    with col4:
        fig_imp = plot_feature_importance(best_model, feature_cols)
        if fig_imp:
            st.markdown("#### 🎯 Feature Importance")
            st.pyplot(fig_imp, use_container_width=True)

    st.markdown("---")
    st.markdown(f"#### 📄 Classification Report — {best_name}")
    st.code(best["report"], language="text")


# ─────────────────────────────────────────────────────────────────────────────
# ─── FRONTEND: Tab — Live Prediction ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def tab_prediction(best_model, scaler):
    st.markdown('<div class="section-header">🔮 Live Diabetes Risk Prediction</div>', unsafe_allow_html=True)
    st.info("Enter the patient's clinical data below and click **Predict** to assess diabetes risk.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demographics**")
        gender  = st.selectbox("Gender", GENDER_OPTIONS, index=0)
        age     = st.slider("Age (years)", min_value=1, max_value=100, value=45, step=1)
        smoking = st.selectbox("Smoking History", SMOKING_OPTIONS, index=0)

    with col2:
        st.markdown("**🫀 Clinical Indicators**")
        hypertension  = st.radio("Hypertension",  ["No", "Yes"], horizontal=True)
        heart_disease = st.radio("Heart Disease",  ["No", "Yes"], horizontal=True)
        bmi           = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=80.0, value=27.5, step=0.1, format="%.1f")

    with col3:
        st.markdown("**🧪 Lab Results**")
        hba1c   = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.7, step=0.1, format="%.1f")
        glucose = st.number_input("Blood Glucose (mg/dL)", min_value=50, max_value=350, value=140, step=1)

    st.markdown("---")

    input_dict = {
        "gender":              gender,
        "age":                 age,
        "hypertension":        1 if hypertension == "Yes" else 0,
        "heart_disease":       1 if heart_disease == "Yes" else 0,
        "smoking_history":     smoking,
        "bmi":                 bmi,
        "HbA1c_level":         hba1c,
        "blood_glucose_level": glucose,
    }

    if st.button("🔍 Predict Diabetes Risk", use_container_width=True, type="primary"):
        with st.spinner("Running prediction…"):
            time.sleep(0.4)
            label, proba = predict_diabetes(best_model, scaler, input_dict)

        st.markdown("---")
        st.markdown("### 🩺 Prediction Result")

        col_res, col_gauge = st.columns([2, 1])

        with col_res:
            if label == 1:
                st.markdown(
                    f'<div class="result-positive">'
                    f'⚠️ <strong>HIGH RISK — Diabetes Detected</strong><br>'
                    f'Predicted probability of diabetes: <strong>{proba * 100:.1f}%</strong><br>'
                    f'<small>Recommendation: Consult a healthcare professional immediately.</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result-negative">'
                    f'✅ <strong>LOW RISK — No Diabetes Detected</strong><br>'
                    f'Predicted probability of diabetes: <strong>{proba * 100:.1f}%</strong><br>'
                    f'<small>Recommendation: Maintain a healthy lifestyle for continued wellbeing.</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with col_gauge:
            # Probability gauge
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            theta = np.linspace(0, np.pi, 300)
            ax.plot(np.cos(theta), np.sin(theta), color="#E2E8F0", lw=18, solid_capstyle="round")
            end = np.pi - proba * np.pi
            theta_fill = np.linspace(end, np.pi, 300)
            colour = CLR_POSITIVE if proba >= 0.5 else CLR_SUCCESS
            ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=colour, lw=18, solid_capstyle="round")
            ax.text(0, -0.2, f"{proba * 100:.1f}%", ha="center", va="center",
                    fontsize=22, fontweight="bold", color="#1E293B")
            ax.text(0, -0.5, "Risk Score", ha="center", va="center",
                    fontsize=10, color="#64748B")
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-0.8, 1.3)
            ax.axis("off")
            fig.patch.set_facecolor("white")
            st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📝 Input Summary")
        summary_df = pd.DataFrame([{
            "Feature":  k.replace("_", " ").title(),
            "Value":    str(v),
        } for k, v in input_dict.items()])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Risk factor callouts
        st.markdown("#### ⚡ Key Risk Indicators")
        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
        risk_col1.metric("HbA1c Level",    f"{hba1c:.1f}%",   delta="High" if hba1c >= 6.5 else "Normal",
                         delta_color="inverse" if hba1c >= 6.5 else "normal")
        risk_col2.metric("Blood Glucose",  f"{glucose} mg/dL", delta="High" if glucose >= 200 else "Normal",
                         delta_color="inverse" if glucose >= 200 else "normal")
        risk_col3.metric("BMI",            f"{bmi:.1f}",        delta="Overweight" if bmi >= 25 else "Normal",
                         delta_color="inverse" if bmi >= 25 else "normal")
        risk_col4.metric("Age",            str(age),           delta="Senior" if age >= 60 else "Under 60",
                         delta_color="off")


# ─────────────────────────────────────────────────────────────────────────────
# ─── MAIN ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Title ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='text-align:center; padding:10px 0 4px 0;'>
            <span style='font-size:2.4rem; font-weight:800; color:#1E293B;'>
                🩺 Diabetes Prediction System
            </span><br>
            <span style='font-size:1rem; color:#64748B;'>
                Machine-Learning–Powered Clinical Risk Assessment
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Load data ──────────────────────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        st.error(f"❌ Dataset not found: `{DATASET_PATH}`")
        st.info("Place the CSV file in the same directory as `app.py` and restart.")
        return

    with st.spinner("Loading dataset…"):
        df = load_data(DATASET_PATH)

    # ── Train models ───────────────────────────────────────────────────────
    with st.spinner("Training models (first run only — results are cached)…"):
        best_model, best_name, scaler, results, X_test, y_test, feature_cols = train_models(df)

    # ── Sidebar ────────────────────────────────────────────────────────────
    render_sidebar(df)

    # ── Navigation tabs ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Dataset Explorer",
        "📊 Visualisations",
        "🤖 Model Performance",
        "🔮 Live Prediction",
    ])

    with tab1:
        tab_dataset(df)
    with tab2:
        tab_visualisations(df)
    with tab3:
        tab_model_performance(results, best_name, y_test, feature_cols, best_model)
    with tab4:
        tab_prediction(best_model, scaler)


if __name__ == "__main__":
    main()
