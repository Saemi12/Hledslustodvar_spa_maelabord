# Þetta er grunnkóði til þess að hlaða inn map af Íslandi með EV hleðslustöðvunum
# Til að keyra: python -m streamlit run maelabordid_main.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import plotly.express as px
from functools import reduce
import re

st.set_page_config(layout="wide")

# --- Load and reshape data ---
@st.cache_data
def load_data():
    lgbm = pd.read_csv("Módel/LightGBM_nidurstodur.csv")
    lgbm_vedur = pd.read_csv("Módel/LGBM_vedur_nidurstodur.csv")
    lgbm_hour = pd.read_csv("Módel/LGBM_hour_nidurstodur.csv")
    lgbm_vedur_hour = pd.read_csv("Módel/LGBM_vedur_hour_nidurstodur.csv")
    lgbm_veg = pd.read_csv("Módel/LGBM_veg_nidurstodur.csv")
    einfaldar = pd.read_csv("Módel/einfold_model_nidurstodur.csv")

    url = "https://drive.google.com/uc?id=1bl3hUhh3EoPdRR7TXmS87nIvSiqPszPm"
    all_stats = pd.read_csv(url)
    #all_stats = pd.read_csv("Módel/all_data_output.csv")

    # Rename columns for clarity
    def rename_model(df, new_name):
        df = df.rename(columns={c: c.replace("LGBMRegressor", new_name) for c in df.columns if "LGBMRegressor" in c})
        return df

    lgbm = rename_model(lgbm, "LGBM")
    lgbm_hour = rename_model(lgbm_hour, "LGBM_hour")
    lgbm_vedur = rename_model(lgbm_vedur, "LGBM_vedur")
    lgbm_vedur_hour = rename_model(lgbm_vedur_hour, "LGBM_vedur_hour")
    lgbm_veg = rename_model(lgbm_veg, "LGBM_umferdargogn")

    # Add source column
    lgbm["source"] = "LGBM"
    lgbm_hour["source"] = "LGBM_hour"
    lgbm_vedur["source"] = "LGBM_vedur"
    lgbm_vedur_hour["source"] = "LGBM_vedur_hour"
    lgbm_veg["source"] = "LGBM_umferdargogn"
    einfaldar["source"] = "Statistical"
    all_stats["source"] = "Actual"

    data = pd.concat([lgbm, lgbm_vedur, lgbm_hour, lgbm_vedur_hour, lgbm_veg, einfaldar], ignore_index=True)
    return data, all_stats

@st.cache_data
def load_evaluation():
    df_eval_einfalt = pd.read_csv("Módel/evaluation_einfold_results.csv", usecols=[1,2,3,4,5])  
    df_eval_LGBM = pd.read_csv("Módel/evaluation_results_LightGBM.csv", usecols=[1,2,3,4,5,6])
    df_eval_LGBM_hour = pd.read_csv("Módel/LGBM_hour_eval.csv", usecols=[1,2,3,4,5,6])
    df_eval_LGBM_vedur = pd.read_csv("Módel/LGBM_vedur_eval.csv", usecols=[1,2,3,4,5,6])
    df_eval_vedur_hour = pd.read_csv("Módel/LGBM_vedur_hour_eval.csv", usecols=[1,2,3,4,5,6])
    df_eval_veg = pd.read_csv("Módel/LGBM_veg_eval.csv", usecols=[1,2,3,4,5])  

    def add_prefix(df, prefix):
        df = df.rename(columns={
            c: f"{prefix}{c.replace('LGBMRegressor', '')}" if c not in ["level", "metric"] else c
            for c in df.columns
        })
        return df

    dfs = [
        add_prefix(df_eval_LGBM, "LGBM"),
        add_prefix(df_eval_LGBM_hour, "LGBM_hour"),
        add_prefix(df_eval_LGBM_vedur, "LGBM_vedur"),
        add_prefix(df_eval_vedur_hour, "LGBM_vedur_hour"),
        add_prefix(df_eval_veg, "LGBM_umferdargogn"),
    ]

    eval_df = reduce(lambda left, right: pd.merge(left, right, on=["level", "metric"], how="outer"), dfs)
    eval_df = pd.merge(df_eval_einfalt, eval_df, on=["level", "metric"], how="outer")
    return eval_df

@st.cache_data
def load_feature_importance_lgbm():
    df_feature_importance_lgbm = pd.read_csv("Módel/lgbm_feature_importance.csv")
    df_feature_importance_lgbm_vedur = pd.read_csv("Módel/lgbm_vedur_features.csv")
    df_feature_importance_lgbm_hour = pd.read_csv("Módel/lgbm_hour_features.csv")
    df_feature_importance_lgbm_vedur_hour = pd.read_csv("Módel/lgbm_vedur_hour_features.csv")
    df_feature_importance_lgbm_veg = pd.read_csv("Módel/lgbm_veg_features.csv")
    return df_feature_importance_lgbm, df_feature_importance_lgbm_vedur, df_feature_importance_lgbm_hour, df_feature_importance_lgbm_vedur_hour, df_feature_importance_lgbm_veg

# --- Load all data ---
data, actuals = load_data()
eval_df = load_evaluation()
df_feature_importance_lgbm, df_feature_importance_lgbm_vedur, df_feature_importance_lgbm_hour, df_feature_importance_lgbm_vedur_hour, df_feature_importance_lgbm_veg = load_feature_importance_lgbm()

feature_importance_dict = {
    "LGBM": df_feature_importance_lgbm,
    "LGBM + Weather": df_feature_importance_lgbm_vedur,
    "LGBM + Hourly Average": df_feature_importance_lgbm_hour,
    "LGBM + Weather + Hourly Average": df_feature_importance_lgbm_vedur_hour,
    "LGBM + Hourly Average + Umferð": df_feature_importance_lgbm_veg,
}

# --- Sidebar Controls ---
st.title("🔋 Mælaborð fyrir orkunotkunarspá á hraðhleðslustöðvum")
st.sidebar.header("Stýringar")

# Models
all_pred_cols = [c for c in data.columns if c not in ["unique_id", "ds", "source"]]
base_models = sorted({re.sub(r"-(lo|hi)-\d+$", "", c) for c in all_pred_cols})
chosen_models = st.sidebar.multiselect("Veldu líkan fyrir grafið", base_models, default=base_models[:2])
pred_type = st.sidebar.radio("Gerð Spár", ["Punktspá", "Líkindaspá", "Bæði"])
conf_level = st.sidebar.radio("Öryggisbil", ["80", "90"])

# --- Hierarchy filtering ---
aggregation_level = st.sidebar.selectbox("Stig tímaraðar", ["Heild", "Stöð", "Afl flokkur", "Tengill"])
station_labels = {"727258": "Akureyri Hof", "727316": "Víðigerði", "2539350": "Glerártorg"}

chosen_station = chosen_power = chosen_connector = None

if aggregation_level in ["Stöð", "Afl flokkur", "Tengill"]:
    chosen_station_label = st.sidebar.selectbox("Veldu stöð", list(station_labels.values()))
    chosen_station = [k for k, v in station_labels.items() if v == chosen_station_label][0]

if aggregation_level in ["Afl flokkur", "Tengill"]:
    df_station = data[data["unique_id"].str.startswith(f"total/{chosen_station}")]
    available_powers = df_station["unique_id"].str.split("/").str[2].dropna().unique()
    available_powers = sorted(available_powers, key=float)
    power_labels = {p: f"{float(p):.1f} kW" for p in available_powers}
    chosen_power_label = st.sidebar.selectbox("Veldu afl flokk", list(power_labels.values()))
    chosen_power = [k for k, v in power_labels.items() if v == chosen_power_label][0]

if aggregation_level == "Tengill":
    df_power = df_station[df_station["unique_id"].str.startswith(f"total/{chosen_station}/{chosen_power}")]
    available_connectors = df_power["unique_id"].str.split("/").str[3].dropna().unique()
    chosen_connector = st.sidebar.selectbox("Veldu tengil", available_connectors)

# --- Filter data based on hierarchy ---
pattern = "total"
if aggregation_level == "Stöð":
    pattern = f"total/{chosen_station}"
elif aggregation_level == "Afl flokkur":
    pattern = f"total/{chosen_station}/{chosen_power}"
elif aggregation_level == "Tengill":
    pattern = f"total/{chosen_station}/{chosen_power}/{chosen_connector}"

filtered_df = data[data["unique_id"] == pattern]
filtered_actuals = actuals[actuals["unique_id"] == pattern]

def set_series_level_id(df):
    if df.empty:
        return df
    df = df.copy()
    if aggregation_level == "Heild":
        df["series_level"] = "Heild"
        df["series_id"] = "Heild"
    else:
        df["series_level"] = df["unique_id"].apply(lambda x: x.split('/')[1])
        df["series_id"] = df["unique_id"]
    return df

filtered_df = set_series_level_id(filtered_df)
filtered_actuals = set_series_level_id(filtered_actuals)

# --- Prepare datetime ---
filtered_df["ds"] = pd.to_datetime(filtered_df["ds"])
filtered_actuals["ds"] = pd.to_datetime(filtered_actuals["ds"])

# --- Plot Predictions ---
fig = go.Figure()

# Actuals
if "y" in filtered_actuals.columns:
    fig.add_trace(go.Scatter(
        x=filtered_actuals["ds"], y=filtered_actuals["y"],
        mode="lines", name="Actual", line=dict(color="white", width=2)
    ))

# Define RGBA colors for intervals
colors = [
    "rgba(255,165,0,0.3)",  # orange
    "rgba(0,255,255,0.3)",  # cyan
    "rgba(255,0,255,0.3)",  # magenta
    "rgba(0,255,0,0.3)"     # lime
]

for i, model in enumerate(chosen_models):
    # Point predictions
    if pred_type in ["Punktspá", "Bæði"] and model in filtered_df.columns:
        fig.add_trace(go.Scatter(
            x=filtered_df["ds"], y=filtered_df[model],
            mode="lines", name=f"{model} (punktspá)",
            line=dict(color=colors[i % len(colors)].replace("0.3","1.0"))
        ))

    # Confidence intervals
    if pred_type in ["Líkindaspá", "Bæði"]:
        lo_col = f"{model}-lo-{conf_level}"
        hi_col = f"{model}-hi-{conf_level}"
        if lo_col in filtered_df.columns and hi_col in filtered_df.columns:
            df_fill = filtered_df[["ds", hi_col, lo_col]].dropna().sort_values("ds")
            if not df_fill.empty:
                x_fill = pd.concat([df_fill["ds"], df_fill["ds"][::-1]])
                y_fill = pd.concat([df_fill[hi_col], df_fill[lo_col][::-1]])
                fillcolor = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=x_fill, y=y_fill,
                    fill="toself", fillcolor=fillcolor,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip", showlegend=True,
                    name=f"{model} bil {conf_level}%"
                ))

latest_time = filtered_actuals["ds"].max()
cutoff_time = latest_time - timedelta(days=8)
fig.update_layout(
    title=f"Spá vs Raunveruleg gildi ({aggregation_level})",
    xaxis_title="Tími", yaxis_title="Gildi",
    legend_title="Legend", hovermode="x unified",
    xaxis=dict(range=[cutoff_time, latest_time])
)
st.plotly_chart(fig, use_container_width=True)

# --- Evaluation Metrics with selectable models ---
st.sidebar.header("Samanburður módela")

# Extract model names for each column
col_model_map = {}
for c in eval_df.columns:
    if c not in ["level", "metric"]:
        col_model_map[c] = re.sub(r"[-/].*$", "", c)

# Get unique model list
all_eval_models = sorted(set(col_model_map.values()))

chosen_eval_models = st.sidebar.multiselect(
    "Veldu líkan fyrir samanburð",
    all_eval_models,
    default=all_eval_models
)

# Keep only columns whose extracted model matches chosen models
cols_to_keep = ["level", "metric"] + [
    c for c, m in col_model_map.items() if m in chosen_eval_models
]
filtered_eval_df = eval_df[cols_to_keep]


st.subheader("📊 Tölulegur samanburður módela")

def highlight_best(s):
    is_best = s == s.min()
    return ["background-color: yellow" if v else "" for v in is_best]

numeric_cols = filtered_eval_df.select_dtypes(include=[float, int]).columns
st.dataframe(
    filtered_eval_df.style
        .format("{:.3f}", subset=numeric_cols)
        .apply(highlight_best, axis=1, subset=numeric_cols)
)


# --- Feature Importance Selection ---
st.sidebar.header("Mikilvægi Breyta")
chosen_importance_model = st.sidebar.selectbox(
    "Veldu líkan fyrir mikilvægi breyta",
    list(feature_importance_dict.keys()),
    index=0
)

# --- Feature Importance ---
st.subheader(f"🔎 Mikilvægi breyta ({chosen_importance_model})")
df_selected = feature_importance_dict[chosen_importance_model]

df_selected = feature_importance_dict[chosen_importance_model].rename(
    columns={"Feature": "Breyta", "Importance": "Mikilvægi"}
)

fig_importance = px.bar(
    df_selected,
    x="Mikilvægi",
    y="Breyta",
    orientation="h",
    title=f"Mikilvægi breyta ({chosen_importance_model})",
    height=600
)
st.plotly_chart(fig_importance, use_container_width=True)
