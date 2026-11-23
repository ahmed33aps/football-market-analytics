import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances, silhouette_score


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Transfer Market AI Dashboard",
    page_icon="⚽",
    layout="wide"
)


# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_main_data():
    return pd.read_csv("Full_data_for_analysis.csv")


@st.cache_data
def load_hierarchical_data():
    return pd.read_excel("Data_for_Clustering.xlsx")


@st.cache_data
def load_premier_league_data():
    return pd.read_csv("premier_league.csv")


@st.cache_resource
def load_labels():
    from sklearn.preprocessing import LabelEncoder
    raw = load_main_data()
    return {col: LabelEncoder().fit(raw[col].fillna('Unknown').astype(str))
            for col in ['position', 'sub_position', 'foot']}

@st.cache_resource
def load_model():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    raw = load_main_data().copy()
    feature_cols = ['age_player','total_goals_player','total_assists_player','avg_player_efficiency','total_impact_score','avg_player_rating','avg_player_attack_contribution','avg_influence','highest_market_value_in_eur','position','sub_position','foot']
    for col, encoder in load_labels().items():
        raw[col] = encoder.transform(raw[col].fillna('Unknown').astype(str))
    raw = raw.dropna(subset=['market_value_in_eur']).drop_duplicates(subset=feature_cols+['market_value_in_eur'])
    model = make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=2))
    model.fit(raw[feature_cols],raw['market_value_in_eur'])
    return model

df = load_main_data()
hier_df_raw = load_hierarchical_data()
pl_df_raw = load_premier_league_data()
model = load_model()
labels = load_labels()


# =====================================================
# HELPERS
# =====================================================
def money(x):
    if pd.isna(x):
        return "€0"
    if x >= 1_000_000:
        return f"€{x / 1_000_000:.2f}M"
    if x >= 1_000:
        return f"€{x / 1_000:.1f}K"
    return f"€{x:,.0f}"


def safe_marker_size(data, col):
    if col not in data.columns:
        return np.ones(len(data)) * 12

    s = pd.to_numeric(data[col], errors="coerce").fillna(0)
    s = s - s.min()

    if s.max() == 0:
        return np.ones(len(s)) * 12

    return (s / s.max()) * 35 + 8


def style_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAF3FF"),
        title_font=dict(size=22, color="#FFFFFF"),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")

    return fig


def kpi_card(title, value, sub=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def prediction_card(label, value, sub=""):
    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">{label}</div>
        <div class="prediction-value">{value}</div>
        <div class="prediction-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def premium_dataframe(data, title="Dataset", desc="Professional styled table"):
    html = data.to_html(classes="premium-table", index=False, escape=False)

    st.markdown(f"""
    <div class="premium-table-wrap">
        <div class="table-title">{title}</div>
        <div class="table-desc">{desc}</div>
        <div class="premium-table-scroll">
            {html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# CLEAN DATA
# =====================================================
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["size_market"] = safe_marker_size(df, "market_value_in_eur")
df["size_influence"] = safe_marker_size(df, "avg_influence")
df["size_attack"] = safe_marker_size(df, "avg_player_attack_contribution")
df["size_rating"] = safe_marker_size(df, "avg_player_rating")

for col in hier_df_raw.select_dtypes(include=["int64", "float64"]).columns:
    hier_df_raw[col] = pd.to_numeric(hier_df_raw[col], errors="coerce")

for col in pl_df_raw.select_dtypes(include=["int64", "float64"]).columns:
    pl_df_raw[col] = pd.to_numeric(pl_df_raw[col], errors="coerce")


# =====================================================
# KMEDOIDS CLASS
# =====================================================
class KMedoids:
    def __init__(self, n_clusters=3, metric="manhattan", random_state=42, max_iter=300):
        self.n_clusters = n_clusters
        self.metric = metric
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        D = pairwise_distances(X, metric=self.metric)
        n = len(X)

        medoid_idx = [rng.randint(n)]

        for _ in range(self.n_clusters - 1):
            dist_to_nearest = D[:, medoid_idx].min(axis=1)
            probs = dist_to_nearest ** 2

            if probs.sum() == 0:
                medoid_idx.append(rng.randint(n))
            else:
                probs = probs / probs.sum()
                medoid_idx.append(rng.choice(n, p=probs))

        medoid_idx = list(medoid_idx)

        for _ in range(self.max_iter):
            labels_ = D[:, medoid_idx].argmin(axis=1)
            new_medoids = list(medoid_idx)
            changed = False

            for k in range(self.n_clusters):
                cluster_pts = np.where(labels_ == k)[0]

                if len(cluster_pts) == 0:
                    continue

                costs = D[np.ix_(cluster_pts, cluster_pts)].sum(axis=1)
                best = cluster_pts[costs.argmin()]

                if best != new_medoids[k]:
                    new_medoids[k] = best
                    changed = True

            medoid_idx = new_medoids

            if not changed:
                break

        self.medoid_indices_ = np.array(medoid_idx)
        self.labels_ = D[:, medoid_idx].argmin(axis=1)
        self.cluster_centers_ = X[medoid_idx]
        self.inertia_ = D[np.arange(n), self.medoid_indices_[self.labels_]].sum()

        return self

    def fit_predict(self, X):
        return self.fit(X).labels_


# =====================================================
# FUZZY FUNCTIONS
# =====================================================
def trimf(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x < c:
        return (c - x) / (c - b)
    return 1


def trapmf(x, a, b, c, d):
    if x <= a or x >= d:
        return 0
    elif b <= x <= c:
        return 1
    elif a < x < b:
        return (x - a) / (b - a)
    elif c < x < d:
        return (d - x) / (d - c)
    return 0


def fuzzy_wage_prediction(attack, cluster_id):
    attack = float(np.clip(attack, 0, 100))
    cluster_id = float(np.clip(cluster_id, 0, 2))

    attack_low = trapmf(attack, 0, 0, 15, 35)
    attack_medium = trimf(attack, 25, 50, 75)
    attack_high = trapmf(attack, 60, 80, 100, 100)

    utility = trapmf(cluster_id, 0, 0, 0.35, 0.65)
    star = trimf(cluster_id, 0.7, 1.0, 1.3)
    midtier = trapmf(cluster_id, 1.35, 1.65, 2.0, 2.0)

    rules = [
        (min(star, attack_high), 180),
        (min(star, attack_medium), 150),
        (min(star, attack_low), 110),
        (min(midtier, attack_high), 150),
        (min(midtier, attack_medium), 110),
        (min(midtier, attack_low), 75),
        (min(utility, attack_high), 110),
        (min(utility, attack_medium), 75),
        (min(utility, attack_low), 40),
        (attack_high, 110),
    ]

    numerator = sum(w * v for w, v in rules)
    denominator = sum(w for w, v in rules)

    if denominator == 0:
        return 75

    return numerator / denominator


def wage_interpretation(wage):
    if wage >= 160:
        return "Elite — very high wage recommendation"
    elif wage >= 120:
        return "Strong — high wage recommendation"
    elif wage >= 85:
        return "Average — medium wage recommendation"
    elif wage >= 50:
        return "Below average — low wage recommendation"
    return "Minimal — very low wage recommendation"


# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 5%, rgba(0,255,135,0.30), transparent 28%),
        radial-gradient(circle at 92% 8%, rgba(0,184,255,0.25), transparent 30%),
        radial-gradient(circle at 50% 100%, rgba(255,215,0,0.10), transparent 35%),
        linear-gradient(135deg, #02060C 0%, #071827 42%, #020A12 100%);
    color: white;
}

.block-container {
    max-width: 1540px;
    padding-top: 1.1rem;
    padding-bottom: 3rem;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.nav-shell {
    position: sticky;
    top: 0;
    z-index: 999;
    background: linear-gradient(135deg, rgba(2,18,34,0.96), rgba(0,70,54,0.90));
    backdrop-filter: blur(24px);
    border: 1px solid rgba(0,255,135,0.34);
    border-radius: 30px;
    padding: 18px 22px;
    margin-bottom: 18px;
    box-shadow:
        0 25px 85px rgba(0,0,0,0.48),
        inset 0 0 35px rgba(0,255,135,0.045);
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-logo {
    width: 54px;
    height: 54px;
    border-radius: 19px;
    background: linear-gradient(135deg, #00FF87, #00B8FF);
    display: flex;
    justify-content: center;
    align-items: center;
    color: #04101C;
    font-size: 27px;
    font-weight: 900;
    box-shadow: 0 18px 50px rgba(0,255,135,0.42);
}

.brand-text h3 {
    margin: 0;
    font-size: 23px;
    font-weight: 900;
    background: linear-gradient(90deg, #FFFFFF, #00FF87);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-text p {
    margin: 0;
    color: #A4B8CD;
    font-size: 13px;
    font-weight: 800;
}

div[role="radiogroup"] {
    background: linear-gradient(135deg, rgba(2,18,34,0.96), rgba(0,70,54,0.90));
    border: 1px solid rgba(0,255,135,0.34);
    border-radius: 30px;
    padding: 18px;
    display: flex;
    gap: 16px;
    justify-content: center;
    box-shadow:
        0 25px 85px rgba(0,0,0,0.48),
        inset 0 0 35px rgba(0,255,135,0.045);
    backdrop-filter: blur(24px);
    margin-bottom: 30px;
    flex-wrap: wrap;
}

div[role="radiogroup"] label {
    background: linear-gradient(135deg, rgba(255,255,255,0.13), rgba(255,255,255,0.045)) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    border-radius: 999px !important;
    padding: 14px 25px !important;
    color: #EAF3FF !important;
    font-weight: 900 !important;
    letter-spacing: 0.35px;
    transition: all 0.25s ease !important;
    box-shadow: 0 14px 42px rgba(0,0,0,0.26);
}

div[role="radiogroup"] label p {
    color: #EAF3FF !important;
    font-size: 15px !important;
    font-weight: 900 !important;
}

div[role="radiogroup"] label:hover {
    transform: translateY(-4px) scale(1.03);
    background: linear-gradient(135deg, rgba(0,255,135,0.24), rgba(0,184,255,0.20)) !important;
    border-color: rgba(0,255,135,0.68) !important;
    box-shadow: 0 22px 68px rgba(0,255,135,0.26);
}

div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(135deg, #00FF87, #00B8FF) !important;
    border: none !important;
    color: #03101C !important;
    box-shadow:
        0 20px 64px rgba(0,255,135,0.48),
        inset 0 0 18px rgba(255,255,255,0.38);
}

div[role="radiogroup"] label[data-checked="true"] p {
    color: #03101C !important;
}

.hero {
    background:
        linear-gradient(135deg, rgba(5,20,35,0.96), rgba(2,10,18,0.96)),
        url("https://images.unsplash.com/photo-1431324155629-1a6deb1dec8d");
    background-size: cover;
    background-position: center;
    border-radius: 42px;
    padding: 54px;
    border: 1px solid rgba(0,255,135,0.38);
    box-shadow: 0 40px 130px rgba(0,0,0,0.70);
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: "";
    position: absolute;
    width: 620px;
    height: 620px;
    right: -230px;
    top: -240px;
    background: radial-gradient(circle, rgba(0,255,135,0.38), transparent 70%);
}

.badge {
    display: inline-block;
    padding: 10px 22px;
    border-radius: 999px;
    background: rgba(0,255,135,0.14);
    border: 1px solid rgba(0,255,135,0.55);
    color: #00FF87;
    font-size: 13px;
    font-weight: 900;
    margin-bottom: 18px;
    box-shadow: 0 12px 42px rgba(0,255,135,0.20);
}

.hero-title {
    font-size: 64px;
    line-height: 1.02;
    font-weight: 900;
    margin: 0;
    letter-spacing: -2px;
    background: linear-gradient(90deg, #FFFFFF, #00FF87, #00B8FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-text {
    color: #B9C8D8;
    font-size: 17px;
    line-height: 1.8;
    max-width: 900px;
    margin-top: 20px;
}

.hero-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-top: 28px;
}

.hero-pill {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 14px 18px;
    min-width: 160px;
    backdrop-filter: blur(14px);
}

.hero-pill span {
    color: #8FA4BA;
    font-size: 12px;
    font-weight: 800;
}

.hero-pill strong {
    display: block;
    color: white;
    font-size: 18px;
    font-weight: 900;
    margin-top: 5px;
}

.panel {
    background:
        linear-gradient(135deg, rgba(255,255,255,0.078), rgba(255,255,255,0.035));
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 30px;
    padding: 26px;
    box-shadow: 0 26px 80px rgba(0,0,0,0.38);
    backdrop-filter: blur(24px);
    margin-bottom: 25px;
}

.panel-title {
    font-size: 25px;
    font-weight: 900;
    color: white;
    margin-bottom: 7px;
}

.panel-desc {
    color: #A5B9CF;
    font-size: 14px;
    margin-bottom: 18px;
}

.kpi-card {
    background:
        linear-gradient(135deg, rgba(0,255,135,0.18), rgba(0,184,255,0.11));
    border: 1px solid rgba(0,255,135,0.30);
    border-radius: 28px;
    padding: 25px;
    box-shadow: 0 22px 72px rgba(0,0,0,0.37);
    min-height: 138px;
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
    margin-bottom: 18px;
}

.kpi-card:hover {
    transform: translateY(-5px);
    border-color: rgba(0,255,135,0.58);
    box-shadow: 0 32px 95px rgba(0,255,135,0.20);
}

.kpi-title {
    color: #A9BED3;
    font-size: 13px;
    font-weight: 900;
    text-transform: uppercase;
}

.kpi-value {
    color: white;
    font-size: 32px;
    font-weight: 900;
    margin-top: 10px;
}

.kpi-sub {
    color: #7F93AA;
    font-size: 12px;
    margin-top: 6px;
}

[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label {
    color: #EAF3FF !important;
    font-weight: 900 !important;
}

.stNumberInput input {
    background: rgba(4,14,26,0.96) !important;
    color: white !important;
    border: 1px solid rgba(0,255,135,0.26) !important;
    border-radius: 16px !important;
}

div[data-baseweb="select"] > div {
    background: rgba(4,14,26,0.96) !important;
    border: 1px solid rgba(0,255,135,0.26) !important;
    border-radius: 16px !important;
}

.stButton > button {
    width: 100%;
    height: 62px;
    border-radius: 20px;
    border: none;
    background: linear-gradient(135deg, #00FF87, #00B8FF);
    color: #04101C;
    font-size: 18px;
    font-weight: 900;
    box-shadow: 0 24px 65px rgba(0,255,135,0.34);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-4px);
    box-shadow: 0 34px 90px rgba(0,255,135,0.50);
    color: #04101C;
}

.prediction-box {
    background:
        linear-gradient(135deg, rgba(0,255,135,0.25), rgba(0,184,255,0.19));
    border: 1px solid rgba(0,255,135,0.60);
    border-radius: 36px;
    padding: 40px;
    text-align: center;
    box-shadow: 0 35px 100px rgba(0,255,135,0.26);
    margin-top: 22px;
}

.prediction-label {
    color: #BFFFF0;
    font-weight: 900;
    font-size: 15px;
    text-transform: uppercase;
}

.prediction-value {
    color: #00FF87;
    font-size: 54px;
    font-weight: 900;
    margin-top: 10px;
}

.prediction-sub {
    color: #CFE7F8;
    font-size: 15px;
    margin-top: 10px;
    font-weight: 700;
}

button[data-baseweb="tab"] {
    background: rgba(255,255,255,0.07);
    border-radius: 16px;
    margin-right: 8px;
    color: #C9D8E8;
    font-weight: 900;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #00FF87, #00B8FF);
    color: #04101C;
}

.premium-table-wrap {
    background: linear-gradient(135deg, rgba(2,14,26,0.96), rgba(3,35,34,0.92));
    border: 1px solid rgba(0,255,135,0.30);
    border-radius: 30px;
    padding: 18px;
    box-shadow:
        0 28px 90px rgba(0,0,0,0.56),
        inset 0 0 45px rgba(0,255,135,0.04);
    overflow: hidden;
    margin-top: 18px;
    margin-bottom: 25px;
}

.premium-table-scroll {
    max-height: 540px;
    overflow: auto;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.09);
}

.premium-table {
    width: 100%;
    border-collapse: collapse;
    color: #EAF3FF;
    font-size: 14px;
    min-width: 1200px;
}

.premium-table thead th {
    position: sticky;
    top: 0;
    z-index: 5;
    background: linear-gradient(135deg, #031827, #023B34);
    color: #00FF87;
    text-transform: uppercase;
    font-size: 12px;
    letter-spacing: 0.8px;
    padding: 16px 14px;
    border-bottom: 1px solid rgba(0,255,135,0.38);
    text-align: left;
}

.premium-table tbody td {
    padding: 14px;
    border-bottom: 1px solid rgba(255,255,255,0.075);
    color: #DCEBFA;
    white-space: nowrap;
}

.premium-table tbody tr:nth-child(even) {
    background: rgba(255,255,255,0.047);
}

.premium-table tbody tr:hover {
    background: linear-gradient(135deg, rgba(0,255,135,0.15), rgba(0,184,255,0.11));
}

.table-title {
    font-size: 22px;
    font-weight: 900;
    color: #FFFFFF;
    margin-bottom: 6px;
}

.table-desc {
    color: #9FB4CA;
    font-size: 14px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# NAVBAR
# =====================================================
st.markdown("""
<div class="nav-shell">
    <div class="nav-brand">
        <div class="brand-logo">⚽</div>
        <div class="brand-text">
            <h3>Transfer Market AI</h3>
            <p>Football Analytics, Clustering, Fuzzy Logic & Valuation Engine</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

page = st.radio(
    "",
    [
        "Overview",
        "Visualization & Analysis",
        "Dynamic Filtering",
        "Machine Learning",
        "Hierarchical Clustering",
        "K-Medoids",
        "Fuzzy Logic"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


# =====================================================
# HERO
# =====================================================
st.markdown("""
<div class="hero">
    <div class="badge">⚽ AI TRANSFER MARKET INTELLIGENCE</div>
    <h1 class="hero-title">Football Player Valuation Dashboard</h1>
    <p class="hero-text">
        A premium football intelligence platform for player market value prediction,
        clustering analysis, transfer segmentation, fuzzy wage recommendation,
        performance scouting, club comparison, and advanced football analytics.
    </p>
    <div class="hero-metrics">
        <div class="hero-pill"><span>MODEL</span><strong>Random Forest</strong></div>
        <div class="hero-pill"><span>CLUSTERING</span><strong>Hierarchical + K-Medoids</strong></div>
        <div class="hero-pill"><span>FUZZY</span><strong>Wage Recommendation</strong></div>
        <div class="hero-pill"><span>DASHBOARD</span><strong>Professional BI</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# OVERVIEW
# =====================================================
if page == "Overview":

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Dataset Overview</div>
        <div class="panel-desc">
            General information about player records, clubs, positions, market value, age distribution, and performance indicators.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Total Records", f"{len(df):,}")

    with c2:
        kpi_card("Unique Players", f"{df['player_name'].nunique():,}")

    with c3:
        kpi_card("Clubs", f"{df['club_name'].nunique():,}")

    with c4:
        kpi_card("Avg Market Value", money(df["market_value_in_eur"].mean()))

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x="market_value_in_eur",
            nbins=50,
            title="Market Value Distribution"
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with col2:
        fig = px.histogram(
            df,
            x="age_player",
            nbins=25,
            title="Age Distribution"
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    premium_dataframe(
        df.head(100),
        title="Dataset Preview",
        desc="A premium preview of the football transfer market dataset."
    )


# =====================================================
# VISUALIZATION
# =====================================================
elif page == "Visualization & Analysis":

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Advanced Visualization & Analysis</div>
        <div class="panel-desc">
            Professional football market analysis divided into tabs.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["Market Value", "Performance", "Positions", "Clubs", "Correlation"])

    with tabs[0]:

        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(
                df,
                x="highest_market_value_in_eur",
                y="market_value_in_eur",
                color="position",
                size="size_rating",
                hover_data=["player_name", "club_name"],
                title="Current vs Highest Market Value"
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

        with col2:
            fig = px.box(
                df,
                x="position",
                y="market_value_in_eur",
                color="position",
                title="Market Value by Position"
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

    with tabs[1]:

        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(
                df,
                x="total_goals_player",
                y="market_value_in_eur",
                color="position",
                size="size_market",
                hover_data=["player_name", "club_name"],
                title="Goals vs Market Value"
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

        with col2:
            fig = px.scatter(
                df,
                x="avg_player_rating",
                y="market_value_in_eur",
                color="foot",
                size="size_influence",
                hover_data=["player_name", "club_name"],
                title="Rating vs Market Value"
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

    with tabs[2]:

        pos_avg = df.groupby("position", as_index=False).agg(
            avg_market_value=("market_value_in_eur", "mean"),
            players=("player_name", "nunique"),
            avg_rating=("avg_player_rating", "mean")
        )

        fig = px.bar(
            pos_avg,
            x="position",
            y="avg_market_value",
            color="avg_rating",
            title="Average Market Value by Position"
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with tabs[3]:

        club_stats = df.groupby("club_name", as_index=False).agg(
            players=("player_name", "nunique"),
            avg_market_value=("market_value_in_eur", "mean"),
            total_market_value=("market_value_in_eur", "sum"),
            avg_rating=("avg_player_rating", "mean")
        ).sort_values("total_market_value", ascending=False)

        fig = px.bar(
            club_stats.head(20),
            x="club_name",
            y="total_market_value",
            color="avg_rating",
            title="Top Clubs by Total Market Value"
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

        premium_dataframe(
            club_stats.head(50),
            title="Club Market Analytics",
            desc="Club-level value and performance analysis."
        )

    with tabs[4]:

        corr_cols = [
            "market_value_in_eur",
            "highest_market_value_in_eur",
            "age_player",
            "goals",
            "assists",
            "minutes_played",
            "player_efficiency",
            "Player_Impact_Score",
            "player_rating",
            "attack_contribution",
            "influence",
            "Club_International_Power",
            "total_goals_player",
            "total_assists_player",
            "avg_player_rating",
            "avg_influence",
            "avg_player_efficiency",
            "avg_player_attack_contribution"
        ]

        corr_cols = [c for c in corr_cols if c in df.columns]

        corr = df[corr_cols].corr()

        fig = px.imshow(
            corr,
            aspect="auto",
            title="Correlation Heatmap"
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)


# =====================================================
# DYNAMIC FILTERING
# =====================================================
elif page == "Dynamic Filtering":

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Dynamic Filtering Center</div>
        <div class="panel-desc">Filter the dataset dynamically and monitor instant KPIs.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        selected_positions = st.multiselect(
            "Position",
            sorted(df["position"].dropna().unique()),
            default=sorted(df["position"].dropna().unique())
        )

    with c2:
        selected_foot = st.multiselect(
            "Foot",
            sorted(df["foot"].dropna().unique()),
            default=sorted(df["foot"].dropna().unique())
        )

    with c3:
        selected_years = st.multiselect(
            "Year",
            sorted(df["year"].dropna().unique()),
            default=sorted(df["year"].dropna().unique())
        )

    selected_clubs = st.multiselect(
        "Club Name",
        sorted(df["club_name"].dropna().unique()),
        default=sorted(df["club_name"].dropna().unique())[:20]
    )

    filtered_df = df[
        (df["position"].isin(selected_positions)) &
        (df["foot"].isin(selected_foot)) &
        (df["year"].isin(selected_years)) &
        (df["club_name"].isin(selected_clubs))
    ]

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        kpi_card("Filtered Records", f"{len(filtered_df):,}")

    with k2:
        kpi_card("Players", f"{filtered_df['player_name'].nunique():,}")

    with k3:
        kpi_card("Avg Market Value", money(filtered_df["market_value_in_eur"].mean()))

    with k4:
        kpi_card("Max Market Value", money(filtered_df["market_value_in_eur"].max()))

    show_cols = [
        "player_name",
        "club_name",
        "position",
        "sub_position",
        "foot",
        "age_player",
        "total_goals_player",
        "total_assists_player",
        "avg_player_rating",
        "market_value_in_eur"
    ]

    show_cols = [c for c in show_cols if c in filtered_df.columns]

    premium_dataframe(
        filtered_df[show_cols].sort_values("market_value_in_eur", ascending=False),
        title="Filtered Player Dataset",
        desc="Dynamic filtered results."
    )


# =====================================================
# MACHINE LEARNING
# =====================================================
elif page == "Machine Learning":

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Machine Learning Prediction Engine</div>
        <div class="panel-desc">Predict player market value using Random Forest Regressor.</div>
    </div>
    """, unsafe_allow_html=True)

    selected_features = [
        "age_player",
        "total_goals_player",
        "total_assists_player",
        "avg_player_efficiency",
        "total_impact_score",
        "avg_player_rating",
        "avg_player_attack_contribution",
        "avg_influence",
        "highest_market_value_in_eur",
        "position",
        "sub_position",
        "foot"
    ]

    left, right = st.columns([3, 1])

    with left:

        a, b, c = st.columns(3)

        with a:
            age_player = st.number_input("Age", 15, 45, 25)
            total_goals_player = st.number_input("Total Goals", 0, 1000, 10)
            total_assists_player = st.number_input("Total Assists", 0, 1000, 5)

        with b:
            avg_player_efficiency = st.number_input("Avg Player Efficiency", 0.0, 10.0, 5.0)
            total_impact_score = st.number_input("Total Impact Score", 0.0, 10000.0, 100.0)
            avg_player_rating = st.number_input("Avg Player Rating", 0.0, 10.0, 7.0)

        with c:
            avg_player_attack_contribution = st.number_input("Avg Attack Contribution", -10.0, 10.0, 2.0)
            avg_influence = st.number_input("Avg Influence", -10.0, 10.0, 2.0)
            highest_market_value_in_eur = st.number_input("Highest Market Value (€)", 0, 300000000, 1000000)

    with right:

        position_input = st.selectbox("Position", labels["position"].classes_)
        sub_position_input = st.selectbox("Sub Position", labels["sub_position"].classes_)
        foot_input = st.selectbox("Preferred Foot", labels["foot"].classes_)

        predict_btn = st.button("⚡ Predict Market Value")

    if predict_btn:

        position_encoded = labels["position"].transform([position_input])[0]
        sub_position_encoded = labels["sub_position"].transform([sub_position_input])[0]
        foot_encoded = labels["foot"].transform([foot_input])[0]

        input_data = pd.DataFrame([[
            age_player,
            total_goals_player,
            total_assists_player,
            avg_player_efficiency,
            total_impact_score,
            avg_player_rating,
            avg_player_attack_contribution,
            avg_influence,
            highest_market_value_in_eur,
            position_encoded,
            sub_position_encoded,
            foot_encoded
        ]], columns=selected_features)

        prediction = model.predict(input_data)[0]

        prediction_card(
            "Estimated Player Market Value",
            f"€ {prediction:,.0f}",
            "Random Forest prediction"
        )


# =====================================================
# HIERARCHICAL CLUSTERING
# =====================================================
elif page == "Hierarchical Clustering":

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Hierarchical Clustering Engine</div>
        <div class="panel-desc">
            This page uses Data_for_Clustering.xlsx only.
            Select numeric features, build Agglomerative Clustering, and predict the nearest cluster for a new player.
        </div>
    </div>
    """, unsafe_allow_html=True)

    hier_numeric_cols = hier_df_raw.select_dtypes(include="number").columns.tolist()

    if len(hier_numeric_cols) == 0:
        st.error("Data_for_Clustering.xlsx has no numeric columns.")
        st.stop()

    default_hier_features = hier_numeric_cols[:min(7, len(hier_numeric_cols))]

    hier_features = st.multiselect(
        "Select Hierarchical Features",
        hier_numeric_cols,
        default=default_hier_features
    )

    if len(hier_features) == 0:
        st.warning("Please select at least one feature.")
        st.stop()

    hier_data = hier_df_raw[hier_features].dropna().reset_index(drop=True)

    if len(hier_data) == 0:
        st.error("No data after dropping missing values.")
        st.stop()

    k = st.slider("Number of Clusters", 2, 8, 3)

    hier_scaler = StandardScaler()
    X_hier = hier_scaler.fit_transform(hier_data)

    h_model = AgglomerativeClustering(
        n_clusters=k,
        metric="euclidean",
        linkage="ward"
    )

    h_labels = h_model.fit_predict(X_hier)

    h_result = hier_data.copy()
    h_result["hierarchical_cluster"] = h_labels

    centers_raw = h_result.groupby("hierarchical_cluster")[hier_features].mean()
    centers_scaled = hier_scaler.transform(centers_raw)

    c1, c2, c3 = st.columns(3)

    with c1:
        kpi_card("Clusters", k)

    with c2:
        kpi_card("Records", f"{len(h_result):,}")

    with c3:
        kpi_card("Features", len(hier_features))

    tab1, tab2, tab3 = st.tabs(["Visualization", "Predict Cluster", "Cluster Profile"])

    with tab1:

        x_col = st.selectbox("X Axis", hier_features, index=0)
        y_col = st.selectbox("Y Axis", hier_features, index=min(1, len(hier_features) - 1))

        h_result["safe_size"] = safe_marker_size(h_result, x_col)

        fig = px.scatter(
            h_result,
            x=x_col,
            y=y_col,
            color="hierarchical_cluster",
            size="safe_size",
            title=f"Hierarchical Clustering: {x_col} vs {y_col}",
            hover_data=hier_features
        )

        st.plotly_chart(style_fig(fig), use_container_width=True)

    with tab2:

        st.markdown("""
        <div class="panel">
            <div class="panel-title">Predict Hierarchical Cluster</div>
            <div class="panel-desc">Enter new player values to assign him to the nearest hierarchical cluster centroid.</div>
        </div>
        """, unsafe_allow_html=True)

        input_values = {}
        cols = st.columns(3)

        for i, feat in enumerate(hier_features):
            with cols[i % 3]:
                input_values[feat] = st.number_input(
                    feat,
                    value=float(hier_data[feat].median()),
                    key=f"hier_{feat}"
                )

        if st.button("🔮 Predict Hierarchical Cluster"):

            input_df = pd.DataFrame([input_values])
            input_scaled = hier_scaler.transform(input_df)

            distances = pairwise_distances(
                input_scaled,
                centers_scaled,
                metric="euclidean"
            )[0]

            pred_cluster = int(np.argmin(distances))

            prediction_card(
                "Predicted Hierarchical Cluster",
                f"Cluster {pred_cluster}",
                f"Nearest centroid distance: {distances[pred_cluster]:.4f}"
            )

    with tab3:

        profile = h_result.groupby("hierarchical_cluster")[hier_features].mean().round(2)

        premium_dataframe(
            profile.reset_index(),
            title="Hierarchical Cluster Profile",
            desc="Mean values per cluster."
        )

        profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

        fig = px.imshow(
            profile_norm,
            text_auto=True,
            aspect="auto",
            title="Hierarchical Cluster Normalized Profile"
        )

        st.plotly_chart(style_fig(fig), use_container_width=True)


# =====================================================
# K-MEDOIDS
# =====================================================
elif page == "K-Medoids":

    st.markdown("""
    <div class="panel">
        <div class="panel-title">K-Medoids Clustering Engine</div>
        <div class="panel-desc">
            This page uses premier_league.csv only.
            PAM-style K-Medoids clustering with Manhattan distance and prediction interface.
        </div>
    </div>
    """, unsafe_allow_html=True)

    kmedoids_features = [
        "goals",
        "assists",
        "market_value_in_eur",
        "player_efficiency",
        "attack_contribution",
        "minutes_played",
        "Club_International_Power"
    ]

    kmedoids_features = [c for c in kmedoids_features if c in pl_df_raw.columns]

    if len(kmedoids_features) == 0:
        st.error("premier_league.csv does not contain the required K-Medoids features.")
        st.stop()

    km_data = pl_df_raw[kmedoids_features].dropna().reset_index(drop=True)

    if len(km_data) == 0:
        st.error("No data after dropping missing values.")
        st.stop()

    km_scaler = StandardScaler()
    X_km = km_scaler.fit_transform(km_data)

    k = st.slider("Number of Clusters", 2, 8, 3, key="kmedoids_k")

    sample_size = min(5000, len(X_km))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_km), sample_size, replace=False)
    X_sample = X_km[sample_idx]

    kmed = KMedoids(n_clusters=k, metric="manhattan", random_state=42)
    kmed.fit(X_sample)

    medoids = kmed.cluster_centers_

    distances_all = pairwise_distances(
        X_km,
        medoids,
        metric="manhattan"
    )

    all_labels = distances_all.argmin(axis=1)

    km_result = km_data.copy()
    km_result["kmedoids_cluster"] = all_labels
    km_result["size_market"] = safe_marker_size(km_result, "market_value_in_eur")

    try:
        sil = silhouette_score(
            X_km[sample_idx],
            all_labels[sample_idx],
            metric="manhattan"
        )
    except Exception:
        sil = np.nan

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Clusters", k)

    with c2:
        kpi_card("Inertia", f"{kmed.inertia_:.2f}")

    with c3:
        kpi_card("Silhouette", f"{sil:.4f}" if not pd.isna(sil) else "N/A")

    with c4:
        kpi_card("Sample Size", f"{sample_size:,}")

    tab1, tab2, tab3 = st.tabs(["Visualization", "Predict Cluster", "Cluster Profile"])

    with tab1:

        fig = px.scatter(
            km_result,
            x="goals" if "goals" in km_result.columns else kmedoids_features[0],
            y="market_value_in_eur" if "market_value_in_eur" in km_result.columns else kmedoids_features[1],
            color="kmedoids_cluster",
            size="size_market",
            title="K-Medoids: Goals vs Market Value",
            hover_data=kmedoids_features
        )

        st.plotly_chart(style_fig(fig), use_container_width=True)

        if "attack_contribution" in km_result.columns and "market_value_in_eur" in km_result.columns:
            fig2 = px.scatter(
                km_result,
                x="attack_contribution",
                y="market_value_in_eur",
                color="kmedoids_cluster",
                size="size_market",
                title="K-Medoids: Attack Contribution vs Market Value",
                hover_data=kmedoids_features
            )

            st.plotly_chart(style_fig(fig2), use_container_width=True)

    with tab2:

        st.markdown("""
        <div class="panel">
            <div class="panel-title">Predict K-Medoids Cluster</div>
            <div class="panel-desc">Enter new player values to assign him to the nearest medoid.</div>
        </div>
        """, unsafe_allow_html=True)

        input_values = {}
        cols = st.columns(3)

        for i, feat in enumerate(kmedoids_features):
            with cols[i % 3]:
                input_values[feat] = st.number_input(
                    feat,
                    value=float(km_data[feat].median()),
                    key=f"kmed_{feat}"
                )

        if st.button("🔮 Predict K-Medoids Cluster"):

            input_df = pd.DataFrame([input_values])
            input_scaled = km_scaler.transform(input_df)

            distances = pairwise_distances(
                input_scaled,
                medoids,
                metric="manhattan"
            )[0]

            pred_cluster = int(np.argmin(distances))

            prediction_card(
                "Predicted K-Medoids Cluster",
                f"Cluster {pred_cluster}",
                f"Nearest medoid distance: {distances[pred_cluster]:.4f}"
            )

    with tab3:

        profile = km_result.groupby("kmedoids_cluster")[kmedoids_features].mean().round(2)

        premium_dataframe(
            profile.reset_index(),
            title="K-Medoids Cluster Profile",
            desc="Mean feature values per cluster."
        )

        profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

        fig = px.imshow(
            profile_norm,
            text_auto=True,
            aspect="auto",
            title="K-Medoids Normalized Cluster Profile"
        )

        st.plotly_chart(style_fig(fig), use_container_width=True)


# =====================================================
# FUZZY LOGIC
# =====================================================
elif page == "Fuzzy Logic":

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Fuzzy Wage Recommendation System</div>
        <div class="panel-desc">
            This page uses premier_league.csv only.
            Fuzzy logic predicts wage recommendation using attack contribution and player cluster.
        </div>
    </div>
    """, unsafe_allow_html=True)

    fuzzy_features = [
        "goals",
        "assists",
        "market_value_in_eur",
        "player_efficiency",
        "attack_contribution",
        "minutes_played",
        "Club_International_Power"
    ]

    fuzzy_features = [c for c in fuzzy_features if c in pl_df_raw.columns]

    if "attack_contribution" not in pl_df_raw.columns:
        st.error("attack_contribution column is required in premier_league.csv for Fuzzy Logic.")
        st.stop()

    fuzzy_data = pl_df_raw[fuzzy_features].dropna().reset_index(drop=True)

    fuzzy_scaler = StandardScaler()
    X_fuzzy = fuzzy_scaler.fit_transform(fuzzy_data)

    kmed_fuzzy = KMedoids(n_clusters=3, metric="manhattan", random_state=42)

    sample_size = min(5000, len(X_fuzzy))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_fuzzy), sample_size, replace=False)

    kmed_fuzzy.fit(X_fuzzy[sample_idx])

    fuzzy_distances = pairwise_distances(
        X_fuzzy,
        kmed_fuzzy.cluster_centers_,
        metric="manhattan"
    )

    fuzzy_data["cluster"] = fuzzy_distances.argmin(axis=1)
    fuzzy_data["attack_contribution"] = fuzzy_data["attack_contribution"].clip(0, 100)

    fuzzy_data["wage_recommendation"] = fuzzy_data.apply(
        lambda row: fuzzy_wage_prediction(row["attack_contribution"], row["cluster"]),
        axis=1
    )

    fuzzy_data["interpretation"] = fuzzy_data["wage_recommendation"].apply(wage_interpretation)

    tab1, tab2, tab3 = st.tabs(["Fuzzy Predictor", "Membership Functions", "Validation Samples"])

    with tab1:

        left, right = st.columns([2, 1])

        with left:

            st.markdown("""
            <div class="panel">
                <div class="panel-title">Fuzzy Prediction Inputs</div>
                <div class="panel-desc">
                    Enter attack contribution and select cluster type.
                </div>
            </div>
            """, unsafe_allow_html=True)

            attack_input = st.slider(
                "Attack Contribution",
                0.0,
                100.0,
                50.0
            )

            cluster_name = st.selectbox(
                "Player Cluster",
                ["Utility Player", "Star Player", "Mid-Tier Contributor"]
            )

            cluster_map = {
                "Utility Player": 0,
                "Star Player": 1,
                "Mid-Tier Contributor": 2
            }

            cluster_input = cluster_map[cluster_name]

            if st.button("⚡ Predict Wage Recommendation"):

                wage_pct = fuzzy_wage_prediction(attack_input, cluster_input)
                interp = wage_interpretation(wage_pct)

                prediction_card(
                    "Recommended Wage Percentage",
                    f"{wage_pct:.1f}%",
                    interp
                )

        with right:

            kpi_card("Input Attack", f"{attack_input:.1f}")
            kpi_card("Cluster Type", cluster_name)
            kpi_card("Fuzzy Rules", "10 Rules")

        x = np.arange(0, 201, 1)

        very_low = [trapmf(i, 0, 0, 40, 65) for i in x]
        low = [trimf(i, 50, 75, 100) for i in x]
        medium = [trimf(i, 85, 110, 135) for i in x]
        high = [trimf(i, 120, 150, 175) for i in x]
        very_high = [trapmf(i, 160, 180, 200, 200) for i in x]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=very_low, mode="lines", name="Very Low"))
        fig.add_trace(go.Scatter(x=x, y=low, mode="lines", name="Low"))
        fig.add_trace(go.Scatter(x=x, y=medium, mode="lines", name="Medium"))
        fig.add_trace(go.Scatter(x=x, y=high, mode="lines", name="High"))
        fig.add_trace(go.Scatter(x=x, y=very_high, mode="lines", name="Very High"))

        fig.update_layout(title="Wage Recommendation Membership Functions")

        st.plotly_chart(style_fig(fig), use_container_width=True)

    with tab2:

        attack_x = np.arange(0, 101, 1)

        low = [trapmf(i, 0, 0, 15, 35) for i in attack_x]
        medium = [trimf(i, 25, 50, 75) for i in attack_x]
        high = [trapmf(i, 60, 80, 100, 100) for i in attack_x]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=attack_x, y=low, mode="lines", name="Low Attack"))
        fig.add_trace(go.Scatter(x=attack_x, y=medium, mode="lines", name="Medium Attack"))
        fig.add_trace(go.Scatter(x=attack_x, y=high, mode="lines", name="High Attack"))

        fig.update_layout(title="Attack Contribution Membership Functions")

        st.plotly_chart(style_fig(fig), use_container_width=True)

        cluster_x = np.arange(0, 2.01, 0.01)

        utility = [trapmf(i, 0, 0, 0.35, 0.65) for i in cluster_x]
        star = [trimf(i, 0.7, 1.0, 1.3) for i in cluster_x]
        midtier = [trapmf(i, 1.35, 1.65, 2.0, 2.0) for i in cluster_x]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=cluster_x, y=utility, mode="lines", name="Utility"))
        fig.add_trace(go.Scatter(x=cluster_x, y=star, mode="lines", name="Star"))
        fig.add_trace(go.Scatter(x=cluster_x, y=midtier, mode="lines", name="Mid-Tier"))

        fig.update_layout(title="Cluster Membership Functions")

        st.plotly_chart(style_fig(fig), use_container_width=True)

    with tab3:

        fig = px.scatter(
            fuzzy_data,
            x="attack_contribution",
            y="wage_recommendation",
            color="cluster",
            title="Fuzzy Wage Recommendation vs Attack Contribution",
            hover_data=fuzzy_features
        )

        st.plotly_chart(style_fig(fig), use_container_width=True)

        premium_dataframe(
            fuzzy_data[
                ["attack_contribution", "cluster", "wage_recommendation", "interpretation"]
            ].head(100),
            title="Fuzzy Validation Samples",
            desc="Real samples scored using fuzzy wage recommendation system."
        )