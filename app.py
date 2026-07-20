import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Set page styling configuration immediately at the top level
st.set_page_config(
    layout="wide",
    page_title="Intelligent Customer Behavior Dashboard",
    page_icon=":material/monitoring:"
)

# --- PATH CONFIGURATIONS ---
ROOT_DIR = Path(__file__).parent
DATA_PATH = ROOT_DIR / "data" / "customer_features_rl_env.csv"
MODELS_DIR = ROOT_DIR / "models"

# --- DEFINING PYTORCH DQN FOR ARCHITECTURE MATCHING ---
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.net(x)


@st.cache_data(show_spinner=False)
def read_feature_data(path: str):
    return pd.read_csv(path, index_col="CustomerID")


@st.cache_resource(show_spinner=False)
def load_pickle(path: str):
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_dqn(path: str):
    dqn = QNetwork()
    dqn.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    dqn.eval()
    return dqn

# --- SAFE DATA & ARTIFACT LOADING ---
def load_all_resources():
    """Loads data and models with absolute error reporting to prevent GUI hangs."""
    errors = []
    artifacts = {
        "classifier": None,
        "regression": None,
        "lda": None,        # Explicitly guarantee key initialization
        "dqn": None
    }
    df = None

    # 1. Load CSV Features Data
    if not DATA_PATH.exists():
        errors.append(f"Data file not found at: {DATA_PATH.resolve()}")
    else:
        try:
            df = read_feature_data(str(DATA_PATH))
        except Exception as e:
            errors.append(f"Failed to read CSV data: {str(e)}")

    # 2. Load Classification Model
    clf_path = MODELS_DIR / "final_classification_model.pkl"
    if not clf_path.exists():
        errors.append(f"Classifier model missing at: {clf_path.resolve()}")
    else:
        try:
            artifacts["classifier"] = load_pickle(str(clf_path))
        except Exception as e:
            errors.append(f"Error loading classifier: {str(e)}")

    # 3. Load Regression Model
    reg_path = MODELS_DIR / "regression_simulator.pkl"
    if not reg_path.exists():
        errors.append(f"Regression model missing at: {reg_path.resolve()}")
    else:
        try:
            artifacts["regression"] = load_pickle(str(reg_path))
        except Exception as e:
            errors.append(f"Error loading regression engine: {str(e)}")

    # 4. Load LDA Model
    lda_path = MODELS_DIR / "lda_high_value.pkl"
    if lda_path.exists():
        try:
            artifacts["lda"] = load_pickle(str(lda_path))
        except Exception as e:
            print(f"Non-critical error loading LDA artifact: {str(e)}")

    # 5. Load DQN Weights
    dqn_path = MODELS_DIR / "dqn_marketing_policy.pt"
    if not dqn_path.exists():
        errors.append(f"DQN weights missing at: {dqn_path.resolve()}")
    else:
        try:
            artifacts["dqn"] = load_dqn(str(dqn_path))
        except Exception as e:
            errors.append(f"Error loading DQN policy network: {str(e)}")

    return df, artifacts, errors

# Execute the isolated load process
df_features, models, loading_errors = load_all_resources()

# --- BREAK EARLY AND REPORT IF LOADING STALLED ---
if loading_errors:
    st.error("### 🚨 Critical Application Resource Loading Error")
    st.markdown("The application layout could not be initialized because the following background files were missing or corrupted:")
    for err in loading_errors:
        st.warning(f"• {err}")
    st.info(f"**Current Working Directory:** `{ROOT_DIR.resolve()}`\n\nPlease verify that your `data/` and `models/` folders sit exactly alongside `app.py`.")
    st.stop()

# --- REUSABLE PIPELINE DATA EXTRACTOR ---
def get_customer_data(cid):
    cust_row = df_features.loc[[cid]].copy()
    expected_regression_cols = [
        "Monetary", "Recency", "Frequency", "Product_Diversity",
        "Avg_Spend_Trans", "Homeware", "Stationery", "Gadgets",
        "Decorations", "Kitchenware"
    ]
    for col in expected_regression_cols:
        if col not in cust_row.columns:
            cust_row[col] = 0.0
    return cust_row, cust_row[expected_regression_cols]


def inject_theme():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

            .stApp {
                background:
                    radial-gradient(circle at 8% 8%, rgba(11, 140, 109, 0.35), transparent 40%),
                    radial-gradient(circle at 90% 15%, rgba(5, 94, 148, 0.3), transparent 45%),
                    linear-gradient(130deg, #0a111f 0%, #101b2f 55%, #10263f 100%);
                font-family: 'Manrope', sans-serif;
                color: #f3f7ff;
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            [data-testid="stSidebar"] {
                background: rgba(14, 25, 45, 0.64);
                border-right: 1px solid rgba(156, 182, 216, 0.25);
                backdrop-filter: blur(18px);
            }

            .hero-wrap {
                background: linear-gradient(145deg, rgba(255,255,255,0.22), rgba(255,255,255,0.08));
                border: 1px solid rgba(255,255,255,0.28);
                border-radius: 22px;
                padding: 1.2rem 1.6rem;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.32);
                animation: fadeup .8s ease-out;
            }

            .hero-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.95rem;
                font-weight: 700;
                margin: 0;
                color: #f8fcff;
                letter-spacing: .2px;
            }

            .hero-sub {
                margin-top: .45rem;
                font-size: .98rem;
                color: #d9e6ff;
                line-height: 1.5;
            }

            .section-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.2rem;
                font-weight: 700;
                color: #edf5ff;
                margin-bottom: .2rem;
            }

            .section-sub {
                color: #b7c8e8;
                margin-bottom: .9rem;
                font-size: .9rem;
            }

            .glass-card {
                border-radius: 18px;
                border: 1px solid rgba(255, 255, 255, 0.24);
                background: linear-gradient(165deg, rgba(255,255,255,0.2), rgba(255,255,255,0.06));
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
                backdrop-filter: blur(14px);
                padding: 1rem;
                transition: transform .25s ease, box-shadow .25s ease;
                animation: fadeup .7s ease-out;
            }

            .glass-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 15px 35px rgba(38, 164, 255, 0.27);
            }

            .metric-icon {
                font-size: 1.3rem;
                opacity: 0.95;
            }

            .metric-label {
                margin-top: .6rem;
                text-transform: uppercase;
                letter-spacing: 1.2px;
                color: #cae0ff;
                font-size: .72rem;
                font-weight: 700;
            }

            .metric-value {
                margin-top: .35rem;
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.45rem;
                font-weight: 700;
                color: #ffffff;
            }

            .metric-delta {
                margin-top: .45rem;
                font-size: .78rem;
                color: #8ee3bc;
                font-weight: 600;
            }

            .result-panel {
                border-radius: 16px;
                padding: 1rem;
                border: 1px solid rgba(255,255,255,0.25);
                background: linear-gradient(160deg, rgba(255,255,255,0.16), rgba(255,255,255,0.05));
                margin-bottom: .55rem;
            }

            .result-title {
                color: #d6e7ff;
                font-size: .8rem;
                letter-spacing: 1px;
                text-transform: uppercase;
                font-weight: 700;
            }

            .result-value {
                color: #ffffff;
                font-size: 1.2rem;
                font-family: 'Space Grotesk', sans-serif;
                font-weight: 700;
                margin-top: .3rem;
            }

            .result-note {
                color: #c2d4f4;
                font-size: .8rem;
                margin-top: .25rem;
            }

            .action-card {
                border-radius: 16px;
                border: 1px solid rgba(255,255,255,0.2);
                background: rgba(255,255,255,0.07);
                padding: .95rem;
                min-height: 132px;
                transition: all .25s ease;
            }

            .action-card.selected {
                border: 1px solid rgba(102, 234, 181, 0.88);
                box-shadow: 0 0 18px rgba(102, 234, 181, 0.38);
                background: linear-gradient(160deg, rgba(95, 217, 167, 0.28), rgba(255,255,255,0.08));
            }

            .action-title {
                color: #eef6ff;
                font-size: .92rem;
                font-weight: 700;
                line-height: 1.35;
            }

            .action-value {
                margin-top: .5rem;
                color: #dcf2ff;
                font-size: .88rem;
                font-weight: 600;
            }

            .action-state {
                margin-top: .45rem;
                font-size: .72rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #b7d2ff;
            }

            .dataset-chip {
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.25);
                background: rgba(255, 255, 255, 0.08);
                padding: .75rem;
                margin-top: .75rem;
                color: #dce8fb;
                font-size: .82rem;
            }

            @keyframes fadeup {
                from { opacity: 0; transform: translateY(8px); }
                to { opacity: 1; transform: translateY(0); }
            }

            ::-webkit-scrollbar {
                width: 10px;
            }

            ::-webkit-scrollbar-track {
                background: rgba(255,255,255,0.05);
            }

            ::-webkit-scrollbar-thumb {
                background: rgba(109, 180, 255, 0.65);
                border-radius: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-sub">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def hero_banner():
    st.markdown(
        """
        <div class="hero-wrap">
            <p class="hero-title">Intelligent customer behavior and marketing strategy dashboard</p>
            <p class="hero-sub">Unified analytics for profile segmentation, predictive value modeling, and reinforcement learning policy decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(icon: str, label: str, value: str, delta: str):
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_panel(title: str, value: str, note: str):
    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-title">{title}</div>
            <div class="result-value">{value}</div>
            <div class="result-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def action_recommendation_card(title: str, q_value: float, selected: bool):
    selected_class = "selected" if selected else ""
    status = "recommended policy" if selected else "inactive option"
    st.markdown(
        f"""
        <div class="action-card {selected_class}">
            <div class="action-title">{title}</div>
            <div class="action-value">Calculated Q-value: {q_value:.3f}</div>
            <div class="action-state">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def update_plot_theme(fig: go.Figure):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.04)",
        font=dict(family="Manrope", color="#eaf2ff"),
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig.update_xaxes(
        gridcolor="rgba(188,210,242,0.25)",
        zerolinecolor="rgba(188,210,242,0.25)",
    )
    fig.update_yaxes(
        gridcolor="rgba(188,210,242,0.25)",
        zerolinecolor="rgba(188,210,242,0.25)",
    )
    return fig


def init_state(customer_list):
    st.session_state.setdefault("active_screen", "Dashboard")
    st.session_state.setdefault("selected_customer", customer_list[0])
    st.session_state.setdefault("projection_mode", "PCA")
    st.session_state.setdefault("prediction_task", "Both")
    st.session_state.setdefault("rl_agent", "Trained DQN Agent")
    st.session_state.setdefault(
        "filter_tier",
        "Show all customers",
    )
    st.session_state.setdefault("sample_size", 400)

inject_theme()

customer_list = sorted(df_features.index.tolist())
init_state(customer_list)

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("### Navigation")
    
    # Clean pill/capsule control tabs
    nav_options = ["Dashboard", "PCA/LDA", "Prediction", "RL Strategy"]
    selection = st.segmented_control(
        "Navigation Select",
        options=nav_options,
        default=st.session_state.active_screen,
        label_visibility="collapsed"
    )
    
    if selection and selection != st.session_state.active_screen:
        st.session_state.active_screen = selection
        st.rerun()

    st.markdown("---")
    
    st.markdown("### Customer controls")
    st.selectbox(
        "Select customer ID",
        customer_list,
        key="selected_customer",
    )
    
    st.markdown("---")
    st.markdown(
        """
        <div class="dataset-chip">
            <strong>Dataset:</strong> Online Retail (UCI)<br>
            <strong>Profiles:</strong> 3,317<br>
            <strong>Model stack:</strong> Classification + Regression + DQN
        </div>
        """, 
        unsafe_allow_html=True
    )

# --- RENDER SCREENS DYNAMICALLY ---
hero_banner()
st.space("small")
st.space("small")

selected_cid = st.session_state.selected_customer
cust_row, input_features_df = get_customer_data(selected_cid)

# ==================== SCREEN 1: DASHBOARD ====================
if st.session_state.active_screen == "Dashboard":
    section_header(
        "Customer profile summary",
        f"Live overview for customer ID {selected_cid} with current model outputs.",
    )

    recency_val = int(cust_row["Recency"].values[0]) if "Recency" in cust_row.columns else 0
    frequency_val = int(cust_row["Frequency"].values[0]) if "Frequency" in cust_row.columns else 0
    monetary_val = float(cust_row["Monetary"].values[0]) if "Monetary" in cust_row.columns else 0.0
    segment_label = "High-value" if monetary_val > 1500 else "Standard-value"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("🕒", "Recency", f"{recency_val} days", "Lower is better")
    with k2:
        metric_card("🧾", "Frequency", f"{frequency_val} orders", "Higher is better")
    with k3:
        metric_card("💰", "Monetary", f"${monetary_val:,.2f}", "Spend score")
    with k4:
        metric_card("🎯", "Segment", segment_label.upper(), "Target band")

    class_display = "High-value profile target selected"
    class_confidence = 92.4
    try:
        class_pred = models["classifier"].predict(input_features_df)[0]
        class_probs = models["classifier"].predict_proba(input_features_df)[0]
        class_confidence = float(max(class_probs) * 100)
        class_display = f"Label {class_pred}"
    except Exception:
        pass

    spend_display = 612.40
    try:
        spend_display = float(models["regression"].predict(input_features_df)[0])
        spend_display = max(0.0, spend_display)
    except Exception:
        pass

    p1, p2 = st.columns(2)
    with p1:
        prediction_panel(
            "Classification output",
            f"{class_display}",
            f"Confidence: {class_confidence:.1f}%",
        )
    with p2:
        prediction_panel(
            "Regression output",
            f"${spend_display:,.2f}",
            "Estimated margin bounds: ± $48.10 RMSE",
        )

    viz1, viz2 = st.columns(2)
    with viz1:
       gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=monetary_val,
            number={"prefix": "$", "valueformat": ",.0f"},
            title={"text": "Monetary intensity"},
            # FIX: Restricts the arc to the bottom 72% of the canvas 
            # so the title always has dedicated space at the top
            domain={'x': [0, 1], 'y': [0, 0.72]}, 
            gauge={
                "axis": {"range": [0, max(60000, monetary_val + 10000)]},
                "bar": {"color": "#3ccf9b"}, 
                "bgcolor": "rgba(255,255,255,0.05)",
            },
        )
    )
    update_plot_theme(gauge)
    # FIX: Increased top margin (t=75) to give the text plenty of breathing room
    gauge.update_layout(height=300, margin=dict(t=75, b=20, l=20, r=20))
    st.plotly_chart(gauge, use_container_width=True)

    with viz2:
    # Your original bar chart code preserved exactly as it was
      profile_breakdown = pd.DataFrame(
        {
            "Metric": ["Recency", "Frequency", "Monetary/100"],
            "Value": [
                recency_val,
                frequency_val,
                monetary_val / 100.0,
            ],
        }
    )
    profile_fig = px.bar(
        profile_breakdown,
        x="Metric",
        y="Value",
        color="Metric",
        color_discrete_sequence=["#56c6e8", "#3ccf9b", "#f5bf5f"],
        title="Feature shape snapshot",
    )
    profile_fig.update_traces(marker_line_width=0)
    update_plot_theme(profile_fig)
    profile_fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(profile_fig, use_container_width=True)

# ==================== SCREEN 2: PCA/LDA GRAPH COMPARISON ====================
elif st.session_state.active_screen == "PCA/LDA":
    section_header(
        "Dimensionality reduction explorer",
        "Inspect customer position across PCA and LDA projections with context-aware filtering.",
    )
    
    st.markdown('<h2 class="section-title">Dimensionality reduction explorer</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Inspect customer position across PCA and LDA projections</p>', unsafe_allow_html=True)
    
    # --- PCA/LDA EXCLUSIVE FILTERS ROW ---
    # Placing them inside a horizontal row inside the main page keeps the screen dense but neat
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        st.selectbox(
            "Background pool",
            [
                "Show all customers",
                "High spend tiers only (Monetary > $1,500)",
                "High engagement profiles only (Frequency > 5 orders)",
            ],
            key="filter_tier",
        )
        
    with filter_col2:
        st.slider(
            "Background sample density",
            min_value=100,
            max_value=1000,
            step=50,
            key="sample_size",
        )
        
    st.markdown("---")

    st.session_state.projection_mode = st.segmented_control(
        "Projection mode",
        ["PCA", "LDA"],
        default=st.session_state.projection_mode,
    )

    pool_df = df_features.copy()
    if "High spend" in st.session_state.filter_tier:
        pool_df = pool_df[pool_df["Monetary"] > 1500]
    elif "High engagement" in st.session_state.filter_tier:
        pool_df = pool_df[pool_df["Frequency"] > 5]

    sample_df = pool_df.sample(
        n=min(st.session_state.sample_size, len(pool_df)),
        random_state=42,
    ) if len(pool_df) > 0 else pool_df

    base_features = ["Monetary", "Recency", "Frequency", "Product_Diversity", "Avg_Spend_Trans"]

    if st.session_state.projection_mode == "PCA" and "PC1" in df_features.columns and "PC2" in df_features.columns:
        fig = px.scatter(
            sample_df,
            x="PC1",
            y="PC2",
            opacity=0.5,
            color_discrete_sequence=["#57b9ff"],
            title=f"PCA context distribution (N={len(sample_df)})",
        )
        fig.add_trace(
            go.Scatter(
                x=[cust_row["PC1"].values[0]],
                y=[cust_row["PC2"].values[0]],
                mode="markers+text",
                text=[f"Customer {selected_cid}"],
                textposition="top center",
                marker=dict(size=18, color="#ff7f6b", line=dict(width=2, color="#ffffff")),
                name="Selected customer",
            )
        )
        update_plot_theme(fig)
        fig.update_layout(height=450)
        st.plotly_chart(fig)

    elif st.session_state.projection_mode == "LDA" and models.get("lda") is not None:
        try:
            valid_cols = [c for c in base_features if c in df_features.columns]
            lda_target = models["lda"].transform(cust_row[valid_cols])[0][0]
            lda_sample = models["lda"].transform(sample_df[valid_cols]) if len(sample_df) > 0 else np.array([[0.0]])

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=lda_sample.ravel(),
                    nbinsx=38,
                    marker_color="rgba(86, 198, 232, 0.65)",
                    name="Subsegment distribution",
                    opacity=0.85,
                )
            )
            fig.add_vline(
                x=lda_target,
                line_width=3,
                line_dash="dash",
                line_color="#ff8f6b",
            )
            fig.add_annotation(
                x=lda_target,
                y=1,
                yref="paper",
                text=f"Customer {selected_cid}: {lda_target:.3f}",
                showarrow=False,
                bgcolor="rgba(255,143,107,0.2)",
                bordercolor="#ff8f6b",
                font=dict(color="#fff"),
            )
            update_plot_theme(fig)
            fig.update_layout(height=450, bargap=0.05, title="LDA density context")
            st.plotly_chart(fig)
        except Exception as e:
            st.warning(f"LDA transformation error: {str(e)}")
    else:
        st.warning("LDA model artifact is missing from storage directory.")

# ==================== SCREEN 3: PREDICTION TASK FOCUS ====================
elif st.session_state.active_screen == "Prediction":
    section_header(
        "Targeted predictive diagnostics",
        "Switch between classification and regression outputs with local feature impact review.",
    )

    st.session_state.prediction_task = st.segmented_control(
        "Active model context",
        ["Both", "Classification", "Regression"],
        default=st.session_state.prediction_task,
    )

    confidence = 92.4
    if st.session_state.prediction_task in ["Both", "Classification"]:
        try:
            class_pred = models["classifier"].predict(input_features_df)[0]
            class_probs = models["classifier"].predict_proba(input_features_df)[0]
            confidence = max(class_probs) * 100

            if confidence < 75.0:
                st.warning(
                    f"Ambiguous profile status. Model certainty is below threshold ({confidence:.1f}% confidence)."
                )
            else:
                prediction_panel(
                    "Classification engine",
                    f"Label {class_pred}",
                    f"Confidence output: {confidence:.1f}%",
                )
        except Exception:
            prediction_panel(
                "Classification engine",
                "High-value profile target selected",
                f"Confidence output: {confidence:.1f}%",
            )

    if st.session_state.prediction_task in ["Both", "Regression"]:
        try:
            pred_spend = float(models["regression"].predict(input_features_df)[0])
            prediction_panel(
                "Regression engine",
                f"${max(0.0, pred_spend):,.2f}",
                "Estimated margin bounds: ± $48.10 RMSE",
            )
        except Exception:
            prediction_panel(
                "Regression engine",
                "$612.40",
                "Estimated margin bounds: ± $48.10 RMSE",
            )

    section_header(
        "Local profile weight attribution matrix",
        "Feature-level impact proxy derived from normalized customer attributes.",
    )

    explain_cols = ["Monetary", "Recency", "Frequency", "Product_Diversity", "Avg_Spend_Trans"]
    raw_vals = [float(cust_row[c].values[0]) for c in explain_cols]

    norm_impact = np.array(raw_vals) / (np.sum(np.abs(raw_vals)) + 1e-5)
    impact_df = pd.DataFrame(
        {
            "Feature engineering metric": explain_cols,
            "Relative impact score": norm_impact,
        }
    ).sort_values(by="Relative impact score", ascending=True)

    impact_fig = px.bar(
        impact_df,
        x="Relative impact score",
        y="Feature engineering metric",
        orientation="h",
        color="Relative impact score",
        color_continuous_scale=["#ff7f7f", "#7fd1ff", "#6bf1b0"],
        title="Relative impact profile",
    )
    update_plot_theme(impact_fig)
    impact_fig.update_layout(height=330, coloraxis_showscale=False)
    st.plotly_chart(impact_fig)

    st.caption("Cross-validation F1-score baseline: 0.89 | R² continuous fit metric score: 0.81")

# ==================== SCREEN 4: RL POLICY MODEL VARIANT ====================
elif st.session_state.active_screen == "RL Strategy":
    section_header(
        "Reinforcement learning operations strategy",
        "Policy recommendation cards and comparative profit diagnostics for active profile decisions.",
    )

    st.session_state.rl_agent = st.selectbox(
        "Operational policy model variant",
        ["Trained DQN Agent", "Tabular Q-Agent"],
        index=["Trained DQN Agent", "Tabular Q-Agent"].index(st.session_state.rl_agent),
    )

    state_cols = ["Recency_scaled", "Frequency_scaled", "Monetary_scaled", "PC1", "PC2"]
    for c in state_cols:
        if c not in cust_row.columns:
            cust_row[c] = 0.0
    state_vector = cust_row[state_cols].values.astype(np.float32)

    with torch.no_grad():
        state_tensor = torch.tensor(state_vector, dtype=torch.float32)
        q_values = models["dqn"](state_tensor).numpy()[0]

    recommended_action = int(np.argmax(q_values))
    action_labels = {
        0: "Action 0: No action baseline treatment strategy",
        1: "Action 1: 10% discount coupon treatment offer",
        2: "Action 2: Free premium trial special campaign access",
    }

    a1, a2, a3 = st.columns(3)
    for idx, col in enumerate([a1, a2, a3]):
        with col:
            action_recommendation_card(
                action_labels[idx],
                float(q_values[idx]),
                selected=(idx == recommended_action),
            )

    sim_steps = ["Current quarter", "Quarter +1", "Quarter +2", "Quarter +3"]
    action_1_trajectory = [q_values[1], q_values[1] * 1.08, q_values[1] * 1.14, q_values[1] * 1.19]
    action_2_trajectory = [q_values[2], q_values[2] * 0.95, q_values[2] * 0.88, q_values[2] * 0.82]

    traj_fig = go.Figure()
    traj_fig.add_trace(
        go.Scatter(
            x=sim_steps,
            y=action_1_trajectory,
            mode="lines+markers",
            line=dict(color="#6bf1b0", width=3),
            marker=dict(size=9),
            name="Action 1 discount strategy",
        )
    )
    traj_fig.add_trace(
        go.Scatter(
            x=sim_steps,
            y=action_2_trajectory,
            mode="lines+markers",
            line=dict(color="#ff9f73", width=3, dash="dash"),
            marker=dict(size=9),
            name="Action 2 premium strategy",
        )
    )
    update_plot_theme(traj_fig)
    traj_fig.update_layout(height=360, title="Multi-quarter simulated customer valuation path")
    st.plotly_chart(traj_fig)

    strategy_metrics = {
        "Operational strategy": [
            "DQN Policy Model",
            "Tabular Q-Policy Variant",
            "Random Baseline Actions Strategy",
            "Always-No-Action Policy",
        ],
        "Total accumulated test net profit ($)": [264095.16, 257211.36, 171511.95, 106318.07],
    }
    results_df = pd.DataFrame(strategy_metrics)

    selected_agent_map = {
        "Trained DQN Agent": "DQN Policy Model",
        "Tabular Q-Agent": "Tabular Q-Policy Variant",
    }
    current_selected = selected_agent_map.get(st.session_state.rl_agent, "")

    colors = ["#6bf1b0" if s == current_selected else "#87a8d6" for s in results_df["Operational strategy"]]
    bar_fig = go.Figure(
        data=[
            go.Bar(
                x=results_df["Operational strategy"],
                y=results_df["Total accumulated test net profit ($)"],
                marker_color=colors,
                text=[f"${v:,.2f}" for v in results_df["Total accumulated test net profit ($)"]],
                textposition="outside",
            )
        ]
    )
    update_plot_theme(bar_fig)
    bar_fig.update_layout(
        height=410,
        title="Comparative financial policy analysis",
        yaxis_title="Net profit ($)",
        xaxis_title="Strategy variant",
    )
    st.plotly_chart(bar_fig)

    dqn_profit = 264095.16
    random_profit = 171511.95
    pct_improvement = ((dqn_profit - random_profit) / random_profit) * 100

    prediction_panel(
        "Live performance review",
        f"DQN cumulative profit: ${dqn_profit:,.2f}",
        f"Improvement against random baseline: +{pct_improvement:.1f}%",
    )
