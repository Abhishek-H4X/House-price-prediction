import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ============================================================
# HousePriceAI | California Housing Analytics
# Streamlit dashboard for the models created by train_models.py
# ============================================================

st.set_page_config(
    page_title="HousePriceAI | California Housing Analytics",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Ridge Regression": "ridge_regression.pkl",
    "Lasso Regression": "lasso_regression.pkl",
}
SCALER_FILE = "scaler.pkl"

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]
FEATURE_LABELS = {
    "MedInc": "Median income (in $10,000s)",
    "HouseAge": "Median house age (years)",
    "AveRooms": "Average rooms per household",
    "AveBedrms": "Average bedrooms per household",
    "Population": "Block-group population",
    "AveOccup": "Average household occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}

# ----------------------------- Styling -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#102448; --muted:#627493; --blue:#2478ef; --line:#e3ebf7; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(135deg,#f7faff 0%,#edf4ff 52%,#f8fbff 100%); color:var(--ink); }
[data-testid="stAppViewContainer"] .main .block-container,
[data-testid="stAppViewContainer"] .main .block-container p,
[data-testid="stAppViewContainer"] .main .block-container label,
[data-testid="stAppViewContainer"] .main .block-container span,
[data-testid="stAppViewContainer"] .main .block-container li { color:#203653; }
[data-testid="stAppViewContainer"] .main .block-container h1,
[data-testid="stAppViewContainer"] .main .block-container h2,
[data-testid="stAppViewContainer"] .main .block-container h3,
[data-testid="stAppViewContainer"] .main .block-container h4 { color:#102448 !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p { color:#344b6b; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color:#657894 !important; }
[data-testid="stDataFrame"] *, [data-testid="stTable"] * { color:#203653 !important; }
div[data-baseweb="input"] input { color:#18304f !important; background:#fff !important; }
div[data-baseweb="select"] * { color:#18304f !important; }
[data-testid="stAlert"] p, [data-testid="stAlert"] li { color:#203653 !important; }
.block-container { padding-top:1.35rem; padding-bottom:2.5rem; max-width:1600px; }
h1,h2,h3 { font-family:'Manrope',sans-serif; color:#102448; letter-spacing:-.035em; }
h1 { font-weight:800; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#10264a 0%,#0b1b37 100%); }
[data-testid="stSidebar"] * { color:#edf4ff !important; }
[data-testid="stSidebar"] .stRadio label { padding:7px 9px; border-radius:10px; color:#edf4ff !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { color:#ffffff !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#c2d0e7 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label p { color:#edf4ff !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#c2d0e7 !important; }
div[data-testid="stMetric"] { background:rgba(255,255,255,.82); border:1px solid #e3ebf7; border-radius:17px; padding:16px 18px; box-shadow:0 8px 24px rgba(39,86,150,.06); }
div[data-testid="stMetricLabel"] { color:#637696; font-weight:600; }
div[data-testid="stMetricValue"] { color:#102448; font-family:'Manrope',sans-serif; font-weight:800; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px; }
div.stButton > button { border-radius:11px; border:0; font-weight:700; min-height:2.7rem; }
div.stButton > button[kind="primary"] { background:linear-gradient(100deg,#2478ef,#4d9bff); color:white; box-shadow:0 7px 18px rgba(36,120,239,.2); }
div[data-baseweb="input"] input, div[data-baseweb="select"] > div { border-radius:10px; }
hr { border-color:#e2eaf6; }
.small-muted { color:#6d7f9c; font-size:.88rem; }
.hero {
  border-radius:22px; padding:30px 34px; min-height:225px; color:white;
  background:linear-gradient(90deg,rgba(9,37,76,.96) 0%,rgba(15,67,123,.83) 48%,rgba(16,68,112,.18) 100%),
  url('https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?auto=format&fit=crop&w=1800&q=85');
  background-size:cover; background-position:center;
  box-shadow:0 14px 34px rgba(25,70,130,.16); margin-bottom:18px;
}
.hero h1 { color:white !important; font-size:2.45rem; margin:.25rem 0 .45rem 0; }
.hero p { color:#e7f1ff; max-width:650px; font-size:1.03rem; }
.hero-kicker { color:#b9d9ff; font-size:.76rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }
.panel-title { font-family:'Manrope',sans-serif; font-size:1.12rem; font-weight:800; color:#142b50; margin-bottom:0; }
.panel-sub { color:#71829e; font-size:.84rem; margin-top:2px; }
.info-card { background:rgba(255,255,255,.75); border:1px solid #e3ebf7; border-radius:16px; padding:16px; box-shadow:0 7px 22px rgba(39,86,150,.045); }
.insight { border-radius:13px; padding:12px 14px; background:linear-gradient(110deg,#f0f7ff,#f8fbff); border:1px solid #e6effb; margin:8px 0; }
.pill { display:inline-block; padding:4px 9px; border-radius:30px; background:#eaf3ff; color:#2478ef; font-size:.75rem; font-weight:700; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Load assets -----------------------------
@st.cache_resource
def load_models():
    missing = [f for f in list(MODEL_FILES.values()) + [SCALER_FILE]
               if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing) +
            ". Run `python train_models.py` first."
        )
    models = {
        name: joblib.load(os.path.join(MODEL_DIR, filename))
        for name, filename in MODEL_FILES.items()
    }
    scaler = joblib.load(os.path.join(MODEL_DIR, SCALER_FILE))
    return models, scaler

@st.cache_data
def load_dataset():
    housing = fetch_california_housing(as_frame=True)
    return housing.frame.copy()

@st.cache_data
def get_evaluation():
    frame = load_dataset()
    X = frame[FEATURES]
    y = frame["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test, y_test

try:
    models, scaler = load_models()
except Exception as exc:
    st.error(f"Model loading problem: {exc}")
    st.info("Check that app.py is in the same project folder as train_models.py and the models/ folder.")
    st.stop()

df = load_dataset()
X_test, y_test = get_evaluation()

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ----------------------------- Helpers -----------------------------
def money(value):
    return f"${value:,.0f}"

def predict_all(input_frame):
    scaled = scaler.transform(input_frame[FEATURES])
    return {
        name: float(model.predict(scaled)[0]) * 100000
        for name, model in models.items()
    }

def section_heading(title, subtitle=None):
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="panel-sub">{subtitle}</div>', unsafe_allow_html=True)

def make_plot_layout(fig, height=330):
    fig.update_layout(
        height=height, margin=dict(l=12,r=12,t=28,b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#536887"),
        legend=dict(orientation="h", y=-0.22),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#eaf0f8", zeroline=False)
    return fig

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## 🏡 HousePriceAI")
    st.caption("ML-powered real estate insights")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Predict House Price", "Model Performance",
         "Geographic Insights", "Data Explorer", "Prediction History", "About"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("### Project status")
    st.markdown('<span class="pill">3 models loaded</span>', unsafe_allow_html=True)
    st.caption("California Housing Dataset")
    st.caption("Linear • Ridge • Lasso")
    st.markdown("---")
    st.caption("Built with Python & Streamlit")

# ----------------------------- Header -----------------------------
top1, top2 = st.columns([5,1])
with top1:
    st.markdown("**CALIFORNIA HOUSING MARKET ANALYSIS**")
    st.markdown('<div class="small-muted">Explore housing patterns, estimate values and compare machine-learning models.</div>', unsafe_allow_html=True)
with top2:
    st.markdown(f'<div style="text-align:right;color:#71829e;font-size:.82rem;padding-top:8px">● Models ready<br>{datetime.now().strftime("%d %b %Y")}</div>', unsafe_allow_html=True)

# ============================= OVERVIEW =============================
if page == "Overview":
    st.markdown("""
    <div class="hero">
      <div class="hero-kicker">Intelligent property valuation</div>
      <h1>Understand the market.<br>Estimate with confidence.</h1>
      <p>Explore California housing data and compare three regression models in one interactive analytics workspace.</p>
      <span class="pill">Linear Regression</span>&nbsp;&nbsp;
      <span class="pill">Ridge Regression</span>&nbsp;&nbsp;
      <span class="pill">Lasso Regression</span>
    </div>
    """, unsafe_allow_html=True)

    eval_rows = []
    scaled_test = scaler.transform(X_test)
    for name, model in models.items():
        pred = model.predict(scaled_test)
        eval_rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred),
        })
    eval_df = pd.DataFrame(eval_rows)
    best = eval_df.loc[eval_df["R²"].idxmax()]
    avg_price = df["MedHouseVal"].mean() * 100000
    total_predictions = len(st.session_state.prediction_history)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Dataset records", f"{len(df):,}", "California census areas")
    m2.metric("Average median value", money(avg_price), "Dataset average")
    m3.metric("Top model", best["Model"].replace(" Regression",""), f"R² {best['R²']:.3f}")
    m4.metric("Your predictions", f"{total_predictions:,}", "This session")

    st.write("")
    left, right = st.columns([1.35,1])
    with left:
        section_heading("Model performance snapshot", "Evaluation on the same held-out test set")
        fig = go.Figure()
        for metric, label in [("MAE","MAE"),("RMSE","RMSE")]:
            fig.add_trace(go.Bar(name=label, x=eval_df["Model"], y=eval_df[metric]))
        st.plotly_chart(make_plot_layout(fig,300), use_container_width=True)
    with right:
        section_heading("What influences value?", "Explore relationships in the dataset")
        corr = df[FEATURES + ["MedHouseVal"]].corr(numeric_only=True)["MedHouseVal"].drop("MedHouseVal").sort_values()
        fig2 = px.bar(x=corr.values, y=corr.index, orientation="h",
                      labels={"x":"Correlation with median value","y":"Feature"})
        fig2.update_traces(marker_color="#4c91f0")
        st.plotly_chart(make_plot_layout(fig2,300), use_container_width=True)

    st.write("")
    a,b = st.columns([1,1])
    with a:
        section_heading("California housing value distribution", "Target variable converted to dollars")
        fig3 = px.histogram(df, x=df["MedHouseVal"]*100000, nbins=45,
                            labels={"x":"Median house value (USD)"})
        fig3.update_traces(marker_color="#6b8ff2", opacity=.86)
        st.plotly_chart(make_plot_layout(fig3,290), use_container_width=True)
    with b:
        section_heading("Market signals", "High-level patterns from the training dataset")
        inc_corr = df["MedInc"].corr(df["MedHouseVal"])
        age_corr = df["HouseAge"].corr(df["MedHouseVal"])
        pop_corr = df["Population"].corr(df["MedHouseVal"])
        st.markdown(f"""
        <div class="insight"><b>💼 Median income</b><br><span class="small-muted">Correlation with value: {inc_corr:.2f}. Income is an important signal in this dataset.</span></div>
        <div class="insight"><b>🏘️ House age</b><br><span class="small-muted">Correlation with value: {age_corr:.2f}. Age has a comparatively weaker linear relationship.</span></div>
        <div class="insight"><b>👥 Population</b><br><span class="small-muted">Correlation with value: {pop_corr:.2f}. Population alone is not a reliable valuation measure.</span></div>
        """, unsafe_allow_html=True)
    st.caption("Important: values represent median house values for California census block groups, not the sale price of an individual property.")

# ============================= PREDICT =============================
elif page == "Predict House Price":
    st.title("Predict a house value")
    st.markdown("Enter census-area characteristics below. The same saved scaler and feature order used during training are applied.")
    with st.container(border=True):
        st.markdown("#### 📍 Location & neighbourhood")
        c1,c2,c3 = st.columns(3)
        with c1:
            latitude = st.number_input("Latitude", 32.0, 42.0, 34.05, 0.01, help="California latitude range.")
        with c2:
            longitude = st.number_input("Longitude", -125.0, -114.0, -118.24, 0.01, help="California longitude range.")
        with c3:
            population = st.number_input("Population", 1.0, 40000.0, 1000.0, 10.0)
        st.markdown("#### 🏠 Housing characteristics")
        c1,c2,c3 = st.columns(3)
        with c1:
            house_age = st.number_input("Median house age (years)", 1.0, 60.0, 20.0, 1.0)
            ave_rooms = st.number_input("Average rooms per household", 1.0, 20.0, 5.0, 0.1)
        with c2:
            ave_bedrms = st.number_input("Average bedrooms per household", 0.5, 10.0, 1.0, 0.1)
            ave_occup = st.number_input("Average household occupancy", 0.5, 20.0, 3.0, 0.1)
        with c3:
            med_inc = st.number_input("Median income (in $10,000s)", 0.5, 20.0, 5.0, 0.1,
                                      help="For example, 5.0 represents approximately $50,000.")
            st.info("The dataset uses area-level statistics, not room counts for one specific home.")

        selected_model = st.selectbox("Primary model for highlighted estimate",
                                      list(models.keys()), index=1)
        submitted = st.button("✨ Generate price estimate", type="primary", use_container_width=True)

    if submitted:
        input_frame = pd.DataFrame([{
            "MedInc":med_inc, "HouseAge":house_age, "AveRooms":ave_rooms,
            "AveBedrms":ave_bedrms, "Population":population,
            "AveOccup":ave_occup, "Latitude":latitude, "Longitude":longitude
        }], columns=FEATURES)
        preds = predict_all(input_frame)
        primary = preds[selected_model]
        record = {
            "Date & time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Latitude":latitude, "Longitude":longitude,
            "Median income":med_inc, "House age":house_age,
            "Average rooms":ave_rooms, "Average bedrooms":ave_bedrms,
            "Population":population, "Average occupancy":ave_occup,
            "Selected model":selected_model,
            "Estimated value":primary,
            **{f"{k} estimate":v for k,v in preds.items()}
        }
        st.session_state.prediction_history.insert(0,record)
        st.success("Estimate generated. The result has been added to this session's prediction history.")
        st.markdown("### Estimated median house value")
        st.markdown(f"<h1 style='color:#2478ef;font-size:2.8rem'>{money(primary)}</h1>", unsafe_allow_html=True)
        st.caption(f"Highlighted model: {selected_model} • California census-area estimate")
        p1,p2,p3 = st.columns(3)
        for col,(name,value) in zip([p1,p2,p3],preds.items()):
            col.metric(name, money(value))
        fig = px.bar(x=list(preds.keys()),y=list(preds.values()),
                     labels={"x":"Model","y":"Estimated value (USD)"},text=[money(v) for v in preds.values()])
        fig.update_traces(marker_color=["#3987ef","#17ae9a","#8c6ce8"],textposition="outside")
        st.plotly_chart(make_plot_layout(fig,320),use_container_width=True)
        st.warning("This is a model estimate of a census block group's median value. It is not a formal appraisal, listing price, or guaranteed sale price.")

# ============================= MODEL PERFORMANCE =============================
elif page == "Model Performance":
    st.title("Model performance")
    st.markdown("A consistent test split is used to compare all three saved models.")
    scaled_test = scaler.transform(X_test)
    rows, pred_map = [], {}
    for name,model in models.items():
        pred = model.predict(scaled_test)
        pred_map[name] = pred
        rows.append({"Model":name,
                     "MAE":mean_absolute_error(y_test,pred),
                     "RMSE":np.sqrt(mean_squared_error(y_test,pred)),
                     "R²":r2_score(y_test,pred)})
    metrics = pd.DataFrame(rows).sort_values("R²",ascending=False)
    best = metrics.iloc[0]
    c1,c2,c3 = st.columns(3)
    c1.metric("Best model",best["Model"])
    c2.metric("Best R²",f'{best["R²"]:.4f}')
    c3.metric("Best RMSE",f'{best["RMSE"]:.4f}',"Target units: $100,000s")
    st.markdown("#### Evaluation metrics")
    display_metrics = metrics.copy()
    for col in ["MAE","RMSE","R²"]:
        display_metrics[col] = display_metrics[col].map(lambda x:f"{x:.4f}")
    st.dataframe(display_metrics,use_container_width=True,hide_index=True)
    l,r = st.columns(2)
    with l:
        section_heading("Actual vs predicted","Closer to the diagonal indicates more accurate predictions")
        chosen = st.selectbox("Choose model",list(models.keys()))
        actual_dollars = np.asarray(y_test)*100000
        pred_dollars = pred_map[chosen]*100000
        sample = pd.DataFrame({"Actual":actual_dollars,"Predicted":pred_dollars})
        fig=px.scatter(sample.sample(min(1200,len(sample)),random_state=42),x="Actual",y="Predicted",
                       opacity=.45,labels={"Actual":"Actual value (USD)","Predicted":"Predicted value (USD)"})
        lo=float(min(sample.min())); hi=float(max(sample.max()))
        fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="Ideal prediction",line=dict(color="#ef8b45",dash="dash")))
        st.plotly_chart(make_plot_layout(fig,360),use_container_width=True)
    with r:
        section_heading("Residual distribution","Residual = actual value − predicted value")
        residuals = (np.asarray(y_test)-pred_map[chosen])*100000
        fig=px.histogram(x=residuals,nbins=45,labels={"x":"Residual (USD)"})
        fig.update_traces(marker_color="#8b72e8")
        st.plotly_chart(make_plot_layout(fig,360),use_container_width=True)
    st.caption("MAE and RMSE are measured in the dataset target scale (MedHouseVal units of $100,000). R² is unitless.")

# ============================= GEOGRAPHIC INSIGHTS =============================
elif page == "Geographic Insights":
    st.title("Geographic insights")
    st.markdown("Explore the geographic distribution of California census block groups.")
    map_df = df.sample(min(5000,len(df)),random_state=42).copy()
    map_df["Median value (USD)"] = map_df["MedHouseVal"]*100000
    # Use Plotly's built-in geo projection instead of Mapbox tiles.
    # This avoids blank maps caused by missing Mapbox tokens or blocked tile servers.
    fig=px.scatter_geo(
        map_df,
        lat="Latitude",
        lon="Longitude",
        color="Median value (USD)",
        size="Population",
        size_max=9,
        color_continuous_scale="Blues",
        scope="usa",
        projection="albers usa",
        hover_data={
            "MedInc":":.2f",
            "HouseAge":":.1f",
            "Median value (USD)":":$,.0f",
            "Latitude":":.3f",
            "Longitude":":.3f",
            "Population":":,.0f"
        },
        labels={"MedInc":"Median income ($10,000s)","HouseAge":"Median house age"}
    )
    fig.update_geos(
        visible=True,
        showcountries=True,
        showsubunits=True,
        subunitcolor="#c8d7ea",
        countrycolor="#aebfd5",
        showland=True,
        landcolor="#f2f6fc",
        showocean=True,
        oceancolor="#e8f2ff",
        lataxis_range=[32, 42],
        lonaxis_range=[-125, -114],
        bgcolor="rgba(0,0,0,0)"
    )
    fig.update_layout(
        height=610,
        margin=dict(l=0,r=0,t=8,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Map displays a sample of up to 5,000 California census block groups. The geographic projection does not require Mapbox credentials or external map tiles.")
    a,b=st.columns(2)
    with a:
        section_heading("Income and median value","Each point represents a census block group")
        fig2=px.scatter(map_df.sample(min(2500,len(map_df)),random_state=7),x="MedInc",y="MedHouseVal",
                        color="HouseAge",opacity=.55,color_continuous_scale="Viridis",
                        labels={"MedInc":"Median income ($10,000s)","MedHouseVal":"Median value ($100,000s)"})
        st.plotly_chart(make_plot_layout(fig2,330),use_container_width=True)
    with b:
        section_heading("Population distribution","Population varies substantially by area")
        fig3=px.histogram(df,x="Population",nbins=45)
        fig3.update_traces(marker_color="#20ad9c")
        st.plotly_chart(make_plot_layout(fig3,330),use_container_width=True)

# ============================= DATA EXPLORER =============================
elif page == "Data Explorer":
    st.title("Data explorer")
    st.markdown("Inspect the original California Housing dataset used to train and evaluate the models.")
    c1,c2,c3=st.columns(3)
    c1.metric("Rows",f"{len(df):,}")
    c2.metric("Features",str(len(FEATURES)))
    c3.metric("Missing values",str(int(df.isna().sum().sum())))
    with st.expander("Dataset preview",expanded=True):
        st.dataframe(df.head(100),use_container_width=True)
    l,r=st.columns(2)
    with l:
        feature=st.selectbox("Select a feature to inspect",FEATURES+["MedHouseVal"])
        fig=px.histogram(df,x=feature,nbins=45,marginal="box")
        fig.update_traces(marker_color="#4b8cf0")
        st.plotly_chart(make_plot_layout(fig,350),use_container_width=True)
    with r:
        section_heading("Feature correlation matrix","Pearson correlation across numeric columns")
        corr=df[FEATURES+["MedHouseVal"]].corr(numeric_only=True)
        fig=px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,aspect="auto")
        st.plotly_chart(make_plot_layout(fig,350),use_container_width=True)
    st.markdown("#### Descriptive statistics")
    st.dataframe(df.describe().T,use_container_width=True)

# ============================= PREDICTION HISTORY =============================
elif page == "Prediction History":
    st.title("Prediction history")
    history=st.session_state.prediction_history
    st.markdown("Predictions are stored in the current browser session. They are not permanently saved.")
    if history:
        hist_df=pd.DataFrame(history)
        c1,c2=st.columns([1,5])
        with c1:
            if st.button("Clear history"):
                st.session_state.prediction_history=[]
                st.rerun()
        st.dataframe(hist_df,use_container_width=True,hide_index=True)
        csv=hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download prediction history (CSV)",csv,
                           file_name="house_price_prediction_history.csv",mime="text/csv")
    else:
        st.info("No predictions yet. Open **Predict House Price** to create your first estimate.")

# ============================= ABOUT =============================
elif page == "About":
    st.title("About HousePriceAI")
    st.markdown("""
    <div class="info-card">
      <h3>🏡 A machine-learning housing analytics project</h3>
      <p>This application demonstrates how regression models can estimate median house values from California census-area characteristics.</p>
    </div>
    """,unsafe_allow_html=True)
    st.write("")
    a,b=st.columns(2)
    with a:
        st.markdown("#### Machine-learning algorithms")
        st.write("- Linear Regression\n- Ridge Regression (L2 regularization)\n- Lasso Regression (L1 regularization)")
        st.markdown("#### Input features")
        st.write("Median income, house age, average rooms, average bedrooms, population, average occupancy, latitude and longitude.")
    with b:
        st.markdown("#### Technology stack")
        st.write("- Python and Streamlit\n- Scikit-learn\n- Pandas and NumPy\n- Plotly\n- Joblib")
        st.markdown("#### Dataset")
        st.write("California Housing dataset distributed through scikit-learn.")
    st.warning("Educational demonstration only. The model predicts census block-group median values and should not be used as a real-world property appraisal or financial decision tool.")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#8190a8;font-size:.8rem">HousePriceAI · California Housing Dataset · Built with Streamlit, Scikit-learn & Plotly</div>',unsafe_allow_html=True)
