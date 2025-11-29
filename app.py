import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- App config ---
st.set_page_config(page_title="EV Range Predictor", page_icon="🔋", layout="wide")

# --- Sidebar Navigation ---
page = st.sidebar.radio("📂 Navigate", ["Home", "Predict Range", "Model Performance"])

# --- Safe loaders (show gentle errors if files are missing) ---
def safe_load_joblib(path, label):
    try:
        return joblib.load(path)
    except Exception as e:
        st.sidebar.warning(f"Missing or unreadable {label}: {path}")
        return None

def safe_read_csv(path, label):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.sidebar.warning(f"Missing or unreadable {label}: {path}")
        return None

# --- Load Models & Scaler ---
scaler = safe_load_joblib("scaler.pkl", "scaler")
models = {
    "Random Forest": safe_load_joblib("rf_model.pkl", "Random Forest model"),
    "Linear Regression": safe_load_joblib("lr_model.pkl", "Linear Regression model"),
    "SVR": safe_load_joblib("svr_model.pkl", "SVR model"),
    "KNN": safe_load_joblib("knn_model.pkl", "KNN model")
}
# Filter out None entries
models = {k: v for k, v in models.items() if v is not None}

# --- Load Data for Evaluation ---
df = safe_read_csv("electric_vehicles_spec_2025.csv", "EV dataset")

# Prepare X, y if df and scaler exist
X, y = None, None
if df is not None and scaler is not None:
    # Basic cleaning
    if "torque_nm" in df.columns:
        df["torque_nm"].fillna(df["torque_nm"].median(), inplace=True)
    if "fast_charging_power_kw_dc" in df.columns:
        df["fast_charging_power_kw_dc"].fillna(df["fast_charging_power_kw_dc"].median(), inplace=True)
    if "cargo_volume_l" in df.columns:
        df["cargo_volume_l"] = pd.to_numeric(df["cargo_volume_l"], errors="coerce")
        df["cargo_volume_l"].fillna(df["cargo_volume_l"].median(), inplace=True)

    required_features = ["battery_capacity_kWh", "top_speed_kmh", "fast_charging_power_kw_dc", "torque_nm", "length_mm"]
    if all(col in df.columns for col in required_features) and "range_km" in df.columns:
        try:
            X = scaler.transform(df[required_features])
            y = df["range_km"]
        except Exception as e:
            st.sidebar.warning("Scaler/features mismatch. Check feature order and scaler training.")

# --- Home Page ---
if page == "Home":
    st.markdown("""
    <div style="background: linear-gradient(to right, #4f46e5, #9333ea); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h1 style="color: white; text-align: center; margin: 0;">🔋 EV Range Predictor</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🚀 Overview
    Estimate the driving range of an electric vehicle from key specifications using multiple machine learning models.
    Use the sidebar to navigate, input EV specs, choose a model, and view performance metrics.

    ---
    """)

    # Author & Skills
    st.subheader("👨‍💻 Author & profile")
    st.markdown("""
    **Nicholas Mwaniki Musee**  
    PhD Candidate in Computer Science (AI/ML), University of Nairobi  
    Lecturer, Data Scientist, Full‑Stack MERN Developer, and Faith‑Centered Community Leader

    **Links:**  
    - LinkedIn: [www.linkedin.com/in/nmusee2025](https://www.linkedin.com/in/nmusee2025)  
    
    - Project: [https://window-weather-613c.vercel.app/]  
    
    """)

    st.markdown("---")
    st.subheader("🛠️ Key skills")
    st.markdown("""
    - **MERN stack:** MongoDB, Express.js, React.js, Node.js  
    - **Programming & tools:** Python, R, SQL/MySQL, MongoDB, Flask, Streamlit, HTML/CSS, Tailwind, Socket.io  
    - **Data analysis & visualization:** NumPy, Pandas, Matplotlib, Seaborn, Power BI, Tableau, EDA, Feature Engineering, Web Scraping  
    - **Machine learning:** Regression, Classification, Clustering, Ensemble Learning; metrics (MAE, RMSE, R², Accuracy, Precision, Recall, F1)  
    - **Deep learning:** ANN, CNN, RNN, LSTM  
    - **NLP:** Tokenization, Lemmatization, Word2Vec, TF‑IDF, Doc2Vec, spaCy, NLTK  
    - **Computer vision:** Object Detection, Segmentation, Tracking  
    - **Big Data:** Apache Spark, PySpark, Spark SQL, Spark ML, Spark Streaming, Kafka integration  
    - **Cloud & MLOps:** AWS (EC2, S3, IAM, Lambda, SageMaker), Docker, MLflow, PyCaret, GitHub Actions  
    - **Deployment:** Streamlit, Flask; reproducible pipelines and monitoring  
    - **Leadership:** Mentorship, troubleshooting complex environments, transparent documentation, pastoral guidance
    """)

    st.markdown("---")
    st.subheader("🌐 Core competencies")
    st.markdown("""
    - **Data strategy & governance:** Metadata management, ethical data use  
    - **Advanced analytics:** Descriptive, inferential, predictive; Power BI dashboards  
    - **Agentic AI & platforms:** Genkit, ADK, Vertex AI, MLOps  
    - **Humanitarian tech & IoT analytics:** LMMS, Kobo Collect, IoT data workflows  
    - **Capacity building:** Peer learning, faith‑based leadership, cross‑country collaboration  
    - **End‑to‑end development:** Backend & frontend web dev, hackathon mentorship, AI/ML training
    """)

    st.markdown("---")
    st.subheader("📊 Big Data skills agenda")
    st.markdown("""
    - **What is Big Data?** Examples, challenges, benefits  
    - **Distributed systems:** Introduction, architecture, scalability, fault tolerance, benefits  
    - **Spark & PySpark:** RDDs, DataFrames, Spark SQL, ML, Streaming  
    - **PySpark DataFrame:** Basic functions, data handling, analysis tasks  
    - **Project practice:** Problem statement formulation and code implementation for large‑scale analysis
    """)

# --- Predict Range Page ---
elif page == "Predict Range":
    st.header("🔍 Predict EV range")

    if scaler is None or len(models) == 0:
        st.info("Please ensure the scaler and model files are available to enable predictions.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            battery = st.number_input("🔋 Battery Capacity (kWh)", min_value=10.0, max_value=150.0, value=75.0, step=1.0)
            torque = st.number_input("🌀 Torque (Nm)", min_value=100.0, max_value=1000.0, value=350.0, step=10.0)
        with col2:
            speed = st.number_input("🏎️ Top Speed (km/h)", min_value=80.0, max_value=300.0, value=180.0, step=1.0)
            length = st.number_input("📏 Length (mm)", min_value=3000.0, max_value=6000.0, value=4600.0, step=10.0)
        with col3:
            charging = st.number_input("⚡ Fast Charging Power (kW DC)", min_value=20.0, max_value=350.0, value=150.0, step=5.0)
            model_choice = st.selectbox("🧠 Select Prediction Model", list(models.keys()))

        if st.button("🚗 Predict Range"):
            try:
                input_data = np.array([[battery, speed, charging, torque, length]])
                input_scaled = scaler.transform(input_data)
                prediction = models[model_choice].predict(input_scaled)
                st.success(f"✅ Estimated EV range using {model_choice}: **{prediction[0]:.2f} km**")
            except Exception as e:
                st.error("Prediction failed. Verify scaler and model feature alignment.")

# --- Model Performance Page ---
elif page == "Model Performance":
    st.header("📊 Model performance summary")

    if X is None or y is None or len(models) == 0:
        st.info("Performance summary requires dataset, scaler, and at least one loaded model.")
    else:
        summary = []
        for name, model in models.items():
            try:
                pred = model.predict(X)
                mae = mean_absolute_error(y, pred)
                rmse = np.sqrt(mean_squared_error(y, pred))
                r2 = r2_score(y, pred)
                summary.append([name, f"{mae:.2f}", f"{rmse:.2f}", f"{r2:.2f}"])
            except Exception as e:
                summary.append([name, "—", "—", "—"])

        df_summary = pd.DataFrame(summary, columns=["Model", "MAE", "RMSE", "R² Score"])
        st.dataframe(df_summary, use_container_width=True)

        # Optional: quick guidance
        st.caption("Tip: Lower MAE/RMSE and higher R² generally indicate better performance.")
