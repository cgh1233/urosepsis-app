import streamlit as st
import pandas as pd
import joblib

# ================== 页面配置 ==================
st.set_page_config(
    page_title="Urosepsis Risk Prediction System",
    layout="wide",
    page_icon="🩺"
)

# ================== 页面样式 ==================
st.markdown("""
<style>
    .stButton>button {
        background-color: #d9534f;
        color: white;
        border-radius: 10px;
        font-size: 20px;
        padding: 0.5em 1em;
    }
    .stNumberInput>div>input {
        font-size: 18px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ================== 加载模型 ==================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()
explainer = shap.TreeExplainer(model)

# ================== 标题 ==================
st.title("🩺 Urosepsis Risk Prediction System")
st.markdown(
    "This system predicts the risk of **urosepsis** "
    "based on key laboratory and clinical indicators."
)

# ================== 输入表单 ==================
def user_input_features():
    st.markdown("### 👨‍⚕️ Patient Clinical Information")

    left, right = st.columns(2)
    data = {}

    # ===== 左侧 =====
    data["PCT"] = left.number_input(
        "Procalcitonin (ng/mL)",
        0.0, 100.0, 0.5
    )

    data["Degreeofhydronephrosis"] = left.selectbox(
        "Degree of Hydronephrosis",
        [0, 1, 2, 3]
    )

    data["Albumin"] = left.number_input(
        "Albumin (g/L)",
        10.0, 60.0, 40.0
    )

    # ===== 右侧 =====
    data["5-mFI"] = right.number_input(
        "Frailty Score (5-mFI)",
        0, 10, 1
    )

    data["Maximumdiameterofcalculi"] = right.number_input(
        "Max Stone Diameter (mm)",
        0.0, 50.0, 10.0
    )

    data["UrinaryTractInfection"] = right.selectbox(
        "Urinary Tract Infection (UTI)",
        [0, 1]
    )

    return pd.DataFrame([data])


input_df = user_input_features()

# ================== 预测 ==================
if st.button("Start Prediction"):

    # 强制列顺序（极其重要）
    input_df = input_df[
        [
            "PCT",
            "Degreeofhydronephrosis",
            "Albumin",
            "5-mFI",
            "Maximumdiameterofcalculi",
            "UrinaryTractInfection"
        ]
    ]

    # 预测正类概率
    proba = model.predict_proba(input_df)[0][1] * 100

    st.markdown(f"""
    <div style="text-align:center;font-size:24px;color:#b30000;margin-top:20px;">
        <strong>Predicted probability of urosepsis:</strong><br>
        <u>{proba:.2f}%</u>
    </div>
    """, unsafe_allow_html=True)
