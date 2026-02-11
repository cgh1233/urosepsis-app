import streamlit as st
import pandas as pd
import shap
import joblib
import streamlit.components.v1 as components

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
    "This system predicts the risk of **urosepsis** based on clinical, laboratory, "
    "and imaging indicators, and provides model interpretability using SHAP."
)

# ================== 输入表单 ==================
def user_input_features():
    st.markdown("### 👨‍⚕️ Patient Clinical Information")

    left, right = st.columns(2)
    data = {}

    # ===== 左侧 =====
    data["Gender"] = left.selectbox("Gender", options=[0, 1],
                                    format_func=lambda x: "Male" if x == 1 else "Female")

    data["5-mFI"] = left.number_input("Frailty Score (5-mFI)", 0, 10, 1)

    data["UrinaryTractInfection"] = left.selectbox("Urinary Tract Infection (UTI)", [0, 1])

    data["CalculusObstruction"] = left.selectbox("Calculus Obstruction", [0, 1])

    data["Degreeofhydronephrosis"] = left.selectbox(
        "Degree of Hydronephrosis",
        [0, 1, 2, 3]
    )

    data["Locationofcalculi"] = left.selectbox(
        "Location of Calculi",
        [1, 2, 3]
    )

    # ===== 右侧 =====
    data["Maximumdiameterofcalculi"] = right.number_input(
        "Max Stone Diameter (mm)",
        0.0, 50.0, 10.0
    )

    data["Albumin"] = right.number_input(
        "Albumin (g/L)",
        10.0, 60.0, 40.0
    )

    data["CRP"] = right.number_input(
        "C-reactive Protein (mg/L)",
        0.0, 300.0, 20.0
    )

    data["PCT"] = right.number_input(
        "Procalcitonin (ng/mL)",
        0.0, 100.0, 0.5
    )

    data["Urineculture"] = right.selectbox(
        "Urine Culture Positive",
        [0, 1]
    )

    return pd.DataFrame([data])


input_df = user_input_features()

# ================== 预测 & SHAP解释 ==================
if st.button("Start Prediction"):

    # 强制列顺序与训练一致（非常重要）
    input_df = input_df[model.feature_names_in_]

    # 预测正类概率
    proba = model.predict_proba(input_df)[0][1] * 100

    st.markdown(f"""
    <div style="text-align:center;font-size:22px;color:#b30000;margin-top:20px;">
        <strong>Predicted probability of urosepsis: <u>{proba:.2f}%</u></strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔍 Model Explanation (SHAP Force Plot)")

    # 计算 shap_values（旧版兼容方式）
    shap_values = explainer.shap_values(input_df)

    # 二分类模型情况
    if isinstance(shap_values, list):
        # 解释正类 (urosepsis = 1)
        shap_value = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        # 单输出情况
        shap_value = shap_values[0]
        base_value = explainer.expected_value

    # 生成 force plot
    shap_html = shap.plots.force(
        base_value,
        shap_value,
        input_df.iloc[0],
        matplotlib=False
    )

    html_content = f"<head>{shap.getjs()}</head><body>{shap_html.html()}</body>"

    components.html(html_content, height=300)

