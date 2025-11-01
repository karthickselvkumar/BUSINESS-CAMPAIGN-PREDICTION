import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -----------------------------
# 🎯 Streamlit App Config
# -----------------------------
st.set_page_config(page_title="Business Campaign Profit Predictor", page_icon="💼")
st.title("💼 Business Campaign Profit Predictor")
st.write("Predict whether a business campaign will be **Profitable (Yes/No)** using machine learning models.")

# -----------------------------
# 📂 Load Dataset (no upload)
# -----------------------------
data_path = r"C:\Users\Sys\Desktop\ProfitPredict\business_campaign.csv"
data = pd.read_csv(data_path)

# -----------------------------
# 🧹 Data Preprocessing
# -----------------------------
df = data.copy()
target = "Profitable"

# Encode target variable
le_target = LabelEncoder()
df[target] = le_target.fit_transform(df[target])

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    if col != target:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

X = df.drop(columns=[target])
y = df[target]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# -----------------------------
# ⚙️ Train Multiple Models
# -----------------------------
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)

# -----------------------------
# 🧾 Display Model Accuracies
# -----------------------------
st.subheader("📊 Model Accuracies")
for model_name, acc in accuracies.items():
    st.write(f"**{model_name}:** {acc:.2f}")

# Bar Chart for accuracies
fig, ax = plt.subplots()
ax.bar(accuracies.keys(), accuracies.values())
ax.set_ylabel("Accuracy")
ax.set_title("Model Performance Comparison")
st.pyplot(fig)

# Best model selection
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
st.success(f"🏆 Best Model: {best_model_name} (Accuracy: {accuracies[best_model_name]:.2f})")

st.markdown("---")

# -----------------------------
# 🧮 Prediction Section
# -----------------------------
st.subheader("🔮 Predict Campaign Profitability")

user_input = {}
for col in X.columns:
    if data[col].dtype in [np.float64, np.int64]:
        user_input[col] = st.number_input(f"{col}", value=float(data[col].mean()))
    else:
        options = sorted(data[col].unique())
        user_input[col] = st.selectbox(f"{col}", options)

input_df = pd.DataFrame([user_input])

# Encode categorical fields
for col, le in label_encoders.items():
    input_df[col] = le.transform(input_df[col])

# Scale numeric fields
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Profitability"):
    pred = best_model.predict(input_scaled)[0]
    result = "✅ Profitable" if pred == 1 else "❌ Not Profitable"
    st.success(f"**Prediction:** {result}")
