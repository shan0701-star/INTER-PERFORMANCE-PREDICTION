import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Intern Performamce predicition", layout="wide")

# ---------------- CUSTOM CSS 🔥 ----------------
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
    }
    .main-title {
        font-size:40px;
        color:#00ffcc;
        text-align:center;
        font-weight:bold;
    }
    .card {
        padding:20px;
        border-radius:15px;
        background: linear-gradient(135deg, #1f4037, #99f2c8);
        color:black;
        text-align:center;
        font-size:20px;
        font-weight:bold;
        margin:10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🔥 Intern Performance Dashboard</p>', unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("intern_data.csv")
df.columns = df.columns.str.strip()

df['performance'] = df['performance'].map({'High':2,'Medium':1,'Low':0})

# ---------------- MODEL ----------------
X = df[['task_completed','attendance','feedback_score']]
y = df['performance']

model = RandomForestClassifier()
model.fit(X,y)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎯 Select Intern")

intern = st.sidebar.selectbox("Choose Intern", df['intern_name'])

row = df[df['intern_name'] == intern].iloc[0]

task = row['task_completed']
attendance = row['attendance']
feedback = row['feedback_score']

# ---------------- CARDS UI 🔥 ----------------
col1, col2, col3 = st.columns(3)

col1.markdown(f'<div class="card">📌 Task<br>{task}</div>', unsafe_allow_html=True)
col2.markdown(f'<div class="card">📊 Attendance<br>{attendance}</div>', unsafe_allow_html=True)
col3.markdown(f'<div class="card">⭐ Feedback<br>{feedback}</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Performance"):

    result = model.predict([[task,attendance,feedback]])

    if result[0]==2:
        label = "High"
        st.success("🔥 High Performance")
    elif result[0]==1:
        label = "Medium"
        st.warning("⚡ Medium Performance")
    else:
        label = "Low"
        st.error("❌ Low Performance")

    # ---------------- GRAPH ----------------
    st.subheader("📊 Intern Metrics")

    fig, ax = plt.subplots()
    ax.bar(['Task','Attendance','Feedback'], [task,attendance,feedback])
    st.pyplot(fig)

    # ---------------- PERFORMANCE GRAPH ----------------
    st.subheader("🎯 Performance Level")

    perf_map = {"Low":1,"Medium":2,"High":3}

    fig2, ax2 = plt.subplots()
    ax2.bar(["Performance"], [perf_map[label]])
    st.pyplot(fig2)

# ---------------- OVERALL GRAPH ----------------
st.subheader("📈 Overall Performance")

fig3, ax3 = plt.subplots()
df['performance'].value_counts().plot(kind='bar', ax=ax3)
st.pyplot(fig3)

# ---------------- TABLE ----------------
st.subheader("📋 Dataset Preview")
st.dataframe(df.head(15))