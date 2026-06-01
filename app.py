import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("music_data.csv")

# Create encoders
mood_encoder = LabelEncoder()
task_encoder = LabelEncoder()
rec_encoder = LabelEncoder()

# Encode columns
data["Mood"] = mood_encoder.fit_transform(data["Mood"])
data["Task"] = task_encoder.fit_transform(data["Task"])
data["Recommendation"] = rec_encoder.fit_transform(
    data["Recommendation"]
)

# Features and target
X = data[["Mood", "Task", "Energy"]]
y = data["Recommendation"]

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
# ==========================
# PREMIUM UI
# ==========================

st.markdown("""
<style>

.stApp{
    background: #0B1120;
}

.main-title{
    text-align:center;
    font-size:60px;
    font-weight:800;
    color:white;
}

.subtitle{
    text-align:center;
    color:#94A3B8;
    font-size:18px;
    margin-bottom:40px;
}

.card{
    background:#111827;
    padding:20px;
    border-radius:18px;
    border:1px solid #1F2937;
}

.result-card{
    background:linear-gradient(135deg,#16A34A,#22C55E);
    padding:25px;
    border-radius:20px;
    text-align:center;
    color:white;
}

.tip-card{
    background:#1E293B;
    padding:20px;
    border-radius:18px;
    color:white;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    background:#22C55E;
    color:white;
    font-size:18px;
    font-weight:bold;
    border:none;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
"""
<div class='main-title'>
🎧 StudyFlow AI
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='subtitle'>
Your AI Study Companion for Focus, Productivity & Music
</div>
""",
unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    mood = st.selectbox(
        "😊 Mood",
        ["Happy","Stressed","Tired","Motivated","Relaxed"]
    )

with col2:
    task = st.selectbox(
        "📚 Task",
        ["Coding","Reading","Writing","Exam Prep"]
    )

with col3:
    energy = st.slider(
        "⚡ Energy",
        1,
        10,
        5
    )

st.write("")

if st.button("🚀 Generate Study Plan"):

    mood_encoded = mood_encoder.transform([mood])[0]
    task_encoded = task_encoder.transform([task])[0]

    prediction = model.predict(
        [[mood_encoded, task_encoded, energy]]
    )

    recommendation = rec_encoder.inverse_transform(
        prediction
    )[0]

    study_method = {
        "Coding":"Deep Work",
        "Reading":"Active Reading",
        "Writing":"Flow Writing",
        "Exam Prep":"Pomodoro"
    }

    st.markdown(
    f"""
    <div class='result-card'>
        <h2>🎵 {recommendation}</h2>
        <p>Recommended Music Category</p>
    </div>
    """,
    unsafe_allow_html=True
    )

    st.write("")

    c1,c2 = st.columns(2)

    with c1:
        st.markdown(
        f"""
        <div class='tip-card'>
            <h3>📖 Study Method</h3>
            <p>{study_method[task]}</p>
        </div>
        """,
        unsafe_allow_html=True
        )

    with c2:
        st.markdown(
        f"""
        <div class='tip-card'>
            <h3>⏱ Recommended Session</h3>
            <p>{20 + energy*5} Minutes</p>
        </div>
        """,
        unsafe_allow_html=True
        )

    st.write("")

    st.info(
        f"💡 Based on your {mood.lower()} mood and {task.lower()} task, "
        f"we recommend {recommendation.lower()} to maximize focus."
    )