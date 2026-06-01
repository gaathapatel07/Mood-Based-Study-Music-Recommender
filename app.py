import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#111827);
}

.main-title{
    text-align:center;
    color:white;
    font-size:3rem;
    font-weight:800;
    margin-bottom:0px;
}

.subtitle{
    text-align:center;
    color:#94a3b8;
    margin-bottom:30px;
}

.music-card{
    background: rgba(30,41,59,0.85);
    padding:25px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(15px);
    text-align:center;
    margin-top:20px;
}

.music-title{
    color:#22c55e;
    font-size:28px;
    font-weight:bold;
}

.music-subtitle{
    color:white;
    font-size:18px;
}

.stButton>button{
    width:100%;
    background:#22c55e;
    color:white;
    border:none;
    border-radius:12px;
    height:3.2em;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#16a34a;
}

</style>
""", unsafe_allow_html=True)
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Mood-Based Study Music Recommender",
    page_icon="🎧",
    layout="centered"
)

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("music_data.csv")

# -----------------------------
# Encode Data
# -----------------------------
mood_encoder = LabelEncoder()
task_encoder = LabelEncoder()
rec_encoder = LabelEncoder()

data["Mood"] = mood_encoder.fit_transform(data["Mood"])
data["Task"] = task_encoder.fit_transform(data["Task"])
data["Recommendation"] = rec_encoder.fit_transform(
    data["Recommendation"]
)

# -----------------------------
# Features and Target
# -----------------------------
X = data[["Mood", "Task", "Energy"]]
y = data["Recommendation"]

# -----------------------------
# Train Model
# -----------------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# -----------------------------
# Playlist Links
# -----------------------------
playlist_links = {
    "Lo-Fi Beats":
        "https://www.youtube.com/results?search_query=lofi+study+music",

    "Ambient Focus":
        "https://www.youtube.com/results?search_query=ambient+focus+music",

    "Deep Focus Electronic":
        "https://www.youtube.com/results?search_query=deep+focus+electronic",

    "Classical Piano":
        "https://www.youtube.com/results?search_query=classical+piano+study",

    "Nature Sounds":
        "https://www.youtube.com/results?search_query=nature+sounds+study",

    "Instrumental Jazz":
        "https://www.youtube.com/results?search_query=instrumental+jazz+study",

    "White Noise":
        "https://www.youtube.com/results?search_query=white+noise+focus"
}

# -----------------------------
# UI
# -----------------------------
st.markdown(
    """
    <div class='main-title'>
    🎧 Mood Music AI
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='subtitle'>
    Discover your perfect study soundtrack
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    Find the perfect study music based on your mood,
    task type, and energy level.
    """
)

st.divider()

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    """
    This AI-powered recommendation system suggests
    study music categories using Machine Learning.

    Built with:
    - Python
    - Scikit-Learn
    - Streamlit
    """
)

# Inputs
mood = st.selectbox(
    "😊 Select Mood",
    ["Happy", "Stressed", "Tired", "Motivated", "Relaxed"]
)

task = st.selectbox(
    "📚 Select Task",
    ["Coding", "Reading", "Writing", "Exam Prep"]
)

energy = st.slider(
    "⚡ Energy Level",
    min_value=1,
    max_value=10,
    value=5
)

st.write("")

# Prediction
if st.button("🎵 Recommend Music"):

    mood_encoded = mood_encoder.transform([mood])[0]
    task_encoded = task_encoder.transform([task])[0]

    prediction = model.predict(
        [[mood_encoded, task_encoded, energy]]
    )

    recommendation = rec_encoder.inverse_transform(
        prediction
    )[0]

    st.success(
        f"🎧 Recommended Music Category: {recommendation}"
    )

    st.markdown(
        f"### 🎶 Suggested Playlist"
    )

    st.markdown(
        f"[Open Playlist Here]({playlist_links[recommendation]})"
    )

    st.balloons()

# Footer
st.divider()

st.caption(
    "Built using Machine Learning, Streamlit, and Scikit-Learn 🚀"
)