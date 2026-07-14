"""
NeuroScope
--------------------------------
A Streamlit app that collects lifestyle data, measures Reaction Time and
Memory Test Score live (instead of asking the user to type them), then
feeds all 9 features into a pre-trained RandomForestRegressor to predict
a Cognitive Score.

Run with:  streamlit run app2.py

Required files in the same folder:
    model2.pkl              -> trained RandomForestRegressor
    gender_mapping2.pkl      -> dict {category_name: code}
    diet_mapping2.pkl        -> dict {category_name: code}
    exercise_mapping2.pkl    -> dict {category_name: code}
    brain.glb                -> 3D brain model, rendered as the page background
"""

import time
import random

import joblib
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroScope",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Column order MUST match the order used during model training
FEATURE_ORDER = [
    "Gender", "Sleep_Duration", "Stress_Level", "Diet_Type",
    "Daily_Screen_Time", "Exercise_Frequency", "Caffeine_Intake",
    "Reaction_Time", "Memory_Test_Score",
]

# Reasonable "ideal" reference values used only for the radar chart & tips.
# Adjust these if your dataset's typical ranges differ.
IDEAL_VALUES = {
    "Sleep_Duration": 8,        # hours
    "Stress_Level": 3,          # assumed 1-10 scale, lower is better
    "Daily_Screen_Time": 4,     # hours, lower is better
    "Caffeine_Intake": 200,     # mg, moderate/lower is better
    "Reaction_Time": 250,       # ms, lower is better
    "Memory_Test_Score": 90,    # 0-100, higher is better
}

# (min, max, invert) used to normalize each metric onto a 0-100 "goodness" scale
NORMALIZE_RANGES = {
    "Sleep_Duration": (0, 12, False),
    "Stress_Level": (0, 10, True),
    "Daily_Screen_Time": (0, 16, True),
    "Caffeine_Intake": (0, 500, True),
    "Reaction_Time": (100, 700, True),
    "Memory_Test_Score": (0, 100, False),
}


# --------------------------------------------------------------------------
# ASSET LOADING
# --------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("model2.pkl")
    gender_mapping = joblib.load("gender_mapping2.pkl")
    diet_mapping = joblib.load("diet_mapping2.pkl")
    exercise_mapping = joblib.load("exercise_mapping2.pkl")
    return model, gender_mapping, diet_mapping, exercise_mapping


# --------------------------------------------------------------------------
# STYLING — dark futuristic health-tech theme, glassmorphism + glow
# --------------------------------------------------------------------------
def apply_custom_css():
    st.markdown("""
    <style>
    html, body, #root,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"],
    [data-testid="stBottomBlockContainer"],
    .main {
        background: radial-gradient(circle at 20% 15%, #0f3d2e 0%, #0a2318 45%, #04120a 100%) !important;
        color: #E7F5EC;
    }
    h1, h2, h3, h4, p, label, span, div {
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }
    h1 {
        background: linear-gradient(90deg, #6EE7B7, #4ADE80, #A3E635);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }
    h2, h3 { color: #86EFAC !important; }

    /* Glass card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 22px;
        box-shadow: 0 0 25px rgba(34, 197, 94, 0.15);
        transition: box-shadow 0.4s ease;
    }
    .glass-card:hover {
        box-shadow: 0 0 40px rgba(163, 230, 53, 0.3);
    }

    /* Buttons */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #15803D 0%, #22C55E 50%, #4ADE80 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6em 1.4em;
        font-weight: 600;
        letter-spacing: 0.3px;
        box-shadow: 0 0 15px rgba(34, 197, 94, 0.5);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 0 25px rgba(163, 230, 53, 0.85);
    }

    /* Progress bar recolor */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #15803D, #4ADE80);
    }

    /* Big reaction test flash block (fills most of the viewport height) */
    .reaction-flash {
        width: 100%;
        min-height: 50vh;
        border-radius: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 10px 0 20px 0;
    }
    .reaction-flash.red { background: radial-gradient(circle, #ff2d55 0%, #7a0d24 100%); }
    .reaction-flash.green { background: radial-gradient(circle, #2dff8f 0%, #0d7a3f 100%); }
    .reaction-flash-text {
        color: white;
        font-size: 40px;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(0,0,0,0.4);
    }

    /* Memory test sequence display */
    .test-screen {
        border-radius: 20px;
        padding: 60px 20px;
        text-align: center;
        font-size: 34px;
        font-weight: 700;
        letter-spacing: 8px;
        color: white;
        margin: 10px 0 20px 0;
    }

    .badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(163, 230, 53, 0.4);
        color: #4ADE80;
        font-size: 14px;
        margin-bottom: 10px;
    }

    .insight-tip {
        background: rgba(255,255,255,0.04);
        border-left: 4px solid #22C55E;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# SESSION STATE INITIALISATION
# --------------------------------------------------------------------------
def init_state():
    defaults = {
        "stage": "lifestyle",       # lifestyle -> reaction -> memory -> results
        "lifestyle": {},
        "reaction_stage": "idle",   # idle -> countdown -> ready -> done
        "reaction_time_ms": None,
        "reaction_start": None,
        "memory_stage": "idle",     # idle -> showing -> input -> done
        "memory_sequence": [],
        "memory_score": None,
        "prediction": None,
        "trigger_shower": False,  # set True right before a main-stage transition
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# --------------------------------------------------------------------------
# PROGRESS BAR
# --------------------------------------------------------------------------
def render_progress():
    progress_map = {"lifestyle": 0.25, "reaction": 0.50, "memory": 0.75, "results": 1.0}
    labels = {"lifestyle": "Step 1 of 4 — Lifestyle Info",
              "reaction": "Step 2 of 4 — Reaction Test",
              "memory": "Step 3 of 4 — Memory Test",
              "results": "Step 4 of 4 — Results"}
    st.markdown(f"<div class='badge'>{labels[st.session_state.stage]}</div>", unsafe_allow_html=True)
    st.progress(progress_map[st.session_state.stage])


# --------------------------------------------------------------------------
# PAGE 1 — LIFESTYLE INFORMATION
# --------------------------------------------------------------------------
def lifestyle_page(gender_mapping, diet_mapping, exercise_mapping):
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🧬 Tell us about your lifestyle")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", list(gender_mapping.keys()))
        sleep = st.slider("Sleep Duration (hours/night)", 0.0, 12.0, 7.0, 0.5)
        stress = st.slider("Stress Level (1 = low, 10 = high)", 1, 10, 5)
        diet = st.selectbox("Diet Type", list(diet_mapping.keys()))
    with col2:
        screen_time = st.slider("Daily Screen Time (hours)", 0.0, 16.0, 5.0, 0.5)
        exercise = st.selectbox("Exercise Frequency", list(exercise_mapping.keys()))
        caffeine = st.slider("Caffeine Intake (mg/day)", 0, 600, 150, 10)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Continue to Reaction Test →"):
        st.session_state.lifestyle = {
            "Gender": gender_mapping[gender],
            "Sleep_Duration": sleep,
            "Stress_Level": stress,
            "Diet_Type": diet_mapping[diet],
            "Daily_Screen_Time": screen_time,
            "Exercise_Frequency": exercise_mapping[exercise],
            "Caffeine_Intake": caffeine,
        }
        st.session_state.stage = "reaction"
        st.session_state.trigger_shower = True
        st.rerun()


# --------------------------------------------------------------------------
# PAGE 2 — REACTION TIME TEST
# --------------------------------------------------------------------------
def render_gold_dust_shower():
    """One-shot gold dust particle shower, falling from the top of the screen.
    Triggered right after a successful reaction-time click."""
    st.markdown("""
    <style>
    .gold-dust {
        position: fixed;
        top: -20px;
        border-radius: 50%;
        background: radial-gradient(circle, #FFE9A8 0%, #FFD700 45%, #B8860B 100%);
        box-shadow: 0 0 8px 2px rgba(255, 215, 0, 0.75);
        z-index: 500;
        pointer-events: none;
        animation-name: goldFall;
        animation-timing-function: ease-in;
        animation-fill-mode: forwards;
    }
    @keyframes goldFall {
        0%   { transform: translateY(0) rotate(0deg); opacity: 0; }
        8%   { opacity: 1; }
        100% { transform: translateY(115vh) rotate(360deg); opacity: 0; }
    }
    </style>
    """, unsafe_allow_html=True)

    particles_html = "".join(
        f"<div class='gold-dust' style='left:{random.uniform(0, 100):.1f}vw; "
        f"width:{random.uniform(4, 9):.1f}px; height:{random.uniform(4, 9):.1f}px; "
        f"animation-duration:{random.uniform(1.8, 3.2):.2f}s; "
        f"animation-delay:{random.uniform(0, 0.6):.2f}s;'></div>"
        for _ in range(40)
    )
    st.markdown(particles_html, unsafe_allow_html=True)


def reaction_time_test():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("⚡ Reaction Time Test")
    st.write("Click **Start**. The whole screen will flash red, then green — click the button the instant it turns green.")

    if st.session_state.reaction_stage == "idle":
        if st.button("Start Reaction Test"):
            st.session_state.reaction_stage = "countdown"
            st.rerun()

    elif st.session_state.reaction_stage == "countdown":
        st.markdown("""
        <div class='reaction-flash red'>
            <div class='reaction-flash-text'>🔴 Wait for green...</div>
        </div>
        """, unsafe_allow_html=True)
        delay = random.uniform(2, 5)
        time.sleep(delay)
        st.session_state.reaction_start = time.time()
        st.session_state.reaction_stage = "ready"
        st.rerun()

    elif st.session_state.reaction_stage == "ready":
        st.markdown("""
        <div class='reaction-flash green'>
            <div class='reaction-flash-text'>🟢 GO!</div>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("CLICK NOW!", key="reaction_click", use_container_width=True):
                elapsed_ms = (time.time() - st.session_state.reaction_start) * 1000
                st.session_state.reaction_time_ms = round(elapsed_ms, 1)
                st.session_state.reaction_stage = "done"
                st.rerun()

    elif st.session_state.reaction_stage == "done":
        render_gold_dust_shower()
        st.success(f"Your reaction time: **{st.session_state.reaction_time_ms} ms**")
        if st.button("Retake Test"):
            st.session_state.reaction_stage = "idle"
            st.session_state.reaction_time_ms = None
            st.rerun()
        if st.button("Continue to Memory Test →"):
            st.session_state.stage = "memory"
            st.session_state.trigger_shower = True
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# PAGE 3 — MEMORY TEST
# --------------------------------------------------------------------------
def memory_test():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🧩 Memory Test")
    st.write("Memorize the sequence of numbers below — it will disappear after 5 seconds.")

    if st.session_state.memory_stage == "idle":
        if st.button("Start Memory Test"):
            length = 4
            st.session_state.memory_sequence = [random.randint(0, 9) for _ in range(length)]
            st.session_state.memory_stage = "showing"
            st.rerun()

    elif st.session_state.memory_stage == "showing":
        sequence_str = "   ".join(str(d) for d in st.session_state.memory_sequence)
        st.markdown(
            f"<div class='test-screen' style='background:linear-gradient(135deg,#15803D,#22C55E);'>"
            f"{sequence_str}</div>",
            unsafe_allow_html=True,
        )
        time.sleep(5)
        st.session_state.memory_stage = "input"
        st.rerun()

    elif st.session_state.memory_stage == "input":
        st.info("Now type the sequence you saw, digits only (e.g. 4 8 1 9).")
        user_input = st.text_input("Your answer", key="memory_answer")
        if st.button("Submit Answer"):
            guess = [c for c in user_input if c.isdigit()]
            actual = [str(d) for d in st.session_state.memory_sequence]
            total = len(actual)
            correct = sum(1 for i in range(min(len(guess), total)) if guess[i] == actual[i])
            score = round((correct / total) * 100, 1) if total else 0
            st.session_state.memory_score = score
            st.session_state.memory_stage = "done"
            st.rerun()

    elif st.session_state.memory_stage == "done":
        render_gold_dust_shower()
        st.success(f"Your Memory Test Score: **{st.session_state.memory_score} / 100**")
        actual_str = " ".join(str(d) for d in st.session_state.memory_sequence)
        st.caption(f"Correct sequence was: {actual_str}")
        if st.button("Retake Test"):
            st.session_state.memory_stage = "idle"
            st.session_state.memory_score = None
            st.rerun()
        if st.button("See My Results →"):
            st.session_state.stage = "results"
            st.session_state.trigger_shower = True
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# PREDICTION
# --------------------------------------------------------------------------
def run_prediction(model):
    data = dict(st.session_state.lifestyle)
    data["Reaction_Time"] = st.session_state.reaction_time_ms
    data["Memory_Test_Score"] = st.session_state.memory_score

    ordered_values = [[data[feat] for feat in FEATURE_ORDER]]
    prediction = model.predict(np.array(ordered_values))[0]
    return round(float(prediction), 1), data


# --------------------------------------------------------------------------
# VISUAL HELPERS
# --------------------------------------------------------------------------
def gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100", "font": {"color": "#4ADE80", "size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4ADE80"},
            "bar": {"color": "#22C55E"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 60], "color": "rgba(224,64,64,0.35)"},
                {"range": [60, 75], "color": "rgba(230,170,50,0.35)"},
                {"range": [75, 90], "color": "rgba(90,200,120,0.35)"},
                {"range": [90, 100], "color": "rgba(212,175,55,0.45)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E7F5EC"},
        height=320,
        margin=dict(t=30, b=10, l=30, r=30),
    )
    return fig


def interpretation(score):
    if score >= 90:
        return "Excellent", "#22C55E"
    elif score >= 75:
        return "Very Good", "#2ecc71"
    elif score >= 60:
        return "Average", "#e6aa32"
    else:
        return "Needs Improvement", "#e04040"


COGNITIVE_BOOST_TIPS = [
    {"icon": "🧍", "title": "Power Posture",
     "text": "Standing tall in an open, confident posture for a couple of minutes before a demanding task can boost alertness and lower stress."},
    {"icon": "🤲", "title": "Hand & Finger Exercises",
     "text": "Simple finger-tapping or hand-coordination drills stimulate the motor cortex and can sharpen focus in minutes."},
    {"icon": "🥗", "title": "Balanced Diet",
     "text": "Diets rich in omega-3s, leafy greens, and whole grains are linked to better long-term memory and brain health."},
    {"icon": "🧘", "title": "Stress Management",
     "text": "Regular breathing exercises, short walks, or mindfulness breaks help lower cortisol and protect cognitive performance."},
    {"icon": "🛌", "title": "Sleep Hygiene",
     "text": "A consistent sleep schedule and 7–9 hours a night are strongly linked to sharper memory and faster reaction times."},
    {"icon": "🚶", "title": "Stay Active",
     "text": "Even light daily movement, like a 20-minute walk, increases blood flow to the brain and supports focus."},
]


def render_cognitive_boost_tips():
    """User-facing suggestions for improving cognitive performance —
    general wellness habits, not a technical model breakdown."""
    st.markdown("#### 🌱 Ways to Boost Your Cognitive Score")
    cols = st.columns(2)
    for i, tip in enumerate(COGNITIVE_BOOST_TIPS):
        with cols[i % 2]:
            st.markdown(f"""
            <div class='insight-tip' style='margin-bottom:14px;'>
                <div style='font-size:20px;'>{tip['icon']} <strong>{tip['title']}</strong></div>
                <div style='font-size:14px; opacity:0.85; margin-top:4px;'>{tip['text']}</div>
            </div>
            """, unsafe_allow_html=True)
    st.caption("General wellness suggestions — not medical advice.")


def normalize(value, feature):
    lo, hi, invert = NORMALIZE_RANGES[feature]
    pct = (value - lo) / (hi - lo) * 100
    pct = max(0, min(100, pct))
    return 100 - pct if invert else pct


def radar_chart(data):
    radar_features = ["Sleep_Duration", "Stress_Level", "Daily_Screen_Time",
                       "Caffeine_Intake", "Reaction_Time", "Memory_Test_Score"]
    user_scores = [normalize(data[f], f) for f in radar_features]
    ideal_scores = [normalize(IDEAL_VALUES[f], f) for f in radar_features]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=user_scores + [user_scores[0]],
        theta=radar_features + [radar_features[0]],
        fill="toself", name="You",
        line=dict(color="#22C55E"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=ideal_scores + [ideal_scores[0]],
        theta=radar_features + [radar_features[0]],
        fill="toself", name="Ideal",
        line=dict(color="#4ADE80", dash="dot"),
        opacity=0.5,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], color="#E7F5EC"),
            angularaxis=dict(color="#E7F5EC"),
        ),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E7F5EC"},
        height=420,
        margin=dict(t=30, b=30, l=40, r=40),
    )
    return fig


def generate_insights(data):
    tips = []
    if data["Sleep_Duration"] < 6:
        tips.append("😴 Your sleep duration is on the low side — aiming for 7–9 hours may support better focus.")
    if data["Stress_Level"] >= 7:
        tips.append("🧘 Your stress level is high — activities like breathing exercises or short walks may help you unwind.")
    if data["Caffeine_Intake"] > 300:
        tips.append("☕ Your caffeine intake is fairly high — cutting back gradually may improve sleep quality.")
    if data["Daily_Screen_Time"] > 8:
        tips.append("📱 Your screen time is high — regular breaks (e.g. the 20-20-20 rule) may reduce fatigue.")
    if data["Reaction_Time"] > 400:
        tips.append("⚡ Your reaction time was a bit slow today — simple reflex games can help you practice.")
    if data["Memory_Test_Score"] < 50:
        tips.append("🧩 Your memory score was on the lower side — memory games and consistent sleep can help.")
    if not tips:
        tips.append("🌟 Your lifestyle metrics look well balanced — keep up the good habits!")
    return tips


# --------------------------------------------------------------------------
# PAGE 4 — RESULTS DASHBOARD
# --------------------------------------------------------------------------
def results_dashboard(model):
    if st.session_state.prediction is None:
        score, data = run_prediction(model)
        st.session_state.prediction = score
        st.session_state.full_data = data
    else:
        score = st.session_state.prediction
        data = st.session_state.full_data

    label, color = interpretation(score)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🧠 Your Cognitive Score")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(gauge_chart(score), use_container_width=True)
    with col2:
        st.markdown(f"<h1 style='color:{color};'>{label}</h1>", unsafe_allow_html=True)
        st.write(f"Reaction Time: **{data['Reaction_Time']} ms**")
        st.write(f"Memory Test Score: **{data['Memory_Test_Score']} / 100**")
        st.caption("This score is generated by a machine learning model for informational and wellness purposes only. It is not a medical or clinical assessment.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("💡 Personalized Insights")
    for tip in generate_insights(data):
        st.markdown(f"<div class='insight-tip'>{tip}</div>", unsafe_allow_html=True)
    st.caption("These are general wellness tips, not medical advice.")
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        render_cognitive_boost_tips()
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.plotly_chart(radar_chart(data), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔄 Start New Assessment"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    apply_custom_css()
    init_state()

    st.markdown("<h1 style='text-align:center;'>🧠 NeuroScope</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; opacity:0.8;'>An AI-powered wellness dashboard that measures your reaction time and memory live.</p>", unsafe_allow_html=True)

    render_progress()

    if st.session_state.trigger_shower:
        render_gold_dust_shower()
        st.session_state.trigger_shower = False

    with st.spinner("Loading assets..."):
        model, gender_mapping, diet_mapping, exercise_mapping = load_assets()

    if st.session_state.stage == "lifestyle":
        lifestyle_page(gender_mapping, diet_mapping, exercise_mapping)
    elif st.session_state.stage == "reaction":
        reaction_time_test()
    elif st.session_state.stage == "memory":
        memory_test()
    elif st.session_state.stage == "results":
        results_dashboard(model)


if __name__ == "__main__":
    main()

# streamlit run C:\Users\afeefa\PyCharmMiscProject\neroscope\app_2.py