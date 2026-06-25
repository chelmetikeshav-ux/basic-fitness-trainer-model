import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import sqlite3
import hashlib

# ==========================================
# DATABASE HELPER FUNCTIONS
# ==========================================
def init_db():
    conn = sqlite3.connect('fitness_tracker.db')
    c = conn.cursor()
    # Create Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT)''')
    # Create Workout Progress Table
    c.execute('''CREATE TABLE IF NOT EXISTS progress 
                 (username TEXT, date TEXT, exercise TEXT, reps INTEGER)''')
    conn.commit()
    conn.close()

def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def create_user(username, password):
    conn = sqlite3.connect('fitness_tracker.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users(username, password) VALUES (?,?)', (username, make_hash(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect('fitness_tracker.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username =? AND password =?', (username, make_hash(password)))
    data = c.fetchall()
    conn.close()
    return data

def save_workout(username, exercise, reps):
    if reps > 0:
        conn = sqlite3.connect('fitness_tracker.db')
        c = conn.cursor()
        date_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
        c.execute('INSERT INTO progress(username, date, exercise, reps) VALUES (?,?,?,?)', 
                  (username, date_str, exercise, reps))
        conn.commit()
        conn.close()

def get_user_progress(username):
    conn = sqlite3.connect('fitness_tracker.db')
    df = pd.read_sql_query('SELECT date, exercise, reps FROM progress WHERE username=?', conn, params=(username,))
    conn.close()
    return df

# Initialize Database on startup
init_db()

# --- Core Angle Calculation Function ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="AI Personal Trainer Pro", layout="wide")
st.title("🚀 AI Personal Fitness Ecosystem")

# --- Custom CSS Injection to Hide the Password Eye Toggle Icon ---
st.markdown("""
    <style>
    button[aria-label="Show password"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# AUTHENTICATION SIDEBAR SYSTEM
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

st.sidebar.title("🔐 Account Access")

if not st.session_state.logged_in:
    auth_mode = st.sidebar.radio("Choose Action", ["Login", "Register"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password") # Hides text behind bullet characters
    
    if auth_mode == "Register":
        if st.sidebar.button("Create Account"):
            if username and password:
                if create_user(username, password):
                    st.sidebar.success("Account created successfully! Switch to Login.")
                else:
                    st.sidebar.error("Username already exists.")
            else:
                st.sidebar.warning("Please fill in all fields.")
                
    elif auth_mode == "Login":
        if st.sidebar.button("Sign In"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.sidebar.error("Invalid Username or Password.")

else:
    st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# ==========================================
# APPLICATION BODY (GATED BY LOGIN STATE)
# ==========================================
if not st.session_state.logged_in:
    st.info("👋 Welcome! Please Register or Login using the sidebar to start tracking your dynamic workout progress.")
else:
    # Setup Navigation Tabs
    tab1, tab2, tab3 = st.tabs(["🏋️ Live AI Trainer", "📅 Workout Planner", "📊 Analytics & Directory"])

    # ==========================================
    # TAB 1: LIVE AI TRACKER (YOLO ENGINE)
    # ==========================================
    with tab1:
        st.subheader("Real-Time Form Analysis & Tracking")
        exercise_mode = st.selectbox("Choose Exercise Target", ["Bicep Curls", "Squats"])

        if 'counter' not in st.session_state: st.session_state.counter = 0
        if 'stage' not in st.session_state: st.session_state.stage = None
        if 'prev_mode' not in st.session_state: st.session_state.prev_mode = exercise_mode

        if st.session_state.prev_mode != exercise_mode:
            # Auto-save progress of previous exercise before resetting if they got reps
            if st.session_state.counter > 0:
                save_workout(st.session_state.username, st.session_state.prev_mode, st.session_state.counter)
                st.toast(f"Saved {st.session_state.counter} reps of {st.session_state.prev_mode} to your profile!")
            st.session_state.counter = 0
            st.session_state.stage = None
            st.session_state.prev_mode = exercise_mode

        col1, col2, col3 = st.columns(3)
        with col1: st.metric(label="Selected Workout", value=exercise_mode)
        with col2: rep_stat = st.empty(); rep_stat.metric(label="Rep Count", value=st.session_state.counter)
        with col3: stage_stat = st.empty(); stage_stat.metric(label="Current Stage", value=str(st.session_state.stage).upper() if st.session_state.stage else "-")

        feedback_text = st.empty()
        frame_window = st.image([])

        @st.cache_resource
        def load_model(): return YOLO('yolov8n-pose.pt')
        model = load_model()

        # Safe toggle switch to run/kill webcam loop dynamically inside Streamlit
        run_camera = st.checkbox("Toggle Webcam Engine", value=False)
        
        # Explicit button to manually log workout set summary
        if st.button("💾 Save Current Reps & Finish Set"):
            if st.session_state.counter > 0:
                save_workout(st.session_state.username, exercise_mode, st.session_state.counter)
                st.success(f"Successfully stored {st.session_state.counter} reps of {exercise_mode}!")
                st.session_state.counter = 0
                rep_stat.metric(label="Rep Count", value=0)
            else:
                st.warning("No complete reps detected to save yet.")

        if run_camera:
            cap = cv2.VideoCapture(0)
            while cap.isOpened() and run_camera:
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.flip(frame, 1) # Mirror image output
                results = model(frame, verbose=False)
                annotated_frame = results[0].plot()
                image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                feedback = "Keep standard form within camera framework"

                try:
                    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                        keypoints = results[0].keypoints.xy[0].cpu().numpy()
                        
                        # --- CONDITION A: BICEP CURL LOGIC ---
                        if exercise_mode == "Bicep Curls":
                            l_shoulder, l_elbow, l_wrist = keypoints[5], keypoints[7], keypoints[9]
                            r_shoulder, r_elbow, r_wrist = keypoints[6], keypoints[8], keypoints[10]
                            
                            if all(v[0] > 0 for v in [l_shoulder, l_elbow, l_wrist]): angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                            elif all(v[0] > 0 for v in [r_shoulder, r_elbow, r_wrist]): angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                            else: angle = None

                            if angle is not None:
                                if angle > 150: st.session_state.stage = "down"; feedback = "Squeeze up!"
                                if angle < 40 and st.session_state.stage == 'down':
                                    st.session_state.stage = "up"; st.session_state.counter += 1
                                    feedback = "Great extension! Lower slowly."

                        # --- CONDITION B: SQUAT LOGIC ---
                        elif exercise_mode == "Squats":
                            l_hip, l_knee, l_ankle = keypoints[11], keypoints[13], keypoints[15]
                            r_hip, r_knee, r_ankle = keypoints[12], keypoints[14], keypoints[16]
                            
                            if all(v[0] > 0 for v in [l_hip, l_knee, l_ankle]): angle = calculate_angle(l_hip, l_knee, l_ankle)
                            elif all(v[0] > 0 for v in [r_hip, r_knee, r_ankle]): angle = calculate_angle(r_hip, r_knee, r_ankle)
                            else: angle = None
                            
                            if angle is not None:
                                if angle > 155: st.session_state.stage = "up"; feedback = "Drop down lower into the squat!"
                                if angle < 105 and st.session_state.stage == 'up':
                                    st.session_state.stage = "down"; st.session_state.counter += 1
                                    feedback = "Perfect depth achieved! Stand tall."
                                elif angle < 135 and angle > 105 and st.session_state.stage == 'up':
                                    feedback = "Almost parallel. Go lower!"

                        rep_stat.metric(label="Rep Count", value=st.session_state.counter)
                        stage_stat.metric(label="Current Stage", value=str(st.session_state.stage).upper())
                        feedback_text.success(f"Coach: {feedback}")

                except Exception as e: pass
                frame_window.image(image)
            cap.release()

    # ==========================================
    # TAB 2: WORKOUT PLAN GENERATOR (DYNAMIC GENERATION FIX)
    # ==========================================
    with tab2:
        st.subheader("📅 Dynamic Workout Plan Generator")
        
        with st.form("plan_form"):
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                goal = st.selectbox("Fitness Goal", ["Muscle Building (Hypertrophy)", "Fat Loss", "Strength & Power", "General Endurance"])
                experience = st.select_slider("Experience Level", options=["Beginner", "Intermediate", "Advanced"])
            with col_w2:
                days = st.slider("Training Days Per Week", 1, 7, 5)
                equipment = st.multiselect("Available Equipment", ["Bodyweight", "Dumbbells", "Barbell", "Resistance Bands", "Full Gym"], default=["Bodyweight"])
                
            submit_plan = st.form_submit_button("Generate Blueprint Routine")
            
        if submit_plan:
            st.success(f"Configured custom routine targeting: **{goal}** ({days} Days/Week)")
            st.markdown("### Your Custom Split Program")
            
            # Templates for workout segments
            upper_body = """* **Day Focus: Upper Body Power**\n  * Bicep Curls (3 Sets x 12 Reps) — *Track using AI Tab!*\n  * Push-Ups / Floor Press (3 Sets x 10 Reps)\n  * Dumbbell Rows / Shoulder Taps (4 Sets x 12 Reps)"""
            lower_body = """* **Day Focus: Lower Body & Legs**\n  * Air Squats / Goblet Squats (4 Sets x 15 Reps) — *Track using AI Tab!*\n  * Lunges (3 Sets x 12 Reps per leg)\n  * Calf Raises (3 Sets x 20 Reps)"""
            core_rest = """* **Day Focus: Rest & Core Conditioning**\n  * Planks (3 Sets x 45 Seconds)\n  * Bicycle Crunches (3 Sets x 20 Reps)"""
            cardio_hiit = """* **Day Focus: Cardio & Endurance HIIT**\n  * Jumping Jacks (4 Sets x 40 Seconds)\n  * Mountain Climbers (3 Sets x 30 Seconds)\n  * High Knees (3 Sets x 30 Seconds)"""

            # Handle structural generation for days 1 through 7 dynamically
            if days == 1:
                st.markdown(f"#### Day 1\n{upper_body}\n{lower_body}")
            elif days == 2:
                st.markdown(f"#### Day 1\n{upper_body}\n\n#### Day 2\n{lower_body}")
            elif days == 3:
                st.markdown(f"#### Day 1\n{upper_body}\n\n#### Day 2\n{core_rest}\n\n#### Day 3\n{lower_body}")
            elif days == 4:
                st.markdown(f"#### Day 1\n{upper_body}\n\n#### Day 2\n{lower_body}\n\n#### Day 3\n{core_rest}\n\n#### Day 4\n{cardio_hiit}")
            elif days >= 5:
                st.markdown(f"#### Day 1\n{upper_body}\n\n#### Day 2\n{lower_body}\n\n#### Day 3\n{core_rest}\n\n#### Day 4\n{cardio_hiit}\n\n#### Day 5\n{upper_body if 'Muscle' in goal else cardio_hiit}")
                if days >= 6:
                    st.markdown(f"#### Day 6\n{lower_body}")
                if days == 7:
                    st.markdown(f"#### Day 7 Active Recovery\n* Light Stretching & Mobility Work (20 Mins)")

    # ==========================================
    # TAB 3: ANALYTICS (REAL SQL PROGRESS DATA)
    # ==========================================
    with tab3:
        st.subheader(f"📊 Real Progress Analytics for {st.session_state.username}")
        
        # Pull progress specific to this logged in user profile
        history_df = get_user_progress(st.session_state.username)
        
        if not history_df.empty:
            fig = px.line(
                history_df, 
                x='date', 
                y='reps', 
                color='exercise',
                title='Your Performance Improvement Curve',
                labels={'reps': 'Total Reps Tracked', 'date': 'Timestamp'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)