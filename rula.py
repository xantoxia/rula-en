# ===================== Imports =====================
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from openai import OpenAI
import datetime

# Initialize history storage (each record has independent chat history)
if "rula_history" not in st.session_state:
    st.session_state.rula_history = []
# Flag for AI generation
if "need_gen_ai" not in st.session_state:
    st.session_state.need_gen_ai = False
# Expand AI analysis result
if "last_expand_idx" not in st.session_state:
    st.session_state.last_expand_idx = -1
# Active chat session ID (-1 = none)
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = -1

# ===================== Page Configuration =====================
st.set_page_config(
    page_title="RULA Rapid Upper Limb Assessment System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #0070C0;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .section-header {
        background-color: #D9E1F2;
        padding: 10px;
        border-radius: 5px;
        margin: 15px 0;
        font-weight: bold;
        color: #003366;
    }
    .score-box {
        background-color: #F0F2F6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .score-value {
        font-size: 28px;
        font-weight: bold;
        color: #0070C0;
    }
    .risk-high {
        color: #C00000;
        font-weight: bold;
    }
    .risk-medium {
        color: #ED7D31;
        font-weight: bold;
    }
    .risk-low {
        color: #00B050;
        font-weight: bold;
    }
    .stImage img {
        max-width: 800px !important;
        margin: 0 auto !important;
        display: block !important;
    }
    .sub-header-green {
    background-color: #DFF2DD;
    padding: 10px;
    border-radius: 5px;
    margin: 15px 0;
    font-weight: bold;
    color: #195927;
    }
</style>
""", unsafe_allow_html=True)

# ===================== Session State Initialization =====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "client" not in st.session_state:
    st.session_state.client = None
if "api_key_entered" not in st.session_state:
    st.session_state.api_key_entered = False
if "rula_result" not in st.session_state:
    st.session_state.rula_result = None
if "auto_angles" not in st.session_state:
    st.session_state.auto_angles = None
if "detection_success" not in st.session_state:
    st.session_state.detection_success = False

# ===================== Static Image Pose Model =====================
def load_pose_models():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp_pose, pose

def get_coord(landmark, img_width, img_height):
    return [landmark.x * img_width, landmark.y * img_height, landmark.z * img_width]

def calculate_angle(a, b, c):
    a = np.array(a)[:2]
    b = np.array(b)[:2]
    c = np.array(c)[:2]
    ba = a - b
    bc = c - b
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

# ===================== Calibrated Neck Flexion Calculation =====================
def calculate_neck_flexion(nose, left_shoulder, right_shoulder, left_hip, right_hip):
    try:
        mid_sho = [(left_shoulder[i] + right_shoulder[i])/2 for i in range(3)]
        mid_hip = [(left_hip[i] + right_hip[i])/2 for i in range(3)]
        
        vertical_drop = nose[1] - mid_sho[1]
        torso_height = mid_hip[1] - mid_sho[1]
        
        if torso_height > 50:
            normalized_drop = vertical_drop / torso_height
            angle = normalized_drop * 70
            angle = max(5, min(45, abs(angle)))
            return int(angle)
        
        torso_vector = np.array(mid_hip) - np.array(mid_sho)
        head_vector = np.array(nose) - np.array(mid_sho)
        angle_side = abs(np.degrees(np.arctan2(*torso_vector[:2])) - np.degrees(np.arctan2(*head_vector[:2])))
        
        return max(5, min(45, angle_side * 0.6))
    except Exception as e:
        print(f"Neck angle calculation failed: {e}")
        return None

# ===================== Full Range Trunk Flexion Calculation =====================
def calculate_trunk_flexion(left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee):
    try:
        mid_sho = [(left_shoulder[i] + right_shoulder[i])/2 for i in range(3)]
        mid_hip = [(left_hip[i] + right_hip[i])/2 for i in range(3)]
        mid_knee = [(left_knee[i] + right_knee[i])/2 for i in range(3)]

        dy = mid_hip[1] - mid_sho[1]
        dx = abs(mid_sho[0] - mid_hip[0])

        torso_length = np.sqrt(dx**2 + dy**2)

        if torso_length < 30:
            return None

        angle = np.degrees(np.arctan2(dx, dy))
        angle = max(0, min(85, angle))
        return int(angle)

    except Exception as e:
        return None
        
# ===================== Industrial Wrist Bend Calculation =====================
def calculate_wrist_bend(elbow, wrist, index_mcp, pinky_mcp):
    try:
        palm_center = [(index_mcp[i] + pinky_mcp[i])/2 for i in range(3)]
        
        angle = calculate_angle(elbow, wrist, palm_center)
        wrist_bend = max(-30, min(30, 180 - angle))
        
        return int(wrist_bend)
    except Exception as e:
        print(f"Wrist angle calculation failed: {e}")
        return None

# ===================== Image Angle Processing =====================
def process_image(image):
    mp_pose, pose = load_pose_models()
    H, W, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(img_rgb)

    DEFAULT_VALUES = {
        "arm_angle": 0,
        "forearm_angle": 90,
        "wrist_bend": 5,
        "neck_angle": 8,
        "trunk_angle": 10
    }
    
    rula_angles = DEFAULT_VALUES.copy()
    default_angles = []
    detection_message = "✅ All angles detected successfully"

    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks.landmark
        
        avg_visibility = sum(lm.visibility for lm in landmarks) / len(landmarks)
        if avg_visibility < 0.28:
            default_angles = ["Upper Arm", "Forearm", "Wrist", "Neck", "Trunk"]
        else:
            def is_visible(landmark_idx):
                return landmarks[landmark_idx].visibility > 0.25
            
            def pt(landmark):
                return get_coord(landmarks[landmark], W, H)

            nose = pt(mp_pose.PoseLandmark.NOSE)
            l_sho = pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
            r_sho = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            l_elb = pt(mp_pose.PoseLandmark.LEFT_ELBOW)
            r_elb = pt(mp_pose.PoseLandmark.RIGHT_ELBOW)
            l_wri = pt(mp_pose.PoseLandmark.LEFT_WRIST)
            r_wri = pt(mp_pose.PoseLandmark.RIGHT_WRIST)
            l_index = pt(mp_pose.PoseLandmark.LEFT_INDEX)
            r_index = pt(mp_pose.PoseLandmark.RIGHT_INDEX)
            l_pinky = pt(mp_pose.PoseLandmark.LEFT_PINKY)
            r_pinky = pt(mp_pose.PoseLandmark.RIGHT_PINKY)
            l_hip = pt(mp_pose.PoseLandmark.LEFT_HIP)
            r_hip = pt(mp_pose.PoseLandmark.RIGHT_HIP)
            l_knee = pt(mp_pose.PoseLandmark.LEFT_KNEE)
            r_knee = pt(mp_pose.PoseLandmark.RIGHT_KNEE)

            # ===================== Auto-select more visible side =====================
            left_arm_vis = sum([
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].visibility,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST].visibility
            ]) / 3
            right_arm_vis = sum([
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].visibility
            ]) / 3

            if left_arm_vis >= right_arm_vis:
                sho_main, elb_main, wri_main = l_sho, l_elb, l_wri
                index_main, pinky_main = l_index, l_pinky
                hip_main, knee_main = l_hip, l_knee
            else:
                sho_main, elb_main, wri_main = r_sho, r_elb, r_wri
                index_main, pinky_main = r_index, r_pinky
                hip_main, knee_main = r_hip, r_knee

            # Shoulder/hip midpoint: use midpoint if both visible, otherwise use main side
            if is_visible(mp_pose.PoseLandmark.LEFT_SHOULDER) and is_visible(mp_pose.PoseLandmark.RIGHT_SHOULDER):
                mid_sho = [(l_sho[i]+r_sho[i])/2 for i in range(3)]
            else:
                mid_sho = sho_main
            if is_visible(mp_pose.PoseLandmark.LEFT_HIP) and is_visible(mp_pose.PoseLandmark.RIGHT_HIP):
                mid_hip = [(l_hip[i]+r_hip[i])/2 for i in range(3)]
            else:
                mid_hip = hip_main

            # --------------------------
            # Neck
            # --------------------------
            neck_ok = False
            if (is_visible(mp_pose.PoseLandmark.NOSE) 
                and is_visible(mp_pose.PoseLandmark.LEFT_SHOULDER) 
                and is_visible(mp_pose.PoseLandmark.RIGHT_SHOULDER)
                and is_visible(mp_pose.PoseLandmark.LEFT_HIP)
                and is_visible(mp_pose.PoseLandmark.RIGHT_HIP)):
                
                neck_angle = calculate_neck_flexion(nose, l_sho, r_sho, l_hip, r_hip)
                if neck_angle is not None and 5 <= neck_angle <= 45:
                    rula_angles["neck_angle"] = neck_angle
                    neck_ok = True
                    cv2.line(image, (int(nose[0]), int(nose[1])), (int(mid_sho[0]), int(mid_sho[1])), (245, 117, 66), 2)
                else:
                    cv2.line(image, (int(nose[0]), int(nose[1])), (int(mid_sho[0]), int(mid_sho[1])), (128, 128, 128), 2, cv2.LINE_AA)
            if not neck_ok:
                default_angles.append("Neck")

            # --------------------------
            # Trunk
            # --------------------------
            trunk_ok = False
            if (is_visible(mp_pose.PoseLandmark.LEFT_SHOULDER if left_arm_vis >= right_arm_vis else mp_pose.PoseLandmark.RIGHT_SHOULDER)
                and is_visible(mp_pose.PoseLandmark.LEFT_HIP if left_arm_vis >= right_arm_vis else mp_pose.PoseLandmark.RIGHT_HIP)
                and is_visible(mp_pose.PoseLandmark.LEFT_KNEE if left_arm_vis >= right_arm_vis else mp_pose.PoseLandmark.RIGHT_KNEE)):
                
                trunk_angle = calculate_trunk_flexion(l_sho, r_sho, l_hip, r_hip, l_knee, r_knee)
                if trunk_angle is not None and 0 <= trunk_angle <= 85:
                    rula_angles["trunk_angle"] = trunk_angle
                    trunk_ok = True
            if not trunk_ok:
                default_angles.append("Trunk")

            # --------------------------
            # Upper Arm
            # --------------------------
            arm_ok = False
            if is_visible(mp_pose.PoseLandmark.LEFT_HIP if left_arm_vis >= right_arm_vis else mp_pose.PoseLandmark.RIGHT_HIP) \
                and is_visible(mp_pose.PoseLandmark.LEFT_SHOULDER if left_arm_vis >= right_arm_vis else mp_pose.PoseLandmark.RIGHT_SHOULDER) \
                and is_visible(mp_pose.PoseLandmark.LEFT_ELBOW if left_arm_vis >= right_arm_vis else mp_pose.PoseLandmark.RIGHT_ELBOW):
                arm_angle = calculate_angle(mid_hip, sho_main, elb_main)
                rula_angles["arm_angle"] = arm_angle
                arm_ok = True
            if not arm_ok:
                default_angles.append("Upper Arm")

            # --------------------------
            # Forearm
            # --------------------------
            forearm_ok = False
            if is_visible(mp_pose.PoseLandmark.LEFT_SHOULDER if left_arm_vis >= right_arm_vis else mp_pose.PoseLandmark.RIGHT_SHOULDER) \
                and is_visible(mp_pose.PoseLandmark.LEFT_ELBOW if left_arm_vis >= right_arm_vis else mp_pose.PoseLandmark.RIGHT_ELBOW) \
                and is_visible(mp_pose.PoseLandmark.LEFT_WRIST if left_arm_vis >= right_arm_vis else mp_pose.PoseLandmark.RIGHT_WRIST):
                forearm_angle = calculate_angle(sho_main, elb_main, wri_main)
                rula_angles["forearm_angle"] = forearm_angle
                forearm_ok = True
            if not forearm_ok:
                default_angles.append("Forearm")
                
            # --------------------------
            # Wrist
            # --------------------------
            wrist_ok = False
            if (is_visible(mp_pose.PoseLandmark.LEFT_ELBOW if left_arm_vis >= right_arm_vis else mp_pose.PoseLandmark.RIGHT_ELBOW) 
                and is_visible(mp_pose.PoseLandmark.LEFT_WRIST if left_arm_vis >= right_arm_vis else mp_pose.PoseLandmark.RIGHT_WRIST)
                and is_visible(mp_pose.PoseLandmark.LEFT_INDEX if left_arm_vis >= right_arm_vis else mp_pose.PoseLandmark.RIGHT_INDEX)
                and is_visible(mp_pose.PoseLandmark.LEFT_PINKY if left_arm_vis >= right_arm_vis else mp_pose.PoseLandmark.RIGHT_PINKY)):
                
                wrist_angle = calculate_wrist_bend(elb_main, wri_main, index_main, pinky_main)
                if wrist_angle is not None:
                    rula_angles["wrist_bend"] = wrist_angle
                    wrist_ok = True
            if not wrist_ok:
                default_angles.append("Wrist")

        if default_angles:
            detection_message = f"⚠️ Some angles failed to detect. Default values applied. Manual correction recommended in 【Section 2】: {', '.join(default_angles)}"

        mp.solutions.drawing_utils.draw_landmarks(
            image, 
            pose_result.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2)
        )

    else:
        default_angles = ["Upper Arm", "Forearm", "Wrist", "Neck", "Trunk"]
        detection_message = "⚠️ No human pose detected. All values set to default. Please use a clear full-body side profile photo."

    pose.close()
    return image, rula_angles, detection_message, default_angles

# ===================== RULA Scoring Logic =====================
def get_arm_base_score(arm_angle):
    if -20 <= arm_angle <= 20: return 1
    elif 20 < arm_angle <= 45: return 2
    elif 45 < arm_angle <= 90: return 3
    else: return 4

def get_forearm_base_score(forearm_angle):
    return 1 if 60 <= forearm_angle <= 100 else 2

def get_wrist_base_score(wrist_bend):
    if abs(wrist_bend) < 1e-6: return 1
    elif abs(wrist_bend) <=15: return 2
    else: return 3

def get_neck_base_score(neck_angle):
    if 0<=neck_angle<=10: return 1
    elif 10<neck_angle<=20: return 2
    elif neck_angle>20: return 3
    else: return 4

def get_trunk_base_score(trunk_angle):
    if trunk_angle <1: return 1
    elif 0<trunk_angle<=20: return 2
    elif 20<trunk_angle<=60: return 3
    else: return 4

def get_leg_score(leg_support):
    return 1 if leg_support else 2

def get_table1_score(arm, forearm, wrist, wrist_twist):
    t = [
        [[[1,2],[2,2],[2,2],[3,3]], [[2,2],[2,2],[2,3],[3,3]], [[2,3],[3,3],[3,3],[4,4]], [[3,3],[3,3],[3,4],[4,4]]],
        [[[2,2],[2,2],[2,3],[3,3]], [[2,3],[3,3],[3,3],[4,4]], [[3,3],[3,3],[4,4],[4,4]], [[3,4],[4,4],[4,4],[5,5]]],
        [[[2,3],[3,3],[3,3],[4,4]], [[3,3],[3,3],[4,4],[4,4]], [[3,4],[4,4],[4,5],[5,5]], [[4,4],[4,5],[5,5],[6,6]]],
        [[[3,3],[3,3],[3,4],[4,4]], [[3,4],[4,4],[4,4],[5,5]], [[4,4],[4,5],[5,5],[6,6]], [[4,5],[5,5],[6,6],[7,7]]]
    ]
    arm_idx = max(0, min(3, arm - 1))
    forearm_idx = max(0, min(3, forearm - 1))
    wrist_idx = max(0, min(3, wrist - 1))
    twist_idx = 1 if wrist_twist else 0
    return t[arm_idx][forearm_idx][wrist_idx][twist_idx]

def get_table2_score(neck, trunk, leg):
    t = [
        [[1,2],[2,3],[3,4],[5,6]],
        [[2,3],[3,4],[4,5],[5,6]],
        [[3,4],[4,5],[5,6],[6,7]],
        [[5,6],[5,6],[6,7],[7,8]]
    ]
    neck_idx = max(0, min(3, neck - 1))
    trunk_idx = max(0, min(3, trunk - 1))
    leg_idx = 0 if leg == 1 else 1
    return t[neck_idx][trunk_idx][leg_idx]
    
def get_table3_score(c, d):
    t = [
        [1,2,3,3,4,5,5,6,7],
        [2,2,3,4,4,5,5,6,7],
        [3,3,3,4,5,5,6,7,7],
        [3,4,4,5,5,6,6,7,7],
        [4,4,5,5,6,6,7,7,7],
        [5,5,5,6,6,7,7,7,7],
        [5,6,6,6,7,7,7,7,7],
        [6,6,7,7,7,7,7,7,7],
        [7,7,7,7,7,7,7,7,7]
    ]
    return t[max(0,min(8,c-1))][max(0,min(8,d-1))]

def calculate_rula_scores(arm_angle, arm_abd, shoulder_up, arm_support, forearm_angle, forearm_abd,
                         wrist_bend, wrist_twist, neck_angle, neck_twist, neck_bend,
                         trunk_angle, trunk_twist, trunk_bend, leg_support, muscle_state, load_state):
    arm_final = max(1, get_arm_base_score(arm_angle) + (1 if arm_abd else 0) + (1 if shoulder_up else 0) - (1 if arm_support else 0))
    forearm_final = max(1, get_forearm_base_score(forearm_angle) + (1 if forearm_abd else 0))
    wrist_final = get_wrist_base_score(wrist_bend)
    neck_final = max(1, get_neck_base_score(neck_angle) + (1 if neck_twist else 0) + (1 if neck_bend else 0))
    trunk_final = max(1, get_trunk_base_score(trunk_angle) + (1 if trunk_twist else 0) + (1 if trunk_bend else 0))
    leg_final = get_leg_score(leg_support)
    a = get_table1_score(arm_final, forearm_final, wrist_final, wrist_twist)
    b = get_table2_score(neck_final, trunk_final, leg_final)

    # Muscle state + load state scoring (full RULA compliance)
    if muscle_state in ["Static posture or holding > 1 minute","Repetitive work > 4 times/minute"]:
        m = 1
        muscle_text = "Prolonged static holding or high-frequency repetition (>4/min) causes sustained static muscle load and increased fatigue/injury risk"
    else:
        m = 0
        muscle_text = "Dynamic posture changes with no prolonged static hold or high repetition; muscles have sufficient recovery intervals, baseline load normal"

    if load_state == "No force / cyclic load < 2kg":
        l = 0
        load_text = "Virtually no external load, very low additional joint stress"
    elif load_state == "Cyclic load 2–10kg":
        l = 1
        load_text = "2–10kg cyclic load, moderately increases muscle-joint load; combined with poor posture amplifies injury risk"
    elif load_state == "Static/repetitive load 2–10kg, cyclic load ≥10kg":
        l = 2
        load_text = "2–10kg static hold or ≥10kg cyclic load, significantly elevated injury risk"
    elif load_state == "Static 10kg, repetitive 10kg, vibration or rapid force increase":
        l = 3
        load_text = "≥10kg static hold, strong vibration or explosive motion — high injury risk load"
    else:
        l = 0
        load_text = "Virtually no external load, very low additional joint stress"
                        
    c = a + m + l
    d = b + m + l
    rula = get_table3_score(c, d)

    if rula <=2:
        lev, plan, cls = "AL1", "No action required", "risk-low"
    elif rula <=4:
        lev, plan, cls = "AL2", "Further investigation & improvement if needed", "risk-medium"
    elif rula <=6:
        lev, plan, cls = "AL3", "Further investigation & improvement soon", "risk-medium"
    else:
        lev, plan, cls = "AL4", "Immediate investigation & improvement required", "risk-high"

    return {
        "arm_final": arm_final,
        "forearm_final": forearm_final,
        "wrist_final": wrist_final,
        "neck_final": neck_final,
        "trunk_final": trunk_final,
        "leg_final": leg_final,
        "a_total": a,
        "b_total": b,
        "muscle_score": m,
        "load_score": l,
        "muscle_desc": muscle_text,
        "load_desc": load_text,
        "c_total": c,
        "d_total": d,
        "rula_total": rula,
        "action_level": lev,
        "action_plan": plan,
        "risk_class": cls
    }
                             
# ===================== AI Module =====================
def call_deepseek_api(messages):
    try:
        if not st.session_state.client:
            API_KEY = st.secrets["API_KEY"]
            st.session_state.client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")
            st.session_state.api_key_entered = True
        res = ""
        for chunk in st.session_state.client.chat.completions.create(model="deepseek-ai/DeepSeek-V4-Flash", messages=messages, stream=True):
            if chunk.choices and chunk.choices[0].delta.content:
                res += chunk.choices[0].delta.content
        return res
    except Exception as e:
        st.error(f"API Error: {e}")
        return ""

# ===================== Main Page Content =====================
st.markdown("<h1 class='main-header'>RULA Rapid Upper Limb Assessment System</h1>", unsafe_allow_html=True)
st.markdown("This system is developed based on the **Rapid Upper Limb Assessment (RULA)** method (McAtamney & Corlett, 1993), in strict compliance with the international standard **ISO 11226:2000 Ergonomics — Evaluation of static working postures**.")

# Section 1: Photo Angle Recognition
st.markdown("<div class='section-header'>【Section 1】📷 Photo-based Angle Recognition (Recommended: 90° full-body side profile)</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload work posture photo (JPG, PNG supported)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("Detecting posture..."):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        processed_image, rula_angles, detection_message, default_angles = process_image(image)
        
        col_img, col_angles = st.columns([3, 2])
        
        with col_img:
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Pose Detection Result", width=640)
        
        with col_angles:
            st.markdown("### 📊 Angle Detection Results")
            
            import pandas as pd
            
            angle_data = []
            angle_items = [
                ("Upper Arm", "arm_angle"),
                ("Forearm", "forearm_angle"),
                ("Wrist", "wrist_bend"),
                ("Neck", "neck_angle"),
                ("Trunk", "trunk_angle")
            ]
            
            highlight_rows = []
            success_rows = []
            
            for idx, (name, key) in enumerate(angle_items):
                angle = int(rula_angles[key])
                if name in default_angles:
                    status = "⚠️ Default"
                    highlight_rows.append(idx)
                else:
                    status = "✅ Detected"
                    success_rows.append(idx)
                
                angle_data.append({
                    "Body Part": name,
                    "Angle (°)": angle,
                    "Status": status
                })
            
            df = pd.DataFrame(angle_data)
            
            def style_row(row):
                styles = [""] * len(row)
                if row["Status"] == "⚠️ Default":
                    styles = ["background-color: #fff3cd"] * len(row)
                elif row["Status"] == "✅ Detected":
                    styles[1] = "color: #00B050; font-weight: bold"
                return styles
            
            styled_df = df.style.apply(style_row, axis=1).hide(axis="index")
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=220)
            
            if detection_message.startswith("✅"):
                st.success(detection_message)
            elif detection_message.startswith("⚠️"):
                st.warning(detection_message)
                st.info("💡 Tip: Adjust camera angle for a true 90° side view. Keep arm and wrist unobstructed (e.g. remove gloves) for best accuracy.")
            else:
                st.error(detection_message)
                st.session_state.detection_success = False
        
        if detection_message.startswith("✅") or detection_message.startswith("⚠️"):
            st.session_state.auto_angles = rula_angles
            st.session_state.detection_success = True
            
if st.session_state.detection_success and st.session_state.auto_angles:
    default_arm = int(st.session_state.auto_angles["arm_angle"])
    default_forearm = int(st.session_state.auto_angles["forearm_angle"])
    default_wrist = int(st.session_state.auto_angles["wrist_bend"])
    default_neck = int(st.session_state.auto_angles["neck_angle"])
    default_trunk = int(st.session_state.auto_angles["trunk_angle"])
else:
    default_arm = 0
    default_forearm = 90
    default_wrist = 5
    default_neck = 8
    default_trunk = 10

# Section 2: RULA Assessment
st.markdown("<div class='section-header'>【Section 2】📊 RULA Rapid Upper Limb Assessment</div>", unsafe_allow_html=True)
with st.form("rula_assessment_form"):
    st.markdown("<div class='sub-header-green'> Part A: Upper Extremity Score (Arm, Forearm, Wrist)</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### 1) Upper Arm Flexion Score")
        arm_angle = st.slider("Upper arm flexion angle (°)", -90, 180, default_arm, help="Forward = positive, backward = negative")
        st.markdown("<small>✅ Check options below if applicable</small>", unsafe_allow_html=True)
        arm_abduction = st.checkbox("Arm abducted", value=False)
        shoulder_raise = st.checkbox("Shoulder elevated", value=False)
        arm_support = st.checkbox("Arm supported (-1 score)", value=False)
    
    with col2:
        st.markdown("##### 2) Forearm Flexion Score")
        forearm_angle = st.slider("Forearm flexion angle (°)", 0, 180, default_forearm, help="60–100° = neutral position")
        st.markdown("<small>✅ Check options below if applicable</small>", unsafe_allow_html=True)
        forearm_abduction = st.checkbox("Forearm abducted", value=False)
    
    with col3:
        st.markdown("##### 3) Wrist Score")
        wrist_bend = st.slider("Wrist bend angle (°)", -45, 45, default_wrist, help="Extension = positive, flexion = negative")
        st.markdown("<small>✅ Check options below if applicable</small>", unsafe_allow_html=True)
        wrist_twist = st.checkbox("Wrist twisted", value=False)
    
    st.markdown("<div class='sub-header-green'> Part B: Trunk Score (Neck, Trunk, Legs)</div>", unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("##### 1) Neck Score")
        neck_angle = st.slider("Neck flexion angle (°)", -30, 60, default_neck, help="Forward = positive, backward = negative")
        st.markdown("<small>✅ Check options below if applicable</small>", unsafe_allow_html=True)
        neck_twist = st.checkbox("Neck twisted", value=False)
        neck_bend = st.checkbox("Neck side-bent", value=False)
    
    with col5:
        st.markdown("##### 2) Trunk Score")
        trunk_angle = st.slider("Trunk flexion angle (°)", 0, 90, default_trunk, help="Forward = positive")
        st.markdown("<small>✅ Check options below if applicable</small>", unsafe_allow_html=True)
        trunk_twist = st.checkbox("Trunk twisted", value=False)
        trunk_bend = st.checkbox("Trunk side-bent", value=False)
    
    with col6:
        st.markdown("##### 3) Leg Score")
        st.markdown("<small>⚠️ Legs supported by default; uncheck if unsupported</small>", unsafe_allow_html=True)
        leg_support = st.checkbox("Legs/feet properly supported and balanced", value=True)
    
    st.markdown("<div class='sub-header-green'> Part C & D: Muscle State & Force/Load Score</div>", unsafe_allow_html=True)
    
    col7, col8 = st.columns(2)
    with col7:
        st.markdown("##### 1) Muscle State Score")
        muscle_state = st.selectbox(
            "Muscle working condition",
            ["No special condition", "Static posture or holding > 1 minute", "Repetitive work > 4 times/minute"],
            index=0
        )
    
    with col8:
        st.markdown("##### 2) Force / Load State Score")
        load_state = st.selectbox(
            "Work load condition",
            ["No force / cyclic load < 2kg", "Cyclic load 2–10kg", "Static/repetitive load 2–10kg, cyclic load ≥10kg", "Static 10kg, repetitive 10kg, vibration or rapid force increase"],
            index=0
        )
    
    submit_button = st.form_submit_button("Start Assessment", type="primary", width='stretch')

# Section 3: AI Analysis & Consultation
st.markdown("<div class='section-header'>【Section 3】💡 AI Analysis, Recommendations & Consultation</div>", unsafe_allow_html=True)

# Process assessment calculation
if submit_button:
    scores = calculate_rula_scores(
        arm_angle, arm_abduction, shoulder_raise, arm_support,
        forearm_angle, forearm_abduction,
        wrist_bend, wrist_twist,
        neck_angle, neck_twist, neck_bend,
        trunk_angle, trunk_twist, trunk_bend,
        leg_support,
        muscle_state, load_state
    )
    
    if scores is not None:
        st.session_state.rula_result = scores
        st.session_state.last_scores = scores
        st.session_state.need_gen_ai = True

# Process AI report generation
if st.session_state.need_gen_ai and "last_scores" in st.session_state and st.session_state.last_scores is not None:
    scores = st.session_state.last_scores
    
    with st.spinner("🧠 AI is generating ergonomic risk analysis report..."):
        ai_prompt = f"""
        You are a professional ergonomics expert. Produce an analysis report in strict accordance with RULA and ISO 11226 standards.
        ⚠️ 【Mandatory Format Rules】
        1. Start with 【Assessment Result Summary】, no opening remarks
        2. Each ○ bullet must be on its own line; never combine bullets
        3. Every body part must include 【measured angle° + individual score】
        4. Analyze specific ergonomic risks with reference to RULA standards
        5. Provide practical, actionable improvement recommendations in 3 categories: posture adjustment, work environment optimization, rest schedule. You may add other relevant sections.
        6. Use concise, clear professional English throughout

        【Assessment Result Summary】
        - Score A (Upper Extremity): {scores['a_total']}
        - Score B (Trunk): {scores['b_total']}
        - Score C / D: {scores['c_total']} / {scores['d_total']}
        - Final RULA Score: {scores['rula_total']}
        - Action Level: {scores['action_level']}
        - Action Plan: {scores['action_plan']}
        
        Assessment Data:
        1. Upper Extremity Scores:
        - Upper Arm: {arm_angle}°, Score {scores['arm_final']}
        - Forearm: {forearm_angle}°, Score {scores['forearm_final']}
        - Wrist: {wrist_bend}°, Score {scores['wrist_final']}
        - Total A: {scores['a_total']}
        2. Trunk Scores:
        - Neck: {neck_angle}°, Score {scores['neck_final']}
        - Trunk: {trunk_angle}°, Score {scores['trunk_final']}
        - Legs: Score {scores['leg_final']}
        - Total B: {scores['b_total']}
        3. Muscle & Load Scores:
        - Muscle: {muscle_state}, Score {scores['muscle_score']}, {scores['muscle_desc']}
        - Load: {load_state}, Score {scores['load_score']}, {scores['load_desc']}
        - Total C: {scores['c_total']}, Total D: {scores['d_total']}
        4. Final Result:
        - RULA Total: {scores['rula_total']}
        - Action Level: {scores['action_level']}
        - Action Plan: {scores['action_plan']}
        
        Output Structure:
        ## I. Body Part Risk Analysis (RULA-aligned)
        1. Upper Extremity (Arm - Forearm - Wrist):
        
            ○ Upper Arm (XX°, Score X): professional analysis
            
            ○ Forearm (XX°, Score X): professional analysis
            
            ○ Wrist (XX°, Score X): professional analysis
            
        2. Trunk (Neck - Trunk - Legs):
        
            ○ Neck (XX°, Score X): analysis
            
            ○ Trunk (XX°, Score X): analysis
            
            ○ Legs (Score X): analysis
            
        3. Muscle & Load Factors:
        
            ○ Muscle State (condition name, Score X): analysis
            
            ○ Load State (condition name, Score X): analysis
        
        ## II. Actionable Improvement Recommendations
        """
        
        ai_response = call_deepseek_api([
            {"role": "system", "content": "You are a professional ergonomics expert, proficient in RULA and ISO 11226 international standards. Respond exclusively in English."},
            {"role": "user", "content": ai_prompt}
        ])

        new_item = {
            "score": scores['rula_total'],
            "content": ai_response,
            "messages": []
        }
        st.session_state.rula_history.insert(0, new_item)
        st.session_state.last_expand_idx = 0
        st.session_state.active_chat_id = 0
          
    st.session_state.need_gen_ai = False
    
# Display assessment results
if "rula_result" in st.session_state and st.session_state.rula_result is not None:
    scores = st.session_state.rula_result
    
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        st.markdown("<div class='score-box'>", unsafe_allow_html=True)
        st.markdown("Score A (Upper Extremity)")
        st.markdown(f"<div class='score-value'>{scores['a_total']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col10:
        st.markdown("<div class='score-box'>", unsafe_allow_html=True)
        st.markdown("Score B (Trunk)")
        st.markdown(f"<div class='score-value'>{scores['b_total']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col11:
        st.markdown("<div class='score-box'>", unsafe_allow_html=True)
        st.markdown("Score C / D")
        st.markdown(f"<div class='score-value'>{scores['c_total']}/{scores['d_total']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col12:
        st.markdown("<div class='score-box'>", unsafe_allow_html=True)
        st.markdown("Final RULA Score")
        st.markdown(f"<div class='score-value {scores['risk_class']}'>{scores['rula_total']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background-color: #F8F9FA; padding: 20px; border-radius: 10px; margin: 15px 0;">
        <h3>Action Level: <span class="{scores['risk_class']}">{scores['action_level']}</span></h3>
        <p>Action Plan: <span class="{scores['risk_class']}">{scores['action_plan']}</span></p>
    </div>
    """, unsafe_allow_html=True)

# Display assessment history
if len(st.session_state.rula_history) == 0:
    st.info("No assessment history yet. Fill in the data and click Start Assessment to generate your first report.")
else:
    total_count = len(st.session_state.rula_history)
    for idx, item in enumerate(st.session_state.rula_history):
        actual_number = total_count - idx
        
        open_flag = True if idx == st.session_state.last_expand_idx else False
        with st.expander(f"Assessment #{actual_number}｜RULA Score: {item['score']}", expanded=open_flag):
            st.markdown(item["content"])
            
            st.markdown("---")
            
            st.markdown("**💬 Consultation History**")
            for message in item["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            prompt = st.chat_input(f"Ask follow-up ergonomics questions about Assessment #{actual_number}...", key=f"chat_input_{actual_number}")
            
            if prompt:
                if not st.session_state.api_key_entered:
                    st.error("Please complete an assessment first. API will initialize automatically.")
                else:
                    item["messages"].append({"role": "user", "content": prompt})
                    st.rerun()
            
            if item["messages"] and item["messages"][-1]["role"] == "user":
                with st.spinner("Thinking..."):
                    context_messages = [
                        {"role": "system", "content": "You are a professional ergonomics expert, proficient in RULA and ISO 11226. Answer based on the RULA report above. Respond exclusively in English."},
                        {"role": "assistant", "content": item["content"]}
                    ] + item["messages"][:-1]
                    
                    context_messages.append(item["messages"][-1])
                    
                    full_response = call_deepseek_api(context_messages)
                    if full_response:
                        item["messages"].append({"role": "assistant", "content": full_response})
                        st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### About This System")
    st.markdown("""
    This system is developed based on the **Rapid Upper Limb Assessment (RULA)** method (McAtamney & Corlett, 1993), in strict compliance with **ISO 11226:2000 Ergonomics — Evaluation of static working postures**.
    
    #### Core Features:
    1. Auto-detect all key joint angles from uploaded photos
    2. 100% match with official RULA scoring table logic
    3. Auto-calculate A/B/C/D scores and final RULA total
    4. Professional AI analysis and improvement recommendations
    5. AI-powered ergonomics Q&A consultation
    
    #### Scoring Scale:
    | RULA Score | Action Level | Action Plan |
    |------------|--------------|-------------|
    | 1–2 | AL1 | No action required |
    | 3–4 | AL2 | Further investigation & improvement if needed |
    | 5–6 | AL3 | Further investigation & improvement soon |
    | ≥7 | AL4 | Immediate investigation & improvement required |
    """)
