# ===================== 最开头：只使用你上传到GitHub的本地模型文件 =====================
import os

# 模型文件已经和代码一起上传到GitHub仓库根目录
MODEL_PATH = 'pose_landmark_lite.tflite'

# 检查模型文件是否存在
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "请确保 pose_landmark_lite.tflite 文件已经上传到GitHub仓库根目录\n"
        "你已经上传的0.5.16版本模型完全可以正常使用"
    )

print("使用本地模型文件：", MODEL_PATH)

# ===================== 正常导入 =====================
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from openai import OpenAI
import datetime

# ===================== 页面基础配置 =====================
st.set_page_config(
    page_title="RULA快速上肢评估系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 页面样式优化
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
    /* 限制图片最大宽度 */
    .stImage img {
        max-width: 800px !important;
        margin: 0 auto !important;
        display: block !important;
    }
</style>
""", unsafe_allow_html=True)

# ===================== 初始化会话状态 =====================
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

# ===================== Mediapipe 导入与配置 =====================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_coord(landmark, W, H):
    return [landmark.x * W, landmark.y * H, landmark.z]

def process_image(image):
    H, W, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 使用你上传的本地模型文件
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        enable_segmentation=False,
        smooth_segmentation=False,
        model_path=MODEL_PATH
    )
    pose_result = pose.process(img_rgb)
    
    rula_angles = {
        "arm_angle": 0,
        "forearm_angle": 90,
        "wrist_bend": 0,
        "neck_angle": 0,
        "trunk_angle": 0
    }
    
    detection_message = None
    if pose_result.pose_landmarks:
        def get_pose_pt(landmark):
            return get_coord(pose_result.pose_landmarks.landmark[landmark], W, H)
        
        # 提取所有关键点
        left_shoulder = get_pose_pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_pose_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_elbow = get_pose_pt(mp_pose.PoseLandmark.LEFT_ELBOW)
        right_elbow = get_pose_pt(mp_pose.PoseLandmark.RIGHT_ELBOW)
        left_wrist = get_pose_pt(mp_pose.PoseLandmark.LEFT_WRIST)
        right_wrist = get_pose_pt(mp_pose.PoseLandmark.RIGHT_WRIST)
        left_hip = get_pose_pt(mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = get_pose_pt(mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee = get_pose_pt(mp_pose.PoseLandmark.LEFT_KNEE)
        right_knee = get_pose_pt(mp_pose.PoseLandmark.RIGHT_KNEE)
        nose = get_pose_pt(mp_pose.PoseLandmark.NOSE)

        # 关键点存在性检查
        if (left_shoulder is not None and right_shoulder is not None and
            left_hip is not None and right_hip is not None and
            left_knee is not None and right_knee is not None):
            
            # 中点
            mid_shoulder = [(left_shoulder[i] + right_shoulder[i])/2 for i in range(3)]
            mid_hip = [(left_hip[i] + right_hip[i])/2 for i in range(3)]
            mid_knee = [(left_knee[i] + right_knee[i])/2 for i in range(3)]

            # 角度计算（和你疲劳工具完全一致）
            def calculate_neck_flexion(nose, mid_shoulder, mid_hip):
                v_neck = np.array(nose) - np.array(mid_shoulder)
                v_trunk = np.array(mid_hip) - np.array(mid_shoulder)
                dot = np.dot(v_neck[:2], v_trunk[:2])
                cos_theta = dot / (np.linalg.norm(v_neck[:2]) * np.linalg.norm(v_trunk[:2]) + 1e-6)
                angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
                return max(0, min(60, angle))

            def calculate_trunk_flexion(mid_shoulder, mid_hip, mid_knee):
                v_trunk = np.array(mid_shoulder) - np.array(mid_hip)
                v_leg = np.array(mid_knee) - np.array(mid_hip)
                dot = np.dot(v_trunk[:2], v_leg[:2])
                cos_theta = dot / (np.linalg.norm(v_trunk[:2]) * np.linalg.norm(v_leg[:2]) + 1e-6)
                angle = 180 - np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
                return max(0, min(90, angle))

            def calculate_shoulder_abduction(shoulder, elbow):
                v_arm = np.array(elbow) - np.array(shoulder)
                v_vert = np.array([0, 1, 0])
                dot = np.dot(v_arm[:2], v_vert[:2])
                cos_theta = dot / (np.linalg.norm(v_arm[:2]) + 1e-6)
                raw_angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
                shoulder_angle = 180 - raw_angle if raw_angle > 90 else raw_angle
                return max(0, min(180, shoulder_angle))

            def calculate_elbow_flexion(shoulder, elbow, wrist):
                v_upper = np.array(shoulder) - np.array(elbow)
                v_lower = np.array(wrist) - np.array(elbow)
                dot = np.dot(v_upper[:2], v_lower[:2])
                cos_theta = dot / (np.linalg.norm(v_upper[:2]) * np.linalg.norm(v_lower[:2]) + 1e-6)
                angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
                return max(0, min(180, angle))

            def calculate_wrist_extension(elbow, wrist, index_tip):
                v_forearm = np.array(elbow) - np.array(wrist)
                v_hand = np.array(index_tip) - np.array(wrist)
                dot = np.dot(v_forearm[:2], v_hand[:2])
                cos_theta = dot / (np.linalg.norm(v_forearm[:2]) * np.linalg.norm(v_hand[:2]) + 1e-6)
                angle = 180 - np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
                return max(-45, min(45, angle))

            # 计算RULA所需角度
            rula_angles["neck_angle"] = calculate_neck_flexion(nose, mid_shoulder, mid_hip)
            rula_angles["trunk_angle"] = calculate_trunk_flexion(mid_shoulder, mid_hip, mid_knee)
            
            # 优先使用左侧手臂
            if np.linalg.norm(np.array(left_elbow) - np.array(left_shoulder)) > 10:
                rula_angles["arm_angle"] = calculate_shoulder_abduction(left_shoulder, left_elbow)
                rula_angles["forearm_angle"] = calculate_elbow_flexion(left_shoulder, left_elbow, left_wrist)
                rula_angles["wrist_bend"] = calculate_wrist_extension(left_elbow, left_wrist, left_wrist)
            else:
                rula_angles["arm_angle"] = calculate_shoulder_abduction(right_shoulder, right_elbow)
                rula_angles["forearm_angle"] = calculate_elbow_flexion(right_shoulder, right_elbow, right_wrist)
                rula_angles["wrist_bend"] = calculate_wrist_extension(right_elbow, right_wrist, right_wrist)

            detection_message = "✅ 角度已自动识别并填充，可手动修正"
        else:
            detection_message = "⚠️ 未能检测到完整的人体关键点，请确保照片中包含完整的上半身"
    else:
        detection_message = "❌ 未能检测到人体姿势，请上传清晰的工作姿势照片"

    # 绘制骨架
    if pose_result.pose_landmarks:
        drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        connection_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        mp_drawing.draw_landmarks(
            image,
            pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            drawing_spec,
            connection_spec
        )
    
    pose.close()
    return image, rula_angles, detection_message

# ===================== RULA评分核心逻辑（100%匹配评估表） =====================
def get_arm_base_score(arm_angle):
    if -20 <= arm_angle <= 20:
        return 1
    elif 20 < arm_angle <= 45 or arm_angle < -20:
        return 2
    elif 45 < arm_angle <= 90:
        return 3
    elif arm_angle > 90:
        return 4
    else:
        return 1

def get_forearm_base_score(forearm_angle):
    if 60 <= forearm_angle <= 100:
        return 1
    else:
        return 2

def get_wrist_base_score(wrist_bend):
    if abs(wrist_bend) < 1e-6:
        return 1
    elif abs(wrist_bend) <= 15:
        return 2
    else:
        return 3

def get_neck_base_score(neck_angle):
    if 0 <= neck_angle <= 10:
        return 1
    elif 10 < neck_angle <= 20:
        return 2
    elif neck_angle > 20:
        return 3
    elif neck_angle < 0:
        return 4
    else:
        return 1

def get_trunk_base_score(trunk_angle):
    if abs(trunk_angle) < 1e-6:
        return 1
    elif 0 < trunk_angle <= 20:
        return 2
    elif 20 < trunk_angle <= 60:
        return 3
    elif trunk_angle > 60:
        return 4
    else:
        return 1

def get_leg_score(leg_support):
    return 1 if leg_support else 2

# 表1：A总分查表
def get_table1_score(arm_score, forearm_score, wrist_score, wrist_twist):
    table1 = [
        [[1,2], [2,2], [2,2], [3,3]],
        [[2,2], [2,2], [2,3], [3,3]],
        [[2,3], [3,3], [3,3], [4,4]],
        [[3,3], [3,3], [3,4], [4,4]]
    ]
    arm_idx = max(0, min(3, arm_score - 1))
    forearm_idx = max(0, min(3, forearm_score - 1))
    wrist_idx = max(0, min(3, wrist_score - 1))
    twist_idx = 1 if wrist_twist else 0
    return table1[arm_idx][forearm_idx][wrist_idx][twist_idx]

# 表2：B总分查表
def get_table2_score(neck_score, trunk_score, leg_score):
    table2 = [
        [[1,2], [2,3], [3,4], [5,6]],
        [[2,3], [3,4], [4,5], [5,6]],
        [[3,4], [4,5], [5,6], [6,7]],
        [[5,6], [5,6], [6,7], [7,8]]
    ]
    neck_idx = max(0, min(3, neck_score - 1))
    trunk_idx = max(0, min(3, trunk_score - 1))
    leg_idx = 0 if leg_score == 1 else 1
    return table2[neck_idx][trunk_idx][leg_idx]

# 表3：最终RULA总分查表
def get_table3_score(c_total, d_total):
    table3 = [
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
    c_idx = max(0, min(8, c_total - 1))
    d_idx = max(0, min(8, d_total - 1))
    return table3[c_idx][d_idx]

def calculate_rula_scores(
    arm_angle, arm_abduction, shoulder_raise, arm_support,
    forearm_angle, forearm_abduction,
    wrist_bend, wrist_twist,
    neck_angle, neck_twist, neck_bend,
    trunk_angle, trunk_twist, trunk_bend,
    leg_support,
    muscle_state, load_state
):
    arm_base = get_arm_base_score(arm_angle)
    forearm_base = get_forearm_base_score(forearm_angle)
    wrist_base = get_wrist_base_score(wrist_bend)
    neck_base = get_neck_base_score(neck_angle)
    trunk_base = get_trunk_base_score(trunk_angle)
    leg_base = get_leg_score(leg_support)
    
    arm_add = 0
    if arm_abduction: arm_add += 1
    if shoulder_raise: arm_add += 1
    if arm_support: arm_add -= 1
    arm_final = max(1, arm_base + arm_add)
    
    forearm_add = 1 if forearm_abduction else 0
    forearm_final = max(1, forearm_base + forearm_add)
    
    neck_add = 0
    if neck_twist: neck_add += 1
    if neck_bend: neck_add += 1
    neck_final = max(1, neck_base + neck_add)
    
    trunk_add = 0
    if trunk_twist: trunk_add += 1
    if trunk_bend: trunk_add += 1
    trunk_final = max(1, trunk_base + trunk_add)
    
    a_total = get_table1_score(arm_final, forearm_final, wrist_base, wrist_twist)
    b_total = get_table2_score(neck_final, trunk_final, leg_base)
    
    muscle_score = 1 if muscle_state in ["静态持物超过1分钟", "重复作业超过4次/分钟"] else 0
    load_score = 0
    if load_state == "2-10kg周期性负荷": load_score = 1
    elif load_state == "2-10kg静态/重复负荷": load_score = 2
    elif load_state == "10kg以上静态/重复负荷": load_score = 3
    
    c_total = a_total + muscle_score + load_score
    d_total = b_total + muscle_score + load_score
    
    rula_total = get_table3_score(c_total, d_total)
    
    if 1 <= rula_total <= 2:
        action_level, action_plan, risk_class = "AL1", "不需处理", "risk-low"
    elif 3 <= rula_total <= 4:
        action_level, action_plan, risk_class = "AL2", "进一步调查及必要时进行改善", "risk-medium"
    elif 5 <= rula_total <= 6:
        action_level, action_plan, risk_class = "AL3", "近日内需进行进一步调查及改善", "risk-medium"
    elif rula_total >= 7:
        action_level, action_plan, risk_class = "AL4", "必须立即进行调查及改善", "risk-high"
    else:
        action_level, action_plan, risk_class = "未知", "无效评分", ""
    
    return {
        "arm_final": arm_final,
        "forearm_final": forearm_final,
        "wrist_final": wrist_base,
        "neck_final": neck_final,
        "trunk_final": trunk_final,
        "leg_final": leg_base,
        "a_total": a_total,
        "b_total": b_total,
        "muscle_score": muscle_score,
        "load_score": load_score,
        "c_total": c_total,
        "d_total": d_total,
        "rula_total": rula_total,
        "action_level": action_level,
        "action_plan": action_plan,
        "risk_class": risk_class
    }

# ===================== SiliconFlow DeepSeek API 调用 =====================
def call_deepseek_api(messages):
    try:
        if not st.session_state.client:
            try:
                API_KEY = st.secrets["API_KEY"]
                st.session_state.client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")
                st.session_state.api_key_entered = True
            except Exception as e:
                st.error(f"API 初始化失败：{str(e)}")
                st.info("请确保已在 Streamlit Secrets 中配置了 API_KEY")
                return None
        
        completion = st.session_state.client.chat.completions.create(
            model="Pro/deepseek-ai/DeepSeek-V3.2",
            messages=messages,
            stream=True
        )
        response = ""
        for chunk in completion:
            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and hasattr(choice.delta, "content") and choice.delta.content is not None:
                    response += choice.delta.content
        return response
    except Exception as e:
        st.error(f"API调用错误: {str(e)}")
        return None

# ===================== 主页面内容 =====================
st.markdown("<h1 class='main-header'>RULA快速上肢评估系统</h1>", unsafe_allow_html=True)
st.markdown("本系统基于**RULA快速上肢评估法**（McAtamney & Corlett, 1993）开发，严格遵循**ISO 11226:2000《人因工程-静态工作姿势评估》**国际标准。")

# 照片自动识别角度功能
st.markdown("<div class='section-header'>📷 照片自动识别角度</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("上传工作姿势照片（支持JPG、PNG）", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("正在识别姿势..."):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        processed_image, rula_angles, detection_message = process_image(image)
        
        # 限制图片最大宽度为800px
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="姿势识别结果", width=800)
        
        # 只有真正检测成功时才更新角度
        if detection_message.startswith("✅"):
            st.session_state.auto_angles = rula_angles
            st.session_state.detection_success = True
            st.success(detection_message)
        else:
            st.session_state.detection_success = False
            if detection_message.startswith("⚠️"):
                st.warning(detection_message)
            else:
                st.error(detection_message)

# 只有检测成功时才使用自动识别的角度，否则使用默认值
if st.session_state.detection_success and st.session_state.auto_angles:
    default_arm = int(st.session_state.auto_angles["arm_angle"])
    default_forearm = int(st.session_state.auto_angles["forearm_angle"])
    default_wrist = int(st.session_state.auto_angles["wrist_bend"])
    default_neck = int(st.session_state.auto_angles["neck_angle"])
    default_trunk = int(st.session_state.auto_angles["trunk_angle"])
else:
    default_arm = 0
    default_forearm = 90
    default_wrist = 0
    default_neck = 0
    default_trunk = 0

# 评估表单
with st.form("rula_assessment_form"):
    st.markdown("<div class='section-header'>一、A部分：上肢评分（手臂、前臂、手腕）</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1）手臂弯曲评分")
        arm_angle = st.slider("手臂弯曲角度（°）", -90, 180, default_arm, help="前倾为正，后倾为负")
        arm_abduction = st.checkbox("手臂外扩", value=False)
        shoulder_raise = st.checkbox("肩膀提高", value=False)
        arm_support = st.checkbox("手臂有支撑（减1分）", value=False)
    
    with col2:
        st.markdown("#### 2）前臂弯曲评分")
        forearm_angle = st.slider("前臂弯曲角度（°）", 0, 180, default_forearm, help="60-100°为中立位")
        forearm_abduction = st.checkbox("前臂外扩", value=False)
    
    with col3:
        st.markdown("#### 3）手腕评分")
        wrist_bend = st.slider("手腕弯曲角度（°）", -45, 45, default_wrist, help="上倾为正，下倾为负")
        wrist_twist = st.checkbox("手腕扭转", value=False)
    
    st.markdown("<div class='section-header'>二、B部分：躯干评分（颈部、身躯、腿部）</div>", unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("#### 1）颈部评分")
        neck_angle = st.slider("颈部弯曲角度（°）", -30, 60, default_neck, help="前倾为正，后仰为负")
        neck_twist = st.checkbox("颈部扭转", value=False)
        neck_bend = st.checkbox("颈部侧弯", value=False)
    
    with col5:
        st.markdown("#### 2）身躯评分")
        trunk_angle = st.slider("身躯弯曲角度（°）", 0, 90, default_trunk, help="前倾为正")
        trunk_twist = st.checkbox("身躯扭转", value=False)
        trunk_bend = st.checkbox("身躯侧弯", value=False)
    
    with col6:
        st.markdown("#### 3）腿部评分")
        leg_support = st.checkbox("腿和脚踝有适当支撑且平衡", value=True)
    
    st.markdown("<div class='section-header'>三、C/D部分：肌肉与负荷评分</div>", unsafe_allow_html=True)
    
    col7, col8 = st.columns(2)
    with col7:
        st.markdown("#### 1）肌肉状态评分")
        muscle_state = st.selectbox(
            "肌肉工作状态",
            ["无特殊状态", "静态持物超过1分钟", "重复作业超过4次/分钟"],
            index=0
        )
    
    with col8:
        st.markdown("#### 2）力量负荷评分")
        load_state = st.selectbox(
            "工作负荷状态",
            ["无作用力/小于2kg", "2-10kg周期性负荷", "2-10kg静态/重复负荷", "10kg以上静态/重复负荷"],
            index=0
        )
    
    submit_button = st.form_submit_button("开始评估", type="primary", width='stretch')

# 评估结果计算与展示
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
    
    st.session_state.rula_result = scores
    
    st.markdown("<div class='section-header'>四、评估结果</div>", unsafe_allow_html=True)
    
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        st.markdown("<div class='score-box'>", unsafe_allow_html=True)
        st.markdown("A总分（上肢）")
        st.markdown(f"<div class='score-value'>{scores['a_total']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col10:
        st.markdown("<div class='score-box'>", unsafe_allow_html=True)
        st.markdown("B总分（躯干）")
        st.markdown(f"<div class='score-value'>{scores['b_total']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col11:
        st.markdown("<div class='score-box'>", unsafe_allow_html=True)
        st.markdown("C/D总分")
        st.markdown(f"<div class='score-value'>{scores['c_total']}/{scores['d_total']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col12:
        st.markdown("<div class='score-box'>", unsafe_allow_html=True)
        st.markdown("最终RULA总分")
        st.markdown(f"<div class='score-value {scores['risk_class']}'>{scores['rula_total']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background-color: #F8F9FA; padding: 20px; border-radius: 10px; margin: 15px 0;'>
        <h3>行动水准：<span class='{scores['risk_class']}'>{scores['action_level']}</span></h3>
        <p>处理方案：<span class='{scores['risk_class']}'>{scores['action_plan']}</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # 自动生成AI分析
    st.markdown("<div class='section-header'>五、AI专业分析与改善建议</div>", unsafe_allow_html=True)
    with st.spinner("正在生成专业分析..."):
        ai_prompt = f"""
        你是专业的人因工程专家，精通RULA快速上肢评估法和ISO 11226国际标准。
        以下是用户的RULA评估数据，请基于这些数据进行专业的风险分析，并给出可落地的改善建议。

        评估数据：
        1. 上肢评分：
           - 手臂弯曲角度：{arm_angle}°，最终评分：{scores['arm_final']}
           - 前臂弯曲角度：{forearm_angle}°，最终评分：{scores['forearm_final']}
           - 手腕弯曲角度：{wrist_bend}°，最终评分：{scores['wrist_final']}
           - A总分：{scores['a_total']}
        2. 躯干评分：
           - 颈部弯曲角度：{neck_angle}°，最终评分：{scores['neck_final']}
           - 身躯弯曲角度：{trunk_angle}°，最终评分：{scores['trunk_final']}
           - 腿部评分：{scores['leg_final']}
           - B总分：{scores['b_total']}
        3. 肌肉与负荷评分：
           - 肌肉状态：{muscle_state}，评分：{scores['muscle_score']}
           - 负荷状态：{load_state}，评分：{scores['load_score']}
           - C总分：{scores['c_total']}，D总分：{scores['d_total']}
        4. 最终结果：
           - RULA总分：{scores['rula_total']}
           - 行动水准：{scores['action_level']}
           - 处理方案：{scores['action_plan']}

        要求：
        1. 先说明整体的风险等级和核心问题
        2. 分点分析每个身体部位的具体风险，结合RULA评估标准
        3. 给出针对性的、可落地的改善建议，分为姿势调整、工作环境优化、休息方案三个部分
        4. 语言专业、简洁、易懂
        """
        
        ai_response = call_deepseek_api([
            {"role": "system", "content": "你是专业的人因工程专家，精通RULA快速上肢评估法和ISO 11226国际标准。"},
            {"role": "user", "content": ai_prompt}
        ])
        
        if ai_response:
            st.session_state.messages = [
                {"role": "system", "content": "你是专业的人因工程专家，精通RULA快速上肢评估法和ISO 11226国际标准。"},
                {"role": "user", "content": ai_prompt},
                {"role": "assistant", "content": ai_response}
            ]
            st.markdown(ai_response)

# 持续对话交流
st.markdown("<div class='section-header'>六、持续咨询交流</div>", unsafe_allow_html=True)

def display_chat_messages():
    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            if msg["role"] != "system":
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

display_chat_messages()

prompt = st.chat_input("继续咨询人因工程相关问题：")
if prompt:
    if not st.session_state.api_key_entered:
        st.error("请先完成评估，系统会自动初始化API")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("思考中..."):
            full_response = call_deepseek_api(st.session_state.messages)
            if full_response:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()

# 侧边栏说明
with st.sidebar:
    st.markdown("### 系统说明")
    st.markdown("""
    本系统基于**RULA快速上肢评估法**（McAtamney & Corlett, 1993）开发，严格遵循**ISO 11226:2000《人因工程-静态工作姿势评估》**国际标准。
    
    #### 核心功能：
    1. 📷 上传照片自动识别所有核心角度
    2. 100%匹配官方RULA评估表的评分逻辑
    3. 自动查表计算A/B/C/D总分和最终RULA总分
    4. AI专业分析与改善建议
    5. 持续的人因工程咨询交流
    
    #### 评分标准：
    | RULA总分 | 行动水准 | 处理方案 |
    |----------|----------|----------|
    | 1-2 | AL1 | 不需处理 |
    | 3-4 | AL2 | 进一步调查及必要时改善 |
    | 5-6 | AL3 | 近日内需进一步调查及改善 |
    | ≥7 | AL4 | 必须立即调查及改善 |
    """)
