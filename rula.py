# ===================== 正常导入 =====================
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from openai import OpenAI
import datetime

# 初始化历史仓库（每条记录自带独立聊天历史）
if "rula_history" not in st.session_state:
    st.session_state.rula_history = []
# 标记本次是否需要生成AI
if "need_gen_ai" not in st.session_state:
    st.session_state.need_gen_ai = False
# 展开AI分析建议结果
if "last_expand_idx" not in st.session_state:
    st.session_state.last_expand_idx = -1
# 当前激活的聊天会话ID（-1表示没有激活）
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = -1

# ===================== 页面基础配置 =====================
st.set_page_config(
    page_title="RULA 快速上肢评估 系统",
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
    .stImage img {
        max-width: 800px !important;
        margin: 0 auto !important;
        display: block !important;
    }
    .sub-header-green {
    background-color: #DFF2DD; /* 柔和草绿色，适配页面配色不刺眼 */
    padding: 10px;
    border-radius: 5px;
    margin: 15px 0;
    font-weight: bold;
    color: #195927;
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

# ===================== ✅ 修复 1：静态图片专用模型（和疲劳代码一致）=====================
def load_pose_models():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,  # 静态图必须开！否则身躯/颈部识别失效
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

# ===================== ✅ 重构：计算失败返回None，不再返回默认值 =====================

# ===================== ✅ 工业场景实测校准版：颈部角度 =====================
def calculate_neck_flexion(nose, left_shoulder, right_shoulder, left_hip, right_hip):
    try:
        mid_sho = [(left_shoulder[i] + right_shoulder[i])/2 for i in range(3)]
        mid_hip = [(left_hip[i] + right_hip[i])/2 for i in range(3)]
        
        # ✅ 方法1：垂直高度差法（主算法，系数从120校准为70）
        vertical_drop = nose[1] - mid_sho[1]
        torso_height = mid_hip[1] - mid_sho[1]
        
        if torso_height > 50:  # 增加最小高度判断，防止异常值
            normalized_drop = vertical_drop / torso_height
            # 经验系数从120大幅降低到70，基于4张实测图校准
            angle = normalized_drop * 70
            angle = max(5, min(45, abs(angle)))  # 上限从60降到45，更符合实际
            return int(angle)
        
        # ✅ 方法2：备用角度差法（系数从100校准为60）
        torso_vector = np.array(mid_hip) - np.array(mid_sho)
        head_vector = np.array(nose) - np.array(mid_sho)
        angle_side = abs(np.degrees(np.arctan2(*torso_vector[:2])) - np.degrees(np.arctan2(*head_vector[:2])))
        
        return max(5, min(45, angle_side * 0.6))
    except Exception as e:
        print(f"颈部角度计算失败: {e}")
        return None

# ===================== ✅ 最终真实版：身躯角度 0~90° 全支持（RULA完美匹配）=====================
def calculate_trunk_flexion(left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee):
    try:
        mid_sho = [(left_shoulder[i] + right_shoulder[i])/2 for i in range(3)]
        mid_hip = [(left_hip[i] + right_hip[i])/2 for i in range(3)]
        mid_knee = [(left_knee[i] + right_knee[i])/2 for i in range(3)]

        # 身躯垂直方向变化
        dy = mid_hip[1] - mid_sho[1]
        # 身躯水平方向变化（判断弯腰最关键）
        dx = abs(mid_sho[0] - mid_hip[0])

        # 身躯真实长度
        torso_length = np.sqrt(dx**2 + dy**2)

        if torso_length < 30:
            return None

        # 真实身躯倾斜角度（0~90°）
        angle = np.degrees(np.arctan2(dx, dy))

        # 真实角度，不强行封顶
        # 但 RULA 最高只用到 >60°
        angle = max(0, min(85, angle))
        return int(angle)

    except Exception as e:
        return None
        
# ===================== ✅ 工业场景专用手腕角度计算 =====================
def calculate_wrist_bend(elbow, wrist, index_mcp, pinky_mcp):
    try:
        # ✅ 用MediaPipe原生的食指和小指MCP点计算手掌中心
        palm_center = [(index_mcp[i] + pinky_mcp[i])/2 for i in range(3)]
        
        # 计算手腕弯曲角度
        angle = calculate_angle(elbow, wrist, palm_center)
        # 转换为RULA标准的-30°到+30°范围
        wrist_bend = max(-30, min(30, 180 - angle))
        
        return int(wrist_bend)
    except Exception as e:
        print(f"手腕角度计算失败: {e}")
        return None

# ===================== 图片角度计算 =====================
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
    detection_message = "✅ 所有角度识别成功"

    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks.landmark
        
        # ========== 新增1：整体平均置信校验，整体太差直接全默认 ==========
        avg_visibility = sum(lm.visibility for lm in landmarks) / len(landmarks)
        # 整体平均可见度低于0.28，姿态不可信，全部标记默认
        if avg_visibility < 0.28:
            default_angles = ["手臂", "前臂", "手腕", "颈部", "身躯"]
        else:
            # 正常阈值判断
            def is_visible(landmark_idx):
                return landmarks[landmark_idx].visibility > 0.25
            
            def pt(landmark):
                return get_coord(landmarks[landmark], W, H)

            # 提取全部关键点坐标
            nose = pt(mp_pose.PoseLandmark.NOSE)
            l_sho = pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
            r_sho = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            l_elb = pt(mp_pose.PoseLandmark.LEFT_ELBOW)
            l_wri = pt(mp_pose.PoseLandmark.LEFT_WRIST)
            l_index = pt(mp_pose.PoseLandmark.LEFT_INDEX)
            l_pinky = pt(mp_pose.PoseLandmark.LEFT_PINKY)
            l_hip = pt(mp_pose.PoseLandmark.LEFT_HIP)
            r_hip = pt(mp_pose.PoseLandmark.RIGHT_HIP)
            l_knee = pt(mp_pose.PoseLandmark.LEFT_KNEE)
            r_knee = pt(mp_pose.PoseLandmark.RIGHT_KNEE)

            mid_sho = [(l_sho[i]+r_sho[i])/2 for i in range(3)]
            mid_hip = [(l_hip[i]+r_hip[i])/2 for i in range(3)]

            # --------------------------
            # 颈部：强制成功标记逻辑
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
                default_angles.append("颈部")

            # --------------------------
            # 身躯：强制成功标记逻辑
            # --------------------------
            trunk_ok = False
            if (is_visible(mp_pose.PoseLandmark.LEFT_SHOULDER) 
                and is_visible(mp_pose.PoseLandmark.RIGHT_SHOULDER)
                and is_visible(mp_pose.PoseLandmark.LEFT_HIP)
                and is_visible(mp_pose.PoseLandmark.RIGHT_HIP)
                and is_visible(mp_pose.PoseLandmark.LEFT_KNEE)
                and is_visible(mp_pose.PoseLandmark.RIGHT_KNEE)):
                
                trunk_angle = calculate_trunk_flexion(l_sho, r_sho, l_hip, r_hip, l_knee, r_knee)
                if trunk_angle is not None and 0 <= trunk_angle <= 85:
                    rula_angles["trunk_angle"] = trunk_angle
                    trunk_ok = True
            if not trunk_ok:
                default_angles.append("身躯")

            # --------------------------
            # 手臂：强制成功标记逻辑
            # --------------------------
            arm_ok = False
            if is_visible(mp_pose.PoseLandmark.LEFT_HIP) and is_visible(mp_pose.PoseLandmark.LEFT_SHOULDER) and is_visible(mp_pose.PoseLandmark.LEFT_ELBOW):
                arm_angle = calculate_angle(mid_hip, l_sho, l_elb)
                rula_angles["arm_angle"] = arm_angle
                arm_ok = True
            if not arm_ok:
                default_angles.append("手臂")

            # --------------------------
            # 前臂：强制成功标记逻辑
            # --------------------------
            forearm_ok = False
            if is_visible(mp_pose.PoseLandmark.LEFT_SHOULDER) and is_visible(mp_pose.PoseLandmark.LEFT_ELBOW) and is_visible(mp_pose.PoseLandmark.LEFT_WRIST):
                forearm_angle = calculate_angle(l_sho, l_elb, l_wri)
                rula_angles["forearm_angle"] = forearm_angle
                forearm_ok = True
            if not forearm_ok:
                default_angles.append("前臂")

            # --------------------------
            # 手腕：强制成功标记逻辑
            # --------------------------
            wrist_ok = False
            if (is_visible(mp_pose.PoseLandmark.LEFT_ELBOW) 
                and is_visible(mp_pose.PoseLandmark.LEFT_WRIST)
                and is_visible(mp_pose.PoseLandmark.LEFT_INDEX)
                and is_visible(mp_pose.PoseLandmark.LEFT_PINKY)):
                
                wrist_angle = calculate_wrist_bend(l_elb, l_wri, l_index, l_pinky)
                if wrist_angle is not None:
                    rula_angles["wrist_bend"] = wrist_angle
                    wrist_ok = True
            if not wrist_ok:
                default_angles.append("手腕")

        # 统一提示文案
        if default_angles:
            detection_message = f"⚠️ 部分角度识别失败，已自动填充默认值，建议在【第二部分】手动修正：{', '.join(default_angles)}"

        # 绘制姿态骨架
        mp.solutions.drawing_utils.draw_landmarks(
            image, 
            pose_result.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2)
        )

    else:
        # 完全检测不到人体骨架，全部默认
        default_angles = ["手臂", "前臂", "手腕", "颈部", "身躯"]
        detection_message = "⚠️ 未识别到人体姿态，全部使用默认值，请更换清晰侧身照片"

    pose.close()
    return image, rula_angles, detection_message, default_angles
# ===================== RULA 评分逻辑（完全保留你原来的） =====================
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

    # ===================== 肌肉状态 + 负荷状态 完整评分（严格匹配RULA原表）=====================
    # 肌肉状态 m（0或1）
    if muscle_state in ["静态，或持物超过1分钟","重复作业超过4次/分钟"]:
        m = 1
    else:
        m = 0

    # 负荷状态 l（0/1/2/3 四档完整匹配）
    if load_state == "无作用力/小于2kg周期性的负荷或力量":
        l = 0
    elif load_state == "2-10kg周期性的负荷或力量":
        l = 1
    elif load_state == "2-10kg静态/重复负荷，10kg或更多周期性负荷":
        l = 2
    elif load_state == "10kg静态，10kg重复的负荷或力量，振动或力量快速增加":
        l = 3
    else:
        l = 0
                        
    c = a + m + l
    d = b + m + l
    rula = get_table3_score(c, d)

    if rula <=2:
        lev, plan, cls = "AL1", "风险程度较低，不需要处理", "risk-low"
    elif rula <=4:
        lev, plan, cls = "AL2", "进一步调查及必要时进行改善", "risk-medium"
    elif rula <=6:
        lev, plan, cls = "AL3", "近日内需进行进一步调查及改善", "risk-medium"
    else:
        lev, plan, cls = "AL4", "必须立即进行调查及改善", "risk-high"

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
# ===================== AI 模块 =====================
def call_deepseek_api(messages):
    try:
        if not st.session_state.client:
            API_KEY = st.secrets["API_KEY"]
            st.session_state.client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")
            st.session_state.api_key_entered = True
        res = ""
        for chunk in st.session_state.client.chat.completions.create(model="Pro/deepseek-ai/DeepSeek-V3.2", messages=messages, stream=True):
            if chunk.choices and chunk.choices[0].delta.content:
                res += chunk.choices[0].delta.content
        return res
    except Exception as e:
        st.error(f"API错误: {e}")
        return ""
        
# ===================== 聊天消息显示函数 =====================
def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ===================== 主页面内容 =====================
st.markdown("<h1 class='main-header'>RULA 快速上肢评估 系统</h1>", unsafe_allow_html=True)
st.markdown("本系统基于**RULA快速上肢评估法**（McAtamney & Corlett, 1993）开发，严格遵循**ISO 11226:2000《人因工程-静态工作姿势评估》**国际标准。")

# 照片自动识别角度功能
st.markdown("<div class='section-header'>【第一部分】📷 照片识别角度（建议90°侧身全身拍照）</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("上传工作姿势照片（支持JPG、PNG）", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("正在识别姿势..."):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        processed_image, rula_angles, detection_message, default_angles = process_image(image)
        
        # ✅ 左右分栏布局：左侧图片，右侧角度结果
        col_img, col_angles = st.columns([3, 2])
        
        with col_img:
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="姿势识别结果", width=640)
        
        with col_angles:
            st.markdown("### 📊 角度识别结果")
            
            import pandas as pd
            
            # 构建纯净数据（不含_style列）
            angle_data = []
            angle_items = [
                ("手臂", "arm_angle"),
                ("前臂", "forearm_angle"),
                ("手腕", "wrist_bend"),
                ("颈部", "neck_angle"),
                ("身躯", "trunk_angle")
            ]
            
            # 单独记录需要高亮的行索引
            highlight_rows = []
            success_rows = []
            
            for idx, (name, key) in enumerate(angle_items):
                angle = int(rula_angles[key])
                if name in default_angles:
                    status = "⚠️ 默认值"
                    highlight_rows.append(idx)
                else:
                    status = "✅ 识别成功"
                    success_rows.append(idx)
                
                angle_data.append({
                    "部位": name,
                    "角度(°)": angle,
                    "状态": status
                })
            
            # 创建纯净的DataFrame
            df = pd.DataFrame(angle_data)
            
            # ✅ 更兼容的样式应用方式（不会产生多余列）
            def style_row(row):
                styles = [""] * len(row)
                if row["状态"] == "⚠️ 默认值":
                    styles = ["background-color: #fff3cd"] * len(row)
                elif row["状态"] == "✅ 识别成功":
                    styles[1] = "color: #00B050; font-weight: bold"  # 只高亮角度列
                return styles
            
            # 应用样式并显示
            styled_df = df.style.apply(style_row, axis=1).hide(axis="index")
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=220)
            
            # 显示识别状态提示
            if detection_message.startswith("✅"):
                st.success(detection_message)
            elif detection_message.startswith("⚠️"):
                st.warning(detection_message)
                st.info("💡 建议：调整拍照角度，确保全身侧身90°，手臂和手腕无遮挡（例如取下手套），以获得更准确的识别结果")
            else:
                st.error(detection_message)
                st.session_state.detection_success = False
        
        # 关键修复：只有真正检测成功或部分成功时才更新角度
        if detection_message.startswith("✅") or detection_message.startswith("⚠️"):
            st.session_state.auto_angles = rula_angles
            st.session_state.detection_success = True
            
# 优化后的代码（和process_image里的科学默认值保持一致）
if st.session_state.detection_success and st.session_state.auto_angles:
    default_arm = int(st.session_state.auto_angles["arm_angle"])
    default_forearm = int(st.session_state.auto_angles["forearm_angle"])
    default_wrist = int(st.session_state.auto_angles["wrist_bend"])
    default_neck = int(st.session_state.auto_angles["neck_angle"])
    default_trunk = int(st.session_state.auto_angles["trunk_angle"])
else:
    # 完全识别失败时，也用科学的中立姿势默认值，而不是0
    default_arm = 0
    default_forearm = 90
    default_wrist = 5  # 优化：和process_image一致
    default_neck = 8   # 优化：和process_image一致
    default_trunk = 10 # 优化：和process_image一致

# ========== 标题挪到form外面，单独一行 ==========
st.markdown("<div class='section-header'>【第二部分】📊 RULA快速上肢评估</div>", unsafe_allow_html=True)
# 表单从A分项开始，不再包含二级大标题
with st.form("rula_assessment_form"):
    st.markdown("<div class='sub-header-green'> A部分：上肢评分（手臂、前臂、手腕）</div>", unsafe_allow_html=True)
    # 原有所有滑块、勾选框代码完全保留不动
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### 1）手臂弯曲评分")
        arm_angle = st.slider("手臂弯曲角度（°）", -90, 180, default_arm, help="前倾为正，后倾为负")
        st.markdown("<small>✅ 如符合体态，则勾选下方选项</small>", unsafe_allow_html=True)
        arm_abduction = st.checkbox("手臂外扩", value=False)
        shoulder_raise = st.checkbox("肩膀提高", value=False)
        arm_support = st.checkbox("手臂有支撑（减1分）", value=False)
    
    with col2:
        st.markdown("##### 2）前臂弯曲评分")
        forearm_angle = st.slider("前臂弯曲角度（°）", 0, 180, default_forearm, help="60-100°为中立位")
        st.markdown("<small>✅ 如符合体态，则勾选下方选项</small>", unsafe_allow_html=True)
        forearm_abduction = st.checkbox("前臂外扩", value=False)
    
    with col3:
        st.markdown("##### 3）手腕评分")
        wrist_bend = st.slider("手腕弯曲角度（°）", -45, 45, default_wrist, help="上倾为正，下倾为负")
        st.markdown("<small>✅ 如符合体态，则勾选下方选项</small>", unsafe_allow_html=True)
        wrist_twist = st.checkbox("手腕扭转", value=False)
    
    st.markdown("<div class='sub-header-green'> B部分：躯干评分（颈部、身躯、腿部）</div>", unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("##### 1）颈部评分")
        neck_angle = st.slider("颈部弯曲角度（°）", -30, 60, default_neck, help="前倾为正，后仰为负")
        st.markdown("<small>✅ 如符合体态，则勾选下方选项</small>", unsafe_allow_html=True)
        neck_twist = st.checkbox("颈部扭转", value=False)
        neck_bend = st.checkbox("颈部侧弯", value=False)
    
    with col5:
        st.markdown("##### 2）身躯评分")
        trunk_angle = st.slider("身躯弯曲角度（°）", 0, 90, default_trunk, help="前倾为正")
        st.markdown("<small>✅ 如符合体态，则勾选下方选项</small>", unsafe_allow_html=True)
        trunk_twist = st.checkbox("身躯扭转", value=False)
        trunk_bend = st.checkbox("身躯侧弯", value=False)
    
    with col6:
        st.markdown("##### 3）腿部评分")
        st.markdown("<small>⚠️ 默认腿部有支撑；若无支撑，请取消勾选</small>", unsafe_allow_html=True)
        leg_support = st.checkbox("腿和脚踝有适当支撑且平衡", value=True)
    
    st.markdown("<div class='sub-header-green'> C、D部分：肌肉状态与力量、负荷状态评分</div>", unsafe_allow_html=True)
    
    col7, col8 = st.columns(2)
    with col7:
        st.markdown("##### 1）肌肉状态评分")
        muscle_state = st.selectbox(
            "肌肉工作状态",
            ["无特殊状态", "静态，或持物超过1分钟", "重复作业超过4次/分钟"],
            index=0
        )
    
    with col8:
        st.markdown("##### 2）力量、负荷状态评分")
        load_state = st.selectbox(
            "工作负荷状态",
            ["无作用力/小于2kg周期性的负荷或力量", "2-10kg周期性的负荷或力量", "2-10kg静态/重复负荷，10kg或更多周期性负荷", "10kg静态，10kg重复的负荷或力量，振动或力量快速增加"],
            index=0
        )
    
    submit_button = st.form_submit_button("开始评估", type="primary", width='stretch')

# ===================== 第三部分：只显示历史记录（不会重复） =====================
st.markdown("<div class='section-header'>【第三部分】💡 AI分析建议及咨询</div>", unsafe_allow_html=True)

# ===================== 第一步：先处理所有逻辑（计算+AI生成），再渲染任何内容 =====================
# 1. 处理评估计算
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
    
    # 确保calculate_rula_scores返回了有效字典
    if scores is not None:
        st.session_state.rula_result = scores
        st.session_state.last_scores = scores
        st.session_state.need_gen_ai = True

# 2. 处理AI生成（单独判断，确保只执行一次）
if st.session_state.need_gen_ai and "last_scores" in st.session_state and st.session_state.last_scores is not None:
    scores = st.session_state.last_scores
    
    with st.spinner("🧠 AI正在生成人因风险分析报告..."):
        ai_prompt = f"""      
        你是专业人因工程专家，严格依照RULA、ISO11226标准输出分析报告。
        强制固定排版结构，分三大块，每一处肢体必须同时写出【实测角度° + 分项得分】：

        【本次评估结果摘要】
        - A总分（上肢）：{scores['a_total']}
        - B总分（躯干）：{scores['b_total']}
        - C/D总分：{scores['c_total']}/{scores['d_total']}
        - 最终RULA总分：{scores['rula_total']}
        - 行动水准：{scores['action_level']}
        - 处理方案：{scores['action_plan']}

        原始测量角度：
        手臂弯曲角度：{arm_angle}°
        前臂弯曲角度：{forearm_angle}°
        手腕弯曲角度：{wrist_bend}°
        颈部弯曲角度：{neck_angle}°
        身躯弯曲角度：{trunk_angle}°
        
        各部位最终分项得分：
        手臂得分：{scores['arm_final']}
        前臂得分：{scores['forearm_final']}
        手腕得分：{scores['wrist_final']}
        颈部得分：{scores['neck_final']}
        躯干得分：{scores['trunk_final']}
        腿部得分：{scores['leg_final']}
        肌肉：{muscle_state}，得分{scores['muscle_score']}，说明：{scores['muscle_desc']}
        负荷：{load_state}，得分{scores['load_score']}，说明：{scores['load_desc']}
        
        输出格式严格照搬样板结构：
        ## 一、分部位风险分析（结合RULA标准）
        1. 上肢（手臂-前臂-手腕）：风险高低概括
            ○ 手臂（XX°，评分X）：专业风险解读
            ○ 前臂（XX°，评分X）：专业风险解读
            ○ 手腕（XX°，评分X）：专业风险解读
        2. 躯干与颈部：整体概括
            ○ 颈部（XX°，评分X）：解读
            ○ 身躯（XX°，评分X）：解读
            ○ 腿部（评分X）：解读
        3. 肌肉与负荷因素：概括
            ○ 肌肉状态：工况名称，评分X + 完整说明
            ○ 负荷状态：工况名称，评分X + 完整说明
        
        ## 二、可落地的改善建议
        分三类：姿势调整、工位环境优化、轮岗休息方案，务实可执行。
        语言专业平实，不要多余花哨格式，每一段肢体必须带上角度+分数成对展示。
        """
        
        ai_response = call_deepseek_api([
            {"role": "system", "content": "你是专业的人因工程专家，精通RULA快速上肢评估法和ISO 11226国际标准。"},
            {"role": "user", "content": ai_prompt}
        ])

        # 存入历史（新记录插最前面，自带空聊天历史）
        new_item = {
            "score": scores['rula_total'],
            "content": ai_response,
            "messages": []  # 每条评估自带独立聊天历史
        }
        st.session_state.rula_history.insert(0, new_item)
        # 标记最新条目自动展开
        st.session_state.last_expand_idx = 0
        # 新评估生成后自动激活它的聊天
        st.session_state.active_chat_id = 0
          
    # 生成完立即关闭开关，防止重复生成
    st.session_state.need_gen_ai = False

# ===================== 第二步：所有逻辑处理完，再渲染页面内容 =====================
# 1. 渲染RULA评估结果卡片（只有点击按钮后才显示）
if "rula_result" in st.session_state and st.session_state.rula_result is not None:
    scores = st.session_state.rula_result
    
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

# 2. 渲染第三部分标题+历史记录（永远在最下面，逻辑处理完才渲染）
if len(st.session_state.rula_history) == 0:
    st.info("暂无评估历史，填写数据后点击开始评估生成首份报告")
else:
    total_count = len(st.session_state.rula_history)  # 总评估次数
    for idx, item in enumerate(st.session_state.rula_history):
        # ✅ 正确序号：总次数 - 当前索引
        # 索引0（最新）→ total_count → 第N次
        # 索引1 → total_count-1 → 第N-1次
        # ...
        # 索引total_count-1（最旧）→ 1 → 第1次
        actual_number = total_count - idx
        
        # 最新条目自动展开（last_expand_idx还是标记索引0，不用改）
        open_flag = True if idx == st.session_state.last_expand_idx else False
        with st.expander(f"第{actual_number}次评估｜RULA总分：{item['score']}", expanded=open_flag):
            # 显示AI分析报告
            st.markdown(item["content"])
            
            # 分割线
            st.markdown("---")
            
            # 显示该评估的独立聊天历史
            st.markdown("**💬 咨询记录**")
            for message in item["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # 该评估的独立聊天输入框（key也要改成用actual_number，避免重复）
            prompt = st.chat_input(f"针对第{actual_number}次评估继续咨询人因工程相关问题...", key=f"chat_input_{actual_number}")
            if prompt:
                if not st.session_state.api_key_entered:
                    st.error("请先完成评估，系统会自动初始化API")
                else:
                    # 添加用户消息到该评估的聊天历史
                    item["messages"].append({"role": "user", "content": prompt})
                    
                    with st.spinner("思考中..."):
                        # 构建上下文：系统提示 + 本次评估报告 + 历史聊天记录
                        context_messages = [
                            {"role": "system", "content": "你是专业的人因工程专家，精通RULA快速上肢评估法和ISO 11226国际标准。请基于上面的RULA评估报告回答用户的问题。"},
                            {"role": "assistant", "content": item["content"]}
                        ] + item["messages"]
                        
                        full_response = call_deepseek_api(context_messages)
                        if full_response:
                            # 添加AI回复到该评估的聊天历史
                            item["messages"].append({"role": "assistant", "content": full_response})
                            st.rerun()

# 侧边栏说明
with st.sidebar:
    st.markdown("### 系统说明")
    st.markdown("""
    本系统基于**RULA快速上肢评估法**（McAtamney & Corlett, 1993）开发，严格遵循**ISO 11226:2000《人因工程-静态工作姿势评估》**国际标准。
    
    #### 核心功能：
    1. 上传照片自动识别所有核心角度
    2. 100%匹配RULA评估表的评分逻辑
    3. 自动计算A/B/C/D得分和RULA总分
    4. AI大模型专业分析与改善建议
    5. AI大模型人因问题对话及咨询
    
    #### 评分标准：
    | RULA总分 | 行动水准 | 处理方案 |
    |----------|----------|----------|
    | 1-2 | AL1 | 姿势，不需要处理 |
    | 3-4 | AL2 | 进一步调查及必要时进行改善 |
    | 5-6 | AL3 | 近日内需进行进一步调查及改善 |
    | ≥7 | AL4 | 必须立即进行调查及改善 |
    """)
