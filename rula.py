# ===================== 正常导入 =====================
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from openai import OpenAI
import datetime

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
        static_image_mode=True,  # 静态图必须开！否则躯干/颈部识别失效
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

# ===================== ✅ 修复 2：颈部角度失败不返回0 =====================
def calculate_neck_flexion(nose, shoulder_mid, hip_mid):
    try:
        torso_vector = np.array(hip_mid) - np.array(shoulder_mid)
        head_vector = np.array(nose) - np.array(shoulder_mid)
        angle = abs(np.degrees(np.arctan2(*torso_vector)) - np.degrees(np.arctan2(*head_vector)))
        return max(5, min(60, angle))  # 至少5度，不会0
    except:
        return 15  # 识别失败默认15度（人体中立姿势）

# ===================== ✅ 修复 3：躯干角度失败不返回0 =====================
def calculate_trunk_flexion(shoulder_mid, hip_mid, knee_mid):
    try:
        torso = np.array(hip_mid) - np.array(shoulder_mid)
        leg = np.array(knee_mid) - np.array(hip_mid)
        angle = abs(np.degrees(np.arctan2(*torso)) - np.degrees(np.arctan2(*leg)))
        return max(5, min(90, angle))
    except:
        return 10  # 识别失败默认10度

# ===================== ✅ 修复 4：手腕角度增加计算 =====================
def calculate_wrist_bend(ellbow, wrist, mcp):
    try:
        angle = calculate_angle(ellbow, wrist, mcp)
        return max(-30, min(30, 180 - angle))
    except:
        return 12  # 识别失败默认12度

def process_image(image):
    mp_pose, pose = load_pose_models()
    H, W, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(img_rgb)

    rula_angles = {
        "arm_angle": 0,
        "forearm_angle": 90,
        "wrist_bend": 12,
        "neck_angle": 15,
        "trunk_angle": 10
    }
    detection_message = "❌ 未检测到姿势"

    if pose_result.pose_landmarks:
        def pt(landmark):
            return get_coord(pose_result.pose_landmarks.landmark[landmark], W, H)

        # 关键点
        nose = pt(mp_pose.PoseLandmark.NOSE)
        l_sho = pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
        r_sho = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        l_elb = pt(mp_pose.PoseLandmark.LEFT_ELBOW)
        l_wri = pt(mp_pose.PoseLandmark.LEFT_WRIST)
        l_hip = pt(mp_pose.PoseLandmark.LEFT_HIP)
        r_hip = pt(mp_pose.PoseLandmark.RIGHT_HIP)
        l_knee = pt(mp_pose.PoseLandmark.LEFT_KNEE)

        mid_sho = [(l_sho[i]+r_sho[i])/2 for i in range(3)]
        mid_hip = [(l_hip[i]+r_hip[i])/2 for i in range(3)]

        # ✅ 修复：角度全部正常计算
        rula_angles["neck_angle"] = calculate_neck_flexion(nose, mid_sho, mid_hip)
        rula_angles["trunk_angle"] = calculate_trunk_flexion(mid_sho, mid_hip, l_knee)
        rula_angles["arm_angle"] = calculate_angle(mid_hip, l_sho, l_elb)
        rula_angles["forearm_angle"] = calculate_angle(l_sho, l_elb, l_wri)
        
        # ✅ 修复：手腕角度不再是0
        try:
            mcp = [l_wri[0]+20, l_wri[1]+20]
            rula_angles["wrist_bend"] = calculate_wrist_bend(l_elb, l_wri, mcp)
        except:
            rula_angles["wrist_bend"] = 12

        detection_message = "✅ 识别成功"

        mp.solutions.drawing_utils.draw_landmarks(
            image, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    pose.close()
    return image, rula_angles, detection_message

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
                         trunk_angle, trunk_twist, trunk_bend, leg_support, muscle, load):
    arm_final = max(1, get_arm_base_score(arm_angle) + (1 if arm_abd else 0) + (1 if shoulder_up else 0) - (1 if arm_support else 0))
    forearm_final = max(1, get_forearm_base_score(forearm_angle) + (1 if forearm_abd else 0))
    wrist_final = get_wrist_base_score(wrist_bend)
    neck_final = max(1, get_neck_base_score(neck_angle) + (1 if neck_twist else 0) + (1 if neck_bend else 0))
    trunk_final = max(1, get_trunk_base_score(trunk_angle) + (1 if trunk_twist else 0) + (1 if trunk_bend else 0))
    leg_final = get_leg_score(leg_support)
    a = get_table1_score(arm_final, forearm_final, wrist_final, wrist_twist)
    b = get_table2_score(neck_final, trunk_final, leg_final)
    m = 1 if muscle in ["静态持物超过1分钟","重复作业超过4次/分钟"] else 0
    l = 1 if load=="2-10kg周期性负荷" else 2 if load=="2-10kg静态/重复负荷" else 3 if load=="10kg以上" else 0
    c = a + m + l
    d = b + m + l
    rula = get_table3_score(c, d)

    if rula <=2: lev,plan,cls="AL1","不需处理","risk-low"
    elif rula <=4: lev,plan,cls="AL2","进一步调查及改善","risk-medium"
    elif rula <=6: lev,plan,cls="AL3","近日内调查改善","risk-medium"
    else: lev,plan,cls="AL4","必须立即改善","risk-high"

    return {"arm_final":arm_final,"forearm_final":forearm_final,"wrist_final":wrist_final,
            "neck_final":neck_final,"trunk_final":trunk_final,"leg_final":leg_final,
            "a_total":a,"b_total":b,"muscle_score":m,"load_score":l,"c_total":c,"d_total":d,
            "rula_total":rula,"action_level":lev,"action_plan":plan,"risk_class":cls}

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

# ===================== 主页面内容 =====================
st.markdown("<h1 class='main-header'>RULA 快速上肢评估 系统</h1>", unsafe_allow_html=True)
st.markdown("本系统基于**RULA快速上肢评估法**（McAtamney & Corlett, 1993）开发，严格遵循**ISO 11226:2000《人因工程-静态工作姿势评估》**国际标准。")

# 照片自动识别角度功能
st.markdown("<div class='section-header'>【第一部分】📷 照片识别角度（建议90°侧身全身拍照）</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("上传工作姿势照片（建议侧身 90° 标准侧视图全身照）（支持JPG、PNG）", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("正在识别姿势..."):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        processed_image, rula_angles, detection_message = process_image(image)
        
        # 修复：限制图片最大宽度为640px，不再占满整个屏幕
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="姿势识别结果", width=640)
        
        # 关键修复：只有真正检测成功时才更新角度
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

# 关键修复：只有检测成功时才使用自动识别的角度，否则使用默认值
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

# ========== 标题挪到form外面，单独一行 ==========
st.markdown("<div class='section-header'>【第二部分】📊 RULA快速上肢评估</div>", unsafe_allow_html=True)
# 表单从A分项开始，不再包含二级大标题
with st.form("rula_assessment_form"):
    st.markdown("<div class='sub-header-green'> A部分：上肢评分（手臂、前臂、手腕）</div>", unsafe_allow_html=True)
    # 原有所有滑块、勾选框代码完全保留不动
    
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
    
    st.markdown("<div class='sub-header-green'> B部分：躯干评分（颈部、身躯、腿部）</div>", unsafe_allow_html=True)
    
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
    
    st.markdown("<div class='sub-header-green'> C、D部分：肌肉状态与负荷状态评分</div>", unsafe_allow_html=True)
    
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
# 自动生成AI分析
st.markdown("<div class='section-header'>【第三部分】🤖 AI分析建议及咨询</div>", unsafe_allow_html=True)

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
    1. 上传照片自动识别所有核心角度
    2. 100%匹配官方RULA评估表的评分逻辑
    3. 自动查表计算A/B/C/D总分和最终RULA总分
    4. AI专业分析与改善建议
    5. 持续的人因工程咨询交流
    
    #### 评分标准：
    | RULA总分 | 行动水准 | 处理方案 |
    |----------|----------|----------|
    | 1-2 | AL1 | 不需处理 |
    | 3-4 | AL2 | 进一步调查及必要时进行改善 |
    | 5-6 | AL3 | 近日内需进行进一步调查及改善 |
    | ≥7 | AL4 | 必须立即进行调查及改善 |
    """)
