import streamlit as st
import pandas as pd
import re
from paddleocr import PaddleOCR
import cv2
import tempfile
import os

# 初始化 OCR（使用缓存，避免重复加载）
@st.cache_resource
def init_ocr():
    return PaddleOCR(use_angle_cls=True, lang='ch', show_log=False, use_gpu=False)

ocr = init_ocr()

# 健康分析数据（与之前相同）
data_ind = [
    ["空腹血糖", 3.9, 6.1, "高血糖", "低血糖", "苦瓜、燕麦、芹菜、柚子", "精制糖、奶茶、蛋糕、白面包"],
    ["甘油三酯", 0.56, 1.7, "高血脂", "低血脂", "深海鱼、燕麦、木耳、牛油果", "肥肉、动物内脏、油炸食品、酒"],
    ["尿酸", 150, 416, "高尿酸", "低尿酸", "冬瓜、柠檬、芹菜、鸡蛋", "海鲜、动物内脏、啤酒、浓汤、火锅"],
    ["收缩压", 90, 120, "高血压", "低血压", "芹菜、菠菜、香蕉、酸奶", "咸菜、腌肉、加工肉、高盐零食"],
    ["舒张压", 60, 80, "高血压", "低血压", "芹菜、菠菜、香蕉、酸奶", "咸菜、腌肉、加工肉、高盐零食"],
    ["BMI", 18.5, 24, "肥胖", "消瘦", "蔬菜、粗粮、鸡胸肉、鸡蛋", "油炸食品、奶茶、蛋糕、肥肉"]
]
df_ind = pd.DataFrame(data_ind,
                      columns=["indicator", "normal_min", "normal_max", "high_risk", "low_risk",
                               "recommend_food", "avoid_food"])

data_rec = [
    ["苦瓜降糖汤", "苦瓜、排骨、姜片", "每日一次，佐餐食用", "高血糖"],
    ["木耳降脂粥", "木耳、大米、红枣", "早晚食用", "高血脂"],
    ["冬瓜利尿汤", "冬瓜、海带、瘦肉", "每周3次", "高尿酸"],
    ["芹菜降压沙拉", "芹菜、酸奶、香蕉", "每日早餐", "高血压"],
    ["鸡胸蔬菜沙拉", "鸡胸肉、生菜、番茄", "每日晚餐", "肥胖"]
]
df_rec = pd.DataFrame(data_rec, columns=["recipe_name", "material", "usage", "fit_condition"])

def extract_metrics_from_image(image_path):
    """OCR 提取指标（复用之前的逻辑）"""
    img = cv2.imread(image_path)
    result = ocr.ocr(img)
    full_text = ' '.join([word[1][0] for line in result for word in line])

    patterns = {
        '空腹血糖': r'(?:空腹血糖|血糖)\s*[：:]\s*(\d+(?:\.\d+)?)\s*(?:mmol/L|mmol)?',
        '甘油三酯': r'(?:甘油三酯|TG)\s*[：:]\s*(\d+(?:\.\d+)?)',
        '尿酸': r'尿酸\s*[：:]\s*(\d+(?:\.\d+)?)\s*(?:μmol/L|umol/L)?',
        '收缩压': r'(?:收缩压|高压)\s*[：:]\s*(\d+)',
        '舒张压': r'(?:舒张压|低压)\s*[：:]\s*(\d+)',
        'BMI': r'(?:BMI|体质指数)\s*[：:]\s*(\d+(?:\.\d+)?)',
        '身高': r'身高\s*[：:]\s*(\d+(?:\.\d+)?)\s*(?:cm|厘米|CM)?',
        '体重': r'体重\s*[：:]\s*(\d+(?:\.\d+)?)\s*(?:kg|公斤|KG)?',
    }
    bp_pattern = r'血压\s*[：:]\s*(\d+)\s*[/\/]\s*(\d+)'

    metrics = {}
    for key, pat in patterns.items():
        match = re.search(pat, full_text, re.IGNORECASE)
        if match:
            metrics[key] = float(match.group(1))

    bp_match = re.search(bp_pattern, full_text, re.IGNORECASE)
    if bp_match:
        metrics['收缩压'] = int(bp_match.group(1))
        metrics['舒张压'] = int(bp_match.group(2))

    if 'BMI' not in metrics and '身高' in metrics and '体重' in metrics:
        height_m = metrics['身高'] / 100.0
        bmi = metrics['体重'] / (height_m ** 2)
        metrics['BMI'] = round(bmi, 1)

    return {
        "空腹血糖": metrics.get("空腹血糖", None),
        "甘油三酯": metrics.get("甘油三酯", None),
        "尿酸": metrics.get("尿酸", None),
        "收缩压": metrics.get("收缩压", None),
        "舒张压": metrics.get("舒张压", None),
        "BMI": metrics.get("BMI", None),
    }

def analyze(metrics):
    """健康分析（同之前）"""
    data = {k: v for k, v in metrics.items() if v is not None}
    if not data:
        return None, None, None

    abnormal = []
    recommend_food = []
    avoid_food = []
    for ind, val in data.items():
        row = df_ind[df_ind["indicator"] == ind]
        if row.empty:
            continue
        row = row.iloc[0]
        if val > float(row["normal_max"]):
            abnormal.append(f"{ind} 偏高：{row['high_risk']}")
            recommend_food.extend(row["recommend_food"].split("、"))
            avoid_food.extend(row["avoid_food"].split("、"))
        elif val < float(row["normal_min"]):
            abnormal.append(f"{ind} 偏低：{row['low_risk']}")
            recommend_food.extend(row["recommend_food"].split("、"))
            avoid_food.extend(row["avoid_food"].split("、"))

    recommend_food = list(set(recommend_food))
    avoid_food = list(set(avoid_food))

    matched_recipes = []
    for risk in abnormal:
        keyword = risk.split("：")[-1]
        recipes = df_rec[df_rec["fit_condition"] == keyword]
        for _, r in recipes.iterrows():
            matched_recipes.append({
                "name": r["recipe_name"],
                "material": r["material"],
                "usage": r["usage"]
            })

    return abnormal, recommend_food, avoid_food, matched_recipes

# ================= 网页界面 =================
st.set_page_config(page_title="体检报告分析", layout="centered")
st.title(" 体检报告智能分析")
st.markdown("上传体检报告图片（支持 jpg/png），系统将自动识别关键指标并提供健康建议。")

uploaded_file = st.file_uploader("选择图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    with st.spinner("正在识别中，请稍候..."):
        metrics = extract_metrics_from_image(tmp_path)
    os.unlink(tmp_path)  # 删除临时文件

    # 显示识别结果并允许修改
    st.subheader("识别结果（可手动修改）")
    col1, col2 = st.columns(2)
    edited = {}
    with col1:
        for key in ["空腹血糖", "甘油三酯", "尿酸"]:
            val = metrics.get(key)
            if val is None:
                val = ""
            edited[key] = st.text_input(f"{key} (mmol/L 或 μmol/L)", value=str(val))
    with col2:
        for key in ["收缩压", "舒张压", "BMI"]:
            val = metrics.get(key)
            if val is None:
                val = ""
            edited[key] = st.text_input(f"{key} (mmHg 或 kg/m²)", value=str(val))

    # 转换为数值（过滤空值）
    final_metrics = {}
    for k, v in edited.items():
        if v.strip():
            try:
                final_metrics[k] = float(v)
            except:
                st.warning(f"{k} 输入无效，已忽略")
        else:
            # 如果未识别且用户未填，则跳过
            pass

    if st.button("开始分析"):
        if not final_metrics:
            st.error("请至少提供一个有效的指标值")
        else:
            abnormal, rec_food, avoid_food, recipes = analyze(final_metrics)
            st.subheader("健康风险分析")
            if not abnormal:
                st.success("所有指标均在正常范围，继续保持健康生活！")
            else:
                for a in abnormal:
                    st.warning(a)

            st.subheader("推荐食材")
            st.write("、".join(rec_food) if rec_food else "无")

            st.subheader("禁忌食材")
            st.write("、".join(avoid_food) if avoid_food else "无")

            st.subheader("个性化药膳方案")
            if recipes:
                for i, r in enumerate(recipes[:3], 1):
                    with st.expander(f"{i}. {r['name']}"):
                        st.write(f"**材料**：{r['material']}")
                        st.write(f"**建议**：{r['usage']}")
            else:
                st.info("暂无匹配药膳方案")