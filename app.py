import pandas as pd
import sys
import os
import re
import subprocess
import cv2
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR
from PIL import Image
import streamlit as st

# ========== 0. Streamlit Cloud 环境适配（关键！解决系统库缺失） ==========
def fix_streamlit_env():
    """安装Streamlit Cloud缺失的系统库（libgomp、libGL等）"""
    try:
        # 安装PaddleOCR依赖的系统库
        subprocess.run(
            ["apt-get", "update", "-y"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        subprocess.run(
            ["apt-get", "install", "-y", "libgomp1", "libgl1-mesa-glx", "libglib2.0-0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except Exception:
        # 本地运行时跳过（非Linux环境）
        pass

# 初始化时修复环境
fix_streamlit_env()

# ========== 1. OCR 模块（保留核心逻辑，适配Streamlit图片读取） ==========
class HealthReportParser:
    """体检报告解析器（OCR + 指标提取）"""
    def __init__(self, use_gpu=False):
        # 手动指定模型路径（避免Streamlit Cloud下载失败，优先用本地缓存）
        self.ocr = PaddleOCR(
            use_angle_cls=True,      # 文本方向矫正
            lang='ch',
            show_log=False,          # 关闭内部日志
            use_gpu=use_gpu,
            # 模型缓存路径（Streamlit Cloud的临时目录）
            det_model_dir=None,  # 自动使用PaddleOCR默认缓存
            rec_model_dir=None,
            cls_model_dir=None
        )

        # 指标提取的正则表达式（保留原有逻辑）
        self.patterns = {
            '空腹血糖': r'(?:空腹血糖|血糖)\s*[：:]\s*(\d+(?:\.\d+)?)\s*(?:mmol/L|mmol)?',
            '甘油三酯': r'(?:甘油三酯|TG)\s*[：:]\s*(\d+(?:\.\d+)?)',
            '尿酸': r'尿酸\s*[：:]\s*(\d+(?:\.\d+)?)\s*(?:μmol/L|umol/L)?',
            '收缩压': r'(?:收缩压|高压)\s*[：:]\s*(\d+)',
            '舒张压': r'(?:舒张压|低压)\s*[：:]\s*(\d+)',
            'BMI': r'(?:BMI|体质指数)\s*[：:]\s*(\d+(?:\.\d+)?)',
            '身高': r'身高\s*[：:]\s*(\d+(?:\.\d+)?)\s*(?:cm|厘米|CM)?',
            '体重': r'体重\s*[：:]\s*(\d+(?:\.\d+)?)\s*(?:kg|公斤|KG)?',
        }
        self.bp_pattern = r'血压\s*[：:]\s*(\d+)\s*[/\/]\s*(\d+)'

    def extract_text(self, image_bytes):
        """适配Streamlit上传的图片：从字节流提取文本（替代原cv2.imread）"""
        # 将Streamlit上传的Bytes转为cv2可识别的格式
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解析图片，请上传有效格式（JPG/PNG）")
        
        # 原有OCR识别逻辑
        result = self.ocr.ocr(img)
        texts = []
        for line in result:
            if line:  # 兼容PaddleOCR不同版本的返回格式
                for word_info in line:
                    texts.append(word_info[1][0])
        return ' '.join(texts)

    def extract_health_metrics(self, image_bytes):
        """提取健康指标（输入改为图片字节流）"""
        full_text = self.extract_text(image_bytes)
        metrics = {}

        # 原有正则提取逻辑
        for key, pattern in self.patterns.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                metrics[key] = float(match.group(1))

        # 血压组合匹配
        bp_match = re.search(self.bp_pattern, full_text, re.IGNORECASE)
        if bp_match:
            metrics['收缩压'] = int(bp_match.group(1))
            metrics['舒张压'] = int(bp_match.group(2))

        # 自动计算BMI（保留原有逻辑）
        if 'BMI' not in metrics and '身高' in metrics and '体重' in metrics:
            height_m = metrics['身高'] / 100.0
            weight_kg = metrics['体重']
            bmi = weight_kg / (height_m ** 2)
            metrics['BMI'] = round(bmi, 1)

        # 返回6项核心指标
        return {
            "空腹血糖": metrics.get("空腹血糖", None),
            "甘油三酯": metrics.get("甘油三酯", None),
            "尿酸": metrics.get("尿酸", None),
            "收缩压": metrics.get("收缩压", None),
            "舒张压": metrics.get("舒张压", None),
            "BMI": metrics.get("BMI", None),
        }

# ========== 2. 健康分析数据（完全保留原有逻辑） ==========
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

# ========== 3. Streamlit 可视化交互（替代命令行） ==========
st.set_page_config(page_title="体检报告OCR分析工具", layout="wide")
st.title("📝 体检报告图像识别分析工具")
st.subheader("上传体检报告图片，自动识别指标并分析健康风险")

# 步骤1：上传图片
uploaded_file = st.file_uploader("请上传体检报告图片（JPG/PNG）", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 显示上传的图片
    st.image(uploaded_file, caption="上传的体检报告", width=400)
    
    # 步骤2：OCR识别指标
    with st.spinner("正在识别图片中的体检指标..."):
        try:
            # 初始化OCR解析器
            parser = HealthReportParser(use_gpu=False)
            # 读取图片字节流并识别
            image_bytes = uploaded_file.getvalue()
            metrics = parser.extract_health_metrics(image_bytes)
            st.success(" 图片识别完成！")
        except Exception as e:
            st.error(f" 识别失败：{str(e)}")
            st.stop()

    # 步骤3：让用户确认/修改识别结果（替代原命令行输入）
    st.subheader("识别结果（可修改）")
    with st.form("metrics_confirm_form"):
        data = {}
        # 逐个生成输入框，填充识别结果
        blood_glucose = st.number_input("空腹血糖 (mmol/L)", value=metrics["空腹血糖"], placeholder="例如：5.2")
        triglyceride = st.number_input("甘油三酯", value=metrics["甘油三酯"], placeholder="例如：1.3")
        uric_acid = st.number_input("尿酸 (μmol/L)", value=metrics["尿酸"], placeholder="例如：350")
        systolic = st.number_input("收缩压 (mmHg)", value=metrics["收缩压"], placeholder="例如：110")
        diastolic = st.number_input("舒张压 (mmHg)", value=metrics["舒张压"], placeholder="例如：75")
        bmi = st.number_input("BMI", value=metrics["BMI"], placeholder="例如：22.5")
        
        # 提交确认
        submit_btn = st.form_submit_button("确认指标，开始分析")

    # 步骤4：健康分析（完全保留原有逻辑）
    if submit_btn:
        # 整理用户确认后的指标
        data = {
            "空腹血糖": blood_glucose,
            "甘油三酯": triglyceride,
            "尿酸": uric_acid,
            "收缩压": systolic,
            "舒张压": diastolic,
            "BMI": bmi
        }
        # 过滤空值
        data = {k: v for k, v in data.items() if v is not None}
        
        if not data:
            st.error("请至少填写一项有效指标！")
        else:
            # 原有健康分析逻辑
            abnormal = []
            recommend_food = []
            avoid_food = []

            for ind, val in data.items():
                row = df_ind[df_ind["indicator"] == ind]
                if row.empty:
                    continue
                row = row.iloc[0]
                min_val = float(row["normal_min"])
                max_val = float(row["normal_max"])

                if val > max_val:
                    abnormal.append(f"{ind} 偏高：{row['high_risk']}")
                    recommend_food.extend(row["recommend_food"].split("、"))
                    avoid_food.extend(row["avoid_food"].split("、"))
                elif val < min_val:
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

            # 可视化输出结果
            st.subheader(" 健康风险分析")
            if not abnormal:
                st.success(" 所有指标均在正常范围，继续保持健康生活！")
            else:
                for a in abnormal:
                    st.warning(f"- {a}")

            st.subheader(" 推荐食材")
            st.write("、".join(recommend_food))

            st.subheader(" 禁忌食材")
            st.write("、".join(avoid_food))

            st.subheader(" 个性化药膳方案")
            if matched_recipes:
                for i, r in enumerate(matched_recipes[:3], 1):
                    st.write(f"{i}. **{r['name']}**")
                    st.write(f"  材料：{r['material']}")
                    st.write(f"  建议：{r['usage']}\n")
            else:
                st.info("暂无匹配的药膳方案")
