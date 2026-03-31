# ========== 第一步：解决 OpenCV 系统依赖（必加） ==========
import subprocess
import sys
import os

def install_opencv_deps():
    """安装OpenCV所需系统依赖（仅Streamlit Cloud生效）"""
    try:
        if "STREAMLIT_SERVER_BASEURL_PATH" in os.environ:
            # 更新源 + 安装缺失库
            subprocess.run(["apt-get", "update"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(
                ["apt-get", "install", "-y", "libgl1-mesa-glx", "libglib2.0-0"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
     print(f": {e}")

    finally:
        print("")
    install_opencv_deps()

# ========== 第二步：核心库导入 ==========
import pandas as pd
import re
from pathlib import Path
from paddleocr import PaddleOCR
import cv2
import streamlit as st

# ========== 第三步：保留原 OCR 解析类 ==========
class HealthReportParser:
    """体检报告解析器（OCR + 指标提取）"""
    def __init__(self, use_gpu=False):
        self.ocr = PaddleOCR(
            use_angle_cls=True,      # 文本方向矫正
            lang='ch',
            show_log=False,          # 关闭内部日志
            use_gpu=use_gpu
        )

        # 指标提取的正则表达式（包含身高、体重）
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
        # 血压组合匹配（如“血压 120/80”）
        self.bp_pattern = r'血压\s*[：:]\s*(\d+)\s*[/\/]\s*(\d+)'

    def extract_text(self, image_path):
        """从图片中提取所有文本，返回字符串"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        result = self.ocr.ocr(img)
        texts = []
        for line in result:
            for word_info in line:
                texts.append(word_info[1][0])
        return ' '.join(texts)

    def extract_health_metrics(self, image_path):
        """提取健康指标，返回包含6项指标的字典"""
        full_text = self.extract_text(image_path)
        metrics = {}

        # 提取所有模式匹配的指标
        for key, pattern in self.patterns.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                metrics[key] = float(match.group(1))

        # 处理血压组合（优先使用组合形式）
        bp_match = re.search(self.bp_pattern, full_text, re.IGNORECASE)
        if bp_match:
            metrics['收缩压'] = int(bp_match.group(1))
            metrics['舒张压'] = int(bp_match.group(2))

        # 自动计算 BMI（如果有身高体重）
        if 'BMI' not in metrics and '身高' in metrics and '体重' in metrics:
            height_m = metrics['身高'] / 100.0
            weight_kg = metrics['体重']
            bmi = weight_kg / (height_m ** 2)
            metrics['BMI'] = round(bmi, 1)

        # 返回最终需要的6项指标
        return {
            "空腹血糖": metrics.get("空腹血糖", None),
            "甘油三酯": metrics.get("甘油三酯", None),
            "尿酸": metrics.get("尿酸", None),
            "收缩压": metrics.get("收缩压", None),
            "舒张压": metrics.get("舒张压", None),
            "BMI": metrics.get("BMI", None),
        }

# ========== 第四步：初始化健康分析数据 ==========
data_ind = [
    ["空腹血糖", 3.9, 6.1, "高血糖", "低血糖", "苦瓜、燕麦、芹菜、柚子", "精制糖、奶茶、蛋糕、白面包"],
    ["甘油三酯", 0.56, 1.7, "高血脂", "低血脂", "深海鱼、燕麦、木耳、牛油果", "肥肉、动物内脏、油炸食品、酒"],
    ["尿酸", 150, 416, "高尿酸", "低尿酸", "冬瓜、柠檬、芹菜、鸡蛋", "海鲜、动物内脏、啤酒、浓汤、火锅"],
    ["收缩压", 90, 120, "高血压", "低血压", "芹菜、菠菜、香蕉、酸奶", "咸菜、腌肉、加工肉、高盐零食"],
    ["舒张压", 60, 80, "高血压", "低血压", "芹菜、菠菜、香蕉、酸奶", "咸菜、腌肉、加工肉、高盐零食"],
    ["BMI", 18.5, 24, "肥胖", "消瘦", "蔬菜、粗粮、鸡胸肉、鸡蛋", "油炸食品、奶茶、蛋糕、肥肉"]
]
df_ind = pd.DataFrame(
    data_ind,
    columns=["indicator", "normal_min", "normal_max", "high_risk", "low_risk", "recommend_food", "avoid_food"]
)

data_rec = [
    ["苦瓜降糖汤", "苦瓜、排骨、姜片", "每日一次，佐餐食用", "高血糖"],
    ["木耳降脂粥", "木耳、大米、红枣", "早晚食用", "高血脂"],
    ["冬瓜利尿汤", "冬瓜、海带、瘦肉", "每周3次", "高尿酸"],
    ["芹菜降压沙拉", "芹菜、酸奶、香蕉", "每日早餐", "高血压"],
    ["鸡胸蔬菜沙拉", "鸡胸肉、生菜、番茄", "每日晚餐", "肥胖"]
]
df_rec = pd.DataFrame(
    data_rec,
    columns=["recipe_name", "material", "usage", "fit_condition"]
)

# ========== 第五步：Streamlit 网页交互核心逻辑 ==========
def main():
    # 页面基础配置
    st.set_page_config(
        page_title="体检报告健康分析系统",
        page_icon="",
        layout="wide"
    )
    st.title(" 体检报告健康分析系统")
    st.divider()

    # 1. 上传图片（替换原命令行输入图片路径）
    st.subheader("上传体检报告图片")
    uploaded_file = st.file_uploader(
        "支持 JPG/PNG/JPEG 格式",
        type=["jpg", "png", "jpeg"],
        help="请上传清晰的体检报告指标页图片"
    )

    if not uploaded_file:
        st.info("请先上传体检报告图片，系统将自动识别健康指标")
        return

    # 保存上传的图片到临时路径（cv2需要本地文件路径）
    temp_image = Path("/tmp") / uploaded_file.name
    with open(temp_image, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. OCR 识别指标（替换原命令行的识别逻辑）
    st.subheader("图片识别结果")
    with st.spinner("正在识别图片中的健康指标..."):
        try:
            parser = HealthReportParser(use_gpu=False)
            metrics = parser.extract_health_metrics(str(temp_image))
            st.success("图片识别完成！")
        except Exception as e:
            st.error(f"识别失败：{str(e)}")
            return

    # 3. 指标确认/修改（替换原命令行的input输入）
    st.subheader("确认/修改健康指标")
    st.caption("未识别到的指标请手动输入，已识别的可直接回车保留")
    
    # 分两列展示输入框，更美观
    col1, col2 = st.columns(2)
    data = {}  # 存储用户确认后的指标

    with col1:
        # 第一列：空腹血糖、甘油三酯、尿酸
        for key in ["空腹血糖", "甘油三酯", "尿酸"]:
            val = metrics.get(key)
            if val is not None:
                user_input = st.text_input(f"{key} (mmol/L)", value=str(val))
            else:
                user_input = st.text_input(f"{key} (mmol/L)", placeholder="未识别，请输入数值")
            
            if user_input.strip():
                try:
                    data[key] = float(user_input)
                except ValueError:
                    st.warning(f"{key} 输入无效，将跳过该指标")

    with col2:
        # 第二列：收缩压、舒张压、BMI
        for key in ["收缩压", "舒张压", "BMI"]:
            val = metrics.get(key)
            unit = "mmHg" if "压" in key else ""
            if val is not None:
                user_input = st.text_input(f"{key} ({unit})", value=str(val))
            else:
                user_input = st.text_input(f"{key} ({unit})", placeholder="未识别，请输入数值")
            
            if user_input.strip():
                try:
                    data[key] = float(user_input)
                except ValueError:
                    st.warning(f"{key} 输入无效，将跳过该指标")

    if not data:
        st.warning("无有效指标数据，请至少填写一项")
        return

    # 4. 健康分析（替换原命令行的print输出）
    st.divider()
    st.subheader("健康风险分析")
    
    abnormal = []
    recommend_food = []
    avoid_food = []
    matched_recipes = []

    # 指标异常判断
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

    # 去重
    recommend_food = list(set(recommend_food))
    avoid_food = list(set(avoid_food))

    # 匹配药膳方案
    for risk in abnormal:
        keyword = risk.split("：")[-1]
        recipes = df_rec[df_rec["fit_condition"] == keyword]
        for _, r in recipes.iterrows():
            matched_recipes.append({
                "name": r["recipe_name"],
                "material": r["material"],
                "usage": r["usage"]
            })

    # 展示分析结果
    if not abnormal:
        st.success(" 所有指标均在正常范围，继续保持健康生活！")
    else:
        st.warning(" 发现异常指标：")
        for a in abnormal:
            st.markdown(f"- {a}")

    # 饮食建议
    st.subheader(" 饮食建议")
    col_food1, col_food2 = st.columns(2)
    with col_food1:
        st.markdown("** 推荐食材**")
        if recommend_food:
            st.write("、".join(recommend_food))
        else:
            st.write("无特殊推荐食材")
    with col_food2:
        st.markdown("禁忌食材")
        if avoid_food:
            st.write("、".join(avoid_food))
        else:
            st.write("无特殊禁忌食材")

    # 药膳方案
    st.subheader(" 个性化药膳方案")
    if matched_recipes:
        for i, r in enumerate(matched_recipes[:3], 1):
            st.markdown(f"""
            **{i}. {r['name']}**  
            - 材料：{r['material']}  
            - 食用建议：{r['usage']}
            """)
    else:
        st.info("暂无匹配的药膳方案")

# ========== 运行主函数 ==========
if __name__ == "__main__":
    main()
