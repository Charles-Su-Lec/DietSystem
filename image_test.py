import pandas as pd
import sys
import os
import re
from pathlib import Path
from paddleocr import PaddleOCR
import cv2

# ========== 1. OCR 模块（改造为单例，避免重复初始化PDX） ==========
class HealthReportParser:
    """体检报告解析器（OCR + 指标提取）- 单例模式"""
    _instance = None  # 单例实例
    _ocr = None       # 全局唯一的OCR实例

    def __new__(cls, use_gpu=False):
        # 单例逻辑：如果实例不存在，才创建；否则复用已有实例
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 仅第一次创建时初始化PaddleOCR（避免PDX重复初始化）
            if cls._ocr is None:
                cls._ocr = PaddleOCR(
                    use_angle_cls=True,      # 文本方向矫正
                    lang='ch',
                    show_log=False,          # 关闭内部日志
                    use_gpu=use_gpu
                )
        return cls._instance

    def __init__(self, use_gpu=False):
        # 初始化仅执行一次（单例保证）
        if hasattr(self, '_initialized'):
            return
        
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
        
        self._initialized = True  # 标记已初始化

    def extract_text(self, image_path):
        """从图片中提取所有文本，返回字符串"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        # 复用全局OCR实例
        result = self._ocr.ocr(img)
        texts = []
        for line in result:
            for word_info in line:
                texts.append(word_info[1][0])
        return ' '.join(texts)

    def extract_health_metrics(self, image_path):
        """提取健康指标，返回包含6项指标的字典（血糖、血脂、尿酸、收缩压、舒张压、BMI）"""
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

        # 如果 BMI 未识别到，但身高和体重都有，则自动计算 BMI
        if 'BMI' not in metrics and '身高' in metrics and '体重' in metrics:
            height_m = metrics['身高'] / 100.0   # 厘米转米
            weight_kg = metrics['体重']
            bmi = weight_kg / (height_m ** 2)
            metrics['BMI'] = round(bmi, 1)

        # 返回最终需要的6项指标（其他指标不参与分析）
        return {
            "空腹血糖": metrics.get("空腹血糖", None),
            "甘油三酯": metrics.get("甘油三酯", None),
            "尿酸": metrics.get("尿酸", None),
            "收缩压": metrics.get("收缩压", None),
            "舒张压": metrics.get("舒张压", None),
            "BMI": metrics.get("BMI", None),
        }

# ========== 2. 健康分析数据（保持不变） ==========
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

# ========== 3. 主程序（保持不变） ==========
if __name__ == "__main__":
    # 获取图片路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        raw_path = input("请输入体检报告图片的绝对路径：").strip()
        raw_path = raw_path.strip('"').strip("'")
        image_path = str(Path(raw_path))

    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        sys.exit(1)

    print(f"正在识别图片: {image_path}")
    try:
        parser = HealthReportParser(use_gpu=False)   # CPU 模式
        metrics = parser.extract_health_metrics(image_path)
    except Exception as e:
        print(f"识别失败: {e}")
        sys.exit(1)

    # 显示识别结果，允许用户修改
    print("\n识别结果如下（可直接回车确认，或输入新值修改）：")
    data = {}
    for key in ["空腹血糖", "甘油三酯", "尿酸", "收缩压", "舒张压", "BMI"]:
        val = metrics.get(key)
        if val is not None:
            prompt = f"{key} (当前值: {val})，请输入新值（直接回车保留）："
        else:
            prompt = f"{key} (未识别到)，请输入值："
        user_input = input(prompt).strip()
        if user_input == "":
            if val is not None:
                data[key] = val
            else:
                print(f"警告：{key} 未提供，将不参与分析。")
        else:
            try:
                data[key] = float(user_input)
            except ValueError:
                print(f"输入无效，跳过 {key}。")

    if not data:
        print("没有有效的指标数据，程序退出。")
        sys.exit(1)

    # 健康分析
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

    # 输出结果
    print("【健康风险分析】")
    if not abnormal:
        print("所有指标均在正常范围，继续保持健康生活！")
    else:
        for a in abnormal:
            print("-", a)

    print("\n【推荐食材】")
    print("、".join(recommend_food))

    print("\n【禁忌食材】")
    print("、".join(avoid_food))

    print("\n【个性化药膳方案】")
    if matched_recipes:
        for i, r in enumerate(matched_recipes[:3], 1):
            print(f"{i}. {r['name']}")
            print(f"  材料：{r['material']}")
            print(f"  建议：{r['usage']}\n")
    else:
        print("暂无匹配药膳方案")
