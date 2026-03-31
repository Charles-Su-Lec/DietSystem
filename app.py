import gradio as gr
import pandas as pd
import re
import tempfile
import os
from paddleocr import PaddleOCR
import cv2
from flask import Flask, request, render_template_string, jsonify
import numpy as np

app = Flask(__name__)

# 初始化 OCR（使用轻量模型，避免超时）
ocr = PaddleOCR(use_angle_cls=False, lang='ch', show_log=False, use_gpu=False)

# 健康数据
data_ind = [
    ["空腹血糖", 3.9, 6.1, "高血糖", "低血糖", "苦瓜、燕麦、芹菜、柚子", "精制糖、奶茶、蛋糕、白面包"],
    ["甘油三酯", 0.56, 1.7, "高血脂", "低血脂", "深海鱼、燕麦、木耳、牛油果", "肥肉、动物内脏、油炸食品、酒"],
    ["尿酸", 150, 416, "高尿酸", "低尿酸", "冬瓜、柠檬、芹菜、鸡蛋", "海鲜、动物内脏、啤酒、浓汤、火锅"],
    ["收缩压", 90, 120, "高血压", "低血压", "芹菜、菠菜、香蕉、酸奶", "咸菜、腌肉、加工肉、高盐零食"],
    ["舒张压", 60, 80, "高血压", "低血压", "芹菜、菠菜、香蕉、酸奶", "咸菜、腌肉、加工肉、高盐零食"],
    ["BMI", 18.5, 24, "肥胖", "消瘦", "蔬菜、粗粮、鸡胸肉、鸡蛋", "油炸食品、奶茶、蛋糕、肥肉"]
]
df_ind = pd.DataFrame(data_ind, columns=["indicator", "normal_min", "normal_max", "high_risk", "low_risk", "recommend_food", "avoid_food"])

data_rec = [
    ["苦瓜降糖汤", "苦瓜、排骨、姜片", "每日一次，佐餐食用", "高血糖"],
    ["木耳降脂粥", "木耳、大米、红枣", "早晚食用", "高血脂"],
    ["冬瓜利尿汤", "冬瓜、海带、瘦肉", "每周3次", "高尿酸"],
    ["芹菜降压沙拉", "芹菜、酸奶、香蕉", "每日早餐", "高血压"],
    ["鸡胸蔬菜沙拉", "鸡胸肉、生菜、番茄", "每日晚餐", "肥胖"]
]
df_rec = pd.DataFrame(data_rec, columns=["recipe_name", "material", "usage", "fit_condition"])

def extract_metrics_from_image(image_bytes):
    # 将字节流转换为 OpenCV 图像
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return {}
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        tmp_path = tmp.name
    result = ocr.ocr(tmp_path)
    full_text = ' '.join([word[1][0] for line in result for word in line])
    os.unlink(tmp_path)

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

    return metrics

def analyze_metrics(fbg, tg, ua, sbp, dbp, bmi):
    data = {}
    if fbg is not None: data["空腹血糖"] = fbg
    if tg is not None: data["甘油三酯"] = tg
    if ua is not None: data["尿酸"] = ua
    if sbp is not None: data["收缩压"] = sbp
    if dbp is not None: data["舒张压"] = dbp
    if bmi is not None: data["BMI"] = bmi
    if not data:
        return "请至少提供一个有效指标"

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
            matched_recipes.append(f"{r['recipe_name']}（{r['material']}）建议：{r['usage']}")

    result = "【健康风险分析】\n" + ("\n".join(abnormal) if abnormal else "所有指标均在正常范围")
    result += "\n\n【推荐食材】\n" + ("、".join(recommend_food) if recommend_food else "无")
    result += "\n\n【禁忌食材】\n" + ("、".join(avoid_food) if avoid_food else "无")
    result += "\n\n【个性化药膳方案】\n" + ("\n".join(matched_recipes) if matched_recipes else "暂无")
    return result

# HTML 模板（简洁，支持移动端）
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>体检报告分析</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; max-width: 800px; margin: auto; }
        input, button { padding: 10px; margin: 5px; width: 100%; }
        .result { white-space: pre-wrap; background: #f0f0f0; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>体检报告智能分析</h1>
    <p>上传体检报告图片（支持 jpg/png），系统将自动识别关键指标。您也可以手动修改数值。</p>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/jpeg,image/png" required>
        <button type="submit">识别并填写</button>
    </form>
    <hr>
    <form action="/analyze" method="post">
        <h3>指标数值（可修改）</h3>
        空腹血糖 (mmol/L): <input type="number" step="0.1" name="fbg" value="{{ fbg or '' }}">
        甘油三酯 (mmol/L): <input type="number" step="0.1" name="tg" value="{{ tg or '' }}">
        尿酸 (μmol/L): <input type="number" step="1" name="ua" value="{{ ua or '' }}">
        收缩压 (mmHg): <input type="number" step="1" name="sbp" value="{{ sbp or '' }}">
        舒张压 (mmHg): <input type="number" step="1" name="dbp" value="{{ dbp or '' }}">
        BMI: <input type="number" step="0.1" name="bmi" value="{{ bmi or '' }}">
        <button type="submit">开始分析</button>
    </form>
    {% if result %}
    <h3>分析结果</h3>
    <div class="result">{{ result }}</div>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, fbg=None, tg=None, ua=None, sbp=None, dbp=None, bmi=None, result=None)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return "请上传文件", 400
    img_bytes = file.read()
    metrics = extract_metrics_from_image(img_bytes)
    # 返回表单，自动填充识别值
    return render_template_string(HTML_TEMPLATE,
                                  fbg=metrics.get("空腹血糖", ""),
                                  tg=metrics.get("甘油三酯", ""),
                                  ua=metrics.get("尿酸", ""),
                                  sbp=metrics.get("收缩压", ""),
                                  dbp=metrics.get("舒张压", ""),
                                  bmi=metrics.get("BMI", ""),
                                  result=None)

@app.route('/analyze', methods=['POST'])
def analyze():
    def parse_float(v):
        return float(v) if v and v != '' else None
    fbg = parse_float(request.form.get('fbg'))
    tg = parse_float(request.form.get('tg'))
    ua = parse_float(request.form.get('ua'))
    sbp = parse_float(request.form.get('sbp'))
    dbp = parse_float(request.form.get('dbp'))
    bmi = parse_float(request.form.get('bmi'))
    result = analyze_metrics(fbg, tg, ua, sbp, dbp, bmi)
    return render_template_string(HTML_TEMPLATE,
                                  fbg=fbg, tg=tg, ua=ua, sbp=sbp, dbp=dbp, bmi=bmi,
                                  result=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
