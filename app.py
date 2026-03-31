import streamlit as st
import os
import tempfile
from image_test import HealthReportParser, df_ind, df_rec

# ========== 缓存 Parser 实例（核心：避免重复初始化PaddleOCR） ==========
@st.cache_resource(ttl=None)  # 永久缓存，仅初始化一次
def get_parser():
    """获取全局唯一的体检报告解析器实例"""
    return HealthReportParser(use_gpu=False)

# ========== Streamlit 主界面 ==========
st.title("体检报告健康分析系统")
st.subheader("上传体检报告图片，自动解析健康指标并生成饮食建议")

# 上传图片
uploaded_file = st.file_uploader("选择体检报告图片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 保存上传的图片到临时文件（避免路径问题）
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_image_path = tmp_file.name

    try:
        # 获取缓存的Parser实例（仅第一次调用时初始化）
        parser = get_parser()
        
        # 解析图片提取指标
        st.info("正在解析图片中的健康指标...")
        metrics = parser.extract_health_metrics(tmp_image_path)
        
        # 显示解析结果
        st.success("指标解析完成！")
        st.subheader("识别的健康指标")
        for key, val in metrics.items():
            if val is not None:
                st.write(f"• {key}：{val}")
            else:
                st.write(f"• {key}：未识别到")

        # 健康分析逻辑（复用image_test.py中的分析逻辑）
        abnormal = []
        recommend_food = []
        avoid_food = []
        matched_recipes = []

        # 过滤有效指标
        valid_metrics = {k: v for k, v in metrics.items() if v is not None}
        if not valid_metrics:
            st.warning("未识别到任何有效指标，无法分析")
        else:
            # 分析指标是否异常
            for ind, val in valid_metrics.items():
                row = df_ind[df_ind["indicator"] == ind]
                if not row.empty:
                    row = row.iloc[0]
                    min_val = float(row["normal_min"])
                    max_val = float(row["normal_max"])
                    
                    if val > max_val:
                        abnormal.append(f"{ind} 偏高（{val}）：{row['high_risk']}")
                        recommend_food.extend(row["recommend_food"].split("、"))
                        avoid_food.extend(row["avoid_food"].split("、"))
                    elif val < min_val:
                        abnormal.append(f"{ind} 偏低（{val}）：{row['low_risk']}")
                        recommend_food.extend(row["recommend_food"].split("、"))
                        avoid_food.extend(row["avoid_food"].split("、"))

            # 去重食材列表
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

            # 显示分析结果
            st.subheader("健康风险分析")
            if not abnormal:
                st.success("🎉 所有指标均在正常范围，继续保持健康生活！")
            else:
                for a in abnormal:
                    st.warning(f"• {a}")

            st.subheader("推荐食材")
            st.write("、".join(recommend_food) if recommend_food else "无特殊推荐")

            st.subheader("禁忌食材")
            st.write("、".join(avoid_food) if avoid_food else "无特殊禁忌")

            st.subheader("个性化药膳方案")
            if matched_recipes:
                for i, r in enumerate(matched_recipes[:3], 1):
                    st.write(f"{i}. **{r['name']}**")
                    st.write(f"   材料：{r['material']}")
                    st.write(f"   食用建议：{r['usage']}")
            else:
                st.info("暂无匹配的药膳方案")

    except Exception as e:
        st.error(f"解析失败：{str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(tmp_image_path):
            os.remove(tmp_image_path)
