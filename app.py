import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ==========================================================
# 0. Streamlit 페이지 설정 (가장 먼저 와야 함)
# ==========================================================
st.set_page_config(
    page_title="AI 상권 조기경보 시스템",
    page_icon="🚨",
    layout="wide"
)

# 한글 폰트 설정 (Mac은 'AppleGothic', Windows는 'Malgun Gothic')
plt.rcParams["font.family"] = "Malgun Gothic" 
plt.rcParams["axes.unicode_minus"] = False

# ==========================================================
# 1. 데이터 로드 (캐싱 적용으로 속도 최적화)
# ==========================================================
@st.cache_data
def load_data():
    with open("market_warning_data.pkl", "rb") as f:
        data = pickle.load(f)
    return data

try:
    data = load_data()
except FileNotFoundError:
    st.error("🚨 'market_warning_data.pkl' 파일을 찾을 수 없습니다. 먼저 data_pipeline.py를 실행해주세요.")
    st.stop()

# 변수 언패킹
dashboard_result = data["dashboard_result"]
df_latest = data["df_latest"]
roc_auc = data["roc_auc"]
latest_month = data["latest_month"]
risk_rules = data["risk_rules"]
feature_direction = data["feature_direction"]
gam_curve_data = data["gam_curve_data"]

# ==========================================================
# 2. 사이드바 (Sidebar) - 행정동 선택
# ==========================================================
st.sidebar.title("🚨 상권 조기경보 탐색기")
st.sidebar.write(f"**기준 월:** {latest_month}")
st.sidebar.divider()

dongs = sorted(dashboard_result["gu_dong"].unique())
selected_dong = st.sidebar.selectbox("🔎 분석할 행정동을 선택하세요:", dongs)

# 선택된 행정동 데이터 필터링
row = dashboard_result[dashboard_result["gu_dong"] == selected_dong].iloc[0]
current_row = df_latest[df_latest["gu_dong"] == selected_dong].iloc[0]

# ==========================================================
# 3. 메인 화면 구성
# ==========================================================
st.title(f"🏢 {selected_dong} 상권 분석 리포트")
st.write("Random Forest와 GAM을 결합한 AI 기반 상권 위축 조기경보 대시보드입니다.")

# 상단 요약 지표 (Metrics)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="최종 경보 등급", value=row['final_grade'])
with col2:
    st.metric(label="상권 위축 위험 확률", value=f"{row['risk_probability']*100:.1f}%")
with col3:
    st.metric(label="발견된 위험 태그 수", value=f"{row['risk_tag_count']} 개")

st.divider()

# 탭 구성
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 상권 현황", "📊 위험도 분석 (RF)", "🔍 위험요인 분석 (GAM)", "🚨 최종 조기경보", "ℹ️ 시스템 설명"
])

# ----------------------------------------------------------
# Tab 1: 상권 현황
# ----------------------------------------------------------
with tab1:
    st.subheader("📍 상권 페르소나 현황")
    c1, c2, c3 = st.columns(3)
    c1.info(f"**상권 건강도**\n\n{row['health_cluster_name']}")
    c2.success(f"**업종 유형**\n\n{row['industry_cluster_name']}")
    c3.warning(f"**구조 전환**\n\n{row['transform_cluster_name']}")

# ----------------------------------------------------------
# Tab 2: 위험도 분석 (RF)
# ----------------------------------------------------------
with tab2:
    st.subheader("📊 위험도 분석 (Random Forest)")
    st.write(f"현재 소비 구조가 과거 6개월 뒤 위축되었던 상권들과 얼마나 유사한지를 확률로 나타냅니다.")
    
    st.markdown(f"""
    * **현재 상권 위험확률:** `{row['risk_probability']*100:.1f}%`
    * **현재 등급 판정:** `{row['final_grade']}`
    * **모델 설명력(ROC-AUC):** `{roc_auc:.3f}`
    """)
    st.progress(float(row['risk_probability']))

# ----------------------------------------------------------
# Tab 3: 위험요인 분석 (GAM)
# ----------------------------------------------------------
with tab3:
    st.subheader("🔍 위험요인 분석 (GAM)")
    st.write("위험확률이 급격히 증가하는 구간(위험 기준선)과 현재 상권의 상태를 비교합니다.")

    if len(risk_rules) == 0:
        st.success("현재 통계적으로 유의한 위험 요인이 발견되지 않았습니다.")
    else:
        feature_desc = {
            "growth_3m": "최근 3개월 매출 변화율",
            "growth_1m": "최근 1개월 매출 변화율",
            "cv_6m": "최근 6개월 매출 변동성",
            "growth_12m_imputed": "최근 1년 매출 변화율",
            "cv_12m_imputed": "최근 1년 매출 변동성"
        }

        # 2열로 그래프 배치
        cols = st.columns(2)
        idx = 0
        
        for feature, rule in risk_rules.items():
            threshold = rule["threshold"]
            threshold_prob = rule["threshold_probability"]
            value = current_row[feature]
            
            x = np.array(gam_curve_data[feature]["x"])
            prob_curve = np.array(gam_curve_data[feature]["probability"])
            current_prob = np.interp(value, x, prob_curve)
            direction = feature_direction[feature]
            
            is_risk = (direction == "increase" and value >= threshold) or \
                      (direction == "decrease" and value <= threshold)
            status = "🔴 위험" if is_risk else "🟢 정상"
            
            with cols[idx % 2]:
                st.markdown(f"#### {feature_desc.get(feature, feature)}")
                st.write(f"**판정:** {status} | **현재값:** `{value:.4f}` (기준: `{threshold:.4f}`)")
                
                # Matplotlib 그래프 생성
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(x, prob_curve * 100, linewidth=2, color="#1f77b4")
                ax.fill_between(x, prob_curve * 100, alpha=0.2, color="#1f77b4")
                ax.axhline(threshold_prob * 100, linestyle="--", color="red", alpha=0.5)
                ax.axvline(threshold, linestyle=":", color="red", label="위험 기준선")
                ax.scatter(value, current_prob * 100, s=100, color="orange", edgecolor="black", zorder=5, label="현재 상권")
                
                ax.set_ylabel("위축 확률 (%)")
                ax.set_xlabel(feature_desc.get(feature, feature))
                ax.legend()
                
                st.pyplot(fig)
                st.write("---")
            idx += 1

# ----------------------------------------------------------
# Tab 4: 최종 조기경보
# ----------------------------------------------------------
with tab4:
    st.subheader("🚨 종합 조기경보 결과")
    
    # 등급에 따른 색상/아이콘 적용
    grade = row['final_grade']
    if grade == "긴급": st.error(f"### {row['warning_message']}")
    elif grade == "위험": st.warning(f"### {row['warning_message']}")
    elif grade in ["주의", "관찰"]: st.info(f"### {row['warning_message']}")
    else: st.success(f"### {row['warning_message']}")
    
    st.write(f"**부여된 위험 태그:** {row['risk_tags_text']}")

# ----------------------------------------------------------
# Tab 5: 시스템 설명
# ----------------------------------------------------------
with tab5:
    st.subheader("ℹ️ AI 상권 조기경보 시스템이란?")
    st.markdown("""
    **목표:** 현재 소비 구조를 기반으로 **6개월 뒤 해당 상권의 매출이 감소할 가능성**을 예측합니다.
    * **위축(1):** 6개월 후 매출 < 현재 매출
    * **정상(0):** 6개월 후 매출 ≥ 현재 매출
    
    **조기경보 기준 테이블:**
    | 등급 | 조건 |
    |---|---|
    | 안정 | RF < 50% |
    | 관찰 | RF ≥ 50% + 위험태그 0개 |
    | 주의 | RF ≥ 50% + 위험태그 1개 이상 |
    | 위험 | RF ≥ 70% + 위험태그 2개 이상 |
    | 긴급 | RF ≥ 80% + 위험태그 3개 이상 |
    """)