import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ==========================================================
# 0. Streamlit 페이지 설정
# ==========================================================
st.set_page_config(
    page_title="AI 상권 조기경보 시스템",
    page_icon="🚨",
    layout="wide"
)

import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st

def set_korean_font():
    # 1. 현재 파일 위치 기준 fonts/malgun.ttf 절대 경로 생성
    current_dir = os.path.dirname(__file__)
    font_path = os.path.join(current_dir, 'fonts', 'malgun.ttf')
    
    # 2. 파일이 진짜 해당 경로에 있는지 검사
    if not os.path.exists(font_path):
        st.error(f"🚨 폰트 파일을 찾을 수 없습니다. 경로를 확인해주세요: {font_path}")
        return
        
    try:
        # 3. Matplotlib 폰트 매니저에 파일 추가
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        
        # 4. 가장 확실한 rcParams 방식으로 전역 폰트 강제 지정
        plt.rcParams['font.family'] = font_name
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        
        # 💡 성공적으로 로드되었는지 대시보드 상단에 띄워 확인하기 (확인 후 삭제 가능)
        st.sidebar.success(f"🎵 폰트 로드 완료: {font_name}")
        
    except Exception as e:
        st.error(f"폰트 설정 중 오류 발생: {e}")

# 앱 최상단에서 호출
set_korean_font()

# ==========================================================
# 1. 데이터 로드 (캐싱 적용)
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

dashboard_result = data["dashboard_result"]
df_latest = data["df_latest"]
roc_auc = data["roc_auc"]
latest_month = data["latest_month"]
risk_rules = data["risk_rules"]
feature_direction = data["feature_direction"]
gam_curve_data = data["gam_curve_data"]

# ==========================================================
# 2. 사이드바 (Sidebar)
# ==========================================================
st.sidebar.title("🚨 상권 조기경보 탐색기")
st.sidebar.write(f"**기준 월:** {latest_month}")
st.sidebar.divider()

dongs = sorted(dashboard_result["gu_dong"].unique())
selected_dong = st.sidebar.selectbox("🔎 분석할 행정동을 선택하세요:", dongs)

row = dashboard_result[dashboard_result["gu_dong"] == selected_dong].iloc[0]
current_row = df_latest[df_latest["gu_dong"] == selected_dong].iloc[0]

# ==========================================================
# 🌟 3. 최상단 마스터 대시보드 (전체 요약) - 신규 추가 영역
# ==========================================================
st.title("📊 상권 위축 조기경보 통합 모니터링")
with st.expander("ℹ️ 조기경보 시스템 안내 및 모니터링 기법", expanded=False):
    st.markdown("""
    **본 대시보드는 Random Forest와 GAM(일반화 가법 모델) 알고리즘을 융합한 AI 예측 시스템입니다.**
    * **분석 방식:** 최근 매출 증감률, 변동성 등 다각적 지표를 분석해 6개월 뒤의 상권 위축 위험 확률을 도출합니다.
    * **경보 기준:** 위축 확률에 따라 `🚨 긴급`, `🔴 위험`, `🟡 주의`, `🟢 관찰/안정` 단계로 분류합니다.
    * **활용 방법:** 최상단의 리스크 요약표를 통해 우선 대응이 필요한 행정동을 선별한 후, 사이드바에서 해당 동을 선택하여 하단에서 개별 징후를 심층 분석하세요.
    """)

st.write("---")

# 실시간 리스크 요약 KPI 생성
st.subheader("📌 실시간 리스크 요약")
counts = dashboard_result["final_grade"].value_counts()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("🚨 긴급 지역", f"{counts.get('긴급', 0)}개", delta="즉각 조치 요망", delta_color="inverse")
kpi2.metric("🔴 위험 지역", f"{counts.get('위험', 0)}개", delta="집중 모니터링", delta_color="inverse")
kpi3.metric("🟡 주의 지역", f"{counts.get('주의', 0)}개", delta="선제 관찰", delta_color="off")
kpi4.metric("🟢 관찰/안정 지역", f"{counts.get('관찰', 0) + counts.get('안정', 0)}개", delta="안정적", delta_color="normal")
st.write(" ")

st.subheader("🎯 중점 관리 대상 상권 리스트")

# 긴급, 위험, 주의 등급만 필터링 후 확률 높은 순으로 정렬
risk_filtered_df = dashboard_result[dashboard_result["final_grade"].isin(["긴급", "위험", "주의"])].copy()
risk_filtered_df = risk_filtered_df.sort_values("risk_probability", ascending=False)

if not risk_filtered_df.empty:
    st.warning(f"⚠️ 현재 모니터링 대상 중 **총 {len(risk_filtered_df)}개**의 상권에서 위축 징후가 감지되었습니다.")
    
    # UI 출력용 데이터 세팅
    display_df = risk_filtered_df[["gu_dong", "risk_probability", "final_grade", "risk_tag_count", "risk_tags_text"]].copy()
    display_df["risk_probability_pct"] = display_df["risk_probability"] * 100
    display_df = display_df[["gu_dong", "risk_probability_pct", "final_grade", "risk_tag_count", "risk_tags_text"]]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "gu_dong": st.column_config.TextColumn("행정동", width="medium"),
            "risk_probability_pct": st.column_config.ProgressColumn(
                "위축 예측 확률", 
                format="%.1f%%",
                min_value=0, 
                max_value=100
            ),
            "final_grade": st.column_config.TextColumn("상태", width="small"),
            "risk_tag_count": st.column_config.NumberColumn("위험 지표 수", format="%d 개"),
            "risk_tags_text": st.column_config.TextColumn("감지된 핵심 위험 징후", width="large")
        }
    )
else:
    st.success("🎉 현재 '주의' 이상의 리스크가 감지된 상권이 없습니다. 모든 지역이 안정적인 상태입니다.")

st.markdown("<br><br><hr>", unsafe_allow_html=True)

# ==========================================================
# 4. 개별 행정동 상세 화면 구성 (기존 대시보드 영역)
# ==========================================================
st.title(f"🏢 {selected_dong} 상권 상세 분석 리포트")
st.write("사이드바에서 선택한 행정동의 세부 지표와 GAM 곡선을 확인합니다.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="최종 경보 등급", value=row['final_grade'])
with col2:
    st.metric(label="상권 위축 위험 확률", value=f"{row['risk_probability']*100:.1f}%")
with col3:
    st.metric(label="발견된 위험 태그 수", value=f"{row['risk_tag_count']} 개")

st.divider()

# 상세 분석 탭 구성
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 상권 현황", "📊 위험도 분석 (RF)", "🔍 위험요인 분석 (GAM)", "🚨 최종 조기경보", "⚙️ 모델 성능 지표"
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
    """)
    st.progress(float(row['risk_probability']))

# ----------------------------------------------------------
# Tab 3: 위험요인 분석 (GAM)
# ----------------------------------------------------------
with tab3:
    st.subheader("🔍 Risk Factor Analysis (GAM)")
    st.write("Compare the current commercial district status with the threshold where risk increases sharply (Risk Baseline 50%).")

    if len(risk_rules) == 0:
        st.success("No statistically significant risk factors found.")
    else:
        # 🎯 영문 매핑
        feature_desc = {
            "growth_3m": "3M Spending Growth Rate",
            "growth_1m": "1M Spending Growth Rate",
            "cv_6m": "6M Spending Volatility",
            "growth_12m_imputed": "1Y Spending Growth Rate",
            "cv_12m_imputed": "1Y Spending Volatility"
        }

        cols = st.columns(2)
        idx = 0
        
        for feature, rule in risk_rules.items():
            thresholds = rule.get("all_thresholds", [rule["threshold"]])
            value = current_row[feature]
            
            x = np.array(gam_curve_data[feature]["x"])
            prob_curve = np.array(gam_curve_data[feature]["probability"])
            current_prob = np.interp(value, x, prob_curve)
            
            is_risk = (current_prob >= 0.5)
            status = "🔴 RISK" if is_risk else "🟢 NORMAL"
            
            with cols[idx % 2]:
                st.markdown(f"#### {feature_desc.get(feature, feature)}")
                
                if len(thresholds) == 3:
                    threshold_text = f"{thresholds[0]:.3f} / {thresholds[1]:.3f} / {thresholds[2]:.3f} (50% Intersection)"
                elif len(thresholds) == 2:
                    threshold_text = f"{thresholds[0]:.3f} ~ {thresholds[1]:.3f} Range"
                elif len(thresholds) == 1:
                    direction = feature_direction[feature]
                    threshold_text = f"Over {thresholds[0]:.3f}" if direction == "increase" else f"Under {thresholds[0]:.3f}"
                else:
                    threshold_text = "No condition met"
                    
                st.write(f"**Status:** {status} | **Current:** `{value:.4f}` (Thresholds: `{threshold_text}`)")
                
                # 그래프 시각화
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(x, prob_curve * 100, linewidth=2, color="#1f77b4")
                ax.fill_between(x, prob_curve * 100, alpha=0.15, color="#1f77b4")
                
                ax.axhline(50.0, linestyle="--", color="red", alpha=0.7)
                
                for i, th in enumerate(thresholds):
                    if np.isfinite(th):
                        # 🎯 [핵심 수정] 빈 문자열 "" 대신 None을 주어 Matplotlib 범례(legend)에 아예 잡히지 않도록 차단
                        label = "Risk Threshold" if i == 0 else None
                        ax.axvline(th, linestyle=":", color="red", alpha=0.8, label=label)
                
                ax.scatter(value, current_prob * 100, s=100, color="orange", edgecolor="black", zorder=5, label="Current Status")
                
                # 🎯 그래프 내부 라벨 영문 선언
                ax.set_ylabel("Contraction Probability (%)")
                ax.set_xlabel(feature_desc.get(feature, feature))
                ax.set_ylim(-5, 105)
                ax.grid(alpha=0.3)
                
                # 🎯 [핵심 추가] 범례 텍스트에 들어오는 결측치들을 깨끗하게 무시하도록 세팅
                ax.legend(loc="upper right")
                
                # 🎯 [핵심 추가] 스트림릿 고유 테마가 Matplotlib 스타일을 오염시키지 않도록 차단하는 옵션 (st.pyplot 내부에 선언)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig) 
                st.write("---")
            idx += 1

# ----------------------------------------------------------
# Tab 4: 최종 조기경보
# ----------------------------------------------------------
with tab4:
    st.subheader("🚨 종합 조기경보 결과")
    
    grade = row['final_grade']
    if grade == "긴급": st.error(f"### {row['warning_message']}")
    elif grade == "위험": st.warning(f"### {row['warning_message']}")
    elif grade in ["주의", "관찰"]: st.info(f"### {row['warning_message']}")
    else: st.success(f"### {row['warning_message']}")
    
    st.write(f"**부여된 위험 태그:** {row['risk_tags_text']}")

# ----------------------------------------------------------
# Tab 5: 모델 성능 지표
# ----------------------------------------------------------
with tab5:
    st.subheader("⚙️ AI 모델 아키텍처 및 성능 스펙")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(label="머신러닝 모델 설명력 (ROC-AUC)", value=f"{roc_auc:.3f}")
    with m_col2:
        st.metric(label="위험 판단 기준선 (GAM)", value="통계적 유의성 50% 돌파 구간")
        
    st.markdown("""
    **조기경보 의사결정 매트릭스 테이블:**
    
    | 최종 부여 등급 | 예측 모델(RF) 조건 | 통계 모델(GAM) 위험태그 조건 |
    | :--- | :--- | :--- |
    | **🚨 긴급 (Emergency)** | 위축 위험 확률 80% 이상 | 감지된 위험 지표 태그 3개 이상 |
    | **🔴 위험 (Danger)** | 위축 위험 확률 70% 이상 | 감지된 위험 지표 태그 2개 이상 |
    | **🟡 주의 (Warning)** | 위축 위험 확률 50% 이상 | 감지된 위험 지표 태그 1개 이상 |
    | **🟢 관찰 (Observation)** | 위축 위험 확률 50% 이상 | 감지된 위험 지표 태그 없음 (0개) |
    | **🟢 안정 (Normal)** | 위축 위험 확률 50% 미만 | 조건 무관 |
    """)