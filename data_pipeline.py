import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from pygam import LogisticGAM, s
import warnings

warnings.filterwarnings("ignore")

def convert_numeric_columns(df):
    target_cols = ["평균이용금액", "평균이용건수"]
    for col in target_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    if {"행정동명", "행정동코드"}.issubset(df.columns):
        df["행정동명"] = df["행정동명"].astype(str).str.strip().replace("", pd.NA)
        mask = df["행정동명"].isna() & (df["행정동코드"] == 2644059000)
        df.loc[mask, "행정동명"] = "강서구 신호동"
        
    if "기준년월" in df.columns:
        df["기준년월"] = pd.to_datetime(df["기준년월"].astype(str).str.strip(), format="%Y-%m", errors="coerce")
        df["연도"] = df["기준년월"].dt.year
        df["월"] = df["기준년월"].dt.month
    return df

def calculate_growth_by_date(df, month_offset):
    past_df = df[['행정동명', '기준년월', 'total_amount']].copy()
    past_df['기준년월'] = past_df['기준년월'] + pd.DateOffset(months=month_offset)
    past_df = past_df.rename(columns={'total_amount': f'past_{month_offset}m_amount'})
    merged = pd.merge(df, past_df, on=['행정동명', '기준년월'], how='left')
    growth = (merged['total_amount'] - merged[f'past_{month_offset}m_amount']) / merged[f'past_{month_offset}m_amount']
    return growth

def calculate_strict_cv(df, window_months):
    merged = df[['행정동명', '기준년월', 'total_amount']].copy()
    merged = merged.rename(columns={'total_amount': 'amt_0m'})
    amt_cols = ['amt_0m']
    
    for i in range(1, window_months):
        past_df = df[['행정동명', '기준년월', 'total_amount']].copy()
        past_df['기준년월'] = past_df['기준년월'] + pd.DateOffset(months=i)
        col_name = f'amt_{i}m'
        past_df = past_df.rename(columns={'total_amount': col_name})
        merged = pd.merge(merged, past_df, on=['행정동명', '기준년월'], how='left')
        amt_cols.append(col_name)
        
    is_fully_contiguous = merged[amt_cols].notna().all(axis=1)
    rolling_std = merged[amt_cols].std(axis=1)
    rolling_mean = merged[amt_cols].mean(axis=1)
    cv = rolling_std / rolling_mean
    return cv.where(is_fully_contiguous, np.nan)

def get_future_value_by_date(df, month_offset, target_col):
    return df.groupby("gu_dong")[target_col].shift(-month_offset)

def main():
    print("🚀 [1/6] 데이터 로드 및 전처리 시작...")
    # =====================================================
    # 1. 데이터 로드 및 전처리
    # =====================================================
    time_df = pd.read_excel("일별 행정동 시간 소비매출 월별 일평균.xlsx")
    business_df = pd.read_excel("일별 행정동 업종(대) 소비매출 월별 일평균.xlsx")

    time_df = convert_numeric_columns(time_df)
    if "시간대" in time_df.columns:
        time_df["시간대"] = time_df["시간대"].astype(str).str.extract(r"(\d+)").astype(int)
    business_df = convert_numeric_columns(business_df)

    for df in [time_df, business_df]:
        df.loc[:, "기준년월"] = pd.to_datetime(df["기준년월"], format='%Y-%m')

    # base_df (total_amount 생성)
    time_total = time_df.groupby(["기준년월", "행정동명"])["평균이용금액"].sum().rename("time_amount")
    base_df = time_total.reset_index()
    base_df["total_amount"] = base_df["time_amount"]

    # 파생변수 생성
    base_df['growth_1m']  = calculate_growth_by_date(base_df, 1)
    base_df['growth_3m']  = calculate_growth_by_date(base_df, 3)
    base_df['growth_12m'] = calculate_growth_by_date(base_df, 12)
    base_df["cv_6m"] = calculate_strict_cv(base_df, 6)
    base_df["cv_12m"] = calculate_strict_cv(base_df, 12)

    # 업종 비중 (HHI 등)
    business_month = business_df.groupby(["기준년월", "행정동명", "업종대분류"])["평균이용금액"].sum().reset_index()
    business_month["전체"] = business_month.groupby(["기준년월", "행정동명"])["평균이용금액"].transform("sum")
    business_month["ratio"] = business_month["평균이용금액"] / business_month["전체"]

    industry_pivot = business_month.pivot_table(index=["기준년월", "행정동명"], columns="업종대분류", values="ratio").reset_index()
    industry_pivot = industry_pivot.rename(columns={
        "교육":"education_ratio", "미용":"beauty_ratio", "생활":"living_ratio",
        "여가/문화":"culture_ratio", "여행/숙박":"travel_ratio", "유통":"retail_ratio",
        "음식/주점":"restaurant_ratio", "음식료픔":"grocery_ratio", "의료":"medical_ratio",
        "의류/잡화":"fashion_ratio", "자동차":"automobile_ratio"
    })
    
    industry_cols = ["education_ratio", "beauty_ratio", "living_ratio", "culture_ratio", "travel_ratio",
                     "retail_ratio", "restaurant_ratio", "grocery_ratio", "medical_ratio", "fashion_ratio", "automobile_ratio"]
    industry_pivot = industry_pivot.sort_values(["행정동명", "기준년월"])
    industry_pivot[industry_cols] = industry_pivot[industry_cols].fillna(0)
    industry_pivot["hhi_index"] = industry_pivot[industry_cols].pow(2).sum(axis=1)

    # 병합
    df_market_dynamics = base_df.merge(
        industry_pivot[["기준년월", "행정동명"] + industry_cols + ["hhi_index"]],
        on=["기준년월", "행정동명"], how="left"
    )
    df_market_dynamics = df_market_dynamics[df_market_dynamics["기준년월"] >= "2024-01-01"].copy()
    df_market_dynamics = df_market_dynamics.rename(columns={"행정동명":"gu_dong", "기준년월":"y_m"})
    df_market_dynamics = df_market_dynamics.drop(columns=["time_amount"])

    # 반올림 및 결측치 대체
    numeric_cols = df_market_dynamics.select_dtypes(include="number").columns
    round_cols = [col for col in numeric_cols if col not in ["y_m", "gu_dong"]]
    df_market_dynamics[round_cols] = df_market_dynamics[round_cols].round(4)

    growth_12m_median = df_market_dynamics["growth_12m"].median()
    cv_12m_median = df_market_dynamics["cv_12m"].median()
    df_market_dynamics["growth_12m_imputed"] = df_market_dynamics["growth_12m"].fillna(growth_12m_median)
    df_market_dynamics["cv_12m_imputed"] = df_market_dynamics["cv_12m"].fillna(cv_12m_median)

    print("📊 [2/6] K-Means 군집화 진행 중...")
    # =====================================================
    # 2. K-Means 클러스터링
    # =====================================================
    cluster_tasks = {
        "상권 건강도": ["total_amount", "growth_1m", "growth_3m", "growth_12m_imputed", "cv_6m", "cv_12m_imputed", "hhi_index"],
        "업종 구조": industry_cols,
        "구조 전환": ["growth_3m", "growth_12m_imputed", "cv_6m", "cv_12m_imputed", "hhi_index"] + industry_cols,
    }
    best_k = {"상권 건강도": 3, "업종 구조": 4, "구조 전환": 3}

    for task_name, features in cluster_tasks.items():
        col_name = "health_cluster" if task_name == "상권 건강도" else "industry_cluster" if task_name == "업종 구조" else "transform_cluster"
        sub_df = df_market_dynamics.dropna(subset=features).copy()
        X_scaled = StandardScaler().fit_transform(sub_df[features])
        km = KMeans(n_clusters=best_k[task_name], random_state=42, n_init=20)
        sub_df[col_name] = km.fit_predict(X_scaled)
        df_market_dynamics[col_name] = sub_df[col_name]

    df_market_dynamics["health_cluster_name"] = df_market_dynamics["health_cluster"].map({0.0: "대형 정체형 상권", 1.0: "소형 안정형 상권", 2.0: "폭발 성장형 상권"})
    df_market_dynamics["industry_cluster_name"] = df_market_dynamics["industry_cluster"].map({0: "소매 및 생활밀착형", 1: "주거/리빙 특화형", 2: "메디컬 특화형", 3: "외식/음식점 중심형"})
    df_market_dynamics["transform_cluster_name"] = df_market_dynamics["transform_cluster"].map({0.0: "외식·라이프 전환형", 1.0: "메디컬 중심 고착형", 2.0: "리빙·소매 다각화형"})

    print("🤖 [3/6] Random Forest 조기경보 모델 학습 중...")
    # =====================================================
    # 3. 예측 모델링 (Random Forest)
    # =====================================================
    df_market_dynamics = df_market_dynamics.sort_values(["gu_dong", "y_m"]).copy()
    df_market_dynamics["future_total_amount_6m_later"] = get_future_value_by_date(df_market_dynamics, 6, "total_amount")
    df_market_dynamics["target_down_6m"] = np.where(df_market_dynamics["future_total_amount_6m_later"] < df_market_dynamics["total_amount"], 1, 0)

    RF_FEATURES = ["total_amount", "growth_1m", "growth_3m", "growth_12m_imputed", "cv_6m", "cv_12m_imputed", "hhi_index"] + industry_cols
    
    df_model = df_market_dynamics.dropna(subset=["future_total_amount_6m_later"]).copy()
    df_model = df_model.dropna(subset=RF_FEATURES + ["target_down_6m"]).copy()

    unique_dates = sorted(df_model["y_m"].unique())
    split_idx = len(unique_dates) - 6
    train_dates = unique_dates[:split_idx]

    is_train = df_model["y_m"].isin(train_dates)
    X_train, X_test = df_model.loc[is_train, RF_FEATURES], df_model.loc[~is_train, RF_FEATURES]
    y_train, y_test = df_model.loc[is_train, "target_down_6m"], df_model.loc[~is_train, "target_down_6m"]

    rf_classifier = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=15, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)

    y_pred_prob = rf_classifier.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    # 최신월 데이터 추출
    latest_month = df_market_dynamics["y_m"].max()
    df_latest = df_market_dynamics.query("y_m == @latest_month").copy()
    df_latest["risk_probability"] = rf_classifier.predict_proba(df_latest[RF_FEATURES])[:, 1]

    print("📈 [4/6] GAM 위험요인 분석 진행 중...")
    # =====================================================
    # 4. GAM 분석 및 위험 규칙 추출
    # =====================================================
    RF_FEATURES_FINAL = ["growth_3m", "growth_1m", "cv_6m", "growth_12m_imputed", "cv_12m_imputed"]
    X_gam = df_model[RF_FEATURES_FINAL].values
    y_gam = df_model["target_down_6m"].values

    gam = LogisticGAM(s(0) + s(1) + s(2) + s(3) + s(4)).gridsearch(X_gam, y_gam)
    pvals = gam.statistics_["p_values"]
    significant_idx = [i for i, p in enumerate(pvals[:-1]) if p < 0.05]

    risk_rules, feature_direction, gam_curve_data = {}, {}, {}

    for i in significant_idx:
        feature_name = RF_FEATURES_FINAL[i]
        grid = gam.generate_X_grid(term=i)
        effect = gam.partial_dependence(term=i)
        probability = 1 / (1 + np.exp(-effect))
        x = grid[:, i]
        
        gam_curve_data[feature_name] = {"x": x.tolist(), "probability": probability.tolist()}
        
        gradient = np.gradient(effect)
        gradient_smooth = pd.Series(gradient).rolling(5, center=True, min_periods=1).mean().values
        threshold_idx = np.argmax(gradient_smooth)
        threshold = x[threshold_idx]
        threshold_probability = probability[threshold_idx]

        risk_rules[feature_name] = {"threshold": threshold, "threshold_probability": threshold_probability}
        feature_direction[feature_name] = "increase" if effect[-1] > effect[0] else "decrease"

    print("🏷️ [5/6] 최신월 위험 등급 및 메시지 생성 중...")
    # =====================================================
    # 5. 최신월 데이터 처리 (위험태그 및 경보 메시지)
    # =====================================================
    def get_risk_tags(row):
        tags = []
        for feature, rule in risk_rules.items():
            val = row[feature]
            if feature_direction[feature] == "increase" and val >= rule["threshold"]: tags.append(feature)
            elif feature_direction[feature] == "decrease" and val <= rule["threshold"]: tags.append(feature)
        return tags

    df_latest["risk_tags"] = df_latest.apply(get_risk_tags, axis=1)
    df_latest["risk_tag_count"] = df_latest["risk_tags"].apply(len)
    df_latest["risk_tags_text"] = df_latest["risk_tags"].apply(lambda x: ", ".join(x) if len(x) > 0 else "-")

    def final_grade(row):
        prob, tag_count = row["risk_probability"], row["risk_tag_count"]
        if prob >= 0.80 and tag_count >= 3: return "긴급"
        elif prob >= 0.70 and tag_count >= 2: return "위험"
        elif prob >= 0.50 and tag_count >= 1: return "주의"
        elif prob >= 0.50 and tag_count == 0: return "관찰"
        else: return "안정"

    df_latest["final_grade"] = df_latest.apply(final_grade, axis=1)

    def warning_message(row):
        grade, prob, tags = row["final_grade"], row["risk_probability"] * 100, row["risk_tag_count"]
        if grade == "긴급": return f"🔴 긴급 | 위험확률 {prob:.1f}% | 위험태그 {tags}개"
        elif grade == "위험": return f"🟠 위험 | 위험확률 {prob:.1f}% | 위험태그 {tags}개"
        elif grade == "주의": return f"🟡 주의 | 위험확률 {prob:.1f}% | 위험태그 {tags}개"
        elif grade == "관찰": return f"⚠️ 관찰 | 위험확률 {prob:.1f}%"
        return "✅ 안정"

    df_latest["warning_message"] = df_latest.apply(warning_message, axis=1)

    dashboard_result = df_latest[[
        "gu_dong", "y_m", "health_cluster_name", "industry_cluster_name", 
        "transform_cluster_name", "risk_probability", "risk_tag_count", 
        "risk_tags_text", "final_grade", "warning_message"
    ]].sort_values("risk_probability", ascending=False).reset_index(drop=True)

    print("💾 [6/6] 분석 결과 Pickle 저장 중...")
    # =====================================================
    # 6. Streamlit용 결과 파일 저장 (Pickle)
    # =====================================================
    export_data = {
        "dashboard_result": dashboard_result,
        "df_latest": df_latest,
        "roc_auc": roc_auc,
        "latest_month": latest_month.strftime("%Y-%m"), # Streamlit 표시용
        "risk_rules": risk_rules,
        "feature_direction": feature_direction,
        "gam_curve_data": gam_curve_data
    }

    with open("market_warning_data.pkl", "wb") as f:
        pickle.dump(export_data, f)
        
    print("✅ 완료! 'market_warning_data.pkl' 파일이 성공적으로 생성되었습니다.")

if __name__ == "__main__":
    main()