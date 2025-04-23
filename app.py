# --- ファイル名: app.py（OpenAI v1 対応 + GPT-3.5 使用版）

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
from lightgbm import LGBMRegressor
from openai import OpenAI
import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials
import datetime

# --- Google認証（secrets経由に修正） ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials_dict = json.loads(st.secrets["gspread_credentials"])
credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)
client_gs = gspread.authorize(credentials)

# --- Streamlit UI設定 ---
st.set_page_config(page_title="AI営業分析ダッシュボード", layout="wide")
st.title("🧠 AI営業分析ダッシュボード（LightGBM × GPTによる完全支援）")

# --- データ読み込み ---
st.sidebar.header("📥 データ設定")
sheet_url = st.sidebar.text_input("Google SheetsのCSVリンク（export?format=csv）")

if sheet_url and "export?format=csv" in sheet_url:
    try:
        df = pd.read_csv(sheet_url)
        df["日付"] = pd.to_datetime(df["日付"], errors="coerce")

        start_date = st.sidebar.date_input("開始日", df["日付"].min().date())
        end_date = st.sidebar.date_input("終了日", df["日付"].max().date())
        df = df[(df["日付"] >= pd.to_datetime(start_date)) & (df["日付"] <= pd.to_datetime(end_date))]

        st.success(f"✅ データ {len(df)} 件を読み込みました")

        # --- 機械学習セクション ---
        st.header("🤖 機械学習による営業分析")
        X = df[["荷電数", "提案数_有効"]]
        y = df["面談獲得数"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ランダムフォレスト回帰
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        st.subheader("📈 面談数予測：Random Forest")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")

        # ロジスティック回帰（提案力）
        st.subheader("📉 提案力分類：Logistic Regression")
        y_binary = (df["提案数_有効"] > df["提案数_有効"].median()).astype(int)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_binary, test_size=0.2, random_state=42)
        log_model = LogisticRegression().fit(X_train2, y_train2)
        y_pred2 = log_model.predict(X_test2)
        st.text(classification_report(y_test2, y_pred2))

        # クラスタリング
        st.subheader("🔍 営業タイプ分類：KMeansクラスタリング")
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
        df["営業タイプ"] = kmeans.labels_
        st.dataframe(df[["営業名", "営業タイプ"]].drop_duplicates())

        # 特徴量の重要度
        st.subheader("📊 成果に寄与する指標")
        importance = pd.Series(rf_model.feature_importances_, index=X.columns)
        st.bar_chart(importance)

        # LightGBM予測
        st.subheader("🚀 高精度予測：LightGBM")
        lgb_model = LGBMRegressor(n_estimators=100)
        lgb_model.fit(X_train, y_train)
        y_pred_lgb = lgb_model.predict(X_test)
        st.line_chart(pd.DataFrame({"LGB予測": y_pred_lgb, "実測": y_test.values}))

        # --- GPT戦略分析セクション ---
        st.header("💬 GPTによる戦略アドバイス")
        client = OpenAI(api_key=st.secrets["openai_api_key"])

        issues = "\n".join(df["課題"].dropna().astype(str))
        good_points = "\n".join(df["良かった点"].dropna().astype(str))
        solutions = "\n".join(df["対策"].dropna().astype(str))

        st.subheader("🧠 課題の傾向分析")
        prompt1 = f"以下の営業課題から共通する傾向と頻出テーマを3つ要約してください:\n{issues}"
        st.markdown(client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt1}]).choices[0].message.content)

        st.subheader("✨ 成功要因の抽出")
        prompt2 = f"以下の『良かった点』から、営業チームにとっての成功パターンを3つ抽出してください:\n{good_points}"
        st.markdown(client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt2}]).choices[0].message.content)

        st.subheader("🔧 対策案の効果と改善")
        prompt3 = f"以下の対策案を読み、効果が高い順に3つと改善提案を教えてください:\n{solutions}"
        st.markdown(client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt3}]).choices[0].message.content)

        st.subheader("📆 今月の傾向要約")
        recent_prompt = f"以下の課題と成果内容から、今月の営業活動の全体傾向を要約してください:\n{issues}\n{good_points}"
        st.markdown(client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": recent_prompt}]).choices[0].message.content)

        st.subheader("📌 営業別AIアドバイス")
        for name in df["営業名"].unique():
            personal_data = df[df["営業名"] == name]
            prompt5 = f"{name}さんの営業活動データをもとに、個別に改善アドバイスを出してください:\n{personal_data[['荷電数', '面談獲得数', '提案数_有効']].describe()}"
            result = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt5}])
            st.markdown(f"#### 👤 {name} さんへの提案:\n" + result.choices[0].message.content)

    except Exception as e:
        st.error(f"❌ エラー：{e}")
else:
    st.info("左のサイドバーに有効なGoogle SheetsのCSVリンクを入力してください。")