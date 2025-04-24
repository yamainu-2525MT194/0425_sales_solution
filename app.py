# --- ファイル名: app.py（OpenAI v1 対応 + GPT-3.5 使用版）

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from openai import OpenAI
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
from lightgbm import LGBMRegressor
from google.oauth2 import service_account
from googleapiclient.discovery import build

st.set_page_config(page_title="営業AI分析ツール（ローカル対応）", layout="wide")
st.title("📊 営業AI分析ダッシュボード（CSV or Google Docs対応）")

# --- 認証情報の読み込み（ローカル .streamlit/secrets.toml） ---
credentials_dict = json.loads(st.secrets["gspread_credentials"])
credentials = service_account.Credentials.from_service_account_info(
    credentials_dict,
    scopes=["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/documents.readonly"]
)
client_openai = OpenAI(api_key=st.secrets["openai_api_key"])

# --- 入力形式の選択 ---
input_type = st.radio("データの種類を選択", ["Googleスプレッドシート（CSV）", "Googleドキュメント（日報形式）"])

# --- データ読み込み ---
df = None
free_text = ""

if input_type == "Googleスプレッドシート（CSV）":
    sheet_url = st.text_input("📎 Googleスプレッドシート（CSVリンク）")
    if sheet_url and "export?format=csv" in sheet_url:
        try:
            df = pd.read_csv(sheet_url)
            df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
            start_date = st.date_input("開始日", df["日付"].min().date())
            end_date = st.date_input("終了日", df["日付"].max().date())
            df = df[(df["日付"] >= pd.to_datetime(start_date)) & (df["日付"] <= pd.to_datetime(end_date))]
            st.success(f"✅ データ {len(df)} 件を読み込みました")
        except Exception as e:
            st.error(f"CSVの読み込みエラー：{e}")

elif input_type == "Googleドキュメント（日報形式）":
    doc_input = st.text_input("📝 GoogleドキュメントID または URL を入力")
    match = re.search(r'/d/([a-zA-Z0-9-_]+)', doc_input)
    doc_id = match.group(1) if match else doc_input.strip()

    if doc_id:
        try:
            service = build('docs', 'v1', credentials=credentials)
            doc = service.documents().get(documentId=doc_id).execute()
            content = doc.get("body", {}).get("content", [])
            for c in content:
                for e in c.get("paragraph", {}).get("elements", []):
                    free_text += e.get("textRun", {}).get("content", "")
            st.text_area("📄 抽出された本文（先頭）", free_text[:1000])
        except Exception as e:
            st.error(f"ドキュメント取得エラー：{e}")

# --- 数値データ分析（CSVのみ） ---
if df is not None:
    st.header("📈 数値データ分析（CSV）")
    try:
        X = df[["荷電数", "提案数_有効"]]
        y = df["面談獲得数"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
        st.subheader("📉 面談数予測（Random Forest）")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test))):.2f}")

        y_binary = (df["提案数_有効"] > df["提案数_有効"].median()).astype(int)
        log_model = LogisticRegression().fit(X_train, y_binary)
        st.subheader("📊 提案力の分類（ロジスティック回帰）")
        st.text(classification_report(y_binary, log_model.predict(X_train)))

        kmeans = KMeans(n_clusters=3).fit(X)
        df["営業タイプ"] = kmeans.labels_
        st.subheader("🔍 営業タイプの分類（KMeans）")
        st.dataframe(df[["営業名", "営業タイプ"]].drop_duplicates())

    except Exception as e:
        st.error(f"数値分析中のエラー：{e}")

# --- GPT要約・提案セクション（共通） ---
st.header("💬 GPTによるAI要約・アドバイス")

if df is not None:
    try:
        issues = "\n".join(df["課題"].dropna().astype(str))
        good = "\n".join(df["良かった点"].dropna().astype(str))
        counter = "\n".join(df["課題を解決する為の対策"].dropna().astype(str))

        prompt = f"以下の営業報告から、成功要因・課題・対策を要約してください。\n良かった点:\n{good}\n課題:\n{issues}\n対策案:\n{counter}"

        result = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        st.markdown(result.choices[0].message.content)

    except Exception as e:
        st.error(f"GPT分析中のエラー：{e}")

elif free_text:
    prompt = f"以下の営業日報を分析し、数値成果・面談所感・顧客所感・課題・対策をまとめ、AIからのアドバイスも追加してください：\n{free_text[:3500]}"
    try:
        result = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        st.markdown(result.choices[0].message.content)
    except Exception as e:
        st.error(f"GPT処理エラー：{e}")
else:
    st.info("左からCSVまたはGoogleドキュメントを選択し、データを読み込んでください。")
