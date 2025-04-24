# --- ファイル名: app.py（OpenAI v1 対応 + GPT-3.5 使用版）


import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from openai import OpenAI
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from google.oauth2 import service_account
from googleapiclient.discovery import build
import gspread
from gspread_dataframe import set_with_dataframe
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

st.set_page_config(page_title="営業日報AI解析ダッシュボード（拡張300行）", layout="wide")
st.title("📊 営業日報AI解析ツール（全方位拡張）")

# --- 認証設定 ---
credentials_dict = json.loads(st.secrets["gspread_credentials"])
credentials = service_account.Credentials.from_service_account_info(
    credentials_dict,
    scopes=["https://www.googleapis.com/auth/documents.readonly", "https://www.googleapis.com/auth/spreadsheets"]
)
client_openai = OpenAI(api_key=st.secrets["openai_api_key"])

# --- ドキュメント入力 ---
doc_input = st.text_input("📄 GoogleドキュメントID または URL")
doc_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', doc_input)
doc_id = doc_id_match.group(1) if doc_id_match else doc_input.strip()

if doc_id:
    try:
        service = build('docs', 'v1', credentials=credentials)
        doc = service.documents().get(documentId=doc_id).execute()
        content = doc.get("body", {}).get("content", [])

        full_text = ""
        for c in content:
            for e in c.get("paragraph", {}).get("elements", []):
                full_text += e.get("textRun", {}).get("content", "")

        st.text_area("📋 ドキュメント内容（先頭）", full_text[:1000], height=200)

        # --- 複数人の抽出処理 ---
        sections = re.split(r"名前：(.+?)\n", full_text)
        reports = []

        for i in range(1, len(sections), 2):
            name = sections[i].strip()
            text = sections[i + 1] if i + 1 < len(sections) else ""
            data = {"名前": name}
            matches = re.findall(r"(オファー数|面談獲得数|提案数（有効）|提案数（有効・無効・不明）|配信数|荷電数|面談実施数)：([0-9]+)件", text)
            for label, value in matches:
                label = label.replace("（有効）", "_有効").replace("（有効・無効・不明）", "_合計")
                data[label] = int(value)
            reports.append(data)

        df = pd.DataFrame(reports).fillna(0)
        st.dataframe(df)

        # --- ダウンロード & 出力 ---
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 CSVとして保存", csv, "report_data.csv", "text/csv")

        sheet_url = st.text_input("📎 スプレッドシートURL（追記先）")
        if sheet_url:
            try:
                gc = gspread.authorize(credentials)
                sheet_id = re.search(r"/d/([\w-]+)", sheet_url).group(1)
                sh = gc.open_by_key(sheet_id).sheet1
                set_with_dataframe(sh, df, row=sh.row_count + 1)
                st.success("✅ 追記完了")
            except Exception as e:
                st.warning(f"スプレッドシートエラー：{e}")

        # --- 分析・可視化 ---
        st.subheader("📈 面談数回帰予測")
        if '面談獲得数' in df.columns:
            features = df.drop(columns=['面談獲得数', '名前'], errors='ignore').fillna(0)
            model = LinearRegression().fit(features, df['面談獲得数'])
            df['予測面談数'] = model.predict(features)
            st.dataframe(df[['名前', '面談獲得数', '予測面談数']])

        st.subheader("🔍 クラスタリング + PCA 可視化")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df.drop(columns=['名前'], errors='ignore'))
        kmeans = KMeans(n_clusters=2, n_init='auto').fit(scaled)
        df['営業タイプ'] = kmeans.labels_

        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled)
        pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
        pca_df['名前'] = df['名前']
        pca_df['営業タイプ'] = df['営業タイプ']

        fig, ax = plt.subplots()
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='営業タイプ', style='名前', ax=ax)
        st.pyplot(fig)

        st.subheader("📊 相関マトリクス")
        corr = df.drop(columns=['名前'], errors='ignore').corr()
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax2)
        st.pyplot(fig2)

        st.subheader("🧠 GPT：要約・改善・インサイト抽出")
        for i in range(1, len(sections), 2):
            name = sections[i].strip()
            content = sections[i + 1][:1500]
            prompt = f"以下は{name}さんの営業日報です。成果、課題、対策、工夫点、そして改善案を整理してください。\n{content}"
            response = client_openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown(f"### {name} さんへのAI分析")
            st.markdown(response.choices[0].message.content)

        # --- 追加機能：目標との乖離チェック ---
        st.subheader("🎯 各営業の目標達成度分析")
        target_offer = st.number_input("目標オファー数（全営業共通）", value=3)
        if 'オファー数' in df.columns:
            df['オファー達成率'] = df['オファー数'] / target_offer
            fig3, ax3 = plt.subplots()
            sns.barplot(data=df, x='名前', y='オファー達成率', ax=ax3)
            ax3.axhline(1, color='red', linestyle='--')
            st.pyplot(fig3)

    except Exception as e:
        st.error(f"ドキュメント処理エラー：{e}")
else:
    st.info("GoogleドキュメントURLまたはIDを入力してください。")
