# --- ファイル名: app.py（OpenAI v1 対応 + GPT-3.5 使用版）

import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import openai
from sklearn.preprocessing import StandardScaler
from google.oauth2 import service_account
from googleapiclient.discovery import build
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="営業日報ダッシュボード", layout="wide")
st.title("🧠 国家品質：完全営業ダッシュボード（全軸ラベル・偏差値・AI要約）")

# Google & OpenAI 認証
credentials_dict = json.loads(st.secrets["gspread_credentials"])
credentials = service_account.Credentials.from_service_account_info(
    credentials_dict, scopes=["https://www.googleapis.com/auth/documents.readonly"])
openai.api_key = st.secrets["openai_api_key"]

# グラフサイズ調整
def auto_figsize(n, base=4.5, height=3, max_width=10):
    return (min(base + n * 0.4, max_width), height)

# ドキュメント入力欄
doc_input = st.sidebar.text_input("📄 GoogleドキュメントID または URL")
doc_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', doc_input)
doc_id = doc_id_match.group(1) if doc_id_match else doc_input.strip()

if doc_id:
    try:
        service = build('docs', 'v1', credentials=credentials)
        doc = service.documents().get(documentId=doc_id).execute()
        content = doc.get("body", {}).get("content", [])
        full_text = "".join(e.get("textRun", {}).get("content", "") for c in content for e in c.get("paragraph", {}).get("elements", []))

        pattern = r"(?:名前[：: ]*(.+?)\s*日付[：: ]*([0-9]{4}[-/][0-9]{2}[-/][0-9]{2}))|(?:日付[：: ]*([0-9]{4}[-/][0-9]{2}[-/][0-9]{2})\s*名前[：: ]*(.+?))"
        matches = list(re.finditer(pattern, full_text))
        reports = []
        for i, match in enumerate(matches):
            name = match.group(1) or match.group(4)
            date = match.group(2) or match.group(3)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            text = full_text[start:end]

            data = {"日付": date, "名前": name.strip(), "全文": text.strip()}
            matches_data = re.findall(r"(オファー数|面談獲得数|提案数（有効）|提案数（有効・無効・不明）|配信数|荷電数|面談実施数)：([0-9]+)件", text)
            for label, value in matches_data:
                label = label.replace("（有効）", "_有効").replace("（有効・無効・不明）", "_合計")
                data[label] = int(value)
            reports.append(data)

        df_all = pd.DataFrame(reports)
        df_all['日付'] = pd.to_datetime(df_all['日付'], errors='coerce')
        latest_date = df_all['日付'].max()
        df = df_all[df_all['日付'] == latest_date].copy()

        st.header(f"📊 TOP6分析（{latest_date.strftime('%Y-%m-%d')} のみ）")

        action_cols = ['オファー数', '面談獲得数', '提案数_有効', '提案数_合計', '配信数']
        stats_df = df[['名前'] + action_cols].set_index('名前')
        zscore = StandardScaler().fit_transform(stats_df)
        deviation = pd.DataFrame(zscore * 10 + 50, columns=stats_df.columns, index=stats_df.index).round(1)
        summary = stats_df.describe().T[['50%', 'std']].rename(columns={'50%': '中央値', 'std': '標準偏差'})

        st.subheader("📐 統計＆偏差値表")
        st.dataframe(pd.concat([summary, deviation.T], axis=1))

        # ① 提案成功率
        df['提案成功率'] = df['提案数_有効'] / df['提案数_合計']
        fig1, ax1 = plt.subplots(figsize=auto_figsize(len(df)))
        sns.barplot(data=df, x='提案成功率', y='名前', ax=ax1, palette='Blues_d')
        ax1.set_xlabel("提案成功率", fontsize=11)
        ax1.set_ylabel("営業名", fontsize=11)
        ax1.set_title("① 営業別 提案成功率", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig1)

        # ② アクション合計
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        df[action_cols].sum().plot(kind='bar', ax=ax2, color='skyblue')
        ax2.set_xlabel("アクション項目", fontsize=11)
        ax2.set_ylabel("合計件数", fontsize=11)
        ax2.set_title("② 本日のアクション合計", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig2)

        # ③ ヒートマップ
        heat = df.groupby('名前')[action_cols].sum()
        fig3, ax3 = plt.subplots(figsize=(6, 0.6 * len(heat)))
        sns.heatmap(heat, annot=True, cmap='YlGnBu', fmt=".0f", ax=ax3)
        ax3.set_xlabel("アクション項目", fontsize=11)
        ax3.set_ylabel("営業名", fontsize=11)
        ax3.set_title("③ 営業別 アクションヒートマップ", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig3)

        # ④ 有効提案数
        fig4, ax4 = plt.subplots(figsize=auto_figsize(len(df)))
        sns.barplot(data=df, x='名前', y='提案数_有効', palette='Oranges', ax=ax4)
        ax4.set_xlabel("営業名", fontsize=11)
        ax4.set_ylabel("有効提案数", fontsize=11)
        ax4.set_title("④ 営業別 有効提案件数", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig4)

        # ⑤ 構成比
        total = df[action_cols].sum()
        fig5, ax5 = plt.subplots()
        ax5.pie(total, labels=total.index, autopct='%1.1f%%', startangle=90)
        ax5.set_title("⑤ アクション構成比（全体）", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig5)

        # ⑥ 偏差値ヒートマップ
        fig6, ax6 = plt.subplots(figsize=(6, 3))
        sns.heatmap(deviation.T, annot=True, cmap='coolwarm', center=50, ax=ax6)
        ax6.set_xlabel("営業名", fontsize=11)
        ax6.set_ylabel("アクション項目", fontsize=11)
        ax6.set_title("⑥ 営業別 偏差値ヒートマップ", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig6)

        # GPT要約
        st.header("🧠 ChatGPT AI要約・フィードバック")
        for _, row in df.iterrows():
            with st.expander(f"{row['日付'].strftime('%Y-%m-%d')}：{row['名前']}"):
                prompt = f"あなたは営業マネージャーです。以下の営業日報を読んで、(1)要約 (2)改善点 (3)提案アクション (4)上司のフィードバック を100文字ずつ日本語で出力してください。\n営業日報:\n{row['全文']}"
                try:
                    res = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.markdown(res.choices[0].message.content.strip())
                except Exception as e:
                    st.error(f"GPTエラー：{e}")

    except Exception as e:
        st.error(f"読込エラー：{e}")
else:
    st.info("📥 サイドバーに GoogleドキュメントID または URL を入力してください")

