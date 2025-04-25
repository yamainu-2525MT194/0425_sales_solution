# --- ファイル名: app.py（OpenAI v1 対応 + GPT-3.5 使用版）

import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import openai
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm

# ───────── 日本語フォントの登録 ─────────
font_path = "fonts/ipaexm.ttf"              # プロジェクト内のパス
fm.fontManager.addfont(font_path)            # フォントをMatplotlibに追加

# フォントプロパティ経由で実際のフォント名を取得
jp_font = fm.FontProperties(fname=font_path).get_name()

# MatplotlibのrcParamsに設定
plt.rcParams['font.family'] = jp_font        # デフォルトフォントを日本語フォントへ
plt.rcParams['axes.unicode_minus'] = False   # マイナス記号を正しく表示
# ───────────────────────────────────────
# Windows環境であれば Meiryo を指定
matplotlib.rcParams['font.family'] = 'Meiryo'
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ページ設定
st.set_page_config(page_title="営業日報ダッシュボード", layout="wide")
st.title("🧠 国家品質：完全営業ダッシュボード（全軸ラベル・偏差値・AI要約）")

# サイドバー設定
st.sidebar.header("⚙️ 設定")
doc_input = st.sidebar.text_input("📄 GoogleドキュメントID または URL")
enable_ai = st.sidebar.checkbox("AI要約・感情分析を有効にする", value=True)

def auto_figsize(n, base=4.5, height=3, max_width=10):
    return (min(base + n * 0.4, max_width), height)

# 認証設定
credentials_dict = json.loads(st.secrets["gspread_credentials"])
credentials = service_account.Credentials.from_service_account_info(
    credentials_dict,
    scopes=["https://www.googleapis.com/auth/documents.readonly"]
)
openai.api_key = st.secrets["openai_api_key"]

doc_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', doc_input)
doc_id = doc_id_match.group(1) if doc_id_match else doc_input.strip()

if doc_id:
    try:
        # ドキュメント取得
        service = build('docs', 'v1', credentials=credentials)
        doc = service.documents().get(documentId=doc_id).execute()
        content = doc.get("body", {}).get("content", [])
        full_text = "".join(
            e.get("textRun", {}).get("content", "")
            for c in content
            for e in c.get("paragraph", {}).get("elements", [])
        )

        # データ抽出
        pattern = (
            r"(?:名前[：: ]*(.+?)\s*日付[：: ]*([0-9]{4}[-/][0-9]{2}[-/][0-9]{2}))"
            r"|(?:日付[：: ]*([0-9]{4}[-/][0-9]{2}[-/][0-9]{2})\s*名前[：: ]*(.+?))"
        )
        matches = list(re.finditer(pattern, full_text))
        reports = []
        for i, m in enumerate(matches):
            name = m.group(1) or m.group(4)
            date = m.group(2) or m.group(3)
            start = m.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(full_text)
            text = full_text[start:end]
            data = {"日付": date, "名前": name.strip(), "全文": text.strip()}
            found = re.findall(
                r"(オファー数|面談獲得数|提案数（有効）|提案数（有効・無効・不明）|配信数|荷電数|面談実施数)：([0-9]+)件",
                text
            )
            for label, val in found:
                key = label.replace("（有効）", "_有効").replace("（有効・無効・不明）", "_合計")
                data[key] = int(val)
            reports.append(data)

        df_all = pd.DataFrame(reports)
        df_all['日付'] = pd.to_datetime(df_all['日付'], errors='coerce')
        latest = df_all['日付'].max()
        df = df_all[df_all['日付'] == latest].copy()

        # 全文表示
        st.subheader("📄 抽出された営業日報原文（全文）")
        st.caption("Googleドキュメントから抽出した全文を表示しています。")
        st.text_area("日報全文", value=full_text, height=300)

        # CSV表示
        st.subheader("📑 営業日報データ（CSV形式）")
        st.caption("抽出した日報データを表形式で表示しています。")
        st.dataframe(df_all)

        # 集計・準備
        action_cols = ['オファー数','面談獲得数','提案数_有効','提案数_合計','配信数']
        stats_df = df[['名前']+action_cols].set_index('名前')
        z = StandardScaler().fit_transform(stats_df)
        deviation = pd.DataFrame(z*10+50, columns=stats_df.columns, index=stats_df.index).round(1)
        summary = stats_df.describe().T[['50%','std']].rename(columns={'50%':'中央値','std':'標準偏差'})

        # 統計＆偏差値
        st.subheader("📐 統計＆偏差値表")
        st.caption("熱意と数字の裏付け。中央値と偏差値を一覧で確認できます。  ")
        st.dataframe(pd.concat([summary,deviation.T],axis=1))
        df['提案成功率'] = df['提案数_有効']/df['提案数_合計']

        # ① 提案成功率
        st.subheader("① 提案成功率")
        st.caption("有効提案率を営業別に比較します。")
        fig1,ax1=plt.subplots(figsize=auto_figsize(len(df)))
        sns.barplot(data=df,x='提案成功率',y='名前',ax=ax1,palette='Blues_d')
        ax1.set_xlabel('提案成功率')
        ax1.set_ylabel('営業名')
        ax1.set_title('営業別 提案成功率')
        st.pyplot(fig1)

        # ② アクション合計
        st.subheader("② アクション合計")
        st.caption("本日実施された各アクションの合計件数を棒グラフで表示します。")
        fig2,ax2=plt.subplots(figsize=(7,4))
        df[action_cols].sum().plot(kind='bar',ax=ax2,color='skyblue')
        ax2.set_xlabel('アクション項目')
        ax2.set_ylabel('合計件数')
        ax2.set_title('本日のアクション合計')
        st.pyplot(fig2)

        # ③ アクションヒートマップ
        st.subheader("③ アクションヒートマップ")
        st.caption("営業×アクション項目ごとの実施数をヒートマップで表示します。")
        heat=df.groupby('名前')[action_cols].sum()
        fig3,ax3=plt.subplots(figsize=(6,0.6*len(heat)))
        sns.heatmap(heat,annot=True,cmap='YlGnBu',fmt=".0f",ax=ax3)
        ax3.set_xlabel('アクション項目')
        ax3.set_ylabel('営業名')
        ax3.set_title('営業別 アクションヒートマップ')
        st.pyplot(fig3)

        # ④ 有効提案件数
        st.subheader("④ 有効提案件数")
        st.caption("営業ごとの有効提案件数を件数で比較できます。")
        fig4,ax4=plt.subplots(figsize=auto_figsize(len(df)))
        sns.barplot(data=df,x='名前',y='提案数_有効',palette='Oranges',ax=ax4)
        ax4.set_xlabel('営業名')
        ax4.set_ylabel('有効提案数')
        ax4.set_title('営業別 有効提案件数')
        st.pyplot(fig4)

        # ⑤ アクション構成比
        st.subheader("⑤ アクション構成比")
        st.caption("全アクションにおける割合を円グラフで可視化します。")
        total=df[action_cols].sum()
        fig5,ax5=plt.subplots()
        ax5.pie(total,labels=total.index,autopct='%1.1f%%',startangle=90)
        ax5.set_title('アクション構成比（全体）')
        ax5.legend(title='アクション項目',loc='upper right')
        st.pyplot(fig5)

        # ⑥ 偏差値ヒートマップ
        st.subheader("⑥ 偏差値ヒートマップ")
        st.caption("各営業のアクション偏差値をヒートマップで表示します。")
        fig6,ax6=plt.subplots(figsize=(6,3))
        sns.heatmap(deviation.T,annot=True,cmap='coolwarm',center=50,ax=ax6)
        ax6.set_xlabel('営業名')
        ax6.set_ylabel('アクション項目')
        ax6.set_title('営業別 偏差値ヒートマップ')
        st.pyplot(fig6)

        # ⑦ 相関係数マトリクス
        st.subheader("⑦ 相関係数マトリクス")
        st.caption("アクション項目同士の相関係数を可視化します。1に近いほど正の相関。")
        corr=df[action_cols].corr()
        fig7,ax7=plt.subplots(figsize=(6,5))
        sns.heatmap(corr,annot=True,cmap='coolwarm',vmin=-1,vmax=1,ax=ax7)
        ax7.set_xlabel('アクション項目')
        ax7.set_ylabel('アクション項目')
        ax7.set_title('アクション相関マトリクス')
        st.pyplot(fig7)

        # ⑧ 回帰分析
        st.subheader("⑧ 回帰分析（成功要因）")
        st.caption("他のアクション項目が提案成功率に与える影響を回帰係数で示します。")
        X=df[action_cols]
        y=df['提案成功率']
        lr=LinearRegression().fit(X,y)
        coefs=pd.Series(lr.coef_,index=X.columns)
        fig8,ax8=plt.subplots()
        coefs.sort_values().plot(kind='barh',ax=ax8,color='salmon')
        ax8.set_xlabel('回帰係数')
        ax8.set_ylabel('アクション項目')
        ax8.set_title('提案成功率への影響要因')
        st.pyplot(fig8)

        # ⑨ クラスタリング
        st.subheader("⑨ 営業クラスタリング")
        st.caption("K-meansにより営業タイプを3つのクラスタに分類し、人数を棒グラフで表示します。")
        X_std=StandardScaler().fit_transform(df[action_cols])
        k3=KMeans(n_clusters=3,random_state=42).fit(X_std)
        df['クラスタ']=k3.labels_
        fig9,ax9=plt.subplots()
        sns.countplot(data=df,x='クラスタ',palette='Set2',ax=ax9)
        ax9.set_xlabel('クラスタ番号')
        ax9.set_ylabel('営業人数')
        ax9.set_title('営業スタイル分類')
        st.pyplot(fig9)

        # ⑩ レーダーチャート
        st.subheader("⑩ 営業プロファイル（レーダーチャート）")
        st.caption("選択した営業のアクションプロファイルを平均値と比較して表示します。")
        selected=st.selectbox('営業担当選択',df['名前'].unique())
        ur=df[df['名前']==selected][action_cols].iloc[0]
        av=df[action_cols].mean()
        fig10=go.Figure()
        fig10.add_trace(go.Scatterpolar(r=ur,theta=action_cols,fill='toself',name=selected))
        fig10.add_trace(go.Scatterpolar(r=av,theta=action_cols,fill='toself',name='平均'))
        fig10.update_layout(
            polar=dict(
                radialaxis=dict(title='件数',visible=True)
            ),
            title='営業アクションプロファイル'
        )
        st.plotly_chart(fig10)

        # ⑪ 感情分析
        if enable_ai:
            st.subheader("⑪ 感情分析")
            st.caption("GPTにより営業日報の感情トーンを分析します。")
            for _,row in df.iterrows():
                with st.expander(f"{row['日付'].strftime('%Y-%m-%d')}：{row['名前']}"):
                    emo_p = (
                        "以下の営業日報の感情を'ポジティブ'/'中立'/'ネガティブ'で判定してください：\n"+row['全文']
                    )
                    try:
                        emo_res=openai.ChatCompletion.create(model='gpt-3.5-turbo',messages=[{'role':'user','content':emo_p}])
                        st.write(f"🧠 感情スコア：**{emo_res.choices[0].message.content.strip()}**")
                    except Exception as e:
                        st.error(f"感情分析エラー：{e}")
        else:
            st.info("⚠️ AI分析はオフです。サイドバーで有効化してください。")

        # ⑫ バイアス補正ランキング
        st.subheader("⑫ 偏差値補正ランキング")
        st.caption("偏差値スコアの平均により営業パフォーマンスを公平にランキングします。")
        bias=deviation[action_cols].mean(axis=1).sort_values(ascending=False)
        st.dataframe(bias.reset_index().rename(columns={0:'偏差値スコア平均'}))

        # AI要約
        if enable_ai:
            st.header("🧠 ChatGPT AI要約・フィードバック")
            st.caption("GPTによる要約・改善点・提案アクション・フィードバックを表示します。")
            for _,row in df.iterrows():
                with st.expander(f"{row['日付'].strftime('%Y-%m-%d')}：{row['名前']}"):
                    sum_p = (
                        "あなたは営業マネージャーです。以下の営業日報を読んで、"
                        +"(1)要約 (2)改善点 (3)提案アクション (4)上司のフィードバック を100文字ずつ日本語で出力してください。\n"
                        +row['全文']
                    )
                    try:
                        res=openai.ChatCompletion.create(model='gpt-3.5-turbo',messages=[{'role':'user','content':sum_p}])
                        st.markdown(res.choices[0].message.content.strip())
                    except Exception as e:
                        st.error(f"GPTエラー：{e}")
    except Exception as e:
        st.error(f"読込エラー：{e}")
else:
    st.info("📥 サイドバーに GoogleドキュメントID または URL を入力してください")


