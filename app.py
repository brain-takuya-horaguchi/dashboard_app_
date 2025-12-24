import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import calendar
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv
from utils import (
    validate_data, generate_alerts, generate_recommendations, 
    calculate_conversion_funnel,
    setup_openai_client, query_data_with_ai, get_suggested_questions,
    save_chat_history, export_chat_history, read_csv_with_encoding,
    create_company_introduction_contract_chart,
    create_job_introduction_contract_chart,
    create_avg_recommendations_chart,
    create_leadtime_chart,
    create_ca_interviews_chart,
    create_scouter_performance_chart
)

# .envファイルを読み込む
load_dotenv()

# ページ設定
st.set_page_config(
    page_title="企業採用分析ダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin: 0;
    }
    .filter-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
    }
    .insight-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .chat-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .chat-message {
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 8px;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .ai-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)


def parse_date(date_str):
    """日付文字列をパースする"""
    if pd.isna(date_str) or date_str == "":
        return None
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
    except:
        try:
            return pd.to_datetime(date_str)
        except:
            return None


def calculate_metrics(df, selected_companies=None, selected_months=None):
    """企業ごとの指標を計算する関数（改良版）"""
    if df.empty:
        return pd.DataFrame()
    
    # 日付カラムのパース
    df['書類提出日_parsed'] = df['進捗：書類提出日'].apply(parse_date)
    df['面接日_parsed'] = df['進捗：面接日'].apply(parse_date)
    df['内定日_parsed'] = df['進捗：内定日'].apply(parse_date)
    
    # 企業フィルタリング（「全て」が選択されていない場合のみ）
    if selected_companies and "全て" not in selected_companies:
        df = df[df['企業：企業名'].isin(selected_companies)]
    
    # 月フィルタリング（「全て」が選択されていない場合のみ）
    if selected_months and "全て" not in selected_months:
        mask = pd.Series(False, index=df.index)
        for month in selected_months:
            year, month_num = month.split('-')
            year, month_num = int(year), int(month_num)
            
            # 書類提出日、面接日、内定日のいずれかが指定月に含まれる場合
            for date_col in ['書類提出日_parsed', '面接日_parsed', '内定日_parsed']:
                mask |= (df[date_col].dt.year == year) & (df[date_col].dt.month == month_num)
        
        df = df[mask]
    
    if df.empty:
        return pd.DataFrame()
    
    company_metrics = []
    
    for company in df['企業：企業名'].unique():
        company_data = df[df['企業：企業名'] == company]
        
        # 基本指標の計算
        推薦人数 = company_data['求職者：求職者ID'].nunique()
        書類提出数 = company_data['書類提出日_parsed'].notna().sum()
        書類結果回収済 = company_data[
            (company_data['書類提出日_parsed'].notna()) & 
            (company_data['進捗：ステータス'].notna())
        ].shape[0]
        
        # 書類通過率
        面接進行数 = company_data['面接日_parsed'].notna().sum()
        書類通過率 = (面接進行数 / 書類提出数 * 100) if 書類提出数 > 0 else 0.0
        
        # 面接関連指標
        一次面接数 = company_data[
            (company_data['進捗：面接回数'] == 1) & 
            (company_data['面接日_parsed'].notna())
        ].shape[0]
        
        一次面接通過数 = company_data[
            ((company_data['進捗：面接回数'] > 1) | 
             (company_data['進捗：最終面接フラグ'] == 1)) &
            (company_data['面接日_parsed'].notna())
        ].shape[0]
        
        一次面接通過率 = (一次面接通過数 / 一次面接数 * 100) if 一次面接数 > 0 else 0.0
        
        最終面接 = company_data[
            (company_data['進捗：最終面接フラグ'] == 1) &
            (company_data['面接日_parsed'].notna())
        ].shape[0]
        
        内定数 = company_data['内定日_parsed'].notna().sum()
        
        # 内定率（推薦人数に対する内定数の割合）
        内定率 = (内定数 / 推薦人数 * 100) if 推薦人数 > 0 else 0.0
        
        # 平均処理時間（書類提出から内定まで）
        内定者_data = company_data[company_data['内定日_parsed'].notna()]
        平均処理時間 = 0
        if not 内定者_data.empty and 内定者_data['書類提出日_parsed'].notna().any():
            処理時間 = (内定者_data['内定日_parsed'] - 内定者_data['書類提出日_parsed']).dt.days
            平均処理時間 = int(処理時間.mean()) if not 処理時間.empty else 0
        
        company_metrics.append({
            '企業名': company,
            '推薦人数': 推薦人数,
            '書類提出数': 書類提出数,
            '書類結果回収済': 書類結果回収済,
            '書類通過率': 書類通過率,
            '1次面接数': 一次面接数,
            '1次面接通過率': 一次面接通過率,
            '最終面接数': 最終面接,
            '内定数': 内定数,
            '内定率': 内定率,
            '平均処理時間': 平均処理時間
        })
    
    return pd.DataFrame(company_metrics)


def create_advanced_dashboard(metrics_df):
    """高度なダッシュボードを作成"""
    if metrics_df.empty:
        return go.Figure()
    
    # 上位企業の選択（内定数 + 内定率でランキング）
    metrics_df['スコア'] = metrics_df['内定数'] * 0.7 + metrics_df['内定率'] * 0.3
    top_companies = metrics_df.nlargest(10, 'スコア')
    
    # 企業名を短縮（表示用）
    def shorten_company_name(name, max_length=15):
        if len(name) <= max_length:
            return name
        return name[:max_length-3] + "..."
    
    top_companies['企業名_短縮'] = top_companies['企業名'].apply(shorten_company_name)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '📈 パフォーマンス分析（Top 10）',
            '🎯 通過率比較',
            '⏱️ 処理時間分析',
            '🏆 内定効率ランキング'
        ),
        specs=[
            [{"secondary_y": True}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ]
    )
    
    # 1. パフォーマンス分析
    fig.add_trace(
        go.Bar(
            name='推薦人数',
            x=top_companies['企業名_短縮'],
            y=top_companies['推薦人数'],
            marker_color='rgba(102, 126, 234, 0.7)',
            yaxis='y',
            hovertemplate='<b>%{text}</b><br>推薦人数: %{y}<extra></extra>',
            text=top_companies['企業名']  # フルネームをホバーに表示
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            name='内定率',
            x=top_companies['企業名_短縮'],
            y=top_companies['内定率'],
            mode='lines+markers',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=8),
            yaxis='y2',
            hovertemplate='<b>%{text}</b><br>内定率: %{y}%<extra></extra>',
            text=top_companies['企業名']  # フルネームをホバーに表示
        ),
        row=1, col=1
    )
    
    # 2. 通過率比較
    fig.add_trace(
        go.Bar(
            name='書類通過率',
            x=top_companies['企業名_短縮'],
            y=top_companies['書類通過率'],
            marker_color='rgba(76, 175, 80, 0.7)',
            hovertemplate='<b>%{text}</b><br>書類通過率: %{y}%<extra></extra>',
            text=top_companies['企業名']  # フルネームをホバーに表示
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            name='1次面接通過率',
            x=top_companies['企業名_短縮'],
            y=top_companies['1次面接通過率'],
            marker_color='rgba(255, 152, 0, 0.7)',
            hovertemplate='<b>%{text}</b><br>1次面接通過率: %{y}%<extra></extra>',
            text=top_companies['企業名']  # フルネームをホバーに表示
        ),
        row=1, col=2
    )
    
    # 3. 処理時間分析
    fig.add_trace(
        go.Scatter(
            name='平均処理時間',
            x=top_companies['内定数'],
            y=top_companies['平均処理時間'],
            mode='markers',  # テキストラベルを削除
            marker=dict(
                size=top_companies['推薦人数'] * 3,
                color=top_companies['内定率'],
                colorscale='Viridis',
                showscale=False  # 凡例を削除
            ),
            hovertemplate='<b>%{customdata}</b><br>内定数: %{x}<br>平均処理時間: %{y}日<extra></extra>',
            customdata=top_companies['企業名']  # フルネームをホバーに表示
        ),
        row=2, col=1
    )
    
    # 4. 内定効率ランキング
    efficiency_df = metrics_df.nlargest(8, '内定率')
    fig.add_trace(
        go.Bar(
            name='内定効率',
            x=efficiency_df['内定率'],
            y=efficiency_df['企業名'],
            orientation='h',
            marker_color='rgba(156, 39, 176, 0.7)'
        ),
        row=2, col=2
    )
    
    # レイアウト調整
    fig.update_layout(
        height=900,  # 高さを増やして余裕を持たせる
        showlegend=False,  # 凡例を削除
        title_text="<b>📊 企業採用分析ダッシュボード</b>",
        title_font_size=20,
        template="plotly_white",
        margin=dict(b=100)  # 下部のマージンを増やす
    )
    
    # 軸の設定
    fig.update_xaxes(
        tickangle=90,  # 垂直に変更
        tickfont=dict(size=10),  # フォントサイズを小さく
        row=1, col=1
    )
    fig.update_xaxes(
        tickangle=90,  # 垂直に変更
        tickfont=dict(size=10),  # フォントサイズを小さく
        row=1, col=2
    )
    fig.update_xaxes(title_text="内定数", row=2, col=1)
    fig.update_xaxes(title_text="内定率 (%)", row=2, col=2)
    fig.update_yaxes(title_text="処理時間 (日)", row=2, col=1)
    
    return fig


def create_trend_analysis(df, selected_companies=None):
    """トレンド分析のグラフを作成"""
    if df.empty:
        return go.Figure()
    
    # 日付カラムのパース
    df['書類提出日_parsed'] = df['進捗：書類提出日'].apply(parse_date)
    df['内定日_parsed'] = df['進捗：内定日'].apply(parse_date)
    
    # 企業フィルタリング（「全て」が選択されていない場合のみ）
    if selected_companies and "全て" not in selected_companies:
        df = df[df['企業：企業名'].isin(selected_companies)]
    
    # 月別集計
    monthly_stats = []
    
    for date_col, label in [('書類提出日_parsed', '書類提出'), ('内定日_parsed', '内定')]:
        monthly_data = df[df[date_col].notna()].copy()
        if not monthly_data.empty:
            monthly_data['年月'] = monthly_data[date_col].dt.to_period('M')
            monthly_counts = monthly_data.groupby(['年月', '企業：企業名']).size().reset_index(name='件数')
            monthly_counts['種類'] = label
            monthly_stats.append(monthly_counts)
    
    if not monthly_stats:
        return go.Figure()
    
    trend_df = pd.concat(monthly_stats, ignore_index=True)
    trend_df['年月_str'] = trend_df['年月'].astype(str)
    
    fig = px.line(
        trend_df,
        x='年月_str',
        y='件数',
        color='企業：企業名',
        facet_col='種類',
        title='📈 月別トレンド分析',
        markers=True
    )
    
    fig.update_layout(
        height=400,
        template="plotly_white",
        title_font_size=16,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        margin=dict(r=150)  # 右側マージンを追加
    )
    
    return fig


def generate_insights(metrics_df):
    """データからインサイトを生成"""
    if metrics_df.empty:
        return []
    
    insights = []
    
    # 1. 最高パフォーマンス企業
    if not metrics_df.empty:
        best_company = metrics_df.loc[metrics_df['内定率'].idxmax()]
        insights.append({
            'type': 'success',
            'title': '🏆 最高パフォーマンス企業',
            'content': f"{best_company['企業名']} が最も高い内定率 {best_company['内定率']:.1f}% を記録しています。"
        })
    
    # 2. 改善の余地がある企業
    low_performance = metrics_df[metrics_df['内定率'] < 10]
    if not low_performance.empty:
        insights.append({
            'type': 'warning',
            'title': '⚠️ 改善の余地',
            'content': f"{len(low_performance)} 社の内定率が10%を下回っています。選考プロセスの見直しを検討してください。"
        })
    
    # 3. 処理時間に関する洞察
    slow_companies = metrics_df[metrics_df['平均処理時間'] > 30]
    if not slow_companies.empty:
        insights.append({
            'type': 'info',
            'title': '⏱️ 処理時間について',
            'content': f"{len(slow_companies)} 社の平均処理時間が30日を超えています。選考スピードの向上が期待できます。"
        })
    
    # 4. 全体的な統計
    total_applications = metrics_df['書類提出数'].sum()
    total_offers = metrics_df['内定数'].sum()
    overall_rate = (total_offers / total_applications * 100) if total_applications > 0 else 0
    
    insights.append({
        'type': 'info',
        'title': '📊 全体統計',
        'content': f"総書類提出数: {total_applications:,} 件、総内定数: {total_offers:,} 件、全体内定率: {overall_rate:.1f}%"
    })
    
    return insights


def get_available_months(df):
    """データから利用可能な月を取得"""
    if df.empty:
        return []
    
    months = set()
    date_columns = ['進捗：書類提出日', '進捗：面接日', '進捗：内定日']
    
    for col in date_columns:
        dates = df[col].apply(parse_date).dropna()
        if not dates.empty:
            for date in dates:
                months.add(f"{date.year}-{date.month:02d}")
    
    return sorted(list(months), reverse=True)


def render_chat_interface(df, metrics_df, openai_client):
    """チャットインターフェースを描画"""
    st.header("🤖 AI データアナリスト")
    st.markdown("読み込んだデータについて何でも質問してください！")
    
    # OpenAI API キーの設定
    if not openai_client:
        # まず.envファイルから環境変数をチェック
        env_api_key = os.environ.get("OPENAI_API_KEY")
        
        if env_api_key:
            st.info("ℹ️ OpenAI APIキーは環境変数から読み込まれています（.envファイル）")
            st.warning("⚠️ APIキーが検出されましたが、クライアントの初期化に問題があります。")
        else:
            st.warning("⚠️ OpenAI APIキーが設定されていません。")
            st.info("💡 **推奨**: プロジェクトルートに `.env` ファイルを作成し、`OPENAI_API_KEY=your-api-key` を設定してください。")
            
            # API キー入力
            api_key_input = st.text_input(
                "OpenAI API キーを入力してください",
                type="password",
                help="OpenAI APIキーを入力してチャット機能を使用できます"
            )
            
            if api_key_input:
                st.session_state.openai_api_key = api_key_input
                st.success("✅ APIキーが設定されました。ページを再読み込みしてください。")
                st.rerun()
        
        return
    
    # 推奨質問の表示
    suggestions = get_suggested_questions(df, metrics_df)
    
    with st.expander("💡 推奨質問", expanded=True):
        st.markdown("以下の質問をクリックして試してみてください：")
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions[:8]):  # 最初の8つの推奨質問を表示
            with cols[i % 2]:
                if st.button(f"📝 {suggestion}", key=f"suggestion_{i}"):
                    # 推奨質問をクリックしたときの処理
                    st.session_state.selected_question = suggestion
    
    # 質問入力の状態管理
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    # 推奨質問がクリックされた場合
    if 'selected_question' in st.session_state:
        st.session_state.current_question = st.session_state.selected_question
        del st.session_state.selected_question
    
    # 質問入力
    question = st.text_area(
        "質問を入力してください",
        value=st.session_state.current_question,
        height=100,
        placeholder="例: 最もパフォーマンスが良い企業はどこですか？",
        key="question_input"
    )
    
    # 質問送信
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📤 質問する", type="primary"):
            if question and question.strip():
                with st.spinner("🤔 分析中..."):
                    answer = query_data_with_ai(question, df, metrics_df, openai_client)
                    save_chat_history(question, answer)
                    # 質問送信後に入力フィールドをクリア
                    st.session_state.current_question = ""
                    st.rerun()
            else:
                st.warning("質問を入力してください。")
    
    with col2:
        if st.button("🗑️ 履歴をクリア"):
            st.session_state.chat_history = []
            st.success("履歴がクリアされました。")
            st.rerun()
    
    # チャット履歴の表示
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.subheader("💬 チャット履歴")
        
        # 最新の履歴から表示
        for chat in reversed(st.session_state.chat_history[-10:]):  # 最新10件を表示
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🙋‍♀️ あなた ({chat['timestamp']})</strong><br>
                {chat['question']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="chat-message ai-message">
                <strong>🤖 AI アナリスト</strong><br>
                {chat['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
        
        # 履歴エクスポート
        if len(st.session_state.chat_history) > 0:
            try:
                chat_export = export_chat_history()
                if chat_export and len(chat_export) > 0:
                    st.download_button(
                        label="📥 チャット履歴をダウンロード",
                        data=chat_export,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        help="チャット履歴をMarkdown形式でダウンロードします"
                    )
                else:
                    st.warning("⚠️ チャット履歴のエクスポートに失敗しました")
            except Exception as e:
                st.error(f"❌ チャット履歴エクスポートエラー: {str(e)}")


def main():
    # ヘッダー
    st.markdown("""
    <div class="main-header">
        <h1>📊 企業採用分析ダッシュボード</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # OpenAI クライアントの設定
    openai_client = setup_openai_client()
    
    # サイドバー
    with st.sidebar:
        st.header("📁 データ管理")
        
        # ファイルアップロード
        uploaded_file = st.file_uploader(
            "CSVファイルを選択",
            type=['csv'],
            help="企業・求職者データのCSVファイルをアップロードしてください"
        )
        
        if uploaded_file is not None:
            st.success("✅ ファイルがアップロードされました")
            
            # データの読み込み
            try:
                df = read_csv_with_encoding(uploaded_file)
                
                # データ検証
                st.header("🔍 データ検証")
                validation_results = validate_data(df)
                
                if not validation_results['is_valid']:
                    for error in validation_results['errors']:
                        st.error(f"❌ {error}")
                    return
                
                # 警告の表示
                if validation_results['warnings']:
                    with st.expander("⚠️ データ品質の警告", expanded=False):
                        for warning in validation_results['warnings']:
                            st.warning(f"⚠️ {warning}")
                
                # データ品質スコア
                col1, col2, col3 = st.columns(3)
                with col1:
                    completeness = validation_results['statistics']['data_completeness']
                    avg_completeness = sum(completeness.values()) / len(completeness)
                    st.metric("データ完全性", f"{avg_completeness:.1f}%")
                
                with col2:
                    duplicate_rate = (len(df) - len(df.drop_duplicates())) / len(df) * 100
                    st.metric("重複率", f"{duplicate_rate:.1f}%")
                
                with col3:
                    quality_score = max(0, 100 - len(validation_results['warnings']) * 10)
                    st.metric("品質スコア", f"{quality_score}/100")
                
                st.header("🔍 フィルター設定")
                
                # 企業選択（「全て」オプション追加）
                all_companies = sorted(df['企業：企業名'].unique())
                company_options = ["全て"] + all_companies
                
                selected_companies = st.multiselect(
                    "企業を選択",
                    options=company_options,
                    default=["全て"],
                    help="分析対象の企業を選択してください（「全て」を選択すると全企業が対象）"
                )
                
                # 月選択（「全て」オプション追加）
                available_months = get_available_months(df)
                if available_months:
                    month_options = ["全て"] + available_months
                    selected_months = st.multiselect(
                        "月を選択",
                        options=month_options,
                        default=["全て"],
                        help="分析対象の月を選択してください（「全て」を選択すると全月が対象）"
                    )
                else:
                    selected_months = ["全て"]
                
                # 分析実行
                if st.button("🔄 分析を実行", type="primary"):
                    st.session_state.analysis_run = True
                
                # データ概要
                st.header("📋 データ概要")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("総行数", f"{len(df):,}")
                    st.metric("企業数", f"{df['企業：企業名'].nunique():,}")
                with col2:
                    st.metric("求職者数", f"{df['求職者：求職者ID'].nunique():,}")
                    if "全て" in selected_companies:
                        st.metric("選択企業数", "全て")
                    else:
                        st.metric("選択企業数", f"{len(selected_companies):,}")
                
            except Exception as e:
                st.error(f"❌ ファイル読み込みエラー: {str(e)}")
                return
        else:
            st.info("👆 CSVファイルをアップロードしてください")
            return
    
    # メインコンテンツ
    if uploaded_file is not None:
        # タブの作成
        tab1, tab2 = st.tabs(["📊 分析ダッシュボード", "🤖 AI チャット"])
        
        with tab1:
            if hasattr(st.session_state, 'analysis_run'):
                # 指標計算
                with st.spinner("📊 データを分析中..."):
                    metrics_df = calculate_metrics(df, selected_companies, selected_months)
                    
                    if metrics_df.empty:
                        st.warning("⚠️ 選択した条件に該当するデータがありません。フィルター設定を確認してください。")
                        return

                # アラート表示
                # st.header("🚨 アラート & 通知")
                # alerts = generate_alerts(metrics_df)
                
                # # 成功アラートのみを表示
                # success_alerts = [a for a in alerts if a['type'] == 'success']
                
                # if success_alerts:
                #     st.subheader("🎉 成功事例")
                #     for alert in success_alerts:
                #         st.markdown(f"""
                #         <div style="background: #d4edda; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; margin: 1rem 0;">
                #             <h4 style="color: #155724; margin: 0 0 0.5rem 0;">{alert['title']}</h4>
                #             <p style="margin: 0;"><strong>{alert['company']}</strong>: {alert['message']}</p>
                #         </div>
                #         """, unsafe_allow_html=True)
                # else:
                #     st.info("📊 現在、表示するアラートはありません。全て順調です！")
                
                # 改善提案表示
                st.header("💡 改善提案")
                recommendations = generate_recommendations(metrics_df)
                
                if recommendations:
                    for rec in recommendations:
                        with st.expander(f"{rec['title']} ({len(rec['target_companies'])} 社対象)", expanded=False):
                            st.write(rec['description'])
                            
                            st.write("**推奨アクション:**")
                            for action in rec['actions']:
                                st.write(f"• {action}")
                            
                            st.write("**対象企業:**")
                            target_companies_str = ', '.join(rec['target_companies'][:5])
                            if len(rec['target_companies']) > 5:
                                target_companies_str += f" など {len(rec['target_companies'])} 社"
                            st.write(target_companies_str)
                else:
                    st.info("📈 現在、特別な改善提案はありません。")
                
                # インサイト表示
                st.header("📊 分析インサイト")
                insights = generate_insights(metrics_df)
                
                cols = st.columns(len(insights))
                for i, insight in enumerate(insights):
                    with cols[i % len(cols)]:
                        if insight['type'] == 'success':
                            st.markdown(f"""
                            <div class="success-box">
                                <h4>{insight['title']}</h4>
                                <p>{insight['content']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif insight['type'] == 'warning':
                            st.markdown(f"""
                            <div class="warning-box">
                                <h4>{insight['title']}</h4>
                                <p>{insight['content']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="insight-box">
                                <h4>{insight['title']}</h4>
                                <p>{insight['content']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # KPI表示
                st.header("📈 主要指標")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_recommendations = metrics_df['推薦人数'].sum()
                    st.metric("総推薦人数", f"{total_recommendations:,}")
                
                with col2:
                    total_applications = metrics_df['書類提出数'].sum()
                    st.metric("総書類提出数", f"{total_applications:,}")
                
                with col3:
                    total_offers = metrics_df['内定数'].sum()
                    st.metric("総内定数", f"{total_offers:,}")
                
                with col4:
                    avg_success_rate = metrics_df['内定率'].mean()
                    st.metric("平均内定率", f"{avg_success_rate:.1f}%")
                
                # 高度なダッシュボード
                st.header("📊 詳細分析")
                dashboard_fig = create_advanced_dashboard(metrics_df)
                st.plotly_chart(dashboard_fig, use_container_width=True)
                
                # コンバージョンファネル
                st.header("🔄 コンバージョンファネル")
                
                # 企業選択
                # 選択された企業のリストを作成（「全て」を除外）
                available_companies = [c for c in selected_companies if c != "全て"]
                if "全て" in selected_companies:
                    available_companies = sorted(df['企業：企業名'].unique())
                
                if len(available_companies) > 1:
                    funnel_company = st.selectbox(
                        "ファネル表示する企業を選択",
                        options=["全体"] + available_companies,
                        help="特定の企業のファネルを表示します"
                    )
                else:
                    funnel_company = "全体"
                
                # データフィルタリング
                # 月フィルタリングが適用されたデータをベースにする
                filtered_df = df.copy()
                
                # 月フィルタリング
                if selected_months and "全て" not in selected_months:
                    filtered_df['書類提出日_parsed'] = filtered_df['進捗：書類提出日'].apply(parse_date)
                    filtered_df['面接日_parsed'] = filtered_df['進捗：面接日'].apply(parse_date)
                    filtered_df['内定日_parsed'] = filtered_df['進捗：内定日'].apply(parse_date)
                    
                    mask = pd.Series(False, index=filtered_df.index)
                    for month in selected_months:
                        year, month_num = month.split('-')
                        year, month_num = int(year), int(month_num)
                        
                        # 書類提出日、面接日、内定日のいずれかが指定月に含まれる場合
                        for date_col in ['書類提出日_parsed', '面接日_parsed', '内定日_parsed']:
                            mask |= (filtered_df[date_col].dt.year == year) & (filtered_df[date_col].dt.month == month_num)
                    
                    filtered_df = filtered_df[mask]
                
                # 企業フィルタリング
                if funnel_company == "全体":
                    # 全体データを計算
                    if "全て" in selected_companies:
                        # 全ての企業データを使用
                        funnel_data = calculate_conversion_funnel(filtered_df)
                    else:
                        # 選択された企業のみ
                        filtered_df = filtered_df[filtered_df['企業：企業名'].isin(available_companies)]
                        funnel_data = calculate_conversion_funnel(filtered_df)
                else:
                    # 特定企業のデータ
                    funnel_data = calculate_conversion_funnel(filtered_df, funnel_company)
                
                # ファネルデータの確認
                if not funnel_data['funnel']:
                    st.warning("⚠️ コンバージョンファネルのデータがありません。")
                else:
                    # デバッグ情報の表示（開発用）
                    with st.expander("🔍 デバッグ情報（開発用）", expanded=False):
                        st.write(f"**フィルタリング前の総データ数**: {len(df)}")
                        st.write(f"**フィルタリング後のデータ数**: {len(filtered_df)}")
                        st.write(f"**選択企業**: {selected_companies}")
                        st.write(f"**選択月**: {selected_months}")
                        st.write(f"**表示企業**: {funnel_company}")
                        st.write("**ファネルデータ**:")
                        st.json(funnel_data['funnel'])
                        st.write("**通過率データ**:")
                        st.json(funnel_data['conversion_rates'])
                    # ファネルグラフ
                    funnel_values = list(funnel_data['funnel'].values())
                    funnel_labels = list(funnel_data['funnel'].keys())
                    
                    fig_funnel = go.Figure(go.Funnel(
                        y=funnel_labels,
                        x=funnel_values,
                        textinfo="value+percent initial",
                        marker=dict(color=["#667eea", "#764ba2", "#ff6b6b", "#4ecdc4", "#45b7d1", "#f39c12", "#e74c3c"]),
                        connector={"line": {"color": "royalblue", "dash": "solid", "width": 2}},
                        opacity=0.8
                    ))
                    
                    fig_funnel.update_layout(
                        title=f"📊 選考フロー - {funnel_company}",
                        font=dict(size=12),
                        height=500,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=0.98,
                            xanchor="left",
                            x=1.02,
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="rgba(0,0,0,0.1)",
                            borderwidth=1
                        ),
                        margin=dict(r=150)
                    )
                    
                    st.plotly_chart(fig_funnel, use_container_width=True)
                    
                    # 通過率詳細
                    st.subheader("📈 各段階通過率")
                    conv_col1, conv_col2 = st.columns(2)
                    
                    with conv_col1:
                        if funnel_data['conversion_rates']:
                            for i, (stage, rate) in enumerate(funnel_data['conversion_rates'].items()):
                                st.metric(f"{stage}", f"{rate:.1f}%")
                        else:
                            st.info("通過率データがありません。")
                    
                    with conv_col2:
                        # 通過率グラフ
                        if funnel_data['conversion_rates']:
                            stages = list(funnel_data['conversion_rates'].keys())
                            rates = list(funnel_data['conversion_rates'].values())
                            
                            fig_conv = px.bar(
                                x=stages,
                                y=rates,
                                title="段階別通過率",
                                labels={'x': '段階', 'y': '通過率 (%)'},
                                color=rates,
                                color_continuous_scale="viridis"
                            )
                            fig_conv.update_coloraxes(showscale=False)  # 凡例を削除
                            fig_conv.update_layout(
                                height=300,
                                xaxis={'tickangle': 45},
                                legend=dict(
                                    orientation="v",
                                    yanchor="top",
                                    y=0.98,
                                    xanchor="left",
                                    x=1.02,
                                    bgcolor="rgba(255,255,255,0.9)",
                                    bordercolor="rgba(0,0,0,0.1)",
                                    borderwidth=1
                                ),
                                margin=dict(r=150)
                            )
                            st.plotly_chart(fig_conv, use_container_width=True)
                        else:
                            st.info("通過率グラフを表示できません。")
                
                # トレンド分析
                if selected_months:
                    st.header("📈 トレンド分析")
                    trend_fig = create_trend_analysis(df, selected_companies)
                    st.plotly_chart(trend_fig, use_container_width=True)
                
                # 新しい分析グラフ
                st.header("📊 追加分析グラフ")
                
                # タブでグラフを整理
                graph_tab1, graph_tab2, graph_tab3, graph_tab4, graph_tab5, graph_tab6 = st.tabs([
                    "企業ごとの紹介～成約率",
                    "求人ごとの紹介～成約率",
                    "求職者1人当たりの平均推薦数",
                    "面談から推薦までのリードタイム",
                    "面談数（CAごと）",
                    "スカウターのパフォーマンス"
                ])
                
                with graph_tab1:
                    st.subheader("📊 企業ごとの紹介～成約率")
                    company_intro_fig = create_company_introduction_contract_chart(df)
                    st.plotly_chart(company_intro_fig, use_container_width=True)
                    
                    # データテーブル
                    from utils import calculate_company_introduction_to_contract_rate
                    company_intro_df = calculate_company_introduction_to_contract_rate(df)
                    if not company_intro_df.empty:
                        st.dataframe(company_intro_df.sort_values('成約率', ascending=False), use_container_width=True)
                
                with graph_tab2:
                    st.subheader("📊 求人ごとの紹介～成約率")
                    job_intro_fig = create_job_introduction_contract_chart(df)
                    st.plotly_chart(job_intro_fig, use_container_width=True)
                    
                    # データテーブル
                    from utils import calculate_job_introduction_to_contract_rate
                    job_intro_df = calculate_job_introduction_to_contract_rate(df)
                    if not job_intro_df.empty:
                        st.dataframe(job_intro_df.sort_values('成約率', ascending=False), use_container_width=True)
                
                with graph_tab3:
                    st.subheader("📊 求職者1人当たりの平均推薦数")
                    avg_rec_fig = create_avg_recommendations_chart(df)
                    st.plotly_chart(avg_rec_fig, use_container_width=True)
                    
                    # 統計情報
                    from utils import calculate_avg_recommendations_per_candidate
                    avg_stats = calculate_avg_recommendations_per_candidate(df)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("全体平均推薦数", f"{avg_stats['avg_recommendations']:.2f}")
                    with col2:
                        st.metric("総求職者数", f"{avg_stats['total_candidates']:,}")
                    with col3:
                        st.metric("総推薦数", f"{avg_stats['total_recommendations']:,}")
                
                with graph_tab4:
                    st.subheader("⏱️ 面談から推薦までのリードタイム")
                    leadtime_fig = create_leadtime_chart(df)
                    st.plotly_chart(leadtime_fig, use_container_width=True)
                    
                    # データテーブル
                    from utils import calculate_interview_to_recommendation_leadtime
                    leadtime_df = calculate_interview_to_recommendation_leadtime(df)
                    if not leadtime_df.empty:
                        st.dataframe(leadtime_df.sort_values('平均リードタイム', ascending=True), use_container_width=True)
                    else:
                        st.info("💡 面談日データが必要です。CSVファイルに「求職者：面談日」カラムが含まれているか確認してください。")
                
                with graph_tab5:
                    st.subheader("👥 面談数（CAごと）")
                    ca_fig = create_ca_interviews_chart(df)
                    st.plotly_chart(ca_fig, use_container_width=True)
                    
                    # データテーブル
                    from utils import calculate_interviews_by_ca
                    ca_df = calculate_interviews_by_ca(df)
                    if not ca_df.empty:
                        st.dataframe(ca_df.sort_values('面談数', ascending=False), use_container_width=True)
                    else:
                        st.info("💡 CAデータが必要です。CSVファイルに「求職者：担当者」カラムが含まれているか確認してください。")
                
                with graph_tab6:
                    st.subheader("🎯 スカウターのパフォーマンス測定")
                    scouter_fig = create_scouter_performance_chart(df)
                    st.plotly_chart(scouter_fig, use_container_width=True)
                    
                    # データテーブル
                    from utils import calculate_scouter_performance
                    scouter_df = calculate_scouter_performance(df)
                    if not scouter_df.empty:
                        st.dataframe(scouter_df.sort_values('成約率', ascending=False), use_container_width=True)
                    else:
                        st.info("💡 スカウターデータが必要です。CSVファイルに「スカウト担当者」カラムが含まれているか確認してください。")
                
                # 詳細データテーブル
                st.header("📋 詳細データ")
                
                # 表示する指標を選択
                display_columns = st.multiselect(
                    "表示する指標を選択",
                    options=metrics_df.columns.tolist(),
                    default=metrics_df.columns.tolist()
                )
                
                if display_columns:
                    # 並び替え設定
                    st.subheader("🔄 並び替え設定")
                    
                    # プリセット並び替えオプション
                    preset_options = {
                        "カスタム": {"column": None, "order": "降順"},
                        "内定数が多い順": {"column": "内定数", "order": "降順"},
                        "内定率が高い順": {"column": "内定率", "order": "降順"},
                        "推薦人数が多い順": {"column": "推薦人数", "order": "降順"},
                        "書類通過率が高い順": {"column": "書類通過率", "order": "降順"},
                        "1次面接通過率が高い順": {"column": "1次面接通過率", "order": "降順"},
                        "処理時間が短い順": {"column": "平均処理時間", "order": "昇順"},
                        "企業名順": {"column": "企業名", "order": "昇順"}
                    }
                    
                    # プリセット選択
                    preset_choice = st.selectbox(
                        "📊 並び替えプリセット",
                        options=list(preset_options.keys()),
                        help="よく使われる並び替えパターンを選択するか、カスタムで独自の設定を行ってください"
                    )
                    
                    if preset_choice == "カスタム":
                        # カスタム並び替え設定
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            sort_column = st.selectbox(
                                "並び替え基準",
                                options=display_columns,
                                help="データを並び替える基準の列を選択してください"
                            )
                        
                        with col2:
                            sort_order = st.selectbox(
                                "並び替え順序",
                                options=["降順", "昇順"],
                                help="降順：大きい値から小さい値へ、昇順：小さい値から大きい値へ"
                            )
                    else:
                        # プリセット使用
                        preset = preset_options[preset_choice]
                        sort_column = preset["column"]
                        sort_order = preset["order"]
                        
                        # プリセット選択が利用可能な列かチェック
                        if sort_column not in display_columns:
                            st.warning(f"⚠️ '{sort_column}' 列が選択されていません。表示する指標に追加してください。")
                            # フォールバック
                            sort_column = display_columns[0]
                            sort_order = "降順"
                    
                    # 降順の場合はascending=False、昇順の場合はascending=True
                    sort_ascending = sort_order == "昇順"
                    
                    sorted_df = metrics_df[display_columns].sort_values(
                        by=sort_column, 
                        ascending=sort_ascending
                    )
                    
                    # データフレームの表示をフォーマット
                    formatted_df = sorted_df.copy()
                    for col in ['書類通過率', '1次面接通過率', '内定率']:
                        if col in formatted_df.columns:
                            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}%")
                    
                    # 並び替え結果の表示
                    if preset_choice == "カスタム":
                        st.info(f"📊 {sort_column} を基準に {sort_order} で並び替えています")
                    else:
                        st.info(f"📊 プリセット「{preset_choice}」で並び替えています")
                    
                    st.dataframe(
                        formatted_df,
                        use_container_width=True,
                        height=400
                    )
                
                # エクスポート機能
                st.header("💾 データエクスポート")
                
                # CSV エクスポート
                try:
                    csv_buffer = io.StringIO()
                    # データの確認
                    if metrics_df.empty:
                        st.warning("⚠️ データが空のためCSVエクスポートできません")
                    else:
                        metrics_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_data = csv_buffer.getvalue()
                        
                        # ファイルサイズの確認
                        if len(csv_data) > 0:
                            st.download_button(
                                label="📥 指標データをCSVでダウンロード",
                                data=csv_data,
                                file_name=f"企業別指標_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                type="primary",
                                help="企業別の指標データをCSV形式でダウンロードします"
                            )
                        else:
                            st.error("❌ CSVデータの生成に失敗しました")
                except Exception as e:
                    st.error(f"❌ CSVエクスポートエラー: {str(e)}")
                
                # エクスポート統計
                st.info(f"📊 エクスポート可能データ: {len(metrics_df)} 企業、{len(df)} 行の原データ")
        
        with tab2:
            # AIチャット機能
            if hasattr(st.session_state, 'analysis_run'):
                render_chat_interface(df, metrics_df, openai_client)
            else:
                st.info("📊 分析ダッシュボードで「分析を実行」をクリックしてからAIチャットをご利用ください。")
    
    else:
        # 初期画面
        st.header("📋 使い方")
        st.markdown("""
        ### 🚀 このツールの特徴
        - **インテリジェントフィルタリング**: 企業・月別での詳細分析（「全て」オプション対応）
        - **🤖 AI チャット機能**: OpenAI GPT-4o-miniを使用したデータ分析チャット
        - **リアルタイムアラート**: パフォーマンスの問題を即座に検出
        - **AI搭載改善提案**: データに基づいた具体的な改善提案
        - **コンバージョンファネル**: 各段階での通過率を視覚化
        - **高度な可視化**: インタラクティブなダッシュボード
        - **トレンド分析**: 時系列での変化を追跡
        - **データ検証**: アップロード時の自動品質チェック
        - **エクスポート機能**: CSV・チャット履歴出力
        
        ### 📊 分析される指標
        - 推薦人数、書類提出数、内定数
        - 書類通過率、1次面接通過率、内定率
        - 平均処理時間、企業別パフォーマンス
        - コンバージョンファネル、通過率分析
        
        ### 🔧 必要なデータ形式
        CSVファイルに以下の列が必要です：
        - `求職者：求職者ID`, `企業：企業名`
        - `進捗：書類提出日`, `進捗：面接日`, `進捗：内定日`
        - `進捗：面接回数`, `進捗：最終面接フラグ`, `進捗：ステータス`
        
        ### 🎯 新機能
        - **🔍 「全て」フィルター**: 企業・月選択で「全て」を選択可能
        - **🤖 AI チャット**: データについて自然言語で質問
        - **💡 推奨質問**: データに基づいた質問例の提供
        - **📝 チャット履歴**: 質問と回答の履歴管理
        - **🚨 リアルタイムアラート**: 緊急度に応じた通知
        - **💡 AI改善提案**: 具体的なアクションプラン
        - **🔄 コンバージョンファネル**: 選考フロー分析
        - **📊 データ検証**: 品質チェック & スコア
        - **📈 トレンド分析**: 時系列変化の追跡
        - **🔄 強化された並び替え**: 降順・昇順対応、プリセット並び替え
        """)


if __name__ == "__main__":
    main()