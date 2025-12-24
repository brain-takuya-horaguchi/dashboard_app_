import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import openai
import streamlit as st
import os
from dotenv import load_dotenv
import chardet
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# .envファイルを読み込む
load_dotenv()


def detect_encoding(file_bytes: bytes) -> str:
    """
    CSVファイルのエンコーディングを自動検出する関数
    """
    # よく使われる日本語エンコーディングのリスト
    encodings = ['utf-8', 'shift_jis', 'cp932', 'euc-jp', 'iso-2022-jp']
    
    # chardetを使用してエンコーディングを検出
    detected = chardet.detect(file_bytes)
    if detected and detected['encoding']:
        # 検出されたエンコーディングを優先的に試す
        detected_encoding = detected['encoding'].lower()
        if detected_encoding in encodings:
            encodings.insert(0, detected_encoding)
    
    # 各エンコーディングを試す
    for encoding in encodings:
        try:
            # ファイルの先頭部分をデコードしてテスト
            file_bytes[:10000].decode(encoding)
            return encoding
        except (UnicodeDecodeError, LookupError):
            continue
    
    # デフォルトとしてUTF-8を返す
    return 'utf-8'


def read_csv_with_encoding(uploaded_file) -> pd.DataFrame:
    """
    エンコーディングを自動検出してCSVファイルを読み込む関数
    """
    # ファイルをバイトとして読み込む
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # ファイルポインタをリセット
    
    # エンコーディングを検出
    encoding = detect_encoding(file_bytes)
    
    # 検出されたエンコーディングでCSVを読み込む
    try:
        df = pd.read_csv(uploaded_file, encoding=encoding)
        return df
    except Exception as e:
        # エンコーディング検出が失敗した場合、よく使われるエンコーディングを順に試す
        encodings = ['utf-8', 'shift_jis', 'cp932', 'euc-jp', 'latin-1']
        for enc in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                return df
            except:
                continue
        # すべて失敗した場合はエラーを再発生
        raise Exception(f"CSVファイルの読み込みに失敗しました。エンコーディングの検出ができませんでした。エラー: {str(e)}")

def validate_data(df: pd.DataFrame) -> Dict:
    """
    データの品質を検証する関数
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    required_columns = [
        '求職者：求職者ID', '企業：企業名', '進捗：書類提出日', 
        '進捗：面接日', '進捗：面接回数', '進捗：最終面接フラグ', 
        '進捗：内定日', '進捗：ステータス'
    ]
    
    # 必須カラムの確認
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_results['errors'].append(f"必須列が不足しています: {', '.join(missing_columns)}")
        validation_results['is_valid'] = False
    
    if not validation_results['is_valid']:
        return validation_results
    
    # データ品質チェック
    total_rows = len(df)
    
    # 1. 空データの確認
    if total_rows == 0:
        validation_results['errors'].append("データが空です")
        validation_results['is_valid'] = False
        return validation_results
    
    # 2. 重複データの確認
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        validation_results['warnings'].append(f"重複行が {duplicates} 件あります")
    
    # 3. 日付フォーマットの確認
    date_columns = ['進捗：書類提出日', '進捗：面接日', '進捗：内定日']
    for col in date_columns:
        invalid_dates = 0
        for val in df[col].dropna():
            try:
                pd.to_datetime(val)
            except:
                invalid_dates += 1
        
        if invalid_dates > 0:
            validation_results['warnings'].append(f"{col} に無効な日付が {invalid_dates} 件あります")
    
    # 4. 面接回数の確認
    invalid_interview_counts = df[df['進捗：面接回数'].notna() & (df['進捗：面接回数'] < 0)].shape[0]
    if invalid_interview_counts > 0:
        validation_results['warnings'].append(f"面接回数が負の値の行が {invalid_interview_counts} 件あります")
    
    # 5. 最終面接フラグの確認
    invalid_final_flags = df[df['進捗：最終面接フラグ'].notna() & 
                           (~df['進捗：最終面接フラグ'].isin([0, 1]))].shape[0]
    if invalid_final_flags > 0:
        validation_results['warnings'].append(f"最終面接フラグが0または1以外の行が {invalid_final_flags} 件あります")
    
    # 統計情報
    validation_results['statistics'] = {
        'total_rows': total_rows,
        'unique_candidates': df['求職者：求職者ID'].nunique(),
        'unique_companies': df['企業：企業名'].nunique(),
        'data_completeness': {
            '書類提出日': (df['進捗：書類提出日'].notna().sum() / total_rows * 100),
            '面接日': (df['進捗：面接日'].notna().sum() / total_rows * 100),
            '内定日': (df['進捗：内定日'].notna().sum() / total_rows * 100)
        }
    }
    
    return validation_results


def generate_alerts(metrics_df: pd.DataFrame) -> List[Dict]:
    """
    パフォーマンス指標に基づいてアラートを生成する関数
    """
    alerts = []
    
    if metrics_df.empty:
        return alerts
    
    # 1. 低内定率アラート
    low_success_rate = metrics_df[metrics_df['内定率'] < 5]
    if not low_success_rate.empty:
        for _, company in low_success_rate.iterrows():
            alerts.append({
                'type': 'danger',
                'title': '🚨 緊急: 極低内定率',
                'company': company['企業名'],
                'message': f"内定率が {company['内定率']:.1f}% と極端に低い状況です。即座の対策が必要です。",
                'priority': 'high'
            })
    
    # 2. 長期処理時間アラート
    slow_processing = metrics_df[metrics_df['平均処理時間'] > 60]
    if not slow_processing.empty:
        for _, company in slow_processing.iterrows():
            alerts.append({
                'type': 'warning',
                'title': '⏰ 長期処理時間',
                'company': company['企業名'],
                'message': f"平均処理時間が {company['平均処理時間']} 日と長期化しています。プロセスの見直しを検討してください。",
                'priority': 'medium'
            })
    
    # 3. 書類通過率異常アラート
    low_document_rate = metrics_df[
        (metrics_df['書類提出数'] > 5) & 
        (metrics_df['書類通過率'] < 10)
    ]
    if not low_document_rate.empty:
        for _, company in low_document_rate.iterrows():
            alerts.append({
                'type': 'warning',
                'title': '📄 書類通過率低下',
                'company': company['企業名'],
                'message': f"書類通過率が {company['書類通過率']:.1f}% と低下しています。応募書類の質向上が必要です。",
                'priority': 'medium'
            })
    
    # 4. 高パフォーマンスアラート（良いニュース）
    high_performers = metrics_df[
        (metrics_df['内定率'] > 30) & 
        (metrics_df['内定数'] > 2)
    ]
    if not high_performers.empty:
        for _, company in high_performers.iterrows():
            alerts.append({
                'type': 'success',
                'title': '🎉 高パフォーマンス',
                'company': company['企業名'],
                'message': f"内定率 {company['内定率']:.1f}% の優秀な成果を記録しています。このアプローチを他社にも展開検討してください。",
                'priority': 'low'
            })
    
    # 5. データ異常アラート
    data_anomalies = metrics_df[
        (metrics_df['書類提出数'] > 0) & 
        (metrics_df['1次面接数'] > metrics_df['書類提出数'])
    ]
    if not data_anomalies.empty:
        for _, company in data_anomalies.iterrows():
            alerts.append({
                'type': 'info',
                'title': '📊 データ異常',
                'company': company['企業名'],
                'message': "1次面接数が書類提出数を上回っています。データの確認が必要です。",
                'priority': 'low'
            })
    
    # 優先度順でソート
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    alerts.sort(key=lambda x: priority_order[x['priority']])
    
    return alerts


def generate_recommendations(metrics_df: pd.DataFrame) -> List[Dict]:
    """
    データ分析結果に基づいて改善提案を生成する関数
    """
    recommendations = []
    
    if metrics_df.empty:
        return recommendations
    
    # 1. 書類選考改善提案
    low_doc_pass_rate = metrics_df[metrics_df['書類通過率'] < 20]
    if not low_doc_pass_rate.empty:
        recommendations.append({
            'category': '書類選考',
            'title': '📄 書類選考プロセスの改善',
            'description': '書類通過率が低い企業があります。',
            'actions': [
                '応募書類のテンプレートを見直す',
                '企業のニーズに合わせた書類カスタマイズ',
                '書類作成研修の実施',
                '過去の成功事例を分析して共有'
            ],
            'target_companies': low_doc_pass_rate['企業名'].tolist()
        })
    
    # 2. 面接対策提案
    low_interview_pass_rate = metrics_df[metrics_df['1次面接通過率'] < 30]
    if not low_interview_pass_rate.empty:
        recommendations.append({
            'category': '面接対策',
            'title': '🎯 面接対策の強化',
            'description': '1次面接通過率が低い企業があります。',
            'actions': [
                '模擬面接の実施',
                '企業研究の徹底',
                '面接官のフィードバック収集',
                '面接スキル向上研修の実施'
            ],
            'target_companies': low_interview_pass_rate['企業名'].tolist()
        })
    
    # 3. 処理時間短縮提案
    slow_companies = metrics_df[metrics_df['平均処理時間'] > 45]
    if not slow_companies.empty:
        recommendations.append({
            'category': '効率化',
            'title': '⚡ 処理時間の短縮',
            'description': '選考プロセスが長期化している企業があります。',
            'actions': [
                '企業との定期的な進捗確認',
                '選考スケジュールの最適化',
                '中間フォローアップの実施',
                'デジタル化による効率化'
            ],
            'target_companies': slow_companies['企業名'].tolist()
        })
    
    # 4. 成功事例の横展開提案
    top_performers = metrics_df.nlargest(3, '内定率')
    if not top_performers.empty:
        recommendations.append({
            'category': '成功事例',
            'title': '🏆 成功事例の横展開',
            'description': '高い成果を出している企業のアプローチを他社にも適用できます。',
            'actions': [
                '成功企業のプロセス分析',
                'ベストプラクティスの文書化',
                '他社への適用可能性検討',
                '成功要因の共有セッション実施'
            ],
            'target_companies': top_performers['企業名'].tolist()
        })
    
    return recommendations


def calculate_conversion_funnel(df: pd.DataFrame, company_name: str = None) -> Dict:
    """
    コンバージョンファネルを計算する関数
    """
    if company_name:
        df = df[df['企業：企業名'] == company_name]
    
    if df.empty:
        return {
            'funnel': {},
            'conversion_rates': {}
        }
    
    # 各段階の数を計算
    funnel = {
        '推薦': df['求職者：求職者ID'].nunique(),
        '書類提出': df['進捗：書類提出日'].notna().sum(),
        '書類通過': df['進捗：面接日'].notna().sum(),
        '1次面接': df[(df['進捗：面接回数'] >= 1) & (df['進捗：面接日'].notna())].shape[0],
        '2次面接以降': df[(df['進捗：面接回数'] > 1) & (df['進捗：面接日'].notna())].shape[0],
        '最終面接': df[(df['進捗：最終面接フラグ'] == 1) & (df['進捗：面接日'].notna())].shape[0],
        '内定': df['進捗：内定日'].notna().sum()
    }
    
    # 通過率を計算（各段階間の通過率）
    conversion_rates = {}
    
    # 存在する段階のみを対象に通過率を計算
    stage_pairs = [
        ('推薦', '書類提出'),
        ('書類提出', '書類通過'),
        ('書類通過', '1次面接'),
        ('1次面接', '2次面接以降'),
        ('2次面接以降', '最終面接'),
        ('最終面接', '内定')
    ]
    
    for from_stage, to_stage in stage_pairs:
        if from_stage in funnel and to_stage in funnel and funnel[from_stage] > 0:
            conversion_rates[f"{from_stage}→{to_stage}"] = (funnel[to_stage] / funnel[from_stage]) * 100
    
    # 表示用にファネルデータから0の値を削除
    funnel = {k: v for k, v in funnel.items() if v > 0}
    
    return {
        'funnel': funnel,
        'conversion_rates': conversion_rates
    }


def export_summary_report(metrics_df: pd.DataFrame, alerts: List[Dict], recommendations: List[Dict]) -> str:
    """
    サマリーレポートを生成する関数
    """
    try:
        if metrics_df.empty:
            return "# エラー: データが空です\n\n分析対象のデータが見つかりませんでした。"
        
        # 基本統計の計算
        total_companies = len(metrics_df)
        total_recommendations = int(metrics_df['推薦人数'].sum())
        total_applications = int(metrics_df['書類提出数'].sum())
        total_offers = int(metrics_df['内定数'].sum())
        
        # 内定率の計算（ゼロ除算を回避）
        overall_rate = (total_offers / total_recommendations * 100) if total_recommendations > 0 else 0
        
        report = f"""# 企業採用分析レポート
生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

## 📊 全体サマリー
- 分析対象企業数: {total_companies}
- 総推薦人数: {total_recommendations:,}
- 総書類提出数: {total_applications:,}
- 総内定数: {total_offers:,}
- 全体内定率: {overall_rate:.1f}%

## 🎯 トップパフォーマー
"""
        
        # 上位3社の処理
        if len(metrics_df) > 0:
            top_3 = metrics_df.nlargest(min(3, len(metrics_df)), '内定率')
            for i, (_, company) in enumerate(top_3.iterrows(), 1):
                report += f"{i}. {company['企業名']}: 内定率 {company['内定率']:.1f}%\n"
        else:
            report += "データが不足しています。\n"
        
        # アラートの処理
        report += "\n## 🚨 アラート\n"
        if alerts:
            high_priority_alerts = [a for a in alerts if a.get('priority') == 'high']
            if high_priority_alerts:
                for alert in high_priority_alerts:
                    company = alert.get('company', '不明')
                    title = alert.get('title', '不明')
                    message = alert.get('message', '不明')
                    report += f"- {title}: {company} - {message}\n"
            else:
                report += "- 緊急のアラートはありません\n"
        else:
            report += "- アラートデータがありません\n"
        
        # 改善提案の処理
        report += "\n## 💡 改善提案\n"
        if recommendations:
            for rec in recommendations:
                title = rec.get('title', '不明')
                description = rec.get('description', '不明')
                actions = rec.get('actions', [])
                target_companies = rec.get('target_companies', [])
                
                report += f"### {title}\n"
                report += f"{description}\n"
                for action in actions:
                    report += f"- {action}\n"
                
                if target_companies:
                    companies_str = ', '.join(target_companies[:3])
                    if len(target_companies) > 3:
                        companies_str += f" など {len(target_companies)} 社"
                    report += f"対象企業: {companies_str}\n\n"
                else:
                    report += "対象企業: なし\n\n"
        else:
            report += "- 改善提案はありません\n"
        
        # 統計情報の追加
        report += "\n## 📈 統計情報\n"
        if len(metrics_df) > 0:
            report += f"- 最高内定率: {metrics_df['内定率'].max():.1f}%\n"
            report += f"- 最低内定率: {metrics_df['内定率'].min():.1f}%\n"
            report += f"- 平均内定率: {metrics_df['内定率'].mean():.1f}%\n"
            report += f"- 平均書類通過率: {metrics_df['書類通過率'].mean():.1f}%\n"
            report += f"- 平均処理時間: {metrics_df['平均処理時間'].mean():.1f}日\n"
        
        return report
        
    except Exception as e:
        return f"# エラー: レポート生成に失敗しました\n\nエラー詳細: {str(e)}"


def setup_openai_client():
    """OpenAI クライアントを設定"""
    try:
        # .envファイルから環境変数を読み込む
        api_key = os.environ.get("OPENAI_API_KEY")
        
        # フォールバック: Streamlit secretsまたはsession_stateから読み込む
        if not api_key:
            api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
        
        if not api_key:
            return None
        
        client = openai.OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"OpenAI APIの設定でエラーが発生しました: {str(e)}")
        return None


def create_data_summary(df: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    """データサマリーを作成してAIに提供"""
    try:
        # 基本統計
        total_rows = len(df)
        unique_companies = df['企業：企業名'].nunique()
        unique_candidates = df['求職者：求職者ID'].nunique()
        
        # 企業一覧（上位10社）
        company_list = df['企業：企業名'].value_counts().head(10).to_dict()
        
        # 指標データの要約
        metrics_summary = ""
        if not metrics_df.empty:
            metrics_summary = f"""
指標データ（上位5社）:
{metrics_df.head(5).to_string()}

全体統計:
- 総推薦人数: {metrics_df['推薦人数'].sum():,}
- 総書類提出数: {metrics_df['書類提出数'].sum():,}
- 総内定数: {metrics_df['内定数'].sum():,}
- 平均内定率: {metrics_df['内定率'].mean():.1f}%
- 平均書類通過率: {metrics_df['書類通過率'].mean():.1f}%
"""
        
        # 日付範囲
        date_columns = ['進捗：書類提出日', '進捗：面接日', '進捗：内定日']
        date_ranges = {}
        for col in date_columns:
            dates = pd.to_datetime(df[col], errors='coerce').dropna()
            if not dates.empty:
                date_ranges[col] = {
                    'min': dates.min().strftime('%Y-%m-%d'),
                    'max': dates.max().strftime('%Y-%m-%d')
                }
        
        summary = f"""
データ概要:
- 総行数: {total_rows:,}
- 企業数: {unique_companies:,}
- 求職者数: {unique_candidates:,}

主要企業 (応募件数順):
{json.dumps(company_list, ensure_ascii=False, indent=2)}

データ期間:
{json.dumps(date_ranges, ensure_ascii=False, indent=2)}

{metrics_summary}

データ構造:
- 各行は求職者と企業の組み合わせを表す
- 進捗ステータス、書類提出日、面接日、内定日などの情報を含む
- 面接回数、最終面接フラグなどの詳細情報も含む
"""
        
        return summary
    except Exception as e:
        return f"データサマリーの作成でエラーが発生しました: {str(e)}"


def query_data_with_ai(question: str, df: pd.DataFrame, metrics_df: pd.DataFrame, client) -> str:
    """AIを使用してデータに関する質問に回答"""
    try:
        # データサマリーを作成
        data_summary = create_data_summary(df, metrics_df)
        
        # プロンプトを作成
        system_prompt = f"""
あなたは企業採用分析の専門家です。以下のデータに基づいて、ユーザーの質問に正確で有用な回答を提供してください。

{data_summary}

回答の際は以下の点に注意してください:
1. データに基づいた具体的な数値を使用する
2. 分析結果から実用的な洞察を提供する
3. 可能な場合は改善提案も含める
4. 日本語で回答する
5. 不明な点があれば、データの制約を明確にする
"""
        
        user_prompt = f"""
質問: {question}

この質問について、提供されたデータを分析して回答してください。
データから読み取れる具体的な情報や傾向、そして実用的な洞察を含めて回答してください。
"""
        
        # OpenAI APIを呼び出し
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # gpt-4o-miniを使用
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"AI分析でエラーが発生しました: {str(e)}"


def get_suggested_questions(df: pd.DataFrame, metrics_df: pd.DataFrame) -> List[str]:
    """データに基づいて推奨質問を生成"""
    suggestions = [
        "最もパフォーマンスが良い企業はどこですか？",
        "内定率を改善するためのアドバイスを教えてください",
        "書類通過率が低い企業の特徴は何ですか？",
        "処理時間が長い企業について分析してください",
        "月別のトレンドについて教えてください",
        "どの企業に最も力を入れるべきですか？",
        "選考プロセスで最も課題となっている段階はどこですか？",
        "成功している企業の共通点は何ですか？"
    ]
    
    # データに基づいた動的な質問を追加
    if not metrics_df.empty:
        # 最も内定率の高い企業
        best_company = metrics_df.loc[metrics_df['内定率'].idxmax(), '企業名']
        suggestions.append(f"{best_company}が成功している理由は何ですか？")
        
        # 最も内定率の低い企業
        worst_company = metrics_df.loc[metrics_df['内定率'].idxmin(), '企業名']
        suggestions.append(f"{worst_company}の改善点は何ですか？")
    
    return suggestions


def save_chat_history(question: str, answer: str):
    """チャット履歴を保存"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'question': question,
        'answer': answer
    })


def export_chat_history() -> str:
    """チャット履歴をエクスポート"""
    try:
        if 'chat_history' not in st.session_state or not st.session_state.chat_history:
            return "# チャット履歴\n\n**状態**: 履歴がありません\n\n質問を送信してからエクスポートしてください。"
        
        export_text = f"""# チャット履歴
生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
履歴件数: {len(st.session_state.chat_history)} 件

---

"""
        
        for i, chat in enumerate(st.session_state.chat_history, 1):
            timestamp = chat.get('timestamp', '不明')
            question = chat.get('question', '質問データが見つかりません')
            answer = chat.get('answer', '回答データが見つかりません')
            
            export_text += f"## 質問 {i} ({timestamp})\n\n"
            export_text += f"**🙋‍♀️ 質問:**\n{question}\n\n"
            export_text += f"**🤖 回答:**\n{answer}\n\n"
            export_text += "---\n\n"
        
        # 統計情報の追加
        export_text += f"""## 📊 統計情報
- 総質問数: {len(st.session_state.chat_history)}
- 最初の質問: {st.session_state.chat_history[0].get('timestamp', '不明')}
- 最後の質問: {st.session_state.chat_history[-1].get('timestamp', '不明')}
"""
        
        return export_text
        
    except Exception as e:
        return f"# エラー: チャット履歴のエクスポートに失敗しました\n\nエラー詳細: {str(e)}"


def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """
    データフレーム内で可能なカラム名を検索する関数
    """
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def apply_filters(df: pd.DataFrame, selected_companies=None, selected_months=None) -> pd.DataFrame:
    """
    企業と月のフィルターを適用する関数
    """
    if df.empty:
        return df
    
    # 日付カラムのパース
    df = df.copy()
    df['書類提出日_parsed'] = pd.to_datetime(df['進捗：書類提出日'], errors='coerce')
    df['面接日_parsed'] = pd.to_datetime(df['進捗：面接日'], errors='coerce')
    df['内定日_parsed'] = pd.to_datetime(df['進捗：内定日'], errors='coerce')
    
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
    
    return df


def calculate_company_introduction_to_contract_rate(df: pd.DataFrame, selected_companies=None, selected_months=None) -> pd.DataFrame:
    """
    企業ごとの紹介～成約率を計算
    成約 = 内定と仮定
    紹介 = 応募OK日がある求職者、または推薦された求職者（応募OK日がない場合は推薦人数）
    """
    if df.empty:
        return pd.DataFrame()
    
    # フィルターを適用
    df = apply_filters(df, selected_companies, selected_months)
    
    if df.empty:
        return pd.DataFrame()
    
    # 日付のパース
    df['内定日_parsed'] = pd.to_datetime(df['進捗：内定日'], errors='coerce')
    df['応募OK日_parsed'] = pd.to_datetime(df['進捗：応募OK日'], errors='coerce')
    
    company_stats = []
    for company in df['企業：企業名'].unique():
        company_data = df[df['企業：企業名'] == company]
        
        # 紹介数: 応募OK日がある求職者数、なければ推薦人数（ユニークな求職者ID数）
        if '進捗：応募OK日' in df.columns:
            紹介数 = company_data[company_data['応募OK日_parsed'].notna()]['求職者：求職者ID'].nunique()
            if 紹介数 == 0:
                # 応募OK日がない場合は推薦人数を使用
                紹介数 = company_data['求職者：求職者ID'].nunique()
        else:
            紹介数 = company_data['求職者：求職者ID'].nunique()  # 推薦人数 = 紹介数
        
        成約数 = company_data['内定日_parsed'].notna().sum()  # 内定数 = 成約数
        成約率 = (成約数 / 紹介数 * 100) if 紹介数 > 0 else 0.0
        
        company_stats.append({
            '企業名': company,
            '紹介数': 紹介数,
            '成約数': 成約数,
            '成約率': 成約率
        })
    
    return pd.DataFrame(company_stats)


def calculate_job_introduction_to_contract_rate(df: pd.DataFrame, selected_companies=None, selected_months=None) -> pd.DataFrame:
    """
    求人ごとの紹介～成約率を計算
    求人IDが存在しない場合は、企業名+求職者IDの組み合わせで代替
    """
    if df.empty:
        return pd.DataFrame()
    
    # フィルターを適用
    df = apply_filters(df, selected_companies, selected_months)
    
    if df.empty:
        return pd.DataFrame()
    
    # 求人IDカラムを探す
    job_id_col = find_column(df, ['求人：求人ID', '求人ID', '求人：ID', 'job_id', 'JobID'])
    
    if job_id_col:
        # 求人IDが存在する場合
        df['内定日_parsed'] = pd.to_datetime(df['進捗：内定日'], errors='coerce')
        df['応募OK日_parsed'] = pd.to_datetime(df['進捗：応募OK日'], errors='coerce')
        
        job_stats = []
        for job_id in df[job_id_col].dropna().unique():
            job_data = df[df[job_id_col] == job_id]
            
            # 紹介数: 応募OK日がある求職者数、なければ推薦人数
            if '進捗：応募OK日' in df.columns:
                紹介数 = job_data[job_data['応募OK日_parsed'].notna()]['求職者：求職者ID'].nunique()
                if 紹介数 == 0:
                    紹介数 = job_data['求職者：求職者ID'].nunique()
            else:
                紹介数 = job_data['求職者：求職者ID'].nunique()
            
            成約数 = job_data['内定日_parsed'].notna().sum()
            成約率 = (成約数 / 紹介数 * 100) if 紹介数 > 0 else 0.0
            
            job_stats.append({
                '求人ID': job_id,
                '紹介数': 紹介数,
                '成約数': 成約数,
                '成約率': 成約率
            })
        
        return pd.DataFrame(job_stats)
    else:
        # 求人IDが存在しない場合、企業名で代替
        return calculate_company_introduction_to_contract_rate(df, selected_companies, selected_months)


def calculate_avg_recommendations_per_candidate(df: pd.DataFrame, selected_companies=None, selected_months=None) -> Dict:
    """
    求職者1人当たりの平均推薦数を計算
    """
    if df.empty:
        return {'avg_recommendations': 0, 'total_candidates': 0, 'total_recommendations': 0}
    
    # フィルターを適用
    df = apply_filters(df, selected_companies, selected_months)
    
    if df.empty:
        return {'avg_recommendations': 0, 'total_candidates': 0, 'total_recommendations': 0}
    
    total_candidates = df['求職者：求職者ID'].nunique()
    total_recommendations = len(df)
    avg_recommendations = total_recommendations / total_candidates if total_candidates > 0 else 0
    
    return {
        'avg_recommendations': avg_recommendations,
        'total_candidates': total_candidates,
        'total_recommendations': total_recommendations
    }


def calculate_interview_to_recommendation_leadtime(df: pd.DataFrame, selected_companies=None, selected_months=None) -> pd.DataFrame:
    """
    面談から推薦までのリードタイムを計算
    面談日と推薦日（書類提出日）の差を計算
    """
    if df.empty:
        return pd.DataFrame()
    
    # フィルターを適用
    df = apply_filters(df, selected_companies, selected_months)
    
    if df.empty:
        return pd.DataFrame()
    
    # 面談日カラムを探す（実際のカラム名に合わせる）
    interview_date_col = find_column(df, [
        '求職者：面談日',  # 実際のカラム名
        '面談：面談日', '面談日', '面談：日付', 'interview_date', 
        '面談：実施日', 'CA：面談日', '求職者：電話面談日'
    ])
    
    # 推薦日 = 書類提出日と仮定
    df['推薦日_parsed'] = pd.to_datetime(df['進捗：書類提出日'], errors='coerce')
    
    if interview_date_col:
        df['面談日_parsed'] = pd.to_datetime(df[interview_date_col], errors='coerce')
        
        # 両方の日付が存在する行のみ
        valid_data = df[(df['面談日_parsed'].notna()) & (df['推薦日_parsed'].notna())].copy()
        
        if not valid_data.empty:
            valid_data['リードタイム'] = (valid_data['推薦日_parsed'] - valid_data['面談日_parsed']).dt.days
            
            leadtime_stats = []
            for company in valid_data['企業：企業名'].unique():
                company_data = valid_data[valid_data['企業：企業名'] == company]
                avg_leadtime = company_data['リードタイム'].mean()
                
                leadtime_stats.append({
                    '企業名': company,
                    '平均リードタイム': avg_leadtime,
                    '件数': len(company_data)
                })
            
            return pd.DataFrame(leadtime_stats)
    
    return pd.DataFrame()


def calculate_interviews_by_ca(df: pd.DataFrame, selected_companies=None, selected_months=None) -> pd.DataFrame:
    """
    CAごとの面談数を計算
    """
    if df.empty:
        return pd.DataFrame()
    
    # フィルターを適用
    df = apply_filters(df, selected_companies, selected_months)
    
    if df.empty:
        return pd.DataFrame()
    
    # CAカラムを探す（実際のカラム名に合わせる）
    ca_col = find_column(df, [
        '求職者：担当者',  # 実際のカラム名
        'CA：CA名', 'CA名', 'CA：名前', 'CA', 'ca_name', 'CA：担当者',
        'キャリアアドバイザー', 'CA：ID', 'CA_ID', '求職者：担当チーム'
    ])
    
    if ca_col:
        # 面談日カラムを探す（実際のカラム名に合わせる）
        interview_date_col = find_column(df, [
            '求職者：面談日',  # 実際のカラム名
            '面談：面談日', '面談日', '面談：日付', 'interview_date',
            '面談：実施日', 'CA：面談日', '求職者：電話面談日'
        ])
        
        if interview_date_col:
            df['面談日_parsed'] = pd.to_datetime(df[interview_date_col], errors='coerce')
            valid_data = df[df['面談日_parsed'].notna()]
        else:
            # 面談日がなくても、CAごとに集計
            valid_data = df
        
        ca_stats = valid_data.groupby(ca_col).agg({
            '求職者：求職者ID': 'nunique',
            '企業：企業名': 'nunique'
        }).reset_index()
        
        ca_stats.columns = ['CA名', '面談数', '担当企業数']
        
        return ca_stats
    else:
        return pd.DataFrame()


def calculate_scouter_performance(df: pd.DataFrame, selected_companies=None, selected_months=None) -> pd.DataFrame:
    """
    スカウターのパフォーマンス測定
    """
    if df.empty:
        return pd.DataFrame()
    
    # フィルターを適用
    df = apply_filters(df, selected_companies, selected_months)
    
    if df.empty:
        return pd.DataFrame()
    
    # スカウターカラムを探す（実際のカラム名に合わせる）
    scouter_col = find_column(df, [
        'スカウト担当者',  # 実際のカラム名
        'スカウター：スカウター名', 'スカウター名', 'スカウター：名前', 
        'スカウター', 'scouter_name', 'スカウター：担当者',
        'スカウター：ID', 'スカウター_ID'
    ])
    
    if scouter_col:
        df['内定日_parsed'] = pd.to_datetime(df['進捗：内定日'], errors='coerce')
        
        scouter_stats = []
        for scouter in df[scouter_col].dropna().unique():
            scouter_data = df[df[scouter_col] == scouter]
            
            紹介数 = scouter_data['求職者：求職者ID'].nunique()
            成約数 = scouter_data['内定日_parsed'].notna().sum()
            成約率 = (成約数 / 紹介数 * 100) if 紹介数 > 0 else 0.0
            
            # 書類提出数
            書類提出数 = scouter_data['進捗：書類提出日'].notna().sum()
            書類提出率 = (書類提出数 / 紹介数 * 100) if 紹介数 > 0 else 0.0
            
            scouter_stats.append({
                'スカウター名': scouter,
                '紹介数': 紹介数,
                '書類提出数': 書類提出数,
                '書類提出率': 書類提出率,
                '成約数': 成約数,
                '成約率': 成約率
            })
        
        return pd.DataFrame(scouter_stats)
    else:
        return pd.DataFrame()


def create_company_introduction_contract_chart(df: pd.DataFrame, selected_companies=None, selected_months=None, sort_by='紹介数', sort_order='降順', limit=10) -> go.Figure:
    """
    企業ごとの紹介～成約率グラフを作成
    
    Parameters:
    - sort_by: 並び替え基準 ('紹介数', '成約数', '成約率')
    - sort_order: 並び替え順序 ('昇順', '降順')
    - limit: 表示件数
    """
    stats_df = calculate_company_introduction_to_contract_rate(df, selected_companies, selected_months)
    
    if stats_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="データがありません", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # 並び替え
    ascending = (sort_order == '昇順')
    if sort_by in stats_df.columns:
        stats_df = stats_df.sort_values(sort_by, ascending=ascending)
    
    # 指定件数を表示
    top_companies = stats_df.head(limit)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='紹介数',
        x=top_companies['企業名'],
        y=top_companies['紹介数'],
        marker_color='rgba(102, 126, 234, 0.7)',
        text=top_companies['紹介数'],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='成約数',
        x=top_companies['企業名'],
        y=top_companies['成約数'],
        marker_color='rgba(76, 175, 80, 0.7)',
        text=top_companies['成約数'],
        textposition='outside'
    ))
    
    # 成約率を折れ線グラフで追加
    fig.add_trace(go.Scatter(
        name='成約率',
        x=top_companies['企業名'],
        y=top_companies['成約率'],
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title='📊 企業ごとの紹介～成約率',
        xaxis_title='企業名',
        yaxis_title='人数',
        yaxis2=dict(title='成約率 (%)', overlaying='y', side='right'),
        barmode='group',
        height=500,
        template='plotly_white',
        xaxis=dict(tickangle=45)
    )
    
    return fig


def create_job_introduction_contract_chart(df: pd.DataFrame, selected_companies=None, selected_months=None, sort_by='紹介数', sort_order='降順', limit=15) -> go.Figure:
    """
    求人ごとの紹介～成約率グラフを作成
    
    Parameters:
    - sort_by: 並び替え基準 ('紹介数', '成約数', '成約率')
    - sort_order: 並び替え順序 ('昇順', '降順')
    - limit: 表示件数
    """
    stats_df = calculate_job_introduction_to_contract_rate(df, selected_companies, selected_months)
    
    if stats_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="データがありません", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # 並び替え
    ascending = (sort_order == '昇順')
    if sort_by in stats_df.columns:
        stats_df = stats_df.sort_values(sort_by, ascending=ascending)
    
    # 指定件数を表示
    top_jobs = stats_df.head(limit)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='紹介数',
        x=top_jobs['求人ID'].astype(str),
        y=top_jobs['紹介数'],
        marker_color='rgba(102, 126, 234, 0.7)',
        text=top_jobs['紹介数'],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='成約数',
        x=top_jobs['求人ID'].astype(str),
        y=top_jobs['成約数'],
        marker_color='rgba(76, 175, 80, 0.7)',
        text=top_jobs['成約数'],
        textposition='outside'
    ))
    
    # 成約率を折れ線グラフで追加
    fig.add_trace(go.Scatter(
        name='成約率',
        x=top_jobs['求人ID'].astype(str),
        y=top_jobs['成約率'],
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title='📊 求人ごとの紹介～成約率',
        xaxis_title='求人ID',
        yaxis_title='人数',
        yaxis2=dict(title='成約率 (%)', overlaying='y', side='right'),
        barmode='group',
        height=500,
        template='plotly_white',
        xaxis=dict(tickangle=45)
    )
    
    return fig


def create_avg_recommendations_chart(df: pd.DataFrame, selected_companies=None, selected_months=None) -> go.Figure:
    """
    求職者1人当たりの平均推薦数グラフを作成
    """
    stats = calculate_avg_recommendations_per_candidate(df, selected_companies, selected_months)
    
    if stats['total_candidates'] == 0:
        fig = go.Figure()
        fig.add_annotation(text="データがありません", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # フィルターを適用（グラフ表示用）
    filtered_df = apply_filters(df, selected_companies, selected_months)
    
    # 企業ごとの平均推薦数を計算
    company_stats = []
    for company in filtered_df['企業：企業名'].unique():
        company_data = filtered_df[filtered_df['企業：企業名'] == company]
        candidates = company_data['求職者：求職者ID'].nunique()
        recommendations = len(company_data)
        avg = recommendations / candidates if candidates > 0 else 0
        
        company_stats.append({
            '企業名': company,
            '平均推薦数': avg,
            '求職者数': candidates,
            '推薦数': recommendations
        })
    
    company_df = pd.DataFrame(company_stats)
    top_companies = company_df.nlargest(10, '平均推薦数')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_companies['企業名'],
        y=top_companies['平均推薦数'],
        marker=dict(
            color=top_companies['平均推薦数'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="平均推薦数")
        ),
        text=[f"{val:.2f}" for val in top_companies['平均推薦数']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'📊 求職者1人当たりの平均推薦数（全体平均: {stats["avg_recommendations"]:.2f}）',
        xaxis_title='企業名',
        yaxis_title='平均推薦数',
        height=500,
        template='plotly_white',
        xaxis=dict(tickangle=45)
    )
    
    return fig


def create_leadtime_chart(df: pd.DataFrame, selected_companies=None, selected_months=None) -> go.Figure:
    """
    面談から推薦までのリードタイムグラフを作成
    """
    stats_df = calculate_interview_to_recommendation_leadtime(df, selected_companies, selected_months)
    
    if stats_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="面談日データが見つかりません。「求職者：面談日」カラムが存在するか確認してください。",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # 上位10社を表示
    top_companies = stats_df.nlargest(10, '件数')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_companies['企業名'],
        y=top_companies['平均リードタイム'],
        marker=dict(
            color=top_companies['平均リードタイム'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="平均リードタイム (日)")
        ),
        text=[f"{val:.1f}日" for val in top_companies['平均リードタイム']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='⏱️ 面談から推薦までのリードタイム（企業別）',
        xaxis_title='企業名',
        yaxis_title='平均リードタイム (日)',
        height=500,
        template='plotly_white',
        xaxis=dict(tickangle=45)
    )
    
    return fig


def create_ca_interviews_chart(df: pd.DataFrame, selected_companies=None, selected_months=None) -> go.Figure:
    """
    面談数（CAごと）グラフを作成
    """
    stats_df = calculate_interviews_by_ca(df, selected_companies, selected_months)
    
    if stats_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="CAデータが見つかりません。「求職者：担当者」カラムが存在するか確認してください。",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # 面談数でソート
    stats_df = stats_df.sort_values('面談数', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=stats_df['CA名'],
        y=stats_df['面談数'],
        marker=dict(
            color=stats_df['面談数'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="面談数")
        ),
        text=stats_df['面談数'],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='👥 面談数（CAごと）',
        xaxis_title='CA名',
        yaxis_title='面談数',
        height=500,
        template='plotly_white',
        xaxis=dict(tickangle=45)
    )
    
    return fig


def create_scouter_performance_chart(df: pd.DataFrame, selected_companies=None, selected_months=None) -> go.Figure:
    """
    スカウターのパフォーマンス測定グラフを作成
    """
    stats_df = calculate_scouter_performance(df, selected_companies, selected_months)
    
    if stats_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="スカウターデータが見つかりません。「スカウト担当者」カラムが存在するか確認してください。",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # 紹介数でソート
    stats_df = stats_df.sort_values('紹介数', ascending=False)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('紹介数', '成約率', '書類提出率', '成約数'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # 紹介数
    fig.add_trace(
        go.Bar(x=stats_df['スカウター名'], y=stats_df['紹介数'], marker_color='rgba(102, 126, 234, 0.7)'),
        row=1, col=1
    )
    
    # 成約率
    fig.add_trace(
        go.Bar(x=stats_df['スカウター名'], y=stats_df['成約率'], marker_color='rgba(76, 175, 80, 0.7)'),
        row=1, col=2
    )
    
    # 書類提出率
    fig.add_trace(
        go.Bar(x=stats_df['スカウター名'], y=stats_df['書類提出率'], marker_color='rgba(255, 152, 0, 0.7)'),
        row=2, col=1
    )
    
    # 成約数
    fig.add_trace(
        go.Bar(x=stats_df['スカウター名'], y=stats_df['成約数'], marker_color='rgba(156, 39, 176, 0.7)'),
        row=2, col=2
    )
    
    fig.update_layout(
        title='🎯 スカウターのパフォーマンス測定',
        height=700,
        template='plotly_white',
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=2)
    
    fig.update_yaxes(title_text="人数", row=1, col=1)
    fig.update_yaxes(title_text="率 (%)", row=1, col=2)
    fig.update_yaxes(title_text="率 (%)", row=2, col=1)
    fig.update_yaxes(title_text="人数", row=2, col=2)
    
    return fig 