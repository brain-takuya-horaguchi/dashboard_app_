#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MarkdownファイルをPDFに変換するスクリプト
"""
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import os
from pathlib import Path

def markdown_to_pdf(md_file_path, pdf_file_path=None):
    """
    MarkdownファイルをPDFに変換
    
    Args:
        md_file_path: Markdownファイルのパス
        pdf_file_path: 出力PDFファイルのパス（Noneの場合は自動生成）
    """
    # 入力ファイルの読み込み
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # MarkdownをHTMLに変換
    html = markdown.markdown(
        md_content,
        extensions=['extra', 'codehilite', 'tables', 'toc']
    )
    
    # HTMLテンプレートに埋め込む
    html_template = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>企業採用分析ダッシュボード 詳細仕様書</title>
        <style>
            @page {{
                size: A4;
                margin: 2cm;
                @top-center {{
                    content: "企業採用分析ダッシュボード 詳細仕様書";
                    font-size: 10pt;
                    color: #666;
                }}
                @bottom-center {{
                    content: "ページ " counter(page) " / " counter(pages);
                    font-size: 10pt;
                    color: #666;
                }}
            }}
            body {{
                font-family: "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo", sans-serif;
                font-size: 11pt;
                line-height: 1.6;
                color: #333;
            }}
            h1 {{
                font-size: 24pt;
                font-weight: bold;
                color: #2c3e50;
                margin-top: 30pt;
                margin-bottom: 20pt;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10pt;
            }}
            h2 {{
                font-size: 18pt;
                font-weight: bold;
                color: #34495e;
                margin-top: 25pt;
                margin-bottom: 15pt;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 8pt;
            }}
            h3 {{
                font-size: 14pt;
                font-weight: bold;
                color: #555;
                margin-top: 20pt;
                margin-bottom: 12pt;
            }}
            h4 {{
                font-size: 12pt;
                font-weight: bold;
                color: #666;
                margin-top: 15pt;
                margin-bottom: 10pt;
            }}
            h5, h6 {{
                font-size: 11pt;
                font-weight: bold;
                color: #777;
                margin-top: 12pt;
                margin-bottom: 8pt;
            }}
            p {{
                margin-bottom: 10pt;
                text-align: justify;
            }}
            ul, ol {{
                margin-bottom: 12pt;
                padding-left: 25pt;
            }}
            li {{
                margin-bottom: 6pt;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2pt 4pt;
                border-radius: 3pt;
                font-family: "Courier New", "Monaco", monospace;
                font-size: 10pt;
            }}
            pre {{
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 5pt;
                padding: 12pt;
                overflow-x: auto;
                margin-bottom: 15pt;
                font-family: "Courier New", "Monaco", monospace;
                font-size: 9pt;
                line-height: 1.4;
            }}
            pre code {{
                background-color: transparent;
                padding: 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 15pt;
                font-size: 10pt;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8pt;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            strong {{
                font-weight: bold;
                color: #2c3e50;
            }}
            em {{
                font-style: italic;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                padding-left: 15pt;
                margin-left: 0;
                margin-bottom: 15pt;
                color: #555;
                font-style: italic;
            }}
            hr {{
                border: none;
                border-top: 2px solid #ddd;
                margin: 20pt 0;
            }}
            .toc {{
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5pt;
                padding: 15pt;
                margin-bottom: 20pt;
            }}
            .toc ul {{
                list-style-type: none;
                padding-left: 0;
            }}
            .toc li {{
                margin-bottom: 5pt;
            }}
            .toc a {{
                color: #3498db;
                text-decoration: none;
            }}
            .toc a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """
    
    # PDFファイルパスの決定
    if pdf_file_path is None:
        md_path = Path(md_file_path)
        pdf_file_path = md_path.with_suffix('.pdf')
    
    # PDFを生成
    try:
        html_doc = HTML(string=html_template)
        html_doc.write_pdf(pdf_file_path)
        print(f"✅ PDFファイルが正常に生成されました: {pdf_file_path}")
        return pdf_file_path
    except Exception as e:
        print(f"❌ PDF生成エラー: {str(e)}")
        raise

if __name__ == "__main__":
    # 入力ファイル
    md_file = "仕様書.md"
    
    # 出力ファイル
    pdf_file = "仕様書.pdf"
    
    # 変換実行
    if os.path.exists(md_file):
        markdown_to_pdf(md_file, pdf_file)
    else:
        print(f"❌ ファイルが見つかりません: {md_file}")
