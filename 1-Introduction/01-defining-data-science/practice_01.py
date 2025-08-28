# -*- coding: utf-8 -*-
"""
Korean WordCloud from CSV
----------------------------------
1) 读取 CSV -> 指定列
2) 清洗文本：仅保留韩文字符和空格（尽量保留词间空格）
3) soynlp.WordExtractor 学习 -> 用 LTokenizer 分词
4) 统计词频并用 WordCloud 可视化

用法示例：
python practice_01.py --csv "H:/Study/your.csv" --col text --font "C:/Windows/Fonts/malgun.ttf"

提示：
- 如果你的列名不是 text，请用 --col 修改
- Windows 字体通常用 'C:/Windows/Fonts/malgun.ttf'
- Mac 可用 '/System/Library/Fonts/AppleSDGothicNeo.ttc'
- Linux 可安装 Nanum 字体并设置路径
"""

import argparse
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer


def clean_korean(text: str) -> str:
    """
    仅保留韩文相关字符与空格，并合并多余空格
    覆盖范围：
    - Hangul Syllables:      \uAC00-\uD7A3
    - Hangul Jamo:           \u1100-\u11FF
    - Hangul Compatibility:  \u3130-\u318F
    - Halfwidth Jamo:        \uFFA0-\uFFDC
    """
    if not isinstance(text, str):
        return ""
    # 非韩文字符替换为空格（保留原有的词间空格）
    cleaned = re.sub(r"[^\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F\uFFA0-\uFFDC\s]", " ", text)
    # 合并多余空格
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def build_tokenizer(docs, min_count=2, min_cohesion_forward=0.05):
    """
    用 WordExtractor 学习词边界并构建 LTokenizer
    """
    wx = WordExtractor(min_frequency=min_count, min_cohesion_forward=min_cohesion_forward)
    wx.train(docs)
    word_scores = wx.extract()

    # 用 cohesion_forward 作为得分来构建分词器
    scores = {w: s.cohesion_forward for w, s in word_scores.items()}
    tokenizer = LTokenizer(scores)
    return tokenizer

def tokenize_and_count(docs, tokenizer, stopwords=None, min_len=2):
    """
    用 LTokenizer 分词并统计词频
    """
    stopwords = set(stopwords or [])
    counter = Counter()
    for line in docs:
        if not line:
            continue
        tokens = tokenizer.tokenize(line)
        for t in tokens:
            # 统计词长大于最小词长且非停词的词汇出现的次数
            if len(t) >= min_len and t not in stopwords:
                counter[t] += 1
    return counter


def main():
    parser = argparse.ArgumentParser(description="Korean WordCloud from CSV")
    parser.add_argument("--csv", required=True, help="CSV 文件路径")
    parser.add_argument("--col", default="text", help="要读取的文本列名（默认：text）")
    parser.add_argument("--font", default="C:/Windows/Fonts/malgun.ttf", help="韩文字体路径（防乱码）")
    parser.add_argument("--topn", type=int, default=200, help="取前 N 个高频词生成词云（默认：200）")
    parser.add_argument("--min_len", type=int, default=2, help="最短词长过滤（默认：2）")
    parser.add_argument("--min_count", type=int, default=4, help="WordExtractor 最小词频（默认：4）")
    parser.add_argument("--min_cohesion", type=float, default=0.15, help="WordExtractor 词汇紧凑度 cohesion 阈值（默认：0.15）")
    parser.add_argument("--bg", default="white", help="词云背景色（默认：white）")
    args = parser.parse_args()

    # 1) 读取 CSV
    df = pd.read_csv(args.csv)
    if args.col not in df.columns:
        raise ValueError(f"列名 '{args.col}' 不在 CSV 中，请使用 --col 指定正确的列名。现有列：{list(df.columns)}")

    raw_texts = df[args.col].astype(str).tolist()

    # 2) 清洗文本 -> 训练语料
    cleaned = [clean_korean(t) for t in raw_texts]
    # 只保留非空行
    docs = [t for t in cleaned if t]

    if not docs:
        raise ValueError("清洗后没有可用韩文文本，请检查数据或清洗规则。")

    stopwords = {
        # 연결어/접속부사
        "그리고", "그러나", "하지만", "그래서", "또한", "또", "그러므로", "따라서",
        "즉", "그러면", "만약", "왜냐하면",

        # 조사 (토큰화에서 떨어질 수 있는 경우)
        "이", "가", "은", "는", "을", "를", "에", "에서", "에게", "께", 
        "으로", "로", "와", "과", "도", "만", "보다", "부터", "까지", "조차", "처럼",

        # 계사/동사 (이다 시제·활용 포함)
        "이다", "입니다", "이었다", "이었습니다", "인", "인것", "인것처럼",
        "한다", "하다", "했다", "하였다", "하는", "하고", "하니", "하면", "할", "하려고",
        "된다", "되다", "된", "되는", "될", "되었다", "되었습니다",
        "있다", "있습니다", "있었다", "있었던", "있는", "있을", "없다", "없습니다", "없는", "없었던",

        # 빈번한 보조동사 표현
        "것이다", "것입니다", "것이었다", "것이었습니다",
        "수있다", "수있습니다", "수없다", "수없습니다",

        # 빈번副詞
        "정도", "매우", "아주", "너무", "진짜", "정말", "그냥", "모두", "다", "항상",

        # 지시대명사
        "이", "그", "저", "이런", "그런", "저런", "이것", "그것", "저것",
        "여기", "거기", "저기",

        # 시간 관련 흔한 단어
        "오늘", "내일", "어제", "지금", "현재", "당시", "경우", "때문", "동안", "앞으로",
    }

    # 3) 训练 tokenizer
    tokenizer = build_tokenizer(
        docs,
        min_count=args.min_count,
        min_cohesion_forward=args.min_cohesion
    )

    # 4) 分词 + 统计词频
    counter = tokenize_and_count(docs, tokenizer, stopwords=stopwords, min_len=args.min_len)

    if not counter:
        raise ValueError("分词后未统计到词频，请检查停用词或最短词长过滤设置。")

    # 仅取前 N 高频词用于词云（过多词会影响美观与速度）
    most_common = counter.most_common(args.topn)
    freq_dict = dict(most_common)

    # 生成词云
    wc = WordCloud(
        font_path=args.font,  # 必须设置韩文字体
        background_color=args.bg,
        width=1300,
        height=900
    )
    wc.generate_from_frequencies(freq_dict)

    # 展示
    plt.figure(figsize=(20, 10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
