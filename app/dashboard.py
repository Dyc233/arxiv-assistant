"""分析看板 — KPI指标卡 + 图表分组布局"""
from pathlib import Path
import pandas as pd
import streamlit as st

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"


@st.cache_data
def load_kpi():
    conf = pd.read_csv(ANALYSIS_DIR / "confs" / "conference_distribution.csv")
    conf_count = conf["conference"].nunique()
    conf_names = ", ".join(conf.sort_values("count", ascending=False)["conference"].head(5).tolist())
    # 从 parquet 读取实时数据
    df = pd.read_parquet(ANALYSIS_DIR / "cleaned_papers.parquet", columns=["publish_date"])
    total_papers = len(df)
    dates = df["publish_date"].dropna()
    date_min = str(dates.min())[:7] if len(dates) > 0 else "-"
    date_max = str(dates.max())[:10] if len(dates) > 0 else "-"
    return total_papers, date_min, date_max, conf_count, conf_names


def render_dashboard():
    total_papers, date_min, date_max, conf_count, conf_names = load_kpi()

    # KPI 指标行
    metric_cols = st.columns(4)
    metric_cols[0].metric("📄 论文总量", f"{total_papers:,}")
    metric_cols[1].metric("🏆 覆盖顶会", conf_count)
    metric_cols[2].metric("📅 截至时间", f"{date_max}")
    #metric_cols[3].metric("🔥 高频顶会", conf_names[:30])

    st.divider()

    # 布局：每行一个主题区块，图表用列排版
    _section_header("📊 领域热点 & 发文趋势")
    c1, c2 = st.columns(2)
    with c1:
        _chart("词云：领域热点分布", ANALYSIS_DIR / "wordcloud" / "global_wordcloud.png")
    with c2:
        _chart("月度发文量趋势", ANALYSIS_DIR / "trends" / "publish_trend.png")

    _section_header("🏅 顶会录用 & 高产作者")
    c3, c4 = st.columns(2)
    with c3:
        _chart("顶会录用分布", ANALYSIS_DIR / "confs" / "conference_distribution.png")
    with c4:
        _chart("高产作者 Top 20", ANALYSIS_DIR / "authors" / "top_authors.png")

    _section_header("🌐 跨学科渗透分析")
    c5, c6 = st.columns(2)
    with c5:
        _chart("非CS标签 Top 20", ANALYSIS_DIR / "crossdomain" / "crossdomain_top_categories.png")
    with c6:
        _chart("渗透率随时间变化", ANALYSIS_DIR / "crossdomain" / "crossdomain_penetration_rate.png")

    _section_header("🔥 关键词趋势热力图")
    _chart("技术术语兴衰 (行归一化)", ANALYSIS_DIR / "keyword_trend" / "keyword_trend.png")


def _section_header(title: str):
    st.markdown(f"### {title}")


def _chart(caption: str, path: Path):
    if path.exists():
        st.image(str(path), width="stretch")
        st.caption(caption)
    else:
        st.warning(f"图表未生成: {caption}")
