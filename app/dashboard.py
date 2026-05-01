"""分析看板 — Tab分组 + Plotly交互图"""
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
TEMPLATE = "plotly_white"


@st.cache_data
def load_kpi():
    conf = pd.read_csv(ANALYSIS_DIR / "confs" / "conference_distribution.csv")
    conf_count = conf["conference"].nunique()
    df = pd.read_parquet(ANALYSIS_DIR / "cleaned_papers.parquet", columns=["publish_date"])
    total_papers = len(df)
    dates = df["publish_date"].dropna()
    date_min = str(dates.min())[:7] if len(dates) > 0 else "-"
    date_max = str(dates.max())[:10] if len(dates) > 0 else "-"
    return total_papers, date_min, date_max, conf_count


@st.cache_data
def _df_publish_trend():
    return pd.read_csv(ANALYSIS_DIR / "trends" / "publish_trend.csv")


@st.cache_data
def _df_keyword_trend():
    df = pd.read_csv(ANALYSIS_DIR / "keyword_trend" / "keyword_trend.csv", index_col=0)
    df = df.astype(float)
    top_terms = df.max(axis=1).nlargest(18).index
    norm = df.loc[top_terms].copy()
    for idx in norm.index:
        row_max = norm.loc[idx].max()
        if row_max > 0:
            norm.loc[idx] = norm.loc[idx] / row_max
    return norm


@st.cache_data
def _df_confs():
    return pd.read_csv(ANALYSIS_DIR / "confs" / "conference_distribution.csv")


@st.cache_data
def _df_authors():
    return pd.read_csv(ANALYSIS_DIR / "authors" / "top_authors.csv")


@st.cache_data
def _df_submission_heatmap():
    df = pd.read_parquet(ANALYSIS_DIR / "cleaned_papers.parquet",
                         columns=["publish_hour", "publish_date"])
    df = df.dropna(subset=["publish_hour"])
    df["publish_hour"] = df["publish_hour"].astype(int)
    df["year"] = pd.to_datetime(df["publish_date"], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    # 过滤样本太少的年份
    year_counts = df["year"].value_counts()
    valid_years = sorted(y for y in year_counts.index if year_counts[y] > 50)
    df = df[df["year"].isin(valid_years)]
    # year×hour 透视表，行归一化
    pivot = df.pivot_table(index="year", columns="publish_hour", aggfunc="size", fill_value=0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    return pivot


@st.cache_data
def _df_crossdomain_cats():
    df = pd.read_csv(ANALYSIS_DIR / "crossdomain" / "crossdomain_top_categories.csv")
    return df.head(20)


@st.cache_data
def _df_crossdomain_rate():
    df = pd.read_csv(ANALYSIS_DIR / "crossdomain" / "crossdomain_penetration_rate.csv")
    cols = [c for c in ["eess", "q-bio", "q-fin", "stat", "math", "physics"] if c in df.columns]
    return df, cols


def render_dashboard():
    total_papers, date_min, date_max, conf_count = load_kpi()

    metric_cols = st.columns(4)
    metric_cols[0].metric("📄 论文总量", f"{total_papers:,}")
    metric_cols[1].metric("🏆 覆盖顶会/期刊", conf_count)
    metric_cols[2].metric("📅 截止日期", f"{date_max}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 热点与趋势", "🏅 学者与会议", "🌐 跨学科"])

    with tab1:
        st.caption("NLP 领域论文发表趋势与关键词热度变化。")
        c1, c2 = st.columns(2)
        with c1:
            _fig_publish_trend()
        with c2:
            _fig_keyword_heatmap()
        _fig_wordcloud()

    with tab2:
        st.caption("顶会录用分布、高产作者与提交时段分布。")
        c1, c2 = st.columns(2)
        with c1:
            _fig_confs()
        with c2:
            _fig_authors()
        _fig_submission_heatmap()

    with tab3:
        st.caption("NLP 论文向其他学科的渗透情况。")
        c1, c2 = st.columns(2)
        with c1:
            _fig_crossdomain_cats()
        with c2:
            _fig_crossdomain_rate()


# ---- 图表函数 ----


def _fig_publish_trend():
    df = _df_publish_trend()
    fig = px.line(df, x="month_str", y="count", markers=False,
                  labels={"month_str": "", "count": "发文量"})
    fig.update_traces(line_color="#3366CC", hovertemplate="%{x}<br>发文量: %{y}")
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), template=TEMPLATE,
                      title="月度发文量趋势", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def _fig_keyword_heatmap():
    norm = _df_keyword_trend()
    fig = px.imshow(norm.values, x=norm.columns, y=norm.index,
                    color_continuous_scale="Blues",
                    labels={"x": "", "y": "", "color": "相对热度"})
    fig.update_xaxes(tickangle=-45, dtick=4)
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), template=TEMPLATE,
                      title="关键词热度热力图（行归一化）")
    st.plotly_chart(fig, use_container_width=True)


def _fig_wordcloud():
    wc_path = ANALYSIS_DIR / "wordcloud" / "global_wordcloud.png"
    if wc_path.exists():
        st.image(str(wc_path), width="stretch")
        st.caption("词云：领域热点分布")
    else:
        st.warning("词云未生成")


def _fig_confs():
    df = _df_confs()
    fig = px.bar(df, x="count", y="conference", orientation="h",
                 labels={"count": "论文数", "conference": ""},
                 color_discrete_sequence=["#3366CC"] * len(df))
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), template=TEMPLATE,
                      title="顶会/期刊录用分布", yaxis=dict(categoryorder="total ascending"))
    fig.update_traces(hovertemplate="%{y}: %{x} 篇")
    st.plotly_chart(fig, use_container_width=True)


def _fig_authors():
    df = _df_authors()
    fig = px.bar(df, x="paper_count", y="author", orientation="h",
                 labels={"paper_count": "论文数", "author": ""},
                 color_discrete_sequence=["#DC3912"] * len(df))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0), template=TEMPLATE,
                      title="高产作者 Top 20", yaxis=dict(categoryorder="total ascending"))
    fig.update_traces(hovertemplate="%{y}: %{x} 篇")
    st.plotly_chart(fig, use_container_width=True)


def _fig_submission_heatmap():
    pivot = _df_submission_heatmap()
    fig = px.imshow(pivot.values, x=pivot.columns, y=pivot.index,
                    color_continuous_scale="YlOrRd",
                    labels={"x": "UTC 小时", "y": "年份", "color": "占比"})
    fig.update_xaxes(dtick=3, side="top")
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), template=TEMPLATE,
                      title="提交时段分布变化（行归一化）")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("arXiv 未提供作者机构信息，以上通过提交时段间接反映地域特征，不作为精确国家归属。")


def _fig_crossdomain_cats():
    df = _df_crossdomain_cats()
    fig = px.bar(df, x="count", y="category", orientation="h",
                 labels={"count": "论文数", "category": ""},
                 color_discrete_sequence=["#109618"] * len(df))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0), template=TEMPLATE,
                      title="非 CS 标签 Top 20", yaxis=dict(categoryorder="total ascending"))
    fig.update_traces(hovertemplate="%{y}: %{x} 篇")
    st.plotly_chart(fig, use_container_width=True)


def _fig_crossdomain_rate():
    df, cols = _df_crossdomain_rate()
    fig = go.Figure()
    for col in cols:
        fig.add_scatter(x=df["quarter"], y=df[col], mode="lines", name=col,
                        hovertemplate="%{x}<br>" + col + ": %{y:.2%}")
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), template=TEMPLATE,
                      title="渗透率随时间变化", xaxis_tickangle=-45,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
