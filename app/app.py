import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from agent.brain import build_research_agent

st.set_page_config(page_title="arXiv NLP 论文助手", page_icon="📄", layout="wide")

# ── 初始化 session_state ──────────────────────────────────────────────────────
if "agent" not in st.session_state:
    with st.spinner("正在加载模型，请稍候（首次启动约需 30 秒）..."):
        st.session_state.agent = build_research_agent()

if "history" not in st.session_state:
    st.session_state.history = []   # [{query, insights, papers}]

if "page" not in st.session_state:
    st.session_state.page = 0      # 当前展示到第几页（每页10条）

PAGE_SIZE = 10


# ── 论文卡片 ──────────────────────────────────────────────────────────────────
def render_paper_card(i, score, paper_id, doc, meta):
    title    = meta.get("title", "无标题")
    authors  = meta.get("authors", "")
    date     = meta.get("publish_date", "")
    cats     = meta.get("categories", "")
    conf     = meta.get("top_conference", "") or ""
    url      = meta.get("url", "")
    comment  = meta.get("comment", "") or ""

    header = f"**{i}. {title}**"
    if url:
        header = f"**{i}. [{title}]({url})**"

    with st.expander(header, expanded=False):
        cols = st.columns([3, 1])
        with cols[0]:
            if authors:
                st.markdown(f"👤 {authors}")
            if date:
                st.markdown(f"📅 {date}　　🏷️ {cats}")
            if conf:
                st.markdown(f"🏆 {conf}")
            if comment:
                st.markdown(f"💬 {comment}")
        with cols[1]:
            st.markdown(f"`score: {round(float(score), 3)}`")
        st.markdown("---")
        st.markdown(doc[:500] + ("..." if len(doc) > 500 else ""))


# ── 主界面 ────────────────────────────────────────────────────────────────────
st.title("📄 arXiv NLP 论文检索助手")

# 展示历史对话（只展示最后一轮，保持简洁）
if st.session_state.history:
    last = st.session_state.history[-1]

    with st.chat_message("user"):
        st.markdown(last["query"])

    with st.chat_message("assistant"):
        # insights 部分
        if last["insights"]:
            st.markdown(last["insights"])

        # 论文卡片分页
        papers = last["papers"]
        total = len(papers)
        show_until = (st.session_state.page + 1) * PAGE_SIZE

        if total == 0:
            st.info("未检索到相关论文。")
        else:
            st.markdown(f"**共检索到 {total} 篇论文，当前展示前 {min(show_until, total)} 篇：**")
            for i, (score, pid, doc, meta) in enumerate(papers[:show_until], start=1):
                render_paper_card(i, score, pid, doc, meta)

            if show_until < total:
                if st.button(f"加载更多（还有 {total - show_until} 篇）"):
                    st.session_state.page += 1
                    st.rerun()

# 输入框
query = st.chat_input("输入你的检索需求，例如：找一些关于 RAG 的论文")

if query:
    st.session_state.page = 0  # 新查询重置分页

    with st.spinner("检索中..."):
        result = st.session_state.agent.respond(query)

    papers  = result.search_response.results
    insights = result.rendered_text

    st.session_state.history = [{
        "query":    query,
        "insights": insights,
        "papers":   papers,
    }]

    st.rerun()
