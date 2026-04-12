import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from agent.brain import build_research_agent

st.set_page_config(page_title="arXiv NLP 论文助手", page_icon="📄", layout="wide")

PAGE_SIZE = 10


# ── 1. cache_resource：进程级单例，刷新页面不重新加载模型 ─────────────────────
@st.cache_resource
def load_agent():
    return build_research_agent()


with st.spinner("正在加载模型，首次启动请稍候..."):
    agent = load_agent()


# ── session_state 初始化 ───────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []       # [{query, insights, papers, page}]

if "translations" not in st.session_state:
    st.session_state.translations = {}  # paper_id -> 中文摘要


# ── 论文卡片 ──────────────────────────────────────────────────────────────────
def render_paper_card(i, score, paper_id, doc, meta, turn_idx):
    title   = meta.get("title", "无标题")
    authors = meta.get("authors", "")
    date    = meta.get("publish_date", "")
    cats    = meta.get("categories", "")
    conf    = meta.get("top_conference", "") or ""
    url     = meta.get("url", "")
    comment = meta.get("comment", "") or ""

    header = f"**{i}. [{title}]({url})**" if url else f"**{i}. {title}**"

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

        # 3. 翻译按钮：按需翻译，结果缓存在 session_state
        if paper_id in st.session_state.translations:
            st.markdown(st.session_state.translations[paper_id])
        else:
            btn_col, txt_col = st.columns([1, 6])
            with btn_col:
                if st.button("🔤 译", key=f"trans_{turn_idx}_{paper_id}"):
                    with st.spinner("翻译中..."):
                        out = agent.renderer.run(
                            f"请将以下英文摘要翻译成中文，只输出翻译结果，不要加任何前缀：\n{doc}"
                        )
                        zh = out.content if isinstance(out.content, str) else str(out.content)
                        st.session_state.translations[paper_id] = zh
                    st.rerun()
            with txt_col:
                st.markdown(doc[:400] + ("..." if len(doc) > 400 else ""))


# ── 单轮渲染 ──────────────────────────────────────────────────────────────────
def render_turn(turn, turn_idx):
    with st.chat_message("user"):
        st.markdown(turn["query"])

    with st.chat_message("assistant"):
        papers = turn["papers"]
        total = len(papers)
        show_until = (turn["page"] + 1) * PAGE_SIZE

        # 4. 论文列表在上，insights 在下
        if total == 0:
            st.info("未检索到相关论文。")
        else:
            st.markdown(f"**共检索到 {total} 篇，当前展示前 {min(show_until, total)} 篇：**")
            for i, (score, pid, doc, meta) in enumerate(papers[:show_until], start=1):
                render_paper_card(i, score, pid, doc, meta, turn_idx)

            if show_until < total:
                if st.button(f"加载更多（还有 {total - show_until} 篇）", key=f"more_{turn_idx}"):
                    st.session_state.history[turn_idx]["page"] += 1
                    st.rerun()

        if turn["insights"]:
            st.markdown("---")
            st.markdown(turn["insights"])


# ── 主界面 ────────────────────────────────────────────────────────────────────
st.title("📄 arXiv NLP 论文检索助手")

# 5. 渲染全部历史轮次
for idx, turn in enumerate(st.session_state.history):
    render_turn(turn, idx)

query = st.chat_input("输入你想查询的内容，例如：找一些关于 RAG 的论文")

if query:
    with st.spinner("检索中..."):
        result = agent.respond(query)

    # 2. 后端日志：路由决策打印到终端
    print(f"\n[Router] {result.routing.model_dump_json(indent=2)}\n")

    st.session_state.history.append({
        "query":    query,
        "insights": result.rendered_text,
        "papers":   result.search_response.results,
        "page":     0,
    })
    st.rerun()
