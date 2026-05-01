import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from agent.agent import build_research_agent
from app.auth import save_query_log, delete_query_log, clear_user_history

PAGE_SIZE = 6
MODE_LABELS = {"semantic": "语义检索", "metadata": "元数据检索", "hybrid": "混合检索"}


@st.cache_resource
def load_agent():
    return build_research_agent()


def render_search():
    agent = load_agent()

    # clear all button
    if st.session_state.history:
        c1, c2 = st.columns([5, 1])
        with c2:
            if st.button("🗑️ 清空记录", key="clear_all"):
                clear_user_history(st.session_state.username)
                st.session_state.history = []
                st.rerun()

    with st.container(height=520):
        if not st.session_state.history:
            st.info("""👋 欢迎使用 arXiv NLP 论文检索系统

试试这些查询：
- 找2025年关于大语言模型推理的论文
- Instruction tuning 的最新进展
- 对比学习在NLP中的应用 survey
- 作者Yoshua Bengio最近的论文""")

        for idx, turn in enumerate(st.session_state.history):
            with st.chat_message("user"):
                c1, c2 = st.columns([20, 1])
                with c1:
                    st.markdown(turn["query"])
                with c2:
                    if turn.get("id") and st.button("🗑️", key=f"del_{turn['id']}"):
                        delete_query_log(turn["id"])
                        st.session_state.history.pop(idx)
                        st.rerun()

            with st.chat_message("assistant"):
                routing = turn.get("routing")
                if routing:
                    parts = [MODE_LABELS.get(routing["search_mode"], routing["search_mode"])]
                    if routing.get("published"):
                        parts.append(f"时间: {routing['published']}")
                    if routing.get("categories"):
                        parts.append(f"类别: {routing['categories']}")
                    if routing.get("authors"):
                        parts.append(f"作者: {routing['authors']}")
                    if routing.get("comment"):
                        parts.append(f"会议/备注: {routing['comment']}")
                    st.caption("🔍 " + " · ".join(parts))

                papers = turn["papers"]
                total = len(papers)
                show_until = (turn["page"] + 1) * PAGE_SIZE
                if total == 0:
                    st.info("未检索到相关论文。")
                else:
                    st.caption(f"共 {total} 篇")
                    for i, (score, pid, doc, meta) in enumerate(papers[:show_until], start=1):
                        _paper_card(i, score, pid, doc, meta, idx)
                    if show_until < total:
                        if st.button(f"加载更多 ({total - show_until}篇)", key=f"more_{idx}"):
                            st.session_state.history[idx]["page"] += 1
                            st.rerun()
                if turn["insights"]:
                    st.markdown("---")
                    st.markdown(turn["insights"])

    query = st.chat_input("输入查询，如：找2025年NLP与quant-ph交叉的论文")

    if query:
        with st.status("正在分析意图...", expanded=False) as status:
            routing = agent.route(query)
            print(f"\n[Router] {routing.model_dump_json(indent=2)}\n")
            mode_label = MODE_LABELS.get(routing.search_mode, routing.search_mode)
            status.update(label=f"意图: {mode_label}，正在检索...", state="running")
            results = agent._search(routing)
            status.update(label=f"检索到 {len(results)} 篇，正在生成分析...", state="running")
            from agent.prompts import build_render_prompt
            render_prompt = build_render_prompt(query, routing, results)
            run_output = agent.renderer.run(render_prompt)
            rendered_text = run_output.content if isinstance(run_output.content, str) else str(run_output.content)
            status.update(label=f"完成 · {mode_label} · {len(results)} 篇", state="complete")

        qid = save_query_log(st.session_state.username, query, results, rendered_text)
        st.session_state.history.append({
            "id": qid, "query": query, "insights": rendered_text,
            "papers": results, "page": 0,
            "routing": {
                "search_mode": routing.search_mode,
                "published": routing.published or "",
                "categories": routing.categories or "",
                "authors": routing.authors or "",
                "comment": routing.comment or "",
            },
        })
        st.rerun()


def _paper_card(i, score, pid, doc, meta, turn_idx):
    title = meta.get("title", "无标题")
    conf = meta.get("top_conference", "") or ""
    date = meta.get("publish_date", "")
    with st.expander(f"{i}. [{title}]({meta.get('url', '')})" if meta.get("url") else f"{i}. {title}",
                     expanded=False):
        st.markdown(f"👤 {meta.get('authors', '')} ｜ 📅 {date} ｜ 🏷️ {meta.get('categories', '')}")
        if conf:
            st.markdown(f"🏆 {conf}")
        col_md = st.columns([6, 1])[0]
        with col_md:
            if pid in st.session_state.translations:
                st.markdown(st.session_state.translations[pid])
            else:
                if st.button("🔤译", key=f"tr_{turn_idx}_{pid}"):
                    with st.spinner("翻译中..."):
                        agent = load_agent()
                        out = agent.renderer.run(
                            f"将以下英文摘要翻译成中文，只输出翻译结果：\n{doc}"
                        )
                        zh = out.content if isinstance(out.content, str) else str(out.content)
                        st.session_state.translations[pid] = zh
                    st.rerun()
                st.markdown(doc)
