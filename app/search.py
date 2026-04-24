import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from agent.agent import build_research_agent
from app.auth import save_query_log

PAGE_SIZE = 6


@st.cache_resource
def load_agent():
    return build_research_agent()


def render_search():
    agent = load_agent()

    with st.container(height=520):
        for idx, turn in enumerate(st.session_state.history):
            with st.chat_message("user"):
                st.markdown(turn["query"])
            with st.chat_message("assistant"):
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
            status.update(label="意图分析完成，正在检索...", state="running")
            results = agent._search(routing)
            status.update(label=f"检索到 {len(results)} 篇，正在生成分析...", state="running")
            from agent.prompts import build_render_prompt
            render_prompt = build_render_prompt(query, routing, results)
            run_output = agent.renderer.run(render_prompt)
            rendered_text = run_output.content if isinstance(run_output.content, str) else str(run_output.content)
            status.update(label="完成", state="complete")

        st.session_state.history.append({
            "query": query, "insights": rendered_text,
            "papers": results, "page": 0,
        })
        save_query_log(st.session_state.username, query, results, rendered_text)
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
