import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from agent.agent import build_research_agent
from app.dashboard import render_dashboard

st.set_page_config(page_title="arXiv NLP 论文检索系统", page_icon="📄", layout="wide")

USERS_FILE = Path(__file__).resolve().parent.parent / "data" / "users.json"
PAGE_SIZE = 6

# --- init defaults ---
for key, default in [("logged_in", False), ("role", None), ("username", ""),
                     ("history", []), ("translations", {}), ("users", None),
                     ("active_page", "分析看板")]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.users is None:
    if USERS_FILE.exists():
        with open(USERS_FILE, "r") as f:
            st.session_state.users = json.load(f)
    else:
        st.session_state.users = {"admin": {"password": "admin", "role": "admin"}}
        _save_users()


def _save_users():
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(st.session_state.users, f, ensure_ascii=False, indent=2)


@st.cache_resource
def load_agent():
    return build_research_agent()


# ==================== Login Dialog ====================
@st.dialog("🔐 登录系统")
def login_dialog():
    st.markdown("arXiv NLP 论文检索系统")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    role_choice = st.radio("角色", ["用户", "管理员"], horizontal=True)

    if st.button("登录", width="stretch"):
        users = st.session_state.users
        if username in users and users[username]["password"] == password:
            actual_role = users[username].get("role", "用户")
            if role_choice == "管理员" and actual_role != "admin":
                st.error("该账号不是管理员")
                return
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = actual_role
            st.rerun()
        elif username not in users:
            st.error("用户不存在")
        else:
            st.error("密码错误")


# ==================== Admin Sidebar ====================
def render_admin_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ 管理面板")

        tab1, tab2 = st.tabs(["👥 用户管理", "🔄 系统维护"])

        with tab1:
            st.caption(f"当前用户数: {len(st.session_state.users)}")

            with st.expander("➕ 添加用户"):
                new_user = st.text_input("新用户名", key="new_user")
                new_pwd = st.text_input("密码", type="password", key="new_pwd")
                role_map = {"用户": "user", "管理员": "admin"}
                new_role_label = st.selectbox("角色", ["用户", "管理员"], key="new_role")
                if st.button("添加", key="add_user_btn"):
                    if new_user and new_pwd:
                        st.session_state.users[new_user] = {"password": new_pwd, "role": role_map[new_role_label]}
                        _save_users()
                        st.success(f"已添加用户: {new_user}")
                        st.rerun()

            with st.expander("🗑️ 删除用户"):
                users_list = [
                    u for u in st.session_state.users if u != st.session_state.username
                ]
                if users_list:
                    del_user = st.selectbox("选择要删除的用户", users_list, key="del_user")
                    if st.button("确认删除", key="del_user_btn", type="secondary"):
                        del st.session_state.users[del_user]
                        _save_users()
                        st.success(f"已删除用户: {del_user}")
                        st.rerun()
                else:
                    st.caption("无其他用户可删除")

        with tab2:
            st.caption("增量更新论文库")
            if st.button("🔄 执行增量更新", width="stretch"):
                with st.spinner("正在更新论文数据..."):
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, "data/updater.py"],
                        capture_output=True, text=True, timeout=300,
                        cwd=str(Path(__file__).resolve().parent.parent)
                    )
                    if result.returncode == 0:
                        st.success("更新完成")
                        st.text(result.stdout[-500:])
                    else:
                        st.error(f"更新失败: {result.stderr[-300:]}")

            st.divider()
            if st.button("🚪 退出登录", width="stretch"):
                st.session_state.logged_in = False
                st.rerun()


# ==================== Search UI ====================
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


# ==================== Main ====================
if not st.session_state.logged_in:
    login_dialog()
else:
    # Sidebar: admin only
    if st.session_state.role == "admin":
        render_admin_sidebar()

    # Title bar
    st.title("📄 arXiv NLP 论文检索系统")
    role_display = {"admin": "管理员", "user": "用户"}.get(st.session_state.role, st.session_state.role)
    st.caption(f"欢迎, **{st.session_state.username}** ({role_display})")

    # Top navigation
    btn_cols = st.columns([1, 1, 4])
    if btn_cols[0].button("📊 分析看板", width="stretch",
                          type="primary" if st.session_state.active_page == "分析看板" else "secondary"):
        st.session_state.active_page = "分析看板"
        st.rerun()
    if btn_cols[1].button("🔍 论文检索", width="stretch",
                          type="primary" if st.session_state.active_page == "论文检索" else "secondary"):
        st.session_state.active_page = "论文检索"
        st.rerun()

    st.divider()

    if st.session_state.active_page == "分析看板":
        render_dashboard()
    else:
        render_search()
