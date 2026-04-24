import sys, json, sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from app.dashboard import render_dashboard
from app.auth import _init_db, _save_users, login_dialog, render_admin_sidebar, APP_DB
from app.search import render_search

st.set_page_config(page_title="arXiv NLP 论文检索系统", page_icon="📄", layout="wide")

_init_db()

# --- session state ---
for key, default in [("logged_in", False), ("role", None), ("username", ""),
                     ("history", []), ("translations", {}), ("users", None),
                     ("active_page", "分析看板")]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.users is None:
    with sqlite3.connect(str(APP_DB)) as conn:
        rows = conn.execute("SELECT username, password, role FROM users").fetchall()
    if rows:
        st.session_state.users = {u: {"password": p, "role": r} for u, p, r in rows}
    else:
        st.session_state.users = {"admin": {"password": "admin", "role": "admin"}}
        _save_users()


# ==================== Main ====================
if not st.session_state.logged_in:
    login_dialog()
else:
    if st.session_state.role == "admin":
        render_admin_sidebar()

    st.title("📄 arXiv NLP 论文检索系统")
    role_display = {"admin": "管理员", "user": "用户"}.get(st.session_state.role, st.session_state.role)
    st.caption(f"欢迎, **{st.session_state.username}** ({role_display})")

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
