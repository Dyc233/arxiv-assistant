import sys, json, sqlite3
from pathlib import Path

import streamlit as st

APP_DB = Path(__file__).resolve().parent.parent / "data" / "app.db"
USERS_FILE = Path(__file__).resolve().parent.parent / "data" / "users.json"


# ==================== DB ====================
def _init_db():
    APP_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(APP_DB)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                query TEXT NOT NULL,
                results_json TEXT,
                insights TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
            )
        """)
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            uf = USERS_FILE
            if uf.exists():
                with open(uf, "r") as f:
                    old_users = json.load(f)
                for u, data in old_users.items():
                    role = data.get("role", "user")
                    if role not in ("admin", "user"):
                        role = "user"
                    conn.execute(
                        "INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
                        (u, data["password"], role)
                    )


def _save_users():
    with sqlite3.connect(str(APP_DB)) as conn:
        conn.execute("DELETE FROM users")
        for u, data in st.session_state.users.items():
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (u, data["password"], data.get("role", "user"))
            )


def save_query_log(username, query, results, insights):
    def _native(v):
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        if isinstance(v, dict):
            return {kk: _native(vv) for kk, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [_native(vv) for vv in v]
        try:
            return v.item()
        except Exception:
            return str(v)

    try:
        safe = _native(results)
        results_json = json.dumps(safe, ensure_ascii=False)
    except Exception:
        results_json = "[]"

    with sqlite3.connect(str(APP_DB)) as conn:
        conn.execute(
            "INSERT INTO query_log (username, query, results_json, insights) VALUES (?, ?, ?, ?)",
            (username, query, results_json, insights)
        )


def load_user_history(username, limit=50):
    with sqlite3.connect(str(APP_DB)) as conn:
        rows = conn.execute(
            "SELECT query, results_json, insights FROM query_log WHERE username = ? ORDER BY id DESC LIMIT ?",
            (username, limit)
        ).fetchall()

    history = []
    for query, results_json, insights in reversed(rows):
        try:
            results = [tuple(r) for r in json.loads(results_json)]
        except Exception:
            results = []
        history.append({"query": query, "insights": insights, "papers": results, "page": 0})
    return history


# ==================== Login ====================
@st.dialog("🔐 登录")
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
            st.session_state.history = load_user_history(username)
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
                        encoding="utf-8", errors="replace",
                        cwd=str(Path(__file__).resolve().parent.parent)
                    )
                    if result.returncode == 0:
                        st.success("更新完成")
                        st.text(result.stdout[-500:])
                        from app.dashboard import load_kpi
                        load_kpi.clear()
                    else:
                        st.error(f"更新失败: {result.stderr[-300:]}")

            st.divider()
            if st.button("🚪 退出登录", width="stretch"):
                st.session_state.logged_in = False
                st.rerun()
