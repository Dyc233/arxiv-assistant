import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "arxiv_papers.db"

def search_database(db_path, table_name, column_name, keyword):
    try:
        # 1. 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 2. 构造 SQL 语句
        # 注意：列名必须通过 f-string 嵌入，因为 SQL 参数化不支持动态列名
        # 为了安全，这里给列名加上双引号防止特殊字符冲突
        query = f'SELECT * FROM {table_name} WHERE "{column_name}" LIKE ?'
        
        # 3. 执行查询（关键词前后加 % 实现模糊匹配）
        search_term = f"%{keyword}%"
        cursor.execute(query, (search_term,))

        # 4. 获取并打印结果
        results = cursor.fetchall()
        
        print(f"\n--- 在 [{column_name}] 中搜索 '{keyword}' 的结果 ---")
        if results:
            for row in results:
                print(row)
        else:
            print("未找到匹配记录。")

    except sqlite3.Error as e:
        print(f"数据库错误: {e}")
    finally:
        if conn:
            conn.close()

# --- 使用示例 ---
if __name__ == "__main__":
    # 假设你有一个名为 'test.db' 的数据库，表中有一列叫 'name'
    db = DEFAULT_DB_PATH 
    table = "papers"
    col = "title"
    word = input("请输入要搜索的关键词: ")

    search_database(db, table, col, word)