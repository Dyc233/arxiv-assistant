import logging
import sqlite3
from pathlib import Path

import arxiv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('arxiv').setLevel(logging.WARNING) 
logging.getLogger('urllib3').setLevel(logging.WARNING)

def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT,
            summary TEXT,
            published TEXT,
            authors TEXT,
            categories TEXT,
            comment TEXT,
            url TEXT
        )
        '''
    )
    conn.commit()
    return conn

def get_existing_ids(conn: sqlite3.Connection) -> set:
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM papers")
    return {row[0] for row in cursor.fetchall()}

def get_oldest_date(conn: sqlite3.Connection) -> str:
    """查询数据库中存有的最老的一篇论文的日期"""
    cursor = conn.cursor()
    # 假设 published 格式是 '2025-01-01 12:00:00+00:00'
    cursor.execute("SELECT MIN(published) FROM papers")
    result = cursor.fetchone()[0]
    if result:
        # 将日期格式化为 arXiv API 接受的 YYYYMMDDHHMM 格式
        # 简单处理：取前 10 位再去掉横杠
        return result[:10].replace("-", "") + "0000"
    return None

def fetch_arxiv_papers(conn: sqlite3.Connection, max_results: int = 50000, batch_size: int = 50) -> None:
    """从 arXiv 抓取论文并存入 SQLite，支持基于时间切片的断点续爬"""
    cursor = conn.cursor()
    
    # 必须先获取已存在的 ID，否则下面会报 NameError
    existing_ids = get_existing_ids(conn)
    
    # 1. 获取时间锚点
    oldest_date = get_oldest_date(conn)
    
    # 2. 构造查询语句
    # 逻辑：只爬取比数据库里“最老论文”还要老的论文，从而让 API 的 start 永远从 0 开始
    query = 'cat:cs.CL'
    if oldest_date:
        # 使用 submittedDate:[开始时间 TO 结束时间] 语法
        # 注意：arXiv 要求时间格式通常为 YYYYMMDDHHMM
        query += f' AND submittedDate:[199001010000 TO {oldest_date}]'
        logging.info(f"[*] Resuming: Searching for papers older than {oldest_date}...")
    else:
        logging.info("[*] Starting a fresh crawl from the latest papers...")

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    papers_batch = []
    
    with tqdm(total=max_results, desc="Fetching", unit="papers") as pbar:
        try:
            for paper in client.results(search):
                paper_id = paper.get_short_id()

                if paper_id in existing_ids:
                    pbar.update(1)
                    continue

                # 数据清洗与转换 [cite: 1]
                title = paper.title.replace('\n', ' ')
                summary = paper.summary.replace('\n', ' ')
                published = str(paper.published)
                authors = ', '.join(author.name for author in paper.authors)
                categories = ', '.join(paper.categories)
                comment = paper.comment or ''
                url = paper.entry_id

                papers_batch.append(
                    (paper_id, title, summary, published, authors, categories, comment, url)
                )
                existing_ids.add(paper_id)

                # 分批提交
                if len(papers_batch) >= batch_size:
                    cursor.executemany(
                        'INSERT OR IGNORE INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                        papers_batch,
                    )
                    conn.commit()
                    papers_batch = []

                pbar.update(1)

            if papers_batch:
                cursor.executemany('INSERT OR IGNORE INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?)', papers_batch)
                conn.commit()

        except KeyboardInterrupt:
            logging.warning("\n[!] User stopped the process. Data is safe in SQLite.")
        except Exception as exc:
            logging.error(f"\n[X] Critical Error: {exc}")
        finally:
            conn.close()
            logging.info("\n[*] Database safely closed.")

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / 'data' / 'arxiv_papers.db'

    db_conn = init_db(db_path)
    fetch_arxiv_papers(db_conn, max_results=10000, batch_size=50) #每次爬取 10000 篇，分批提交，每批 50 篇