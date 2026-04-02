"""测试简化版检索系统"""
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from search import PaperSearcher


def test_semantic_search():
    """测试语义检索"""
    print("\n" + "="*80)
    print("测试 1: 语义检索")
    print("="*80)

    searcher = PaperSearcher()
    results = searcher.semantic_search("transformer attention mechanism", final_top_k=3)
    searcher.format_results(results)


def test_metadata_search():
    """测试元数据检索"""
    print("\n" + "="*80)
    print("测试 2: 元数据检索")
    print("="*80)

    searcher = PaperSearcher()
    results = searcher.metadata_search(
        title="BERT",
        published="after:2020",
        top_k=3
    )
    searcher.format_results(results)


def test_hybrid_search():
    """测试混合检索"""
    print("\n" + "="*80)
    print("测试 3: 混合检索")
    print("="*80)

    searcher = PaperSearcher()
    results = searcher.hybrid_search(
        query_text="natural language understanding",
        published="after:2022",
        final_top_k=3
    )
    searcher.format_results(results)


if __name__ == "__main__":
    print("开始测试简化版检索系统...")

    try:
        test_semantic_search()
        test_metadata_search()
        test_hybrid_search()
        print("\n✅ 所有测试完成！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
