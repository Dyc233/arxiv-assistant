import argparse
import sys
from pathlib import Path

''' 
  .\.venv\Scripts\python.exe retrieval\test_retrieval.py --preset metadata_title
  .\.venv\Scripts\python.exe retrieval\test_retrieval.py --preset metadata_author
  .\.venv\Scripts\python.exe retrieval\test_retrieval.py --preset semantic
  .\.venv\Scripts\python.exe retrieval\test_retrieval.py --preset hybrid_acl_mt
  .\.venv\Scripts\python.exe retrieval\test_retrieval.py --preset hybrid_recent
  .\.venv\Scripts\python.exe retrieval\test_retrieval.py --preset all

  也支持手动传参测单个场景，比如：

  .\.venv\Scripts\python.exe retrieval\test_retrieval.py --mode metadata --title "Attention Is All You Need"
  .\.venv\Scripts\python.exe retrieval\test_retrieval.py --mode semantic --query "agent reasoning"
  .\.venv\Scripts\python.exe retrieval\test_retrieval.py --mode hybrid --query "machine translation" --categories "cs.CL" --comment "ACL"
  .\.venv\Scripts\python.exe retrieval\test_retrieval.py --mode hybrid --query "retrieval augmented generation" --published "recent:2y" --categories "cs.CL"
'''



THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[0]
this_dir_str = str(THIS_DIR)
if sys.path and sys.path[0] == this_dir_str:
    sys.path.pop(0)
if this_dir_str in sys.path:
    sys.path.remove(this_dir_str)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from retrieval import SearchMode, SearchRequest, format_search_response, get_retrieval_service


PRESET_CASES = {
    "metadata_title": SearchRequest(
        mode=SearchMode.METADATA,
        title="Attention Is All You Need",
        top_k=5,
    ),
    "metadata_author": SearchRequest(
        mode=SearchMode.METADATA,
        authors="Yoshua Bengio",
        top_k=5,
    ),
    "semantic": SearchRequest(
        mode=SearchMode.SEMANTIC,
        query_text="agent reasoning",
        top_k=5,
        recall_top_k=20,
    ),
    "hybrid_acl_mt": SearchRequest(
        mode=SearchMode.HYBRID,
        query_text="machine translation",
        categories="cs.CL",
        comment="ACL",
        top_k=5,
        recall_top_k=30,
    ),
    "hybrid_recent": SearchRequest(
        mode=SearchMode.HYBRID,
        query_text="retrieval augmented generation",
        published="recent:2y",
        categories="cs.CL",
        top_k=5,
        recall_top_k=30,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test retrieval module independently.")
    parser.add_argument(
        "--preset",
        choices=["all", *PRESET_CASES.keys()],
        help="Run one of the built-in retrieval smoke tests.",
    )
    parser.add_argument("--mode", choices=[mode.value for mode in SearchMode], help="Manual mode selection.")
    parser.add_argument("--query", dest="query_text")
    parser.add_argument("--title")
    parser.add_argument("--authors")
    parser.add_argument("--categories")
    parser.add_argument("--comment")
    parser.add_argument("--published")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--recall-top-k", type=int, default=50)
    parser.add_argument("--disable-reranker", action="store_true")
    return parser.parse_args()


def clone_request(template: SearchRequest) -> SearchRequest:
    return SearchRequest(
        mode=template.mode,
        query_text=template.query_text,
        title=template.title,
        authors=template.authors,
        categories=template.categories,
        comment=template.comment,
        published=template.published,
        top_k=template.top_k,
        recall_top_k=template.recall_top_k,
        use_reranker=template.use_reranker,
    )


def request_from_args(args: argparse.Namespace) -> SearchRequest:
    if not args.mode:
        raise ValueError("Manual mode requires --mode.")
    return SearchRequest(
        mode=SearchMode(args.mode),
        query_text=args.query_text,
        title=args.title,
        authors=args.authors,
        categories=args.categories,
        comment=args.comment,
        published=args.published,
        top_k=args.top_k,
        recall_top_k=args.recall_top_k,
        use_reranker=not args.disable_reranker,
    )


def run_case(name: str, request: SearchRequest) -> None:
    service = get_retrieval_service()
    print("\n" + "=" * 80)
    print(f"[CASE] {name}")
    print(f"mode={request.mode.value} query={request.query_text or ''}")
    response = service.search(request)
    print(format_search_response(response))


def main() -> None:
    args = parse_args()
    if args.preset:
        if args.preset == "all":
            for name, template in PRESET_CASES.items():
                request = clone_request(template)
                request.use_reranker = not args.disable_reranker and request.use_reranker
                run_case(name, request)
            return
        request = clone_request(PRESET_CASES[args.preset])
        request.use_reranker = not args.disable_reranker and request.use_reranker
        run_case(args.preset, request)
        return

    request = request_from_args(args)
    run_case("manual", request)


if __name__ == "__main__":
    main()
