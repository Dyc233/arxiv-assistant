import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent.agent import build_research_agent


def main() -> None:
    agent = build_research_agent()

    print("\n" + "=" * 60)
    print("      NLP 论文检索 Agent 测试脚本")
    print("      输入 q / exit / quit 即可结束对话")
    print("=" * 60)

    while True:
        user_input = input("\n请输入你的检索需求: ").strip()
        if user_input.lower() in ["q", "exit", "quit"]:
            print("再见。")
            break

        if not user_input:
            continue

        try:
            print("\n正在分析并检索中...")
            agent.print_response(user_input, stream=True)
        except Exception as exc:
            print(f"发生错误: {exc}")


if __name__ == "__main__":
    main()
