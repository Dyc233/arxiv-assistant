from agent.brain import build_research_agent


if __name__ == "__main__":
    agent = build_research_agent()
    print("=" * 60)
    print("Agent已启动，输入 q / exit / quit 退出")
    print("=" * 60)

    while True:
        user_input = input("\n[用户] 请输入: ").strip()
        if user_input.lower() in ["q", "exit", "quit"]:
            print("再见。")
            break
        if not user_input:
            continue

        print("\n[Agent 思考中...]")
        agent.print_response(user_input, stream=True)
