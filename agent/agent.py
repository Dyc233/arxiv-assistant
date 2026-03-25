import os

from agno.agent import Agent
from agno.models.moonshot import MoonShot

from agent.tools.search_tools import metadata_filtered_search, pure_semantic_search


def build_research_agent() -> Agent:
    llm_model = MoonShot(
        id="kimi-k2-turbo-preview",
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.cn/v1",
    )

    return Agent(
        model=llm_model,
        tools=[pure_semantic_search, metadata_filtered_search],
        description="你是一个 NLP 论文检索助手，需要先理解用户意图，再调用正确的检索工具。",
        instructions=[
            "1. 先识别用户是否包含明确的 metadata 约束，包括发布时间、作者、arXiv 分类、顶会/顶刊或 comment 关键词。",
            "2. 如果没有这些约束，只调用 `pure_semantic_search`。",
            "3. 只要出现 metadata 约束，就调用 `metadata_filtered_search`。",
            "4. `metadata_filtered_search` 的参数规范如下："
            " published 可用 2024 / after:2024 / before:2023 / equal:2022 / since:2024-01-01 / between:2023-01-01,2024-12-31 / recent:2y；"
            " authors、categories、comment 均用英文逗号分隔多个条件。",
            "5. 传给工具的 `query` 只保留研究主题，不要把时间、作者、分类、顶会条件重复塞进 query。",
            "6. 得到工具返回结果后，用专业中文总结，优先概括论文主题、趋势、代表性工作和筛选条件。",
        ],
        markdown=True,
    )


if __name__ == "__main__":
    agent = build_research_agent()
    print("=" * 60)
    print("科研 Agent 已启动，输入 q / exit / quit 退出")
    print("=" * 60)

    while True:
        user_input = input("\n[用户] 请输入你的检索需求: ").strip()
        if user_input.lower() in ["q", "exit", "quit"]:
            print("再见。")
            break
        if not user_input:
            continue

        print("\n[Agent 思考中...]")
        agent.print_response(user_input, stream=True)
