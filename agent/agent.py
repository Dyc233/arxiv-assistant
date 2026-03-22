import os
from agno.agent import Agent
from agno.models.moonshot import MoonShot

# ==========================================
# 第一部分：定义工具 (Tools / Skills)
# ==========================================

def pure_semantic_search(query: str) -> str:
    """
    当用户的查询仅仅是学术概念、研究主题，且**没有任何**时间、年份、作者等硬性限制条件时，调用此工具。
    
    Args:
        query (str): 提取出的纯学术语义搜索关键词（尽量用英文，如 "Agent reasoning", "Attention mechanism"）。
    """
    # 这里未来将接入 analysis.query 中的常规搜索逻辑
    print(f"\n[🔧 工具执行] -> 触发纯语义搜索 | 检索词: '{query}'")
    
    # 模拟返回的数据库结果
    return f"找到了关于 '{query}' 的 5 篇相关论文（模拟数据）。"


def metadata_filtered_search(query: str, year: int = None, operator: str = None, author: str = None) -> str:
    """
    当用户的查询中包含明确的年份限制（如"2024年后"、"去年"）或特定的作者名称时，**必须**调用此工具。
    
    Args:
        query (str): 剔除时间、作者等过滤条件后，剩余的核心研究主题（如 "大模型幻觉"）。
        year (int, optional): 提取出的四位数字年份（如 2024）。
        operator (str, optional): 时间逻辑算子。必须是 "after" (之后/以后/最近), "before" (之前/以前), 或 "equal" (当年)。
        author (str, optional): 提取出的特定作者姓名（拼音或英文）。
    """
    # 这里未来将接入带有 ChromaDB where 过滤器的搜索逻辑
    print(f"\n[🔧 工具执行] -> 触发带物理过滤的搜索")
    print(f"   ├─ 核心语义词: '{query}'")
    print(f"   ├─ 年份限制: {year}")
    print(f"   ├─ 逻辑算子: {operator}")
    print(f"   └─ 作者限制: {author}")
    
    # 模拟返回的数据库结果
    return f"在满足时间/作者过滤条件的情况下，找到了关于 '{query}' 的 3 篇最新论文（模拟数据）。"


# ==========================================
# 第二部分：构建 Agent 大脑
# ==========================================

def build_research_agent():
    # 这里以兼容 OpenAI 格式的 API 为例（如果你用 DeepSeek、Qwen 等国内大模型，修改 base_url 即可）
    # 意图识别强烈建议使用能力较强的模型（如 gpt-4o, deepseek-chat）
    llm_model = MoonShot(
        id="kimi-k2-turbo-preview",
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.cn/v1", 
    )

    agent = Agent(
        model=llm_model,
        tools=[pure_semantic_search, metadata_filtered_search], # 把工具交给 Agent
        description="你是一个顶级的 NLP 领域科研助手，你的任务是精准理解用户的论文检索意图，并调用合适的检索工具提取数据。",
        instructions=[
            "1. 仔细分析用户的输入。",
            "2. 如果用户没有提及明确的年份、时间或作者要求，请调用 `pure_semantic_search`。",
            "3. 如果用户提及了诸如'2024年后'、'最近两年'或特定的作者名，必须拆解出对应的参数，调用 `metadata_filtered_search`。",
            "4. 获取到工具返回的数据后，使用专业、学术的中文为用户进行总结解答。",
            "5. 在回答中，可以适当加上一些学术视角的点评。"
        ],
        markdown=True
    )
    return agent

# ==========================================
# 第三部分：交互测试入口
# ==========================================

if __name__ == "__main__":
    agent = build_research_agent()
    
    print("="*60)
    print("科研 Agent 大脑已启动！(输入 'q' 退出)")
    print("="*60)
    
    while True:
        user_input = input("\n[用户] 请输入你的检索需求: ").strip()
        if user_input.lower() in ['q', 'exit', 'quit']:
            print("再见！")
            break
        if not user_input:
            continue
            
        print("\n[Agent 思考中...]")
        # 这一句会让 Agent 自动分析、调工具、拿数据、并流式输出最终回答
        agent.print_response(user_input, stream=True)