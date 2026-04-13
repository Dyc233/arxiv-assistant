import os
from dataclasses import dataclass
from typing import Any

from agno.agent import Agent
from agno.agent.agent import RunOutput
from agno.models.moonshot import MoonShot
from pydantic import BaseModel

from agent.prompts import ROUTER_DESCRIPTION, ROUTER_INSTRUCTIONS, build_render_prompt
from agent.schemas import RoutingDecision
from search import PaperSearcher


@dataclass(slots=True)
class ResearchAgentResult:
    routing: RoutingDecision
    results: list
    rendered_text: str


class AgnoResearchAssistant:
    def __init__(self) -> None:
        self.searcher = PaperSearcher()
        self.router = Agent(
            model=self._build_model(),
            description=ROUTER_DESCRIPTION,
            instructions=ROUTER_INSTRUCTIONS,
            output_schema=RoutingDecision,
            structured_outputs=True,
            markdown=False,
        )
        self.renderer = Agent(
            model=self._build_model(),
            description="你是一个论文检索结果呈现器，负责把检索结果转成用户可读的中文回答。",
            markdown=True,
        )

    # 把用户输入转成结构化的检索意图
    def route(self, user_input: str) -> RoutingDecision:
        try:
            run_output = self.router.run(user_input)
            return self._coerce_routing_decision(run_output)
        except Exception as e:
            print(f"路由失败 ({e})，已切换至语义检索模式。")
            return RoutingDecision(
                task_type="search",
                search_mode="semantic",
                query_text=user_input,
                response_mode="list_with_insights",
                top_k=5,
            )

    # 根据路由结果执行检索
    def _search(self, routing: RoutingDecision) -> list:
        kw = dict(
            title=self._normalize(routing.title),
            authors=self._normalize(routing.authors),
            categories=self._normalize(routing.categories),
            comment=self._normalize(routing.comment),
            published=self._normalize(routing.published),
        )
        mode = routing.search_mode
        if mode == "semantic":
            return self.searcher.semantic_search(
                self._normalize(routing.query_text), final_top_k=routing.top_k
            )
        elif mode == "metadata":
            return self.searcher.metadata_search(
                query_text=self._normalize(routing.query_text), **kw, top_k=routing.top_k
            )
        else:  # hybrid
            return self.searcher.hybrid_search(
                query_text=self._normalize(routing.query_text), **kw, final_top_k=routing.top_k
            )

    def respond(self, user_input: str) -> ResearchAgentResult:
        routing = self.route(user_input)
        results = self._search(routing)
        render_prompt = build_render_prompt(user_input, routing, results)
        run_output = self.renderer.run(render_prompt)
        rendered_text = self._coerce_text_content(run_output)
        return ResearchAgentResult(routing=routing, results=results, rendered_text=rendered_text)

    def print_response(
        self,
        user_input: str,
        *,
        stream: bool | None = None,
        show_message: bool = True,
        **_: Any,
    ) -> None:
        routing = self.route(user_input)
        results = self._search(routing)
        render_prompt = build_render_prompt(user_input, routing, results)

        if stream:
            self.renderer.print_response(render_prompt, stream=True, show_message=show_message)
            return

        run_output = self.renderer.run(render_prompt)
        print(self._coerce_text_content(run_output))

    @staticmethod
    def _build_model() -> MoonShot:
        return MoonShot(
            id="kimi-k2-turbo-preview",
            api_key=os.getenv("MOONSHOT_API_KEY"),
            base_url="https://api.moonshot.cn/v1",
        )

    # 对输入进行规范化处理，去掉多余空格，保持 None 不变
    @staticmethod
    def _normalize(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    # 把 Agent 的输出内容强制转换成 RoutingDecision
    @staticmethod
    def _coerce_routing_decision(run_output: RunOutput) -> RoutingDecision:
        content = run_output.content
        if isinstance(content, RoutingDecision):
            return content
        if isinstance(content, BaseModel):
            return RoutingDecision.model_validate(content.model_dump())
        if isinstance(content, dict):
            return RoutingDecision.model_validate(content)
        raise TypeError(f"Unexpected router output content type: {type(content)!r}")

    # 把 Agent 的输出内容强制转换成字符串
    @staticmethod
    def _coerce_text_content(run_output: RunOutput) -> str:
        content = run_output.content
        if isinstance(content, str):
            return content
        if isinstance(content, BaseModel):
            return content.model_dump_json(indent=2)
        return str(content)


def build_research_agent() -> AgnoResearchAssistant:
    return AgnoResearchAssistant()
