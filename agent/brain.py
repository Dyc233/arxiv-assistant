import os
from dataclasses import dataclass
from typing import Any

from agno.agent import Agent
from agno.agent.agent import RunOutput
from agno.models.moonshot import MoonShot
from pydantic import BaseModel

from agent.prompts import ROUTER_DESCRIPTION, ROUTER_INSTRUCTIONS, build_render_prompt
from agent.schemas import RoutingDecision
from retrieval import SearchMode, SearchRequest, SearchResponse, format_search_response, get_retrieval_service


@dataclass(slots=True)
class ResearchAgentResult:
    routing: RoutingDecision
    search_response: SearchResponse
    rendered_text: str


class AgnoResearchAssistant:
    def __init__(self) -> None:
        self.retrieval_service = get_retrieval_service()
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

    def route(self, user_input: str) -> RoutingDecision:
        run_output = self.router.run(user_input)
        return self._coerce_routing_decision(run_output)

    def retrieve(self, routing: RoutingDecision) -> SearchResponse:
        request = SearchRequest(
            mode=SearchMode(routing.search_mode),
            query_text=self._normalize_text(routing.query_text),
            title=self._normalize_text(routing.title),
            authors=self._normalize_text(routing.authors),
            categories=self._normalize_text(routing.categories),
            comment=self._normalize_text(routing.comment),
            published=self._normalize_text(routing.published),
            top_k=routing.top_k,
        )
        return self.retrieval_service.search(request)

    def respond(self, user_input: str) -> ResearchAgentResult:
        routing = self.route(user_input)
        search_response = self.retrieve(routing)
        render_prompt = build_render_prompt(user_input, routing, search_response)
        run_output = self.renderer.run(render_prompt)
        rendered_text = self._coerce_text_content(run_output)
        return ResearchAgentResult(
            routing=routing,
            search_response=search_response,
            rendered_text=rendered_text,
        )

    def print_response(
        self,
        user_input: str,
        *,
        stream: bool | None = None,
        show_message: bool = True,
        **_: Any,
    ) -> None:
        routing = self.route(user_input)
        search_response = self.retrieve(routing)
        render_prompt = build_render_prompt(user_input, routing, search_response)

        if stream:
            self.renderer.print_response(
                render_prompt,
                stream=True,
                show_message=show_message,
            )
            return

        run_output = self.renderer.run(render_prompt)
        print(self._coerce_text_content(run_output))

    def debug_search(self, user_input: str) -> str:
        routing = self.route(user_input)
        search_response = self.retrieve(routing)
        return "\n".join(
            [
                "[Routing]",
                routing.model_dump_json(indent=2),
                "",
                "[SearchResponse]",
                format_search_response(search_response),
            ]
        )

    @staticmethod
    def _build_model() -> MoonShot:
        return MoonShot(
            id="kimi-k2-turbo-preview",
            api_key=os.getenv("MOONSHOT_API_KEY"),
            base_url="https://api.moonshot.cn/v1",
        )

    @staticmethod
    def _normalize_text(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

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

    @staticmethod
    def _coerce_text_content(run_output: RunOutput) -> str:
        content = run_output.content
        if isinstance(content, str):
            return content
        if isinstance(content, BaseModel):
            return content.model_dump_json(indent=2)
        if isinstance(content, dict):
            return str(content)
        return str(content)


def build_research_agent() -> AgnoResearchAssistant:
    return AgnoResearchAssistant()
