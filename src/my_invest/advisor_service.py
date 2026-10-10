from dataclasses import dataclass
from time import perf_counter
from typing import Any
from agno.agent import Agent

from my_invest.agents.advisor_agent import create_investment_advisor_agent
from my_invest.settings import settings


@dataclass(frozen=True)
class AdvisoryResult:
    prompt: str
    text: str
    duration_ms: float
    model_name: str
    llm_call_count: int = 1
    tool_call_count: int = 0
    tools_used: tuple[str, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AdvisorService:
    def __init__(self) -> None:
        self.settings = settings
        self._agent: Agent | None = None

    def get_agent(self) -> Agent:
        if self._agent is None:
            self._agent = create_investment_advisor_agent()
        return self._agent

    def run_advisory(self, prompt: str) -> AdvisoryResult:
        if not prompt or not prompt.strip():
            raise ValueError("Investment prompt cannot be empty.")

        started_at = perf_counter()
        agent = self.get_agent()
        response = agent.run(prompt)

        text = getattr(response, "content", None)
        if text is None:
            text = str(response)

        duration_ms = round((perf_counter() - started_at) * 1000, 2)

        # Extract tools used and tool execution count
        tools_list: list[str] = []
        raw_tools = getattr(response, "tools", None)
        if raw_tools:
            for t in raw_tools:
                t_name = getattr(t, "tool_name", None) or (t.get("tool_name") if isinstance(t, dict) else str(t))
                if t_name and t_name not in tools_list:
                    tools_list.append(str(t_name))

        # Check messages for tool calls and LLM iterations
        messages = getattr(response, "messages", None) or []
        assistant_count = 0
        for m in messages:
            role = getattr(m, "role", None)
            if role == "assistant":
                assistant_count += 1
            tc = getattr(m, "tool_calls", None)
            if tc:
                for c in tc:
                    c_name = None
                    if hasattr(c, "function") and hasattr(c.function, "name"):
                        c_name = c.function.name
                    elif isinstance(c, dict):
                        c_name = c.get("function", {}).get("name") or c.get("name")
                    elif hasattr(c, "name"):
                        c_name = c.name
                    if c_name and c_name not in tools_list:
                        tools_list.append(str(c_name))

        llm_call_count = max(assistant_count, 1)
        tool_call_count = len(raw_tools) if raw_tools else len(tools_list)

        # Extract token usage if available from response metrics
        metrics = getattr(response, "metrics", None)
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        if metrics:
            input_tokens = getattr(metrics, "input_tokens", 0) or 0
            output_tokens = getattr(metrics, "output_tokens", 0) or 0
            total_tokens = getattr(metrics, "total_tokens", 0) or (input_tokens + output_tokens)

        return AdvisoryResult(
            prompt=prompt,
            text=text,
            duration_ms=duration_ms,
            model_name=self.settings.model_name,
            llm_call_count=llm_call_count,
            tool_call_count=tool_call_count,
            tools_used=tuple(tools_list),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
