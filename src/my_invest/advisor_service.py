from dataclasses import dataclass
from time import perf_counter
from typing import Any
from agno.agent import Agent

from my_invest.agents.advisor_agent import create_investment_advisor_agent
from my_invest.cache import get_semantic_cache
from my_invest.gateway import estimate_token_cost
from my_invest.guardrails import InputGuardrail, OutputGuardrail
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
    estimated_cost_usd: float = 0.0
    is_cached: bool = False
    similarity_score: float = 0.0
    is_guarded: bool = False


class AdvisorService:
    def __init__(self, temperature: float | None = None, max_tokens: int | None = None) -> None:
        self.settings = settings
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_tokens = max_tokens if max_tokens is not None else settings.max_tokens
        self._agent: Agent | None = None
        self._cache = get_semantic_cache()
        self._input_guardrail = InputGuardrail()
        self._output_guardrail = OutputGuardrail()

    def get_agent(self, temperature: float | None = None, max_tokens: int | None = None) -> Agent:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        recreate = (
            self._agent is None
            or (temperature is not None and temperature != self.temperature)
            or (max_tokens is not None and max_tokens != self.max_tokens)
        )
        if recreate:
            self.temperature = temp
            self.max_tokens = tokens
            self._agent = create_investment_advisor_agent(temperature=temp, max_tokens=tokens)
        return self._agent

    def run_advisory(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AdvisoryResult:
        if not prompt or not prompt.strip():
            raise ValueError("Investment prompt cannot be empty.")

        started_at = perf_counter()

        # Step 1: Input Guardrail Validation (Domain Relevance & Prompt Injection)
        guard_res = self._input_guardrail.validate(prompt)
        if not guard_res.is_valid:
            duration_ms = round((perf_counter() - started_at) * 1000, 2)
            return AdvisoryResult(
                prompt=prompt,
                text=guard_res.message,
                duration_ms=duration_ms,
                model_name=f"My-Invest Guardrail ({self.settings.guard_model_name})",
                llm_call_count=0,
                tool_call_count=0,
                tools_used=(),
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                is_cached=False,
                similarity_score=0.0,
                is_guarded=True,
            )

        # Step 2: Tier-2 Qdrant Semantic Cache Check
        cached_match = self._cache.lookup(prompt)
        if cached_match is not None:
            duration_ms = round((perf_counter() - started_at) * 1000, 2)
            return AdvisoryResult(
                prompt=prompt,
                text=cached_match.text,
                duration_ms=duration_ms,
                model_name=f"{cached_match.model_name} (⚡ Qdrant Semantic Cache Hit: {cached_match.similarity_score * 100:.1f}%)",
                llm_call_count=0,
                tool_call_count=0,
                tools_used=cached_match.tools_used,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                is_cached=True,
                similarity_score=cached_match.similarity_score,
            )

        agent = self.get_agent(temperature=temperature, max_tokens=max_tokens)
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

        cost_usd = estimate_token_cost(self.settings.active_model_name, input_tokens, output_tokens)

        # Step 3: Output Guardrail Enforcement (SEBI & DICGC compliance)
        guarded_output = self._output_guardrail.enforce(text, prompt=prompt)
        text = guarded_output.text

        # Step 4: Tier-2 Store generated advisory in Qdrant Semantic Cache
        self._cache.store(
            prompt=prompt,
            text=text,
            tools_used=tuple(tools_list),
            model_name=self.settings.active_model_name,
        )

        return AdvisoryResult(
            prompt=prompt,
            text=text,
            duration_ms=duration_ms,
            model_name=self.settings.active_model_name,
            llm_call_count=llm_call_count,
            tool_call_count=tool_call_count,
            tools_used=tuple(tools_list),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost_usd,
            is_cached=False,
        )
