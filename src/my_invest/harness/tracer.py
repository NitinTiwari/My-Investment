"""
Execution Trajectory Tracer & Auditor for My-Invest Agentic Harness.

Audits agent lifecycle events, multi-step tool calls, tokens, latency,
and guardrail interventions.
"""
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Optional


@dataclass
class ToolInvocationRecord:
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    status: str = "success"
    output_preview: str = ""


@dataclass
class TrajectoryTrace:
    scenario_id: str
    prompt: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    model_name: str = ""
    llm_iterations: int = 0
    tools_invoked: List[str] = field(default_factory=list)
    detailed_tool_records: List[ToolInvocationRecord] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    is_cached: bool = False
    similarity_score: float = 0.0
    is_guarded: bool = False
    output_text: str = ""
    error: Optional[str] = None


class TrajectoryAuditor:
    """Records and evaluates execution traces for agent interactions."""

    def __init__(self) -> None:
        self.traces: List[TrajectoryTrace] = []

    def start_trace(self, scenario_id: str, prompt: str) -> TrajectoryTrace:
        trace = TrajectoryTrace(
            scenario_id=scenario_id,
            prompt=prompt,
            start_time=perf_counter(),
        )
        self.traces.append(trace)
        return trace

    def finish_trace(self, trace: TrajectoryTrace, result: Any) -> TrajectoryTrace:
        trace.end_time = perf_counter()
        trace.duration_ms = getattr(result, "duration_ms", round((trace.end_time - trace.start_time) * 1000, 2))
        trace.model_name = getattr(result, "model_name", "")
        trace.llm_iterations = getattr(result, "llm_call_count", 1)
        trace.tools_invoked = list(getattr(result, "tools_used", ()))
        trace.input_tokens = getattr(result, "input_tokens", 0)
        trace.output_tokens = getattr(result, "output_tokens", 0)
        trace.total_tokens = getattr(result, "total_tokens", 0)
        trace.estimated_cost_usd = getattr(result, "estimated_cost_usd", 0.0)
        trace.is_cached = getattr(result, "is_cached", False)
        trace.similarity_score = getattr(result, "similarity_score", 0.0)
        trace.is_guarded = getattr(result, "is_guarded", False)
        trace.output_text = getattr(result, "text", str(result))
        return trace

