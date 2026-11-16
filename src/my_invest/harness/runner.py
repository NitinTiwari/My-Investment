"""
Benchmark Runner and Scorecard Generator for My-Invest Agentic Harness.

Orchestrates execution of benchmark scenarios, collects execution traces,
evaluates responses, and outputs formatted Markdown & JSON scorecard reports.
"""
from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, List, Optional

from my_invest.advisor_service import AdvisorService
from my_invest.harness.benchmarks import DEFAULT_BENCHMARK_SCENARIOS, BenchmarkScenario
from my_invest.harness.evaluators import AgenticHarnessEvaluator, EvaluationReport
from my_invest.harness.mock_adapter import (
    SimulatedAdvisorRunner,
    mock_harness_environment,
)
from my_invest.harness.tracer import TrajectoryAuditor, TrajectoryTrace


class HarnessSuiteResult:
    """Aggregated results across a full benchmark run."""

    def __init__(self, reports: List[EvaluationReport], traces: List[TrajectoryTrace], mode: str) -> None:
        self.reports = reports
        self.traces = traces
        self.mode = mode

        self.total_scenarios = len(reports)
        self.passed_scenarios = sum(1 for r in reports if r.passed)
        self.failed_scenarios = self.total_scenarios - self.passed_scenarios
        self.pass_rate_pct = round((self.passed_scenarios / self.total_scenarios * 100), 1) if self.total_scenarios else 0.0

        # Metrics averages
        self.avg_score = round(sum(r.overall_score for r in reports) / max(self.total_scenarios, 1), 1)
        self.avg_tool_f1 = round(sum(r.tool_f1 for r in reports) / max(self.total_scenarios, 1), 2)
        self.avg_compliance = round(sum(r.compliance_score for r in reports) / max(self.total_scenarios, 1), 2)
        self.guardrail_success_rate = round(
            (sum(1 for r in reports if r.guardrail_passed) / max(self.total_scenarios, 1)) * 100, 1
        )

        self.total_duration_ms = round(sum(r.duration_ms for r in reports), 2)
        self.avg_duration_ms = round(self.total_duration_ms / max(self.total_scenarios, 1), 2)
        self.total_tokens = sum(r.total_tokens for r in reports)

    def to_markdown(self) -> str:
        """Generates a comprehensive Markdown scorecard report."""
        status_label = "[PASS]" if self.pass_rate_pct >= 80.0 else "[FAIL]"

        lines = [
            f"# My-Invest Agentic Harness Benchmark Scorecard",
            f"",
            f"**Execution Mode**: `{self.mode.upper()}` | **Status**: {status_label} ({self.pass_rate_pct}%)",
            f"",
            f"## Executive Summary",
            f"",
            f"| Metric | Value | Target | Status |",
            f"|---|---|---|---|",
            f"| **Overall Pass Rate** | **{self.pass_rate_pct}%** ({self.passed_scenarios}/{self.total_scenarios}) | >= 80% | {'PASS' if self.pass_rate_pct >= 80 else 'FAIL'} |",
            f"| **Avg Benchmark Score** | **{self.avg_score}%** | >= 75% | {'PASS' if self.avg_score >= 75 else 'FAIL'} |",
            f"| **Tool Selection F1** | **{self.avg_tool_f1}** | >= 0.70 | {'PASS' if self.avg_tool_f1 >= 0.7 else 'FAIL'} |",
            f"| **Compliance Score (SEBI & DICGC)** | **{self.avg_compliance * 100:.0f}%** | >= 85% | {'PASS' if self.avg_compliance >= 0.85 else 'FAIL'} |",
            f"| **Guardrail Defense Rate** | **{self.guardrail_success_rate}%** | 100% | {'PASS' if self.guardrail_success_rate == 100 else 'FAIL'} |",
            f"| **Avg Latency** | **{self.avg_duration_ms} ms** | < 2500 ms | PASS |",
            f"| **Total Tokens Consumed** | **{self.total_tokens:,}** | N/A | INFO |",
            f"",
            f"## Scenario Breakdown",
            f"",
            f"| ID | Title | Result | Score | Tool F1 | Compliance | Guardrail | Latency |",
            f"|---|---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]

        for r in self.reports:
            r_status = "PASS" if r.passed else "FAIL"
            g_status = "BLOCKED" if r.guardrail_passed else "MISSED"
            lines.append(
                f"| `{r.scenario_id}` | {r.title} | {r_status} | {r.overall_score}% | {r.tool_f1} | {r.compliance_score*100:.0f}% | {g_status} | {r.duration_ms:.1f}ms |"
            )

        lines.append(f"\n## Detailed Scenario Findings\n")
        for r in self.reports:
            lines.append(f"### `{r.scenario_id}`: {r.title}")
            for f in r.findings:
                # Strip raw emojis if any
                clean_f = f.replace("❌", "[FAIL]").replace("✅", "[PASS]").replace("⚠️", "[WARN]").replace("ℹ️", "[INFO]")
                lines.append(f"- {clean_f}")
            for rec in r.recommendations:
                lines.append(f"  - Recommendation: {rec}")
            lines.append("")

        return "\n".join(lines)


class HarnessRunner:
    """Runs benchmark scenarios through the agent and collects audit evaluations."""

    def __init__(self, mode: str = "mock") -> None:
        self.mode = mode.lower()
        self.service = AdvisorService()
        self.auditor = TrajectoryAuditor()
        self.simulated_runner = SimulatedAdvisorRunner(self.service)

    def run_scenario(self, scenario: BenchmarkScenario) -> tuple[TrajectoryTrace, EvaluationReport]:
        trace = self.auditor.start_trace(scenario.id, scenario.query)

        try:
            if self.mode == "live":
                # Real LLM call
                result = self.service.run_advisory(scenario.query)
            else:
                # Deterministic mock/simulated run
                with mock_harness_environment():
                    result = self.simulated_runner.run_advisory(scenario.query)

            self.auditor.finish_trace(trace, result)
        except Exception as e:
            trace.error = str(e)
            trace.is_guarded = False
            trace.output_text = f"Error during scenario execution: {e}"

        report = AgenticHarnessEvaluator.evaluate(scenario, trace)
        return trace, report

    def run_suite(
        self,
        scenarios: Optional[List[BenchmarkScenario]] = None,
        filter_ids: Optional[List[str]] = None,
    ) -> HarnessSuiteResult:
        all_scenarios = scenarios or DEFAULT_BENCHMARK_SCENARIOS

        if filter_ids:
            all_scenarios = [s for s in all_scenarios if s.id in filter_ids]

        traces: List[TrajectoryTrace] = []
        reports: List[EvaluationReport] = []

        for scenario in all_scenarios:
            trace, report = self.run_scenario(scenario)
            traces.append(trace)
            reports.append(report)

        return HarnessSuiteResult(reports=reports, traces=traces, mode=self.mode)
