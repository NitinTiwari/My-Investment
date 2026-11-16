"""
My-Invest Agentic Harness Package.

Provides benchmark suites, deterministic mock environments, execution tracers,
multi-criteria evaluators, and scorecard reporting for autonomous agent verification.
"""
from my_invest.harness.benchmarks import DEFAULT_BENCHMARK_SCENARIOS, BenchmarkScenario
from my_invest.harness.evaluators import AgenticHarnessEvaluator, EvaluationReport
from my_invest.harness.mock_adapter import (
    SimulatedAdvisorRunner,
    mock_harness_environment,
)
from my_invest.harness.runner import HarnessRunner, HarnessSuiteResult
from my_invest.harness.tracer import TrajectoryAuditor, TrajectoryTrace

__all__ = [
    "BenchmarkScenario",
    "DEFAULT_BENCHMARK_SCENARIOS",
    "AgenticHarnessEvaluator",
    "EvaluationReport",
    "HarnessRunner",
    "HarnessSuiteResult",
    "TrajectoryAuditor",
    "TrajectoryTrace",
    "SimulatedAdvisorRunner",
    "mock_harness_environment",
]

