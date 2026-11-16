"""
CLI entry point for My-Invest Agentic Harness.

Usage:
    python -m my_invest.harness.cli [--mode mock|live] [--output REPORT.md] [--fail-under 80]
"""
import argparse
import io
import sys
from pathlib import Path

from my_invest.harness.runner import HarnessRunner


def main() -> int:
    # Ensure stdout handles UTF-8 characters gracefully on all platforms
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="My-Invest Agentic Evaluation & Benchmark Harness CLI"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="Execution mode: 'mock' for offline deterministic testing, 'live' for real LLM inference (default: mock)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the Markdown Scorecard report (e.g. scorecard.md)",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=80.0,
        help="Minimum required benchmark pass rate percentage (default: 80.0)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated scenario IDs to run (e.g. scen_fd_01,scen_bal_03)",
    )

    args = parser.parse_args()

    filter_ids = [s.strip() for s in args.scenarios.split(",")] if args.scenarios else None

    print(f"\n=======================================================")
    print(f"[HARNESS] MY-INVEST AGENTIC HARNESS RUNNER")
    print(f"   Mode: {args.mode.upper()}")
    print(f"   Fail Threshold: {args.fail_under}%")
    print(f"=======================================================\n")

    runner = HarnessRunner(mode=args.mode)
    suite_result = runner.run_suite(filter_ids=filter_ids)

    markdown_report = suite_result.to_markdown()
    print(markdown_report)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.write_text(markdown_report, encoding="utf-8")
        print(f"\nSaved scorecard report to: {out_path}")

    if suite_result.pass_rate_pct < args.fail_under:
        print(
            f"\n[FAIL] Pass rate {suite_result.pass_rate_pct}% is below required threshold of {args.fail_under}%"
        )
        return 1

    print(f"\n[SUCCESS] All harness benchmarks passed above threshold ({suite_result.pass_rate_pct}% >= {args.fail_under}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
