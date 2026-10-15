import sys
from pathlib import Path

# Ensure src is on Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from my_invest.advisor_service import AdvisorService
from my_invest.settings import settings


def main():
    print("=" * 70)
    print("     My-Invest: Indian FD & Mutual Fund Investment Advisor")
    print(f"     Powered by {settings.active_provider} ({settings.active_model_name}) & Agno")
    print("=" * 70)
    print("Example prompts:")
    print("  - I want to invest 2 lacs inr in Indian bank fixed deposits and in Indian mutual funds")
    print("  - Suggest best 3 year FD rates and flexi cap mutual funds for 1.5 lakh rupees")
    print("  - Type 'exit' to quit.")

    service = AdvisorService()

    while True:
        try:
            prompt = input("Enter your investment goal: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("Goodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "q"}:
            print("Exiting My-Invest. Happy investing!")
            break

        print("[Thinking] Analyzing Indian bank FD rates, AMFI mutual funds, and calculating allocations...")
        try:
            result = service.run_advisory(prompt)
            print("" + "=" * 70)
            print(result.text)
            tools_summary = ", ".join(result.tools_used) if result.tools_used else "None"
            print(f"(Completed in {result.duration_ms / 1000:.2f}s | LLM Inferences: {result.llm_call_count} | Tools Used: {result.tool_call_count} [{tools_summary}])")
        except Exception as error:
            print(f"[Error] Unable to complete investment research: {error}")


if __name__ == "__main__":
    main()
