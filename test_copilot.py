import pandas as pd
from copilot.router import answer_question

df = pd.read_csv("data/cookie_cats.csv")

questions = [
    "Is the 7-day retention different between the two gate versions? Is it statistically significant?",
    "Does moving the gate from level 30 to 40 affect how many game rounds players play on average?",
]

for q in questions:
    print("=" * 70)
    print("Q:", q)
    print("=" * 70)
    result = answer_question(df, q)
    print("\nAnswer:")
    print(result["answer"])
    print("\nTool calls made:")
    for tc in result["tool_calls"]:
        print(f"  - {tc['tool']}({tc['args']})")
        r = tc["result"]
        if "p_value" in r:
            print(f"    p_value: {r['p_value']:.4f}")
        elif "status" in r:
            print(f"    status: {r['status']}")
            print(f"    verdict: {r['verdict']}")
    print()
