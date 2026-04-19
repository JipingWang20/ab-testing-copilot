import pandas as pd
from stats.tests import two_sample_test

df = pd.read_csv("data/cookie_cats.csv")

print("=" * 60)
print("Test 1: retention_7 (binary)")
print("=" * 60)
result = two_sample_test(df, metric="retention_7", group_col="version", metric_type="binary")
for k, v in result.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("Test 2: sum_gamerounds (continuous)")
print("=" * 60)
result = two_sample_test(df, metric="sum_gamerounds", group_col="version", metric_type="continuous")
for k, v in result.items():
    print(f"  {k}: {v}")
