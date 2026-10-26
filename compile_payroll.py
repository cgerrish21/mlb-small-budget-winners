"""
Compile team payroll data from config.py constants.
Computes payroll rank (percentile) within each season.
Output: data/payroll.csv
"""

import os
import pandas as pd

from config import YEARS, DATA_DIR, PAYROLL_DATA


def main():
    print("Compiling payroll data...", flush=True)
    rows = []

    for year in YEARS:
        if year not in PAYROLL_DATA:
            print(f"  WARNING: no payroll data for {year}", flush=True)
            continue

        payrolls = PAYROLL_DATA[year]
        # Sort by payroll to compute ranks
        sorted_teams = sorted(payrolls.items(), key=lambda x: x[1])
        n = len(sorted_teams)

        for rank_idx, (team, payroll) in enumerate(sorted_teams):
            # rank_pct: 0 = lowest payroll, 1 = highest
            rank_pct = rank_idx / (n - 1) if n > 1 else 0.5

            # Tier: bottom/middle/top third
            if rank_pct <= 0.333:
                tier = "bottom"
            elif rank_pct >= 0.667:
                tier = "top"
            else:
                tier = "middle"

            rows.append({
                "Season": year,
                "team": team,
                "payroll_mm": payroll,
                "payroll_rank": rank_idx + 1,  # 1 = lowest
                "payroll_rank_pct": round(rank_pct, 3),
                "payroll_tier": tier,
            })

        print(f"  {year}: {n} teams, range ${sorted_teams[0][1]}M - ${sorted_teams[-1][1]}M",
              flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(DATA_DIR, "payroll.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
