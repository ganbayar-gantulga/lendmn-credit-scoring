"""
LendMN Credit Scoring — Synthetic Data Generator
Mongolian lending market-themed dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

def generate_credit_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic Mongolian credit applicant data.
    Features are calibrated to reflect MNT salary ranges and
    typical loan behavior in Mongolia.
    """

    # ── Demographics ──────────────────────────────────────────
    age = np.random.normal(35, 10, n_samples).clip(18, 70).astype(int)

    employment_type = np.random.choice(
        ["government", "private", "self_employed", "freelance"],
        p=[0.30, 0.45, 0.18, 0.07],
        size=n_samples,
    )

    # Monthly income in MNT (₮)
    income_map = {
        "government":   (1_200_000, 400_000),
        "private":      (1_500_000, 600_000),
        "self_employed":(1_800_000, 900_000),
        "freelance":    (900_000,  500_000),
    }
    monthly_income = np.array([
        max(300_000, np.random.normal(*income_map[e]))
        for e in employment_type
    ])

    # Employment duration (months)
    employment_months = np.random.exponential(36, n_samples).clip(0, 360).astype(int)

    # ── Credit Profile ────────────────────────────────────────
    num_existing_loans   = np.random.poisson(1.2, n_samples).clip(0, 6)
    credit_history_months = np.random.exponential(30, n_samples).clip(0, 240).astype(int)
    previous_defaults    = np.random.poisson(0.3, n_samples).clip(0, 5)
    num_dependents       = np.random.poisson(1.5, n_samples).clip(0, 7)

    # Loan requested (₮)
    loan_amount = np.random.lognormal(np.log(2_000_000), 0.8, n_samples).clip(100_000, 50_000_000)

    # ── Derived Features ──────────────────────────────────────
    debt_to_income = (loan_amount * 0.05) / monthly_income          # approx monthly payment ratio
    loan_to_income  = loan_amount / (monthly_income * 12)

    # ── Target: Default Probability ──────────────────────────
    # Logistic model: higher risk → more likely to default
    log_odds = (
        -2.5                                                          # base
        + 0.01  * (35 - age).clip(-10, 10)                           # older = lower risk
        - 0.0000004 * monthly_income                                  # higher income = lower risk
        - 0.005 * employment_months.clip(0, 60)                       # tenure reduces risk
        + 0.4   * (employment_type == "freelance").astype(float)      # freelance riskier
        - 0.3   * (employment_type == "government").astype(float)     # gov safer
        + 0.5   * num_existing_loans                                  # more loans = riskier
        - 0.005 * credit_history_months.clip(0, 60)                  # longer history = safer
        + 1.2   * previous_defaults                                   # defaults = big risk
        + 0.1   * num_dependents                                      # dependents add pressure
        + 0.6   * debt_to_income.clip(0, 5)                          # high DTI = riskier
    )
    prob_default  = 1 / (1 + np.exp(-log_odds))
    noise         = np.random.normal(0, 0.05, n_samples)
    default       = (prob_default + noise > 0.5).astype(int)

    df = pd.DataFrame({
        "age":                  age,
        "employment_type":      employment_type,
        "monthly_income":       monthly_income.astype(int),
        "employment_months":    employment_months,
        "num_existing_loans":   num_existing_loans,
        "credit_history_months":credit_history_months,
        "previous_defaults":    previous_defaults,
        "num_dependents":       num_dependents,
        "loan_amount":          loan_amount.astype(int),
        "debt_to_income":       debt_to_income.round(4),
        "loan_to_income":       loan_to_income.round(4),
        "default":              default,
    })

    return df


if __name__ == "__main__":
    df = generate_credit_dataset(5000)
    out = Path(__file__).parent.parent / "data" / "credit_data.csv"
    df.to_csv(out, index=False)
    print(f"✓ Dataset saved → {out}")
    print(f"  Shape   : {df.shape}")
    print(f"  Defaults: {df['default'].sum()} / {len(df)} ({df['default'].mean():.1%})")
    print(df.describe().round(2))
