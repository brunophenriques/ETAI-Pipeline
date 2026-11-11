from src.data import load_config, load_data
from src.preprocess import (
    prepare_rows,
    fill_structural_missing,
    recover_recency,
)

config = load_config()
train, _, _ = load_data(config)

train = prepare_rows(train, config)

structural = fill_structural_missing(train, config)

processed = recover_recency(
    df=structural,
    history=train,
    config=config,
)

print("\n=== RECENCY RECOVERY TEST ===")

print(
    "NaN before:",
    structural["days_since_previous_purchase"].isna().sum()
)

print(
    "Recovered:",
    processed["recency_was_recovered"].sum()
)

print(
    "NaN after:",
    processed["days_since_previous_purchase"].isna().sum()
)

known = structural["days_since_previous_purchase"].notna()

assert processed.loc[
    known, "days_since_previous_purchase"
].equals(
    structural.loc[
        known, "days_since_previous_purchase"
    ]
)

assert processed["recency_was_recovered"].sum() == 192
assert processed["days_since_previous_purchase"].isna().sum() == 1

print("\nRecency recovery passed.")
