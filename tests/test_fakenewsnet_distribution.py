import csv
from pathlib import Path


# FakeNewsNet contains very large tweet_ids fields.
# Increase Python's default CSV field-size limit.
csv.field_size_limit(50_000_000)


# =========================================================
# LOCATE DATASET
# =========================================================

WEBPAGE_ROOT = Path(__file__).resolve().parents[1]

REPOSITORY_ROOT = WEBPAGE_ROOT.parents[2]

DATASET_DIR = (
    REPOSITORY_ROOT
    / "FakeNewsNet"
    / "dataset"
)


DATASETS = {

    "PolitiFact Fake":
        "politifact_fake.csv",

    "PolitiFact Real":
        "politifact_real.csv",

    "GossipCop Fake":
        "gossipcop_fake.csv",

    "GossipCop Real":
        "gossipcop_real.csv"
}


# =========================================================
# COUNT RECORDS
# =========================================================

def count_records(file_path):

    count = 0

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            title = row.get(
                "title",
                ""
            ).strip()

            if title:

                count += 1

    return count


# =========================================================
# DISPLAY DISTRIBUTION
# =========================================================

print()
print("=" * 70)
print("FAKENEWSNET DATASET DISTRIBUTION")
print("=" * 70)


counts = {}


for name, filename in DATASETS.items():

    path = (
        DATASET_DIR
        / filename
    )

    count = count_records(
        path
    )

    counts[name] = count

    print()
    print(
        f"{name}: {count}"
    )


# =========================================================
# TOTAL FAKE / REAL
# =========================================================

total_fake = (
    counts["PolitiFact Fake"]
    +
    counts["GossipCop Fake"]
)

total_real = (
    counts["PolitiFact Real"]
    +
    counts["GossipCop Real"]
)

total = (
    total_fake
    +
    total_real
)


print()
print("=" * 70)
print("CLASS TOTALS")
print("=" * 70)

print()

print(
    "Total FAKE:",
    total_fake
)

print(
    "Total REAL:",
    total_real
)

print(
    "Total records:",
    total
)


# =========================================================
# PERCENTAGES
# =========================================================

if total > 0:

    fake_percentage = (
        total_fake
        / total
        * 100
    )

    real_percentage = (
        total_real
        / total
        * 100
    )

    print()

    print(
        f"FAKE percentage: "
        f"{fake_percentage:.2f}%"
    )

    print(
        f"REAL percentage: "
        f"{real_percentage:.2f}%"
    )


# =========================================================
# CLASS RATIO
# =========================================================

if total_fake > 0:

    ratio = (
        total_real
        / total_fake
    )

    print()

    print(
        f"REAL-to-FAKE ratio: "
        f"{ratio:.2f}:1"
    )


print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print()