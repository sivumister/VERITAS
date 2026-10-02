from services.model_service import (
    analyze_news
)

from services.attribution_service import (
    attribute_article_text
)


paragraph = """
A university announced a new research programme for students.
The programme includes computer laboratories, academic workshops,
research activities and access to digital resources.
Some online posts also claim that experimental machines at the
institution can predict future events and understand human thoughts.
University officials said those unusual claims have not been verified.
Students will continue attending normal academic programmes.
"""


# Repeat enough times to force multiple BERT chunks.
article = " ".join(
    [paragraph] * 12
)


print()
print("=" * 75)
print("VERITAS CHUNKED ATTRIBUTION CONSISTENCY TEST")
print("=" * 75)


model_result = analyze_news(
    article,
    "article"
)


print()
print(
    "Normal article model Fake score:",
    model_result[
        "fake_probability"
    ],
    "%"
)

print(
    "Normal model chunks:",
    model_result[
        "chunks_used"
    ]
)


print()
print(
    "Running chunk-aware attribution..."
)


# 8 steps is enough for this consistency test.
# The score itself does not depend on the IG step count.
attribution = attribute_article_text(
    article,
    steps=8
)


print()
print(
    "Attribution Fake score:",
    attribution[
        "fake_score"
    ],
    "%"
)

print(
    "Attribution chunks:",
    attribution[
        "chunks_used"
    ]
)

print(
    "Total tokens:",
    attribution[
        "total_tokens"
    ]
)

print(
    "Analysed tokens:",
    attribution[
        "analysed_tokens"
    ]
)

print(
    "Reached max article length:",
    attribution[
        "truncated"
    ]
)


score_difference = abs(

    model_result[
        "fake_probability"
    ]

    - attribution[
        "fake_score"
    ]
)


print()
print(
    "Score difference:",
    round(
        score_difference,
        4
    ),
    "percentage points"
)


if (
    model_result["chunks_used"]
    == attribution["chunks_used"]
    and score_difference <= 0.02
):

    print()
    print(
        "PASS: Attribution matches "
        "the article model."
    )

else:

    print()
    print(
        "FAIL: Attribution does not "
        "match the article model."
    )


print()
print("=" * 75)
print("TEST COMPLETE")
print("=" * 75)