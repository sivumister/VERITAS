from services.highlight_service import (
    score_article_passages
)


article = """
The university opened a new computer laboratory for students.
The facility contains modern computers and networking equipment.
University officials said the laboratory will support practical lessons.
Students are expected to begin using the facility this semester.
Secret technology inside the laboratory can read students' thoughts.
The machines can allegedly predict every student's future.
University management has provided no evidence supporting those claims.
The laboratory was officially opened during a campus ceremony.
"""


print()
print("=" * 70)
print("VERITAS WHOLE-ARTICLE SENTENCE INFLUENCE TEST")
print("=" * 70)


result = score_article_passages(
    article
)


print(
    "\nOriginal article Fake score:",
    result["baseline_fake_score"],
    "%"
)

print(
    "Article decision boundary:",
    result["article_threshold"],
    "%"
)

print(
    "Sentences analysed:",
    result["total_segments"]
)


print()
print("=" * 70)
print("SENTENCE INFLUENCE")
print("=" * 70)


for number, segment in enumerate(
    result["segments"],
    start=1
):

    print()
    print(
        f"[{number}]",
        segment["text"]
    )

    print(
        "Score without sentence:",
        segment["score_without_sentence"],
        "%"
    )

    print(
        "Influence on Fake score:",
        segment["influence"],
        "percentage points"
    )


print()
print("=" * 70)
print("STRONGEST POSITIVE CONTRIBUTORS")
print("=" * 70)


if result["ranked_positive"]:

    for number, segment in enumerate(
        result["ranked_positive"],
        start=1
    ):

        print()
        print(
            f"{number}.",
            segment["text"]
        )

        print(
            "Influence:",
            segment["influence"],
            "percentage points"
        )

else:

    print(
        "\nNo sentence produced a positive "
        "contribution to the article's Fake score."
    )


print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)