from services.attribution_service import (
    attribute_article_text
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
print("=" * 75)
print("VERITAS INTEGRATED GRADIENTS TEST")
print("=" * 75)


result = attribute_article_text(
    article,
    steps=24
)


print()
print(
    "Article Fake score:",
    result["fake_score"],
    "%"
)

print(
    "Article decision boundary:",
    result["article_threshold"],
    "%"
)

print(
    "Integrated Gradient steps:",
    result["steps"]
)

print(
    "Input truncated:",
    result["truncated"]
)


print()
print("=" * 75)
print("TOP POSITIVE CONTRIBUTORS TO FAKE OUTPUT")
print("=" * 75)


positive_words = (
    result["positive_words"][:15]
)


if positive_words:

    for number, word in enumerate(
        positive_words,
        start=1
    ):

        print(
            f"{number:>2}. "
            f"{word['word']:<20} "
            f"{word['contribution']:>8.3f}%"
        )

else:

    print(
        "No positive contributors found."
    )


print()
print("=" * 75)
print("TOP NEGATIVE CONTRIBUTORS TO FAKE OUTPUT")
print("=" * 75)


negative_words = (
    result["negative_words"][:15]
)


if negative_words:

    for number, word in enumerate(
        negative_words,
        start=1
    ):

        print(
            f"{number:>2}. "
            f"{word['word']:<20} "
            f"{word['contribution']:>8.3f}%"
        )

else:

    print(
        "No negative contributors found."
    )


print()
print("=" * 75)
print("FULL WORD ATTRIBUTION")
print("=" * 75)


for word in result["words"]:

    contribution = (
        word["contribution"]
    )


    if contribution > 0:

        marker = "+"

    elif contribution < 0:

        marker = "-"

    else:

        marker = " "


    print(
        f"{marker} "
        f"{word['word']:<20} "
        f"{contribution:>8.3f}%"
    )


print()
print("=" * 75)
print("TEST COMPLETE")
print("=" * 75)