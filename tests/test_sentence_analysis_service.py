from services.sentence_analysis_service import (
    split_into_sentences,
    analyze_article_sentences
)


sample_article = """
The government announced a new health programme on Monday.
Officials said the programme would begin across several districts next month.
Scientists have secretly proved that drinking salt water cures every known viral disease.
Doctors are hiding this treatment from the public because pharmaceutical companies would lose money.
The Ministry of Health later released an official statement concerning the programme.
"""


print()
print("=" * 70)
print("VERITAS SENTENCE ANALYSIS TEST")
print("=" * 70)


# =========================================================
# TEST SENTENCE SPLITTING
# =========================================================

sentences = split_into_sentences(
    sample_article
)

print()
print(
    "SENTENCES FOUND:",
    len(sentences)
)

print()

for number, sentence in enumerate(
    sentences,
    start=1
):
    print(
        f"{number}. {sentence}"
    )


# =========================================================
# TEST MODEL ANALYSIS
# =========================================================

print()
print("=" * 70)
print("ANALYSING SENTENCES")
print("=" * 70)


results = analyze_article_sentences(
    sample_article
)


for result in results:

    print()

    print(
        f"Sentence {result['sentence_number']}:"
    )

    print(
        result["text"]
    )

    print(
        f"Fake probability: "
        f"{result['fake_probability']}%"
    )

    print(
        f"Real probability: "
        f"{result['real_probability']}%"
    )

    print(
        f"Signal: "
        f"{result['strength']} "
        f"{result['direction']}"
    )


print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print()
