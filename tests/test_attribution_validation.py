import re

from services.attribution_service import (
    attribute_article_text
)


# =========================================================
# TEST ARTICLES
# =========================================================

articles = {

    "MIXED ARTICLE": """
The university opened a new computer laboratory for students.
The facility contains modern computers and networking equipment.
University officials said the laboratory will support practical lessons.
Students are expected to begin using the facility this semester.
Secret technology inside the laboratory can read students' thoughts.
The machines can allegedly predict every student's future.
University management has provided no evidence supporting those claims.
The laboratory was officially opened during a campus ceremony.
""",


    "ORDINARY UNIVERSITY ARTICLE": """
The university opened a new library for students this week.
The building contains study rooms, computers and reading areas.
University staff attended the opening ceremony on Monday.
Students will be able to use the facility during normal campus hours.
The institution said additional books will be purchased during the semester.
The library will also provide internet access for academic research.
""",


    "SENSATIONAL ARTICLE": """
A mysterious machine has reportedly appeared inside a private laboratory.
Some online posts claim the machine can read people's thoughts.
Others claim that it can predict events before they happen.
No technical documentation has been published to demonstrate these abilities.
The claims have spread rapidly across several social media platforms.
Researchers have not independently confirmed how the alleged machine works.
""",


    "DEBUNKING STYLE ARTICLE": """
A message circulating online claims that ordinary mobile phones can secretly read human thoughts.
Researchers said there is no scientific evidence supporting the claim.
The technology used in standard smartphones cannot directly access a person's thoughts.
Experts explained that phones can collect behavioural data from apps and sensors.
This information may be used to predict interests, but this is different from reading thoughts.
Readers were advised to verify unusual technology claims before sharing them.
"""
}


# =========================================================
# FIND SENTENCE SPANS
# =========================================================

def get_sentence_spans(text):

    spans = []

    pattern = re.compile(
        r'[^.!?]+[.!?]?'
    )


    for match in pattern.finditer(text):

        sentence = (
            match.group()
            .strip()
        )


        if not sentence:
            continue


        # Remove leading whitespace from position
        raw_text = match.group()

        leading_spaces = (
            len(raw_text)
            - len(raw_text.lstrip())
        )


        start = (
            match.start()
            + leading_spaces
        )


        end = (
            start
            + len(sentence)
        )


        spans.append({
            "text": sentence,
            "start": start,
            "end": end
        })


    return spans


# =========================================================
# AGGREGATE WORD ATTRIBUTIONS INTO SENTENCES
# =========================================================

def calculate_sentence_scores(
    text,
    words
):

    sentences = get_sentence_spans(
        text
    )


    results = []


    for sentence in sentences:

        sentence_words = [

            word

            for word in words

            if (
                word["start"]
                >= sentence["start"]
                and
                word["start"]
                < sentence["end"]
            )
        ]


        positive = sum(

            word["contribution"]

            for word in sentence_words

            if word["contribution"] > 0
        )


        negative = sum(

            word["contribution"]

            for word in sentence_words

            if word["contribution"] < 0
        )


        net = (
            positive
            + negative
        )


        strongest_positive = sorted(

            [
                word
                for word in sentence_words
                if word["contribution"] > 0
            ],

            key=lambda item:
                item["contribution"],

            reverse=True

        )[:5]


        results.append({

            "text":
                sentence["text"],

            "positive":
                round(
                    positive,
                    3
                ),

            "negative":
                round(
                    negative,
                    3
                ),

            "net":
                round(
                    net,
                    3
                ),

            "top_words":
                strongest_positive
        })


    return results


# =========================================================
# RUN VALIDATION
# =========================================================

for article_name, article in articles.items():

    article = article.strip()


    print()
    print("=" * 80)
    print(article_name)
    print("=" * 80)


    result = attribute_article_text(
        article,
        steps=24
    )


    print(
        "\nArticle Fake score:",
        result["fake_score"],
        "%"
    )

    print(
        "Decision boundary:",
        result["article_threshold"],
        "%"
    )


    sentence_results = (
        calculate_sentence_scores(
            article,
            result["words"]
        )
    )


    print()
    print("-" * 80)
    print("PASSAGE ATTRIBUTION")
    print("-" * 80)


    for number, sentence in enumerate(
        sentence_results,
        start=1
    ):

        print()
        print(
            f"[{number}] "
            f"{sentence['text']}"
        )

        print(
            "Positive contribution:",
            sentence["positive"],
            "%"
        )

        print(
            "Negative contribution:",
            sentence["negative"],
            "%"
        )

        print(
            "Net contribution:",
            sentence["net"],
            "%"
        )


        if sentence["top_words"]:

            words_text = ", ".join(

                f"{word['word']} "
                f"({word['contribution']:+.2f})"

                for word
                in sentence["top_words"]
            )


            print(
                "Strongest positive words:",
                words_text
            )


    print()
    print("-" * 80)
    print("STRONGEST SENTENCES TOWARD FAKE OUTPUT")
    print("-" * 80)


    ranked_sentences = sorted(

        sentence_results,

        key=lambda item:
            item["net"],

        reverse=True
    )


    for rank, sentence in enumerate(
        ranked_sentences[:3],
        start=1
    ):

        print()
        print(
            f"{rank}. "
            f"Net {sentence['net']:+.3f}%"
        )

        print(
            sentence["text"]
        )


print()
print("=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)