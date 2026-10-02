from services.attribution_service import (
    attribute_article_text,
    build_influence_highlights
)


articles = {

    "MIXED": """
The university opened a new computer laboratory for students.
The facility contains modern computers and networking equipment.
University officials said the laboratory will support practical lessons.
Students are expected to begin using the facility this semester.
Secret technology inside the laboratory can read students' thoughts.
The machines can allegedly predict every student's future.
University management has provided no evidence supporting those claims.
The laboratory was officially opened during a campus ceremony.
""",


    "ORDINARY": """
The university opened a new library for students this week.
The building contains study rooms, computers and reading areas.
University staff attended the opening ceremony on Monday.
Students will be able to use the facility during normal campus hours.
The institution said additional books will be purchased during the semester.
The library will also provide internet access for academic research.
""",


    "SENSATIONAL": """
A mysterious machine has reportedly appeared inside a private laboratory.
Some online posts claim the machine can read people's thoughts.
Others claim that it can predict events before they happen.
No technical documentation has been published to demonstrate these abilities.
The claims have spread rapidly across several social media platforms.
Researchers have not independently confirmed how the alleged machine works.
""",


    "DEBUNKING": """
A message circulating online claims that ordinary mobile phones can secretly read human thoughts.
Researchers said there is no scientific evidence supporting the claim.
The technology used in standard smartphones cannot directly access a person's thoughts.
Experts explained that phones can collect behavioural data from apps and sensors.
This information may be used to predict interests, but this is different from reading thoughts.
Readers were advised to verify unusual technology claims before sharing them.
"""
}


for name, article in articles.items():

    article = article.strip()


    print()
    print("=" * 80)
    print(name)
    print("=" * 80)


    attribution = attribute_article_text(
        article,
        steps=24
    )


    highlights = (
        build_influence_highlights(
            article,
            attribution
        )
    )


    print(
        "Article Fake score:",
        attribution["fake_score"],
        "%"
    )


    print(
        "Highlighted passages:",
        highlights["highlight_count"]
    )


    for number, segment in enumerate(
        highlights["segments"],
        start=1
    ):

        marker = (
            ">>> AI INFLUENCE"
            if segment["highlighted"]
            else "    "
        )


        print()
        print(
            f"{marker} [{number}]"
        )

        print(
            segment["text"]
        )

        print(
            "Net contribution:",
            segment[
                "net_contribution"
            ],
            "%"
        )

        print(
            "Signal:",
            segment[
                "signal"
            ]
        )


print()
print("=" * 80)
print("HIGHLIGHT TEST COMPLETE")
print("=" * 80)