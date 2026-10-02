from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from services.article_extractor import extract_article
from services.model_service import analyze_news


print()
print("=" * 60)
print("VERITAS URL + AI MODEL TEST")
print("=" * 60)
print()


url = input(
    "Paste a news article URL: "
)


try:

    # -----------------------------------------------------
    # STEP 1 - EXTRACT ARTICLE
    # -----------------------------------------------------

    print()
    print("Extracting article...")

    article = extract_article(
        url
    )


    print(
        "Article extracted successfully."
    )


    print()
    print(
        "Title:",
        article["title"]
    )

    print(
        "Word count:",
        article["word_count"]
    )


    # -----------------------------------------------------
    # STEP 2 - SEND EXTRACTED TEXT TO BERT
    # -----------------------------------------------------

    print()
    print(
        "Sending extracted article to VERITAS..."
    )


    result = analyze_news(
        article["text"],
        "article"
    )


    # -----------------------------------------------------
    # STEP 3 - DISPLAY RESULT
    # -----------------------------------------------------

    print()
    print("=" * 60)

    print(
        "VERITAS ANALYSIS RESULT"
    )

    print("=" * 60)


    print()
    print(
        "Headline:",
        article["title"]
    )


    print()
    print(
        "Model:",
        result["model_used"]
    )


    print(
        "Prediction:",
        result["prediction"]
    )


    print(
        "Fake probability:",
        f'{result["fake_probability"]}%'
    )


    print(
        "Real probability:",
        f'{result["real_probability"]}%'
    )


    print(
        "Decision threshold:",
        f'{result["threshold"]}%'
    )


    print(
        "Chunks analysed:",
        result["chunks_used"]
    )


    print()
    print("=" * 60)


except Exception as error:

    print()
    print(
        "VERITAS URL analysis failed:"
    )

    print(
        error
    )