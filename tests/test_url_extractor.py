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


print()
print("=" * 60)
print("VERITAS URL EXTRACTION TEST")
print("=" * 60)
print()


url = input(
    "Paste a news article URL: "
)


try:

    article = extract_article(
        url
    )


    print()
    print("=" * 60)

    print("ARTICLE EXTRACTED SUCCESSFULLY")

    print("=" * 60)


    print()
    print("Title:")
    print(
        article["title"]
    )


    print()
    print("Final URL:")
    print(
        article["url"]
    )


    print()
    print(
        "Word count:",
        article["word_count"]
    )


    print()
    print("Article preview:")
    print("-" * 60)

    print(
        article["text"][:1500]
    )

    print()

    print("-" * 60)


except Exception as error:

    print()
    print(
        "Extraction failed:"
    )

    print(
        error
    )