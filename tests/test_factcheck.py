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


from services.factcheck_service import search_fact_checks


print()
print("=" * 60)
print("VERITAS FACT CHECK TEST")
print("=" * 60)
print()


claim = input(
    "Enter a claim to fact-check: "
)


try:

    results = search_fact_checks(
        claim
    )


    print()


    if not results:

        print(
            "No matching published fact-checks were found."
        )


    else:

        print(
            f"Found {len(results)} fact-check review(s)."
        )

        print()


        for number, result in enumerate(
            results,
            start=1
        ):

            print("=" * 60)

            print(
                f"RESULT {number}"
            )

            print("=" * 60)


            print(
                "Reviewed claim:",
                result["claim"]
            )


            print(
                "Claimant:",
                result["claimant"]
                or "Unknown"
            )


            print(
                "Publisher:",
                result["publisher"]
            )


            print(
                "Rating:",
                result["rating"]
            )


            print(
                "Title:",
                result["title"]
            )


            print(
                "Review date:",
                result["review_date"]
                or "Unknown"
            )


            print(
                "URL:",
                result["url"]
                or "Unavailable"
            )


            print()


except Exception as error:

    print()

    print(
        "Fact-check test failed:"
    )

    print(
        error
    )