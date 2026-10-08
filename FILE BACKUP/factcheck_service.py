from pathlib import Path
import os

import requests
from dotenv import load_dotenv


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# =========================================================
# LOAD .ENV FILE
# =========================================================

load_dotenv(
    BASE_DIR / ".env"
)


API_KEY = os.getenv(
    "GOOGLE_FACTCHECK_API_KEY"
)


# =========================================================
# GOOGLE FACT CHECK API
# =========================================================

FACTCHECK_URL = (
    "https://factchecktools.googleapis.com/"
    "v1alpha1/claims:search"
)


# =========================================================
# SEARCH FACT CHECKS
# =========================================================

def search_fact_checks(query, max_results=5):

    query = query.strip()

    if not query:
        raise ValueError(
            "Fact-check query cannot be empty."
        )

    if not API_KEY:
        raise ValueError(
            "Google Fact Check API key was not found."
        )


    params = {

        "query": query,

        "languageCode": "en",

        "pageSize": max_results,

        "key": API_KEY
    }


    try:

        response = requests.get(
            FACTCHECK_URL,
            params=params,
            timeout=15
        )

        response.raise_for_status()

    except requests.RequestException as error:

        raise ValueError(
            f"Fact-check search failed: {error}"
        )


    data = response.json()


    claims = data.get(
        "claims",
        []
    )


    results = []


    # =====================================================
    # CLEAN API RESPONSE
    # =====================================================

    for claim in claims:

        claim_text = claim.get(
            "text",
            "Unknown claim"
        )

        claimant = claim.get(
            "claimant"
        )

        reviews = claim.get(
            "claimReview",
            []
        )


        for review in reviews:

            publisher = review.get(
                "publisher",
                {}
            )


            results.append({

                "claim":
                    claim_text,

                "claimant":
                    claimant,

                "publisher":
                    publisher.get(
                        "name",
                        "Unknown publisher"
                    ),

                "publisher_site":
                    publisher.get(
                        "site"
                    ),

                "rating":
                    review.get(
                        "textualRating",
                        "No rating"
                    ),

                "title":
                    review.get(
                        "title",
                        "Fact Check"
                    ),

                "url":
                    review.get(
                        "url"
                    ),

                "review_date":
                    review.get(
                        "reviewDate"
                    )
            })


    return results