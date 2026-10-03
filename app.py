from flask import Flask, render_template, request
from datetime import datetime
import json
import os

from services.model_service import analyze_news
from services.article_extractor import extract_article
from services.factcheck_service import search_fact_checks
from services.evidence_service import (
    rank_fact_checks,
    determine_evidence_relationship
)
from services.prediction_strength import (
    calculate_prediction_strength
)
from services.explanation_service import (
    build_explanation
)
from services.assessment_service import (
    build_overall_assessment
)
from services.attribution_service import (
    attribute_article_text,
    build_influence_highlights
)


app = Flask(__name__)

HISTORY_FILE = "history.json"


# ---------------------------------------------------------
# HISTORY FUNCTIONS
# ---------------------------------------------------------

def load_history():

    if os.path.exists(HISTORY_FILE):

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    return []


def save_history(data):

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ---------------------------------------------------------
# ANALYSE NEWS
# ---------------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    # -------------------------------------------------
    # GET FORM INPUT
    # -------------------------------------------------

    user_input = request.form.get(
        "news_input",
        ""
    ).strip()

    input_type = request.form.get(
        "input_type",
        "article"
    ).strip().lower()


    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------

    if not user_input:

        return render_template(
            "analysis.html",
            error="Please enter some content before analysing."
        )


    allowed_types = [
        "claim",
        "headline",
        "article",
        "url"
    ]


    if input_type not in allowed_types:

        return render_template(
            "analysis.html",
            error="Invalid analysis type selected."
        )


    try:

        # Default values
        source_url = None
        article_title = None
        displayed_content = user_input


        # =================================================
        # URL ANALYSIS
        # =================================================

        if input_type == "url":

            print(
                "Extracting article from URL..."
            )

            extracted_article = extract_article(
                user_input
            )


            source_url = extracted_article[
                "url"
            ]

            article_title = extracted_article[
                "title"
            ]

            article_text = extracted_article[
                "text"
            ]


            print(
                "Article extracted successfully."
            )


            # Analyse extracted ARTICLE TEXT,
            # not the URL itself
            model_result = analyze_news(
                article_text,
                "article"
            )


            displayed_content = article_text


        # =================================================
        # NORMAL TEXT ANALYSIS
        # =================================================

        else:

            model_result = analyze_news(
                user_input,
                input_type
            )


        # -------------------------------------------------
        # MODEL RESULT
        # -------------------------------------------------

        prediction = model_result[
            "prediction"
        ]

        fake_probability = model_result[
            "fake_probability"
        ]

        real_probability = model_result[
            "real_probability"
        ]

        model_used = model_result[
            "model_used"
        ]

        threshold = model_result.get(
            "threshold"
        )

        chunks_used = model_result.get(
            "chunks_used"
        )

        # =========================================================
        # MODEL PREDICTION STRENGTH
        # =========================================================

        prediction_strength = calculate_prediction_strength(
            prediction=prediction,
            fake_probability=fake_probability,
            real_probability=real_probability,
            threshold=threshold
        )

        # =========================================================
        # AI INFLUENCE HIGHLIGHTS
        # =========================================================
        #
        # Only article-based analysis receives passage attribution.
        #
        # Integrated Gradients explains which passages influenced
        # the model's FAKE-side output. It does NOT determine that
        # highlighted passages are factually false.
        # =========================================================

        attribution_info = None

        influence_highlights = None

        attribution_error = None


        if input_type in [
            "article",
            "url"
        ]:

            try:

                print()
                print(
                    "Generating AI influence highlights..."
                )


                attribution_info = (
                    attribute_article_text(
                        displayed_content,
                        steps=24
                    )
                )


                influence_highlights = (
                    build_influence_highlights(

                        displayed_content,

                        attribution_info,

                        minimum_net=1.5,

                        maximum_highlights=5
                    )
                )


                print(
                    "AI influence highlights generated:"
                )

                print(
                    influence_highlights[
                        "highlight_count"
                    ],
                    "passage(s)"
                )

                print(
                    "Attribution chunks:",
                    attribution_info[
                        "chunks_used"
                    ]
                )

                print()


            except Exception as attribution_exception:

                print(
                    "Attribution error:",
                    attribution_exception
                )


                attribution_error = (
                    "AI influence highlights could "
                    "not be generated for this article."
                )


                attribution_info = None

                influence_highlights = None


        # -------------------------------------------------
        # SAVE HISTORY
        # -------------------------------------------------
        # -------------------------------------------------
        # EXTERNAL FACT-CHECK SEARCH
        # -------------------------------------------------

        
        # =========================================================
        # EXTERNAL FACT-CHECK EVIDENCE
        # =========================================================

        factcheck_results = []

        factcheck_error = None

        factcheck_query = None

        evidence_relationship = {
            "status": "NO_MATCH",
            "message": "No matching published fact-check was found."
        }


        try:

            # -----------------------------------------------------
            # CHOOSE SEARCH QUERY
            # -----------------------------------------------------

            if input_type in ["claim", "headline"]:

                factcheck_query = user_input


            elif input_type == "url":

                factcheck_query = (
                    article_title
                    if article_title
                    else " ".join(
                        displayed_content.split()[:25]
                    )
                )


            else:

                words = displayed_content.split()

                factcheck_query = " ".join(
                    words[:25]
                )


            # -----------------------------------------------------
            # SEARCH GOOGLE FACT CHECK
            # -----------------------------------------------------

            raw_factcheck_results = search_fact_checks(
                factcheck_query,
                max_results=5
            )


            # -----------------------------------------------------
            # RANK RESULTS BY RELEVANCE
            # -----------------------------------------------------

            factcheck_results = rank_fact_checks(
                factcheck_query,
                raw_factcheck_results
            )


            # -----------------------------------------------------
            # COMPARE EXTERNAL EVIDENCE WITH AI
            # -----------------------------------------------------

            evidence_relationship = (
                determine_evidence_relationship(
                    prediction,
                    factcheck_results
                )
            )


        except Exception as factcheck_exception:

            print(
                "Fact-check search error:",
                factcheck_exception
            )

            factcheck_error = (
                "External fact-check evidence "
                "could not be retrieved."
            )

            evidence_relationship = {
                "status": "UNAVAILABLE",
                "message": (
                    "The external fact-check service "
                    "was unavailable, so VERITAS could "
                    "not compare the AI prediction with "
                    "external evidence."
                )
            }
        # =========================================================
        # USER-FRIENDLY PREDICTION EXPLANATION
        # =========================================================

        explanation = build_explanation(
            prediction=prediction,
            fake_probability=fake_probability,
            real_probability=real_probability,
            prediction_strength=prediction_strength,
            threshold=threshold,
            evidence_relationship=evidence_relationship,
            factcheck_results=factcheck_results
        )            
       
        # =========================================================
        # OVERALL VERITAS ASSESSMENT
        # =========================================================

        overall_assessment = build_overall_assessment(
            prediction=prediction,
            prediction_strength=prediction_strength,
            evidence_relationship=evidence_relationship,
            factcheck_results=factcheck_results
        )
        print()
        print("OVERALL VERITAS ASSESSMENT:")
        print(overall_assessment)
        print()


        record = {

            "input":
                user_input,

            "input_type":
                input_type,

            "title":
                article_title,

            "source_url":
                source_url,

            "model_used":
                model_used,

            "result":
                prediction,

            "fake_probability":
                fake_probability,

            "real_probability":
                real_probability,

            "threshold":
                threshold,

            "chunks_used":
                chunks_used,
            "factcheck_count":
                 len(factcheck_results),

            "evidence_status":
                evidence_relationship["status"],

            "prediction_strength":
                prediction_strength["strength"],

            "prediction_margin":
                prediction_strength["distance"],

            "influence_highlight_count":
                (
                    influence_highlights[
                        "highlight_count"
                    ]
                    if influence_highlights
                    else 0
                ),

            "attribution_chunks":
                (
                    attribution_info[
                        "chunks_used"
                    ]
                    if attribution_info
                    else None
                ),

            "attribution_truncated":
                (
                    attribution_info[
                        "truncated"
                    ]
                    if attribution_info
                    else None
                ),
            "explanation_label":
                explanation["result_label"],

            "explanation_summary":
                explanation["summary"],

            "assessment_status":
                overall_assessment["status"],

            "assessment_label":
                overall_assessment["label"],

            "assessment_headline":
                overall_assessment["headline"],

            "time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        }


        history = load_history()

        history.append(
            record
        )

        save_history(
            history
        )


        # -------------------------------------------------
        # SEND RESULT TO PAGE
        # -------------------------------------------------
        # -------------------------------------------------
        # CREATE SHORT PREVIEW FOR DISPLAY
        # -------------------------------------------------

        if input_type in ["article", "url"]:

            words = displayed_content.split()

            article_preview = " ".join(
                words[:25]
            )

            if len(words) > 25:
                article_preview += "..."

        else:

            article_preview = displayed_content

        print("PREVIEW:", article_preview)
        return render_template(

            "analysis.html",

            article=
                displayed_content,

            article_preview=
                article_preview,

            input_type=
                input_type,

            article_title=
                article_title,

            source_url=
                source_url,

            model_used=
                model_used,

            result=
                prediction,

            fake_probability=
                fake_probability,

            real_probability=
                real_probability,

            threshold=
                threshold,

            chunks_used=
                chunks_used,

            prediction_strength=prediction_strength,

            factcheck_results=factcheck_results,
            factcheck_error=factcheck_error,
            
            evidence_relationship=evidence_relationship,

            explanation=explanation,
            overall_assessment=overall_assessment,
            influence_highlights=
                influence_highlights,

            attribution_info=
                attribution_info,

            attribution_error=
                attribution_error,
        )


    except Exception as error:

        print(
            "Analysis error:",
            error
        )

        return render_template(

            "analysis.html",

            error=str(error)
        )

# ---------------------------------------------------------
# ABOUT
# ---------------------------------------------------------

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ---------------------------------------------------------
# ANALYSIS PAGE
# ---------------------------------------------------------

@app.route("/analysis")
def analysis():

    return render_template(
        "analysis.html"
    )


# ---------------------------------------------------------
# HISTORY
# ---------------------------------------------------------

@app.route("/history")
def history():

    history_data = load_history()

    return render_template(
        "history.html",
        history=history_data[::-1]
    )


# ---------------------------------------------------------
# START APPLICATION
# ---------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )