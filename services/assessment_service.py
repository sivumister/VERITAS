"""Keep the model prediction separate from external claim verification."""


def build_overall_assessment(prediction, prediction_strength, evidence_relationship,
                             factcheck_results=None):
    prediction = str(prediction).upper()
    relationship = evidence_relationship or {"status": "NO_MATCH"}
    status = relationship.get("status", "NO_MATCH")
    strength = (prediction_strength or {}).get("strength", "UNKNOWN").lower()
    model_label = "FAKE" if prediction == "FAKE" else "REAL"
    model_message = f"The classifier predicted {model_label} with {strength} prediction strength."
    # Do not use the first loosely related result's rating as the external verdict.
    external_rating = relationship.get("external_rating")
    common = {"external_rating": external_rating, "scope": "selected_claim",
              "ai_prediction": prediction, "is_verified_article": False}
    if status in ("AGREEMENT", "CONFLICT") and external_rating in ("TRUE", "FALSE"):
        action = "supported" if external_rating == "TRUE" else "contradicted"
        return dict(common, status="ALIGNED" if status == "AGREEMENT" else "CONFLICT",
                    label=f"Selected claim {action} by a published fact-check",
                    headline=("AI and Claim Review Align" if status == "AGREEMENT" else
                              "Published Claim Review and Model Disagree"),
                    message=f"{model_message} A published review of the same selected claim rated it {external_rating}.",
                    guidance="Read the cited review and its date. This finding applies to the selected claim; it does not verify the entire article.")
    if status == "NOT_CHECKED":
        return dict(common, status="AI_ONLY", label="Unverified — external evidence not checked",
                    headline="External Verification Pending", message=model_message,
                    guidance="Use Check external evidence to search for relevant claim reviews and web sources.")
    if status == "UNAVAILABLE":
        return dict(common, status="AI_ONLY", label="External verification unavailable",
                    headline="External Verification Unavailable", message=model_message,
                    guidance="The lookup could not be completed. The classifier result is unverified; retry the evidence search.")
    if status == "NO_CLEAR_VERDICT":
        return dict(common, status="INCONCLUSIVE", label="Inconclusive — review required",
                    headline="External Evidence Needs Review", message=f"{model_message} {relationship.get('message', '')}",
                    guidance="Retrieved pages and topic similarity do not establish truth. Review the selected claim against the cited passages.")
    label = ("Potentially fake — externally unverified" if prediction == "FAKE" else
             "Unverified — insufficient external evidence")
    return dict(common, status="AI_ONLY", label=label,
                headline="Insufficient External Evidence", message=model_message,
                guidance="No sufficiently specific external verdict was established. Missing evidence does not make the claim true or false.")
