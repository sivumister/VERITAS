from services.model_service import predict_headline


# =========================================================
# VERITAS HEADLINE MODEL DIAGNOSTIC TEST
# =========================================================


test_headlines = [

    # -----------------------------------------------------
    # ORDINARY NEWS-STYLE HEADLINES
    # -----------------------------------------------------

    {
        "category": "ORDINARY",
        "text":
            "Local school opens new science laboratory for students"
    },

    {
        "category": "ORDINARY",
        "text":
            "Heavy rainfall causes flooding in several communities"
    },

    {
        "category": "ORDINARY",
        "text":
            "University introduces new computer science programme"
    },

    {
        "category": "ORDINARY",
        "text":
            "Health officials launch vaccination awareness campaign"
    },

    {
        "category": "ORDINARY",
        "text":
            "Farmers prepare for the start of the rainy season"
    },


    # -----------------------------------------------------
    # SENSATIONAL / MISINFORMATION-STYLE HEADLINES
    # These are synthetic examples for model behaviour tests.
    # -----------------------------------------------------

    {
        "category": "SUSPICIOUS",
        "text":
            "Scientists confirm drinking bleach cures every known disease"
    },

    {
        "category": "SUSPICIOUS",
        "text":
            "Secret government device can read every citizen's thoughts"
    },

    {
        "category": "SUSPICIOUS",
        "text":
            "Doctors reveal one fruit that makes humans live for 300 years"
    },

    {
        "category": "SUSPICIOUS",
        "text":
            "Scientists discover that humans no longer need sleep"
    },

    {
        "category": "SUSPICIOUS",
        "text":
            "New phone application can predict the exact date of your death"
    }
]


print()
print("=" * 80)
print("VERITAS HEADLINE MODEL DIAGNOSTIC")
print("=" * 80)


fake_count = 0
real_count = 0


for number, item in enumerate(
    test_headlines,
    start=1
):

    result = predict_headline(
        item["text"]
    )

    prediction = result[
        "prediction"
    ]

    fake_probability = result[
        "fake_probability"
    ]

    real_probability = result[
        "real_probability"
    ]


    if prediction == "FAKE":
        fake_count += 1
    else:
        real_count += 1


    print()
    print("-" * 80)

    print(
        f"TEST {number} - "
        f"{item['category']}"
    )

    print()

    print(
        item["text"]
    )

    print()

    print(
        "Prediction:",
        prediction
    )

    print(
        "Fake probability:",
        f"{fake_probability}%"
    )

    print(
        "Real probability:",
        f"{real_probability}%"
    )

    print(
        "Margin:",
        f"{abs(fake_probability - real_probability):.2f}",
        "percentage points"
    )


print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    "Headlines classified FAKE:",
    fake_count
)

print(
    "Headlines classified REAL:",
    real_count
)

print()