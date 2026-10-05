from flask import Flask, render_template, request, redirect, url_for, flash, session
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)

import smtplib
import ssl
import hashlib
import hmac

from email.message import EmailMessage

from itsdangerous import (
    URLSafeTimedSerializer,
    BadSignature,
    SignatureExpired
)

from werkzeug.security import generate_password_hash, check_password_hash
import json
import os
import secrets
import re

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from sqlalchemy.exc import IntegrityError

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


# ---------------------------------------------------------
# CREATE FLASK APPLICATION
# ---------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------
# DATABASE CONFIGURATION
# ---------------------------------------------------------

# Set SECRET_KEY in the environment for deployment. Locally, retain a
# randomly generated key so sessions survive restarts. Do not commit this file.
os.makedirs(app.instance_path, exist_ok=True)
secret_key = os.environ.get("SECRET_KEY")
if not secret_key:
    key_path = Path(app.instance_path) / ".secret_key"
    try:
        with key_path.open("x", encoding="utf-8") as key_file:
            key_file.write(secrets.token_hex(32))
        key_path.chmod(0o600)
    except FileExistsError:
        pass
    secret_key = key_path.read_text(encoding="utf-8").strip()
    if not secret_key:
        raise RuntimeError("Set SECRET_KEY or restore instance/.secret_key.")
app.config["SECRET_KEY"] = secret_key
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("COOKIE_SECURE") == "1"

database_url = os.environ.get(
    "DATABASE_URL",
    "sqlite:///veritas.db"
)

if database_url.startswith("postgresql://"):
    database_url = database_url.replace(
        "postgresql://",
        "postgresql+psycopg://",
        1
    )

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# ---------------------------------------------------------
# DATABASE
# ---------------------------------------------------------

db = SQLAlchemy(app)


# ---------------------------------------------------------
# LOGIN MANAGER
# ---------------------------------------------------------

login_manager = LoginManager()
login_manager.init_app(app)

login_manager.login_view = "login"
login_manager.login_message = "Please log in to continue."


# ---------------------------------------------------------
# DATABASE MODELS AND USER LOADER
# ---------------------------------------------------------

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(
        db.String(80),
        unique=True,
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password_hash = db.Column(
        db.String(255),
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    histories = db.relationship(
        "AnalysisHistory",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan"
    )


class AnalysisHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    input_type = db.Column(
        db.String(20),
        nullable=False
    )

    title = db.Column(
        db.String(500)
    )

    article_text = db.Column(
        db.Text
    )

    url = db.Column(
        db.Text
    )

    prediction = db.Column(
        db.String(20),
        nullable=False
    )

    confidence = db.Column(
        db.Float
    )

    prediction_strength = db.Column(
        db.String(20)
    )

    model_used = db.Column(
        db.String(100)
    )

    fact_check_result = db.Column(
        db.Text
    )

    explanation = db.Column(
        db.Text
    )

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )
    # Original history keys and the full analysis-page context are retained.
    result_data = db.Column(db.JSON, nullable=False)


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None

# ---------------------------------------------------------
# PASSWORD RESET
# ---------------------------------------------------------

PASSWORD_RESET_SALT = "veritas-password-reset-v1"


def get_reset_serializer():
    return URLSafeTimedSerializer(
        app.config["SECRET_KEY"]
    )


def get_password_reset_nonce(user):
    """
    Creates a value tied to the user's current password.

    After the password changes, old reset links automatically
    become invalid.
    """

    secret_key = app.config["SECRET_KEY"].encode("utf-8")

    return hmac.new(
        secret_key,
        user.password_hash.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def generate_password_reset_token(user):

    serializer = get_reset_serializer()

    return serializer.dumps(
        {
            "user_id": user.id,
            "nonce": get_password_reset_nonce(user)
        },
        salt=PASSWORD_RESET_SALT
    )


def verify_password_reset_token(token):

    serializer = get_reset_serializer()

    try:
        data = serializer.loads(
            token,
            salt=PASSWORD_RESET_SALT,
            max_age=1800  # 30 minutes
        )

    except (SignatureExpired, BadSignature):
        return None

    user = db.session.get(
        User,
        data.get("user_id")
    )

    if not user:
        return None

    expected_nonce = get_password_reset_nonce(user)

    if not hmac.compare_digest(
        data.get("nonce", ""),
        expected_nonce
    ):
        return None

    return user


def send_password_reset_email(user, reset_url):

    smtp_host = os.environ.get(
        "SMTP_HOST",
        "smtp.gmail.com"
    )

    smtp_port = int(
        os.environ.get(
            "SMTP_PORT",
            "587"
        )
    )

    smtp_username = os.environ.get(
        "SMTP_USERNAME"
    )

    smtp_password = os.environ.get(
        "SMTP_PASSWORD"
    )

    smtp_from = os.environ.get(
        "SMTP_FROM",
        smtp_username
    )


    if not smtp_username or not smtp_password:
        raise RuntimeError(
            "SMTP email settings are not configured."
        )


    message = EmailMessage()

    message["Subject"] = "Reset your VERITAS password"

    message["From"] = smtp_from

    message["To"] = user.email


    message.set_content(
        f"""
Hello {user.username},

A password reset was requested for your VERITAS account.

Use the link below to choose a new password:

{reset_url}

This link expires in 30 minutes.

If you did not request this password reset, you can ignore this email.

VERITAS
Fake News Detector
"""
    )


    context = ssl.create_default_context()


    with smtplib.SMTP(
        smtp_host,
        smtp_port,
        timeout=20
    ) as server:

        server.starttls(
            context=context
        )

        server.login(
            smtp_username,
            smtp_password
        )

        server.send_message(
            message
        )


@app.route(
    "/forgot-password",
    methods=["GET", "POST"]
)
def forgot_password():

    if current_user.is_authenticated:
        return redirect(url_for("home"))


    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()


        user = db.session.execute(
            db.select(User).filter_by(
                email=email
            )
        ).scalar_one_or_none()


        if user:

            token = generate_password_reset_token(
                user
            )

            reset_url = url_for(
                "reset_password",
                token=token,
                _external=True
            )


            try:

                send_password_reset_email(
                    user,
                    reset_url
                )

            except Exception:

                app.logger.exception(
                    "Could not send password reset email."
                )

                flash(
                    "The password reset email could not "
                    "be sent right now. Please try again later.",
                    "danger"
                )

                return render_template(
                    "forgot_password.html"
                )


        # Important:
        # Do not reveal whether the email exists.
        flash(
            "If an account exists with that email address, "
            "a password reset link has been sent.",
            "info"
        )

        return redirect(
            url_for("login")
        )


    return render_template(
        "forgot_password.html"
    )


@app.route(
    "/reset-password/<token>",
    methods=["GET", "POST"]
)
def reset_password(token):

    if current_user.is_authenticated:
        return redirect(url_for("home"))

    # Check whether the reset link is valid
    user = verify_password_reset_token(token)

    if not user:

        flash(
            "This password reset link is invalid or has expired.",
            "danger"
        )

        return redirect(
            url_for("forgot_password")
        )


    if request.method == "POST":

        password = request.form.get(
            "password",
            ""
        )

        confirmation = request.form.get(
            "confirm_password",
            ""
        )


        # Validate password
        if not 8 <= len(password) <= 128:

            flash(
                "Password must contain between 8 and 128 characters.",
                "danger"
            )


        elif password != confirmation:

            flash(
                "Passwords do not match.",
                "danger"
            )


        else:

            # Replace old password with new hashed password
            user.password_hash = generate_password_hash(
                password
            )

            try:

                db.session.commit()

            except Exception:

                db.session.rollback()

                app.logger.exception(
                    "Password reset failed."
                )

                flash(
                    "Your password could not be changed. "
                    "Please try again.",
                    "danger"
                )

            else:

                flash(
                    "Password changed successfully. "
                    "You can now log in with your new password.",
                    "success"
                )

                return redirect(
                    url_for("login")
                )


    return render_template(
        "reset_password.html"
    )
# All models must exist before creating tables. This also runs under a WSGI
# server that imports app.py. create_all creates tables; it does not migrate
# an existing schema. Use migrations if this database already has older tables.
with app.app_context():
    db.create_all()

# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.route("/")
@login_required
def home():

    return render_template(
        "index.html"
    )


# ---------------------------------------------------------
# ANALYSE NEWS
# ---------------------------------------------------------

@app.route("/analyze", methods=["POST"])
@login_required
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
        page_context = dict(

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

        entry = AnalysisHistory(
            user_id=current_user.id,
            input_type=input_type,
            title=article_title,
            article_text=displayed_content,
            url=source_url,
            prediction=prediction,
            confidence=(fake_probability if prediction == "FAKE" else real_probability),
            prediction_strength=prediction_strength["strength"],
            model_used=model_used,
            fact_check_result=json.dumps(factcheck_results, ensure_ascii=False),
            explanation=json.dumps(explanation, ensure_ascii=False),
            result_data={"history": record, "page_context": page_context}
        )
        try:
            db.session.add(entry)
            db.session.commit()
        except Exception:
            db.session.rollback()
            app.logger.exception("Could not save analysis history")
            flash("Analysis completed, but its history could not be saved.", "warning")

        return render_template("analysis.html", **page_context)


    except Exception as error:

        db.session.rollback()
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
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirmation = request.form.get("confirm_password")

        if not 3 <= len(username) <= 80:
            flash("Username must contain between 3 and 80 characters.", "danger")
        elif len(email) > 120 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
            flash("Please enter a valid email address.", "danger")
        elif not 8 <= len(password) <= 128:
            flash("Password must contain between 8 and 128 characters.", "danger")
        elif confirmation is not None and password != confirmation:
            flash("Passwords do not match.", "danger")
        elif db.session.execute(db.select(User).filter_by(username=username)).scalar_one_or_none():
            flash("Username already exists.", "danger")
        elif db.session.execute(db.select(User).filter_by(email=email)).scalar_one_or_none():
            flash("Email already registered.", "danger")
        else:
            user = User(username=username, email=email,
                        password_hash=generate_password_hash(password))
            try:
                db.session.add(user)
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                flash("That username or email is already registered.", "danger")
            else:
                flash("Account created successfully. You can now log in.", "success")
                return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "")

        # Allow login using either username OR email
        user = db.session.execute(
            db.select(User).where(
                (User.username == identifier) |
                (User.email == identifier.lower())
            )
        ).scalar_one_or_none()

        if (
            user
            and len(password) <= 128
            and check_password_hash(user.password_hash, password)
        ):
            session.clear()
            login_user(user)

            flash(
                f"Welcome back, {user.username}!",
                "success"
            )

            return redirect(url_for("home"))

        flash("Invalid username/email or password.", "danger")

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/about")
@login_required
def about():

    return render_template(
        "about.html"
    )


# ---------------------------------------------------------
# ANALYSIS PAGE
# ---------------------------------------------------------

@app.route("/analysis")
@login_required
def analysis():

    return render_template(
        "analysis.html"
    )


# ---------------------------------------------------------
# HISTORY
# ---------------------------------------------------------

@app.route("/history")
@login_required
def history():
    entries = db.session.execute(
        db.select(AnalysisHistory)
        .filter_by(user_id=current_user.id)
        .order_by(AnalysisHistory.created_at.desc(), AnalysisHistory.id.desc())
    ).scalars().all()
    # Preserve dictionary keys used by the existing history.html template.
    history_data = []
    for entry in entries:
        record = dict(entry.result_data["history"])
        record["id"] = entry.id
        record["article"] = entry.article_text
        words = (entry.article_text or "").split()
        record["article_preview"] = " ".join(words[:25]) + ("..." if len(words) > 25 else "")
        history_data.append(record)
    return render_template("history.html", history=history_data)


@app.route("/history/<int:id>")
@login_required
def history_detail(id):
    entry = db.first_or_404(
        db.select(AnalysisHistory).filter_by(id=id, user_id=current_user.id)
    )
    # Reuse the existing result page; never rerun the model to view history.
    return render_template("analysis.html", **entry.result_data["page_context"])


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
