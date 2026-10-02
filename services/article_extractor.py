from urllib.parse import urlparse
import socket
import ipaddress

import requests
from bs4 import BeautifulSoup


# =========================================================
# BASIC URL VALIDATION
# =========================================================

def validate_url(url):

    url = url.strip()

    parsed = urlparse(url)

    # Only allow normal web URLs
    if parsed.scheme not in ["http", "https"]:
        raise ValueError(
            "Only HTTP and HTTPS URLs are supported."
        )

    if not parsed.hostname:
        raise ValueError(
            "The URL does not contain a valid hostname."
        )

    # Prevent requests to localhost/private addresses
    try:

        addresses = socket.getaddrinfo(
            parsed.hostname,
            None
        )

        for address in addresses:

            ip_string = address[4][0]

            ip = ipaddress.ip_address(
                ip_string
            )

            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):

                raise ValueError(
                    "Private or local network URLs are not allowed."
                )

    except socket.gaierror:

        raise ValueError(
            "The website address could not be resolved."
        )

    return url


# =========================================================
# EXTRACT TITLE
# =========================================================

def extract_title(soup):

    # Try Open Graph title first
    og_title = soup.find(
        "meta",
        property="og:title"
    )

    if og_title and og_title.get("content"):

        return og_title[
            "content"
        ].strip()


    # Then try the main H1
    h1 = soup.find("h1")

    if h1:

        title = h1.get_text(
            " ",
            strip=True
        )

        if title:

            return title


    # Finally use HTML title
    if soup.title:

        return soup.title.get_text(
            " ",
            strip=True
        )


    return "Unknown Article"


# =========================================================
# EXTRACT ARTICLE TEXT
# =========================================================

def extract_article(url):

    url = validate_url(url)


    headers = {

        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/152.0 Safari/537.36"
        )
    }


    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=15,
            allow_redirects=True
        )

        response.raise_for_status()

    except requests.RequestException as error:

        raise ValueError(
            f"Could not retrieve the webpage: {error}"
        )


    # Make sure we actually received HTML
    content_type = response.headers.get(
        "Content-Type",
        ""
    ).lower()

    if "html" not in content_type:

        raise ValueError(
            "The URL does not appear to contain an HTML news article."
        )


    soup = BeautifulSoup(
        response.content,
        "html.parser"
    )


    # -----------------------------------------------------
    # REMOVE UNWANTED PAGE ELEMENTS
    # -----------------------------------------------------

    unwanted_tags = [

        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "form",
        "noscript"
    ]


    for tag_name in unwanted_tags:

        for tag in soup.find_all(
            tag_name
        ):

            tag.decompose()


    # -----------------------------------------------------
    # EXTRACT TITLE
    # -----------------------------------------------------

    title = extract_title(
        soup
    )


    # -----------------------------------------------------
    # LOOK FOR ARTICLE CONTAINER
    # -----------------------------------------------------

    article_container = soup.find(
        "article"
    )


    if article_container:

        paragraphs = (
            article_container
            .find_all("p")
        )

    else:

        paragraphs = soup.find_all(
            "p"
        )


    # -----------------------------------------------------
    # CLEAN PARAGRAPHS
    # -----------------------------------------------------

    cleaned_paragraphs = []

    seen = set()


    for paragraph in paragraphs:

        text = paragraph.get_text(
            " ",
            strip=True
        )

        # Very short paragraphs are often
        # menus, captions or unrelated text
        if len(text) < 40:

            continue


        # Remove duplicates
        if text in seen:

            continue


        seen.add(text)

        cleaned_paragraphs.append(
            text
        )


    article_text = "\n\n".join(
        cleaned_paragraphs
    )


    # -----------------------------------------------------
    # VALIDATE EXTRACTED ARTICLE
    # -----------------------------------------------------

    if len(article_text) < 200:

        raise ValueError(
            "VERITAS could not extract enough article "
            "text from this webpage."
        )


    return {

        "url":
            response.url,

        "title":
            title,

        "text":
            article_text,

        "word_count":
            len(
                article_text.split()
            )
    }