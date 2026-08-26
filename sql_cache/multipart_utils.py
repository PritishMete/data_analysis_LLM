# sql_cache/multipart_utils.py
# ─────────────────────────────────────────────────────────────────────────────
# A minimal, self-contained multipart/form-data TEXT FIELD extractor.
#
# Why this exists instead of just calling `await request.form()`: empirically
# verified (see this feature's delivery notes) that calling Starlette's
# `request.form()` inside BaseHTTPMiddleware.dispatch() breaks the SAME
# request's downstream FastAPI File()/Form() parameter parsing entirely —
# both fields come back missing. `await request.body()` (raw bytes) does NOT
# have this problem; it replays correctly for the downstream handler
# regardless of content type. So this module extracts just the one text
# field the cache needs directly from those raw bytes, and NEVER calls
# `.form()` anywhere — the file part is never even decoded, let alone
# touched, so an uploaded file's bytes reach the real route completely
# unmodified.
#
# Deliberately narrow: this is NOT a general multipart parser (it doesn't
# handle nested multipart, doesn't decode file parts, doesn't validate
# encoding beyond utf-8-with-replacement). It does exactly one job — pull
# one named text field's value out of a raw multipart body — because that's
# all the SQL Cache needs to make its decision.
# ─────────────────────────────────────────────────────────────────────────────


def extract_boundary(content_type: str) -> bytes | None:
    """Pulls the boundary token out of a `Content-Type: multipart/form-data;
    boundary=...` header value. Handles both bare and quoted boundary forms."""
    if "boundary=" not in content_type:
        return None
    raw = content_type.split("boundary=", 1)[1].strip()
    raw = raw.split(";", 1)[0].strip()  # drop any trailing header parameters
    raw = raw.strip('"')
    return raw.encode("utf-8") if raw else None


def extract_text_field(raw_body: bytes, boundary: bytes, field_name: str) -> str | None:
    """Returns the decoded value of the first `name="{field_name}"` text part
    found in `raw_body`, or None if it isn't present. Only ever looks at the
    part whose Content-Disposition matches `field_name` — every other part
    (including any file part) is skipped without being decoded."""
    if not raw_body or not boundary:
        return None

    delimiter = b"--" + boundary
    marker = f'name="{field_name}"'.encode("utf-8")

    for part in raw_body.split(delimiter):
        if marker not in part:
            continue
        # A field part looks like:
        #   \r\nContent-Disposition: form-data; name="text"\r\n\r\nVALUE\r\n
        # Skip parts that ALSO declare a filename= — those are file parts
        # whose "name=" happens to match incidentally; a real text field
        # never has a filename parameter.
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        header_section = part[:header_end]
        if b"filename=" in header_section:
            continue

        value_section = part[header_end + 4:]
        value = value_section.rstrip(b"\r\n-")  # trailing CRLF + closing boundary dashes
        return value.decode("utf-8", errors="replace")

    return None
