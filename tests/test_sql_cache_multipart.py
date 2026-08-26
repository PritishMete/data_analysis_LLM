# tests/test_sql_cache_multipart.py
from sql_cache.multipart_utils import extract_boundary, extract_text_field


def _build_multipart_body(boundary: str, fields: dict, file_field: tuple[str, str, bytes] | None = None) -> bytes:
    """Builds a realistic multipart/form-data body for testing — mirrors
    what an actual browser/httpx client produces."""
    b = boundary.encode()
    parts = []
    for name, value in fields.items():
        parts.append(
            b"--" + b + b"\r\n"
            b'Content-Disposition: form-data; name="' + name.encode() + b'"\r\n\r\n'
            + value.encode() + b"\r\n"
        )
    if file_field is not None:
        field_name, filename, content = file_field
        parts.append(
            b"--" + b + b"\r\n"
            b'Content-Disposition: form-data; name="' + field_name.encode() + b'"; filename="' + filename.encode() + b'"\r\n'
            b"Content-Type: text/csv\r\n\r\n"
            + content + b"\r\n"
        )
    parts.append(b"--" + b + b"--\r\n")
    return b"".join(parts)


def test_extract_boundary_from_plain_header():
    assert extract_boundary("multipart/form-data; boundary=abc123") == b"abc123"


def test_extract_boundary_handles_quoted_form():
    assert extract_boundary('multipart/form-data; boundary="abc123"') == b"abc123"


def test_extract_boundary_returns_none_when_absent():
    assert extract_boundary("multipart/form-data") is None


def test_extract_text_field_finds_simple_value():
    body = _build_multipart_body("BOUNDARY", {"text": "total revenue by region"})
    assert extract_text_field(body, b"BOUNDARY", "text") == "total revenue by region"


def test_extract_text_field_alongside_a_file_part():
    body = _build_multipart_body(
        "BOUNDARY",
        {"text": "total revenue by region"},
        file_field=("file", "data.csv", b"a,b\n1,2\n"),
    )
    assert extract_text_field(body, b"BOUNDARY", "text") == "total revenue by region"


def test_extract_text_field_ignores_file_part_with_matching_name():
    # A file field is never mistaken for a text field even if its "name="
    # happens to collide, because file parts always carry filename=.
    body = _build_multipart_body("BOUNDARY", {}, file_field=("text", "not_a_text_field.csv", b"binary,data"))
    assert extract_text_field(body, b"BOUNDARY", "text") is None


def test_extract_text_field_returns_none_when_field_missing():
    body = _build_multipart_body("BOUNDARY", {"other_field": "value"})
    assert extract_text_field(body, b"BOUNDARY", "text") is None


def test_extract_text_field_returns_none_for_empty_body():
    assert extract_text_field(b"", b"BOUNDARY", "text") is None


def test_extract_multiple_fields_independently():
    body = _build_multipart_body(
        "BOUNDARY",
        {"text": "total revenue by region", "dataset_id": "abc-123", "organization_id": "org_1"},
    )
    assert extract_text_field(body, b"BOUNDARY", "text") == "total revenue by region"
    assert extract_text_field(body, b"BOUNDARY", "dataset_id") == "abc-123"
    assert extract_text_field(body, b"BOUNDARY", "organization_id") == "org_1"
