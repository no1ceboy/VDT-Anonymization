"""DOCX text extraction and source-span replacement with run formatting retained."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO


@dataclass
class ParagraphPart:
    paragraph: object
    text: str
    start: int
    end: int


def _walk_paragraphs(container_element, parent, paragraph_class, table_class, qn, seen):
    for child in container_element.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = paragraph_class(child, parent)
            # Keep the XML proxy itself in `seen`; wrappers can be recreated
            # when python-docx traverses merged cells or shared headers.
            key = paragraph._p
            if key not in seen:
                seen.add(key)
                yield paragraph
        elif child.tag == qn("w:tbl"):
            table = table_class(child, parent)
            for row in table.rows:
                for cell in row.cells:
                    yield from _walk_paragraphs(
                        cell._tc, cell, paragraph_class, table_class, qn, seen
                    )


def _all_paragraphs(document):
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    seen = set()
    yield from _walk_paragraphs(
        document.element.body, document._body, Paragraph, Table, qn, seen
    )
    for section in document.sections:
        containers = (
            section.header,
            section.first_page_header,
            section.even_page_header,
            section.footer,
            section.first_page_footer,
            section.even_page_footer,
        )
        for container in containers:
            yield from _walk_paragraphs(
                container._element, container, Paragraph, Table, qn, seen
            )


def load_docx(payload: bytes):
    try:
        from docx import Document
    except ImportError as error:
        raise RuntimeError("DOCX support requires the API extra: pip install '.[api]'") from error
    try:
        return Document(BytesIO(payload))
    except Exception as error:
        raise ValueError("The uploaded file is not a readable DOCX document") from error


def extract_docx_text(document) -> tuple[str, list[ParagraphPart]]:
    parts = []
    cursor = 0
    for paragraph in _all_paragraphs(document):
        value = paragraph.text
        start = cursor
        end = start + len(value)
        parts.append(ParagraphPart(paragraph, value, start, end))
        cursor = end + 1
    text = "\n".join(part.text for part in parts)
    if not text.strip():
        raise ValueError("The DOCX document contains no extractable text")
    return text, parts


def _replace_paragraph_runs(part: ParagraphPart, local_replacements: list[dict]) -> None:
    paragraph = part.paragraph
    runs = list(paragraph.runs)
    original_run_text = "".join(run.text for run in runs)
    if original_run_text != part.text:
        raise ValueError(
            "A paragraph contains text outside ordinary DOCX runs (for example, "
            "a hyperlink or complex field); refusing to produce a partial document"
        )
    if not local_replacements:
        return
    if not runs:
        raise ValueError("A target paragraph has no editable DOCX text runs")

    ordered = sorted(local_replacements, key=lambda item: (item["start"], item["end"]))
    previous_end = 0
    for item in ordered:
        start, end = int(item["start"]), int(item["end"])
        if not previous_end <= start < end <= len(part.text):
            raise ValueError("Invalid or overlapping replacement spans in DOCX paragraph")
        expected = item.get("original")
        if expected is not None and part.text[start:end] != expected:
            raise ValueError("DOCX paragraph source text does not match the planned span")
        previous_end = end

    run_start = 0
    replacement_index = 0
    for run in runs:
        value = run.text
        run_end = run_start + len(value)
        output = []
        for offset, char in enumerate(value):
            absolute = run_start + offset
            while (replacement_index < len(ordered)
                   and int(ordered[replacement_index]["end"]) <= absolute):
                replacement_index += 1
            if replacement_index < len(ordered):
                item = ordered[replacement_index]
                start, end = int(item["start"]), int(item["end"])
                if start <= absolute < end:
                    if absolute == start:
                        output.append(str(item["replacement"]))
                    continue
            output.append(char)
        run.text = "".join(output)
        run_start = run_end


def apply_docx_replacements(document, text: str, parts: list[ParagraphPart],
                            replacements: list[dict]) -> bytes:
    joined_text = "\n".join(part.text for part in parts)
    if joined_text != text:
        raise ValueError("DOCX text changed after entity linking; refusing to alter spans")

    replacements_by_part: dict[int, list[dict]] = {index: [] for index in range(len(parts))}
    for replacement in replacements:
        start, end = int(replacement["start"]), int(replacement["end"])
        matched = False
        for index, part in enumerate(parts):
            if part.start <= start < end <= part.end:
                replacements_by_part[index].append({
                    **replacement,
                    "start": start - part.start,
                    "end": end - part.start,
                })
                matched = True
                break
        if not matched:
            raise ValueError(
                f"Replacement span [{start}, {end}) crosses a paragraph boundary"
            )

    for index, part in enumerate(parts):
        _replace_paragraph_runs(part, replacements_by_part[index])

    output = BytesIO()
    document.save(output)
    return output.getvalue()
