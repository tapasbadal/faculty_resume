
import os
import re
import json
from datetime import date
from pathlib import Path

import pandas as pd
import fitz  # PyMuPDF
import streamlit as st

# Optional AI layer
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# =========================================================
# Configuration
# =========================================================

DEFAULT_FOLDER = ""
OUTPUT_FOLDER = "/tmp"



# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf_text(pdf_path):
    """Extract text from a PDF. Returns page-labelled text."""
    try:
        doc = fitz.open(pdf_path)
        pages = []

        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text")
            pages.append(f"\n--- PAGE {page_no} ---\n{text}")

        doc.close()
        return "\n".join(pages)

    except Exception as e:
        return f"ERROR: {e}"


def clean_text(text):
    text = text.replace("\x00", " ")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2012", "-").replace("\u2212", "-")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


# =========================================================
# NAME
# =========================================================

def extract_name_basic(text, filename):
    lines = [x.strip() for x in text.splitlines() if x.strip()]

    for line in lines[:40]:
        m = re.match(r"^(?:name|candidate name)\s*[:\-]\s*(.+)$", line, re.I)
        if m:
            return m.group(1).strip()

    for line in lines[:20]:
        if (
            2 <= len(line.split()) <= 7
            and len(line) <= 80
            and not re.search(
                r"(curriculum vitae|resume|cv|profile|objective|email|phone|mobile)",
                line,
                re.I,
            )
            and not re.search(r"[@:/\\]", line)
        ):
            return line

    return Path(filename).stem


# =========================================================
# EDUCATION SECTION
# =========================================================

EDUCATION_HEADINGS = [
    r"^\s*education\s*$",
    r"^\s*educational qualifications?\s*$",
    r"^\s*academic qualifications?\s*$",
    r"^\s*academic background\s*$",
    r"^\s*qualifications?\s*$",
    r"^\s*educational background\s*$",
]

STOP_HEADINGS = [
    r"^\s*experience\s*$",
    r"^\s*work experience\s*$",
    r"^\s*work history\s*$",
    r"^\s*professional experience\s*$",
    r"^\s*employment history\s*$",
    r"^\s*research\s*$",
    r"^\s*selected publications\s*$",
    r"^\s*publications?\s*$",
    r"^\s*projects?\s*$",
    r"^\s*consultancy.*$",
    r"^\s*patents?\s*$",
    r"^\s*skills\s*$",
    r"^\s*personal details\s*$",
    r"^\s*references?\s*$",
    r"^\s*awards?.*$",
    r"^\s*achievements?.*$",
]


def is_heading(line, patterns):
    return any(re.match(p, line, re.I) for p in patterns)


def extract_education_section(text):
    lines = text.splitlines()

    start = None

    for i, line in enumerate(lines):
        if is_heading(line.strip(), EDUCATION_HEADINGS):
            start = i
            break

    if start is None:
        for i, line in enumerate(lines):
            if re.search(
                r"\b(education|educational qualifications|academic qualifications)\b",
                line,
                re.I,
            ):
                start = i
                break

    if start is None:
        return ""

    end = len(lines)

    for i in range(start + 1, len(lines)):
        if is_heading(lines[i].strip(), STOP_HEADINGS):
            end = i
            break

    section = "\n".join(lines[start:end]).strip()

    if len(section) > 25000:
        section = section[:25000]

    return section


# =========================================================
# DEGREE PATTERNS
# =========================================================

GRADUATION_PATTERNS = [
    (r"\bB\.?\s*Tech(?:nology)?\b", "B.Tech"),
    (r"\bBTech\b", "B.Tech"),
    (r"\bBachelor\s+of\s+Technology\b", "B.Tech"),
    (r"\bB\.?\s*E\.?\b", "B.E."),
    (r"\bB\.?\s*Eng(?:ineering)?\b", "B.E."),
    (r"\bBachelor\s+of\s+Engineering\b", "B.E."),
    (r"\bB\.?\s*Sc(?:ience)?\b", "B.Sc"),
    (r"\bBachelor\s+of\s+Science\b", "B.Sc"),
    (r"\bB\.?\s*C\.?\s*A\.?\b", "BCA"),
    (r"\bBachelor\s+of\s+Computer\s+Applications\b", "BCA"),
    (r"\bB\.?\s*Com(?:merce)?\b", "B.Com"),
    (r"\bBachelor\s+of\s+Commerce\b", "B.Com"),
]

PG_PATTERNS = [
    (r"\bM\.?\s*Tech(?:nology)?\b", "M.Tech"),
    (r"\bMTech\b", "M.Tech"),
    (r"\bMaster\s+of\s+Technology\b", "M.Tech"),
    (r"\bM\.?\s*E\.?\b", "M.E."),
    (r"\bMaster\s+of\s+Engineering\b", "M.E."),
    (r"\bM\.?\s*Sc(?:ience)?\b", "M.Sc"),
    (r"\bMaster\s+of\s+Science\b", "M.Sc"),
    (r"\bM\.?\s*C\.?\s*A\.?\b", "MCA"),
    (r"\bMaster\s+of\s+Computer\s+Applications\b", "MCA"),
    (r"\bM\.?\s*B\.?\s*A\.?\b", "MBA"),
    (r"\bMaster\s+of\s+Business\s+Administration\b", "MBA"),
    (r"\bM\.?\s*Com(?:merce)?\b", "M.Com"),
    (r"\bMaster\s+of\s+Commerce\b", "M.Com"),
]

PHD_PATTERNS = [
    (r"\bPh\.?\s*D\.?\b", "Ph.D."),
    (r"\bDoctor\s+of\s+Philosophy\b", "Ph.D."),
]


def find_degree(text, patterns):
    for pattern, name in patterns:
        if re.search(pattern, text, re.I):
            return name
    return ""


# =========================================================
# DATE PARSING
# =========================================================

YEAR_RE = r"(?:19\d{2}|20\d{2})"

MONTH_RE = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)"
)

RANGE_PATTERNS = [
    rf"\b{MONTH_RE}\s+({YEAR_RE})\s*-\s*{MONTH_RE}\s+({YEAR_RE})\b",
    rf"\b({YEAR_RE})\s*-\s*({YEAR_RE})\b",
    rf"\b({YEAR_RE})\s+(?:to|TO)\s+({YEAR_RE})\b",
]


def extract_years(segment):
    return [
        int(y)
        for y in re.findall(rf"\b({YEAR_RE})\b", segment)
        if 1950 <= int(y) <= 2035
    ]


def completion_year(segment):
    """
    Completion year:
      - date range -> ending year
      - single year -> that year
    """
    for pattern in RANGE_PATTERNS:
        matches = list(re.finditer(pattern, segment, re.I))
        if matches:
            m = matches[-1]
            groups = m.groups()
            if len(groups) >= 2:
                try:
                    y = int(groups[-1])
                    if 1950 <= y <= 2035:
                        return y
                except Exception:
                    pass

    years = extract_years(segment)
    return years[-1] if years else None


# =========================================================
# EDUCATION RECORDS
# =========================================================

def education_records_rule_based(section):
    lines = [x.strip() for x in section.splitlines() if x.strip()]
    records = []

    all_patterns = [
        ("Graduation", GRADUATION_PATTERNS),
        ("Post Graduation", PG_PATTERNS),
        ("PhD", PHD_PATTERNS),
    ]

    for i, line in enumerate(lines):
        for category, patterns in all_patterns:
            if find_degree(line, patterns):
                windows = [
                    " ".join(lines[max(0, i - 1): min(len(lines), i + 3)]),
                    " ".join(lines[max(0, i): min(len(lines), i + 5)]),
                    " ".join(lines[max(0, i - 2): min(len(lines), i + 5)]),
                ]

                for segment in windows:
                    year = completion_year(segment)
                    if year:
                        degree = find_degree(segment, patterns)
                        records.append({
                            "category": category,
                            "degree": degree,
                            "year": year,
                            "evidence": segment,
                            "method": "Rule-based",
                        })
                        break

    # Deduplicate
    unique = []
    seen = set()

    for r in records:
        key = (r["category"], r["degree"], r["year"], r["evidence"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


# =========================================================
# EXPERIENCE EXTRACTION - RULE BASED
# =========================================================

EXPERIENCE_HEADINGS = [
    r"^\s*experience\s*$",
    r"^\s*work experience\s*$",
    r"^\s*work history\s*$",
    r"^\s*professional experience\s*$",
    r"^\s*employment history\s*$",
]

EXPERIENCE_STOP_HEADINGS = [
    r"^\s*education\s*$",
    r"^\s*research\s*$",
    r"^\s*selected publications\s*$",
    r"^\s*publications?\s*$",
    r"^\s*projects?\s*$",
    r"^\s*consultancy.*$",
    r"^\s*patents?\s*$",
    r"^\s*skills\s*$",
    r"^\s*personal details\s*$",
    r"^\s*references?\s*$",
]


def extract_experience_section(text):
    lines = text.splitlines()

    start = None
    for i, line in enumerate(lines):
        if is_heading(line.strip(), EXPERIENCE_HEADINGS):
            start = i
            break

    if start is None:
        return ""

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if is_heading(lines[i].strip(), EXPERIENCE_STOP_HEADINGS):
            end = i
            break

    return "\n".join(lines[start:end]).strip()


def parse_date_token(month, year):
    month_map = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    return date(int(year), month_map[month.lower()], 1)


def parse_experience_dates(segment):
    """
    Returns (start_date, end_date, evidence) where possible.
    Handles:
      Jan 2018 - Dec 2020
      01-08-2025 - Till date
      July 2017 – December 2018
      Sep 2003 - Dec 2011
    """

    s = segment.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)

    # DD-MM-YYYY / DD/MM/YYYY
    full_dates = re.findall(
        r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", s
    )

    if len(full_dates) >= 2:
        d1, m1, y1 = map(int, full_dates[0])
        d2, m2, y2 = map(int, full_dates[1])
        try:
            return date(y1, m1, d1), date(y2, m2, d2), segment
        except ValueError:
            pass

    # Month Year - Month Year
    month_years = re.findall(
        rf"({MONTH_RE})\s+({YEAR_RE})", s, re.I
    )

    if len(month_years) >= 2:
        m1, y1 = month_years[0]
        m2, y2 = month_years[1]
        try:
            start = parse_date_token(m1, y1)
            end = parse_date_token(m2, y2)
            return start, end, segment
        except Exception:
            pass

    # Year - Year
    years = re.findall(rf"\b({YEAR_RE})\b", s)
    if len(years) >= 2:
        # Only use this fallback when the years are reasonably close.
        y1, y2 = int(years[0]), int(years[1])
        if 1950 <= y1 <= 2035 and 1950 <= y2 <= 2035 and y1 <= y2:
            return date(y1, 1, 1), date(y2, 12, 1), segment

    # Start date/year + Till date / Present
    if re.search(r"\b(till\s+date|till\s+now|present|current)\b", s, re.I):
        m = re.search(rf"({MONTH_RE})\s+({YEAR_RE})", s, re.I)
        if m:
            start = parse_date_token(m.group(1), m.group(2))
            return start, date.today(), segment

        m = re.search(rf"\b({YEAR_RE})\b", s)
        if m:
            return date(int(m.group(1)), 1, 1), date.today(), segment

    return None, None, segment


def months_between(start, end):
    if not start or not end or end < start:
        return 0

    return max(
        1,
        (end.year - start.year) * 12 + (end.month - start.month) + 1
    )


def merge_intervals(intervals):
    """
    Merge overlapping experience intervals so that concurrent roles
    are not double-counted.
    """
    valid = [
        (s, e)
        for s, e in intervals
        if s and e and e >= s
    ]

    if not valid:
        return []

    valid.sort(key=lambda x: x[0])
    merged = [list(valid[0])]

    for start, end in valid[1:]:
        prev_start, prev_end = merged[-1]

        if start <= prev_end:
            if end > prev_end:
                merged[-1][1] = end
        else:
            merged.append([start, end])

    return [(x[0], x[1]) for x in merged]


def extract_experience_rule_based(text):
    section = extract_experience_section(text)

    if not section:
        return {
            "experience_records": [],
            "total_months": None,
            "total_years": None,
            "experience_method": "Not Found",
        }

    lines = [x.strip() for x in section.splitlines() if x.strip()]

    intervals = []
    records = []

    # Build windows around lines containing dates.
    for i, line in enumerate(lines):
        window = " ".join(lines[max(0, i - 1): min(len(lines), i + 4)])

        if not re.search(
            rf"({YEAR_RE}).*({YEAR_RE})|"
            rf"{MONTH_RE}.*({YEAR_RE})|"
            r"\b(till date|present|current)\b",
            window,
            re.I,
        ):
            continue

        start, end, evidence = parse_experience_dates(window)

        if start and end:
            intervals.append((start, end))

            records.append({
                "start": start.isoformat(),
                "end": end.isoformat(),
                "duration_months": months_between(start, end),
                "evidence": evidence,
                "method": "Rule-based",
            })

    merged = merge_intervals(intervals)

    total_months = sum(months_between(s, e) for s, e in merged)

    return {
        "experience_records": records,
        "merged_intervals": merged,
        "total_months": total_months,
        "total_years": round(total_months / 12, 1) if total_months else None,
        "experience_method": "Rule-based",
    }


# =========================================================
# SCI / SCIE PUBLICATION COUNT
# =========================================================

SCI_HEADINGS = [
    r"journal publications?\s*\(.*sci.*scie.*\)",
    r"journal publications?\s*\(.*scie.*sci.*\)",
    r"journal publications?\s*\(.*sci.*\)",
    r"journal publications?\s*\(.*scie.*\)",
    r"scie?\s*/\s*scie",
]


def extract_sci_section(text):
    lines = text.splitlines()

    start = None

    for i, line in enumerate(lines):
        if re.search(r"\bSCI\b|\bSCIE\b", line, re.I) and re.search(
            r"journal|publication|paper", line, re.I
        ):
            start = i
            break

    if start is None:
        return ""

    # Stop at the next major publication category/section.
    end = len(lines)

    for i in range(start + 1, len(lines)):
        line = lines[i].strip()

        if re.search(
            r"^\s*(peer[- ]reviewed|conference|book|patent|"
            r"consultancy|research|teaching|services|article|blog)\b",
            line,
            re.I,
        ):
            end = i
            break

    return "\n".join(lines[start:end]).strip()


def count_numbered_entries(section):
    """
    Count numbered publication entries, e.g.
    1. ...
    2. ...
    3. ...
    """
    if not section:
        return 0

    matches = re.findall(
        r"(?m)^\s*(\d+)\s*[.)]\s+",
        section
    )

    if not matches:
        # Some PDF extraction loses line boundaries.
        matches = re.findall(
            r"(?<!\d)(\d{1,3})\s*[.)]\s+(?=[A-Z][A-Za-z])",
            section
        )

    nums = []
    for n in matches:
        try:
            nums.append(int(n))
        except Exception:
            pass

    return len(sorted(set(nums)))


def extract_sci_scie_rule_based(text):
    section = extract_sci_section(text)

    if not section:
        return {
            "sci_scie_count": None,
            "sci_scie_evidence": "",
            "sci_scie_method": "Not Found",
        }

    count = count_numbered_entries(section)

    return {
        "sci_scie_count": count,
        "sci_scie_evidence": section[:12000],
        "sci_scie_method": "Rule-based",
    }


# =========================================================
# AI EXTRACTION
# =========================================================

AI_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_name": {"type": "string"},
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "Graduation",
                            "Post Graduation",
                            "PhD",
                            "Other"
                        ],
                    },
                    "degree": {"type": "string"},
                    "field": {"type": "string"},
                    "institution": {"type": "string"},
                    "start_date": {"type": "string"},
                    "completion_date": {"type": "string"},
                    "completion_year": {"type": "string"},
                    "evidence": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["High", "Medium", "Low"]
                    },
                },
                "required": [
                    "category",
                    "degree",
                    "field",
                    "institution",
                    "start_date",
                    "completion_date",
                    "completion_year",
                    "evidence",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
        "experience": {
            "type": "object",
            "properties": {
                "total_experience_years": {"type": "string"},
                "academic_experience_years": {"type": "string"},
                "industry_experience_years": {"type": "string"},
                "records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "position": {"type": "string"},
                            "organization": {"type": "string"},
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"},
                            "evidence": {"type": "string"},
                        },
                        "required": [
                            "position",
                            "organization",
                            "start_date",
                            "end_date",
                            "evidence",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "total_experience_years",
                "academic_experience_years",
                "industry_experience_years",
                "records",
            ],
            "additionalProperties": False,
        },
        "publications": {
            "type": "object",
            "properties": {
                "sci_scie_count": {"type": "string"},
                "sci_count": {"type": "string"},
                "scie_count": {"type": "string"},
                "evidence": {"type": "string"},
                "confidence": {
                    "type": "string",
                    "enum": ["High", "Medium", "Low"]
                },
            },
            "required": [
                "sci_scie_count",
                "sci_count",
                "scie_count",
                "evidence",
                "confidence",
            ],
            "additionalProperties": False,
        },
    },
    "required": [
        "candidate_name",
        "education",
        "experience",
        "publications",
    ],
    "additionalProperties": False,
}


def run_ai_extraction(text, model_name="gpt-5.6"):
    """
    AI acts as a second-pass verifier/extractor.

    Important:
    - It is instructed NOT to guess.
    - Every important value must have evidence from the CV.
    - For date ranges, completion_year must be the END year.
    - SCI/SCIE count must be based only on what the CV explicitly supports.
    """
    if not OPENAI_AVAILABLE:
        raise RuntimeError(
            "OpenAI package is not installed. Run: pip install openai"
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in Windows before using AI."
        )

    client = OpenAI(api_key=api_key)

    system_prompt = """
You are an expert CV/resume information extraction system.

Extract information ONLY from the supplied CV text.

CRITICAL RULES:
1. NEVER guess a missing year, degree, institution, experience period, or publication count.
2. Education dates must come from the education/academic qualification evidence.
3. A date in work experience, publication references, project descriptions, course history,
   funding, or unrelated text must NOT be used as an education completion year.
4. If education says "Mar 2008 - Sep 2013 PhD", completion_year is 2013.
5. If education says "Jul 2001 - Apr 2003 MBA", completion_year is 2003.
6. If a qualification gives only one year, use that year as completion_year.
7. For experience, preserve the actual position, organization, start and end dates.
8. "Till date", "present", or "current" means the experience continues to today.
9. Do not double-count overlapping employment periods when estimating total experience.
10. For SCI/SCIE publications, count only papers that the CV itself explicitly places
    under a heading/category indicating SCI, SCIE, SCI/SCIE, or equivalent.
11. Do not infer that every journal paper is SCI/SCIE merely because it is a journal.
12. Evidence must be copied/paraphrased closely enough to allow human verification.
13. If information is absent or ambiguous, return an empty string and Low confidence.
14. Return valid JSON matching the supplied schema.
"""

    user_prompt = f"""
Analyse this faculty CV.

Required outputs:
- candidate name
- graduation qualification and completion year
- post-graduation qualification and completion year
- PhD and completion year
- education evidence for every extracted qualification
- work/academic/industry experience records
- total experience estimate without double-counting overlaps
- number of SCI/SCIE papers explicitly supported by the CV
- separate SCI and SCIE counts only when the CV explicitly distinguishes them

CV TEXT:
{text[:120000]}
"""

    response = client.responses.create(
        model=model_name,
        instructions=system_prompt,
        input=user_prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "faculty_cv_extraction",
                "strict": True,
                "schema": AI_SCHEMA,
            }
        },
    )

    return json.loads(response.output_text)


# =========================================================
# MERGE AI + RULE BASED RESULTS
# =========================================================

def normalize_ai_year(value):
    if value is None:
        return ""

    s = str(value).strip()
    m = re.search(r"\b(19\d{2}|20\d{2})\b", s)

    if not m:
        return ""

    y = int(m.group(1))
    return y if 1950 <= y <= 2035 else ""


def choose_education_record(ai_records, category):
    matches = [
        r for r in ai_records
        if r.get("category") == category
    ]

    if not matches:
        return None

    # Prefer high confidence, then records containing a completion year.
    matches.sort(
        key=lambda x: (
            {"High": 3, "Medium": 2, "Low": 1}.get(
                x.get("confidence", "Low"), 0
            ),
            bool(normalize_ai_year(x.get("completion_year"))),
        ),
        reverse=True,
    )

    return matches[0]


def merge_results(rule_result, ai_result):
    result = dict(rule_result)

    if not ai_result:
        return result

    # AI candidate name
    ai_name = ai_result.get("candidate_name", "").strip()
    if ai_name:
        result["Candidate Name"] = ai_name

    edu = ai_result.get("education", [])

    mapping = {
        "Graduation": (
            "Graduation Degree",
            "Graduation Year",
            "Graduation Evidence",
            "Graduation Confidence",
        ),
        "Post Graduation": (
            "Post Graduation Degree",
            "Post Graduation Year",
            "Post Graduation Evidence",
            "PG Confidence",
        ),
        "PhD": (
            "PhD",
            "PhD Year",
            "PhD Evidence",
            "PhD Confidence",
        ),
    }

    for category, cols in mapping.items():
        rec = choose_education_record(edu, category)

        if not rec:
            continue

        degree_col, year_col, evidence_col, confidence_col = cols

        year = normalize_ai_year(rec.get("completion_year", ""))
        evidence = rec.get("evidence", "").strip()
        degree = rec.get("degree", "").strip()
        confidence = rec.get("confidence", "Low")

        # AI only replaces a rule result when it provides evidence.
        if year or evidence or degree:
            if degree:
                result[degree_col] = degree
            if year:
                result[year_col] = year
            if evidence:
                result[evidence_col] = evidence
            result[confidence_col] = confidence

    # Experience
    exp = ai_result.get("experience", {})

    result["AI Total Experience Years"] = exp.get(
        "total_experience_years", ""
    )
    result["AI Academic Experience Years"] = exp.get(
        "academic_experience_years", ""
    )
    result["AI Industry Experience Years"] = exp.get(
        "industry_experience_years", ""
    )

    # Publications
    pubs = ai_result.get("publications", {})

    result["AI SCI/SCIE Count"] = pubs.get("sci_scie_count", "")
    result["AI SCI Count"] = pubs.get("sci_count", "")
    result["AI SCIE Count"] = pubs.get("scie_count", "")
    result["AI SCI/SCIE Evidence"] = pubs.get("evidence", "")
    result["AI Publication Confidence"] = pubs.get("confidence", "")

    result["AI Status"] = "Completed"

    return result


# =========================================================
# PROCESS ONE RESUME
# =========================================================

def process_resume(pdf_path, use_ai=False, model_name="gpt-5.6"):
    filename = os.path.basename(pdf_path)

    raw_text = extract_pdf_text(pdf_path)

    if raw_text.startswith("ERROR"):
        return {
            "Resume": filename,
            "Candidate Name": "",
            "Status": "PDF Error",
            "Remarks": raw_text,
        }

    text = clean_text(raw_text)

    candidate_name = extract_name_basic(text, filename)

    education_section = extract_education_section(text)
    education_rule_records = education_records_rule_based(
        education_section
    )

    # Rule-based education result
    education = {
        "Education Section Found": "Yes" if education_section else "No",
        "Graduation Degree": "",
        "Graduation Year": "",
        "Graduation Confidence": "Not Found",
        "Graduation Evidence": "",
        "Post Graduation Degree": "",
        "Post Graduation Year": "",
        "PG Confidence": "Not Found",
        "Post Graduation Evidence": "",
        "PhD": "",
        "PhD Year": "",
        "PhD Confidence": "Not Found",
        "PhD Evidence": "",
    }

    for category, degree_col, year_col, conf_col, evidence_col in [
        (
            "Graduation",
            "Graduation Degree",
            "Graduation Year",
            "Graduation Confidence",
            "Graduation Evidence",
        ),
        (
            "Post Graduation",
            "Post Graduation Degree",
            "Post Graduation Year",
            "PG Confidence",
            "Post Graduation Evidence",
        ),
        (
            "PhD",
            "PhD",
            "PhD Year",
            "PhD Confidence",
            "PhD Evidence",
        ),
    ]:
        records = [
            r for r in education_rule_records
            if r["category"] == category
        ]

        if records:
            # Prefer record with a year and shortest evidence.
            records.sort(
                key=lambda r: (
                    bool(r["year"]),
                    -len(r["evidence"]),
                ),
                reverse=True,
            )
            r = records[0]
            education[degree_col] = r["degree"]
            education[year_col] = r["year"]
            education[conf_col] = "Medium"
            education[evidence_col] = r["evidence"]

    # Rule-based experience
    exp_rule = extract_experience_rule_based(text)

    # Rule-based SCI/SCIE
    pub_rule = extract_sci_scie_rule_based(text)

    result = {
        "Resume": filename,
        "Candidate Name": candidate_name,

        **education,

        "Rule Total Experience Years": exp_rule.get(
            "total_years", ""
        ),
        "Rule Experience Months": exp_rule.get(
            "total_months", ""
        ),
        "Experience Method": exp_rule.get(
            "experience_method", ""
        ),

        "Rule SCI/SCIE Count": (
            pub_rule.get("sci_scie_count", "")
        ),
        "Rule SCI/SCIE Method": (
            pub_rule.get("sci_scie_method", "")
        ),
        "Rule SCI/SCIE Evidence": (
            pub_rule.get("sci_scie_evidence", "")
        ),

        "AI Status": "Not Run",
        "AI Total Experience Years": "",
        "AI Academic Experience Years": "",
        "AI Industry Experience Years": "",
        "AI SCI/SCIE Count": "",
        "AI SCI Count": "",
        "AI SCIE Count": "",
        "AI SCI/SCIE Evidence": "",
        "AI Publication Confidence": "",
    }

    # Status before AI
    missing = []
    if not result["Graduation Year"]:
        missing.append("Graduation")
    if not result["Post Graduation Year"]:
        missing.append("Post Graduation")
    if not result["PhD Year"]:
        missing.append("PhD")

    result["Status"] = (
        "Complete" if not missing else "Review Required"
    )
    result["Remarks"] = (
        "" if not missing
        else "Not found/uncertain: " + ", ".join(missing)
    )

    # AI second pass
    if use_ai:
        try:
            ai_result = run_ai_extraction(
                text,
                model_name=model_name
            )
            result = merge_results(result, ai_result)

            # Recalculate status after AI
            missing = []
            if not result.get("Graduation Year"):
                missing.append("Graduation")
            if not result.get("Post Graduation Year"):
                missing.append("Post Graduation")
            if not result.get("PhD Year"):
                missing.append("PhD")

            result["Status"] = (
                "Complete" if not missing else "Review Required"
            )
            result["Remarks"] = (
                "" if not missing
                else "Not found/uncertain: " + ", ".join(missing)
            )

        except Exception as e:
            result["AI Status"] = f"AI Error: {e}"

    return result


# =========================================================
# FOLDER
# =========================================================

def process_uploaded_files(uploaded_files, use_ai=False, model_name="gpt-5.6"):
    results = []
    progress = st.progress(0)

    for i, uploaded_file in enumerate(uploaded_files):
        temp_path = Path("/tmp") / uploaded_file.name
        try:
            temp_path.write_bytes(uploaded_file.getbuffer())
            results.append(
                process_resume(
                    str(temp_path),
                    use_ai=use_ai,
                    model_name=model_name,
                )
            )
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

        progress.progress((i + 1) / max(len(uploaded_files), 1))

    progress.empty()
    return pd.DataFrame(results)


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="Faculty CV Intelligence System",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Faculty CV Intelligence System — Version 3")

st.caption(
    "PDF extraction + rule-based validation + optional AI verification"
)

with st.sidebar:
    st.header("Settings")

    uploaded_files = st.file_uploader(
        "Upload faculty resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more faculty CV/resume PDF files.",
    )

    use_ai = st.checkbox(
        "Use AI verification",
        value=False,
        help=(
            "AI sends extracted CV text to the configured API. "
            "Do not enable this if CV data cannot leave your system."
        ),
    )

    model_name = st.text_input(
        "AI Model",
        value="gpt-5.6",
    )

    if use_ai:
        if not OPENAI_AVAILABLE:
            st.warning(
                "Install the OpenAI package: pip install openai"
            )

        if os.getenv("OPENAI_API_KEY"):
            st.success("OPENAI_API_KEY detected.")
        else:
            st.warning(
                "OPENAI_API_KEY is not configured."
            )

st.markdown(
    """
### What this version extracts

- Graduation degree + completion year
- Post-graduation degree + completion year
- PhD + completion year
- Evidence for every qualification
- Total experience
- Academic / industry experience using AI
- SCI/SCIE publication count
- SCI vs SCIE count when explicitly supported by the CV
- Rule-based result + AI verification
"""
)

st.divider()

if st.button("🔍 Scan Resumes", type="primary"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF resume.")

    elif use_ai and not os.getenv("OPENAI_API_KEY"):
        st.error(
            "AI is enabled but OPENAI_API_KEY is not configured."
        )

    else:
        with st.spinner("Processing CVs..."):
            df = process_uploaded_files(
                uploaded_files,
                use_ai=use_ai,
                model_name=model_name,
            )

        if df.empty:
            st.warning("No PDF resumes found.")

        else:
            st.success(
                f"Processed {len(df)} resume(s)."
            )

            # -------------------------------------------------
            # Summary
            # -------------------------------------------------

            total = len(df)
            complete = len(
                df[df["Status"] == "Complete"]
            )
            review = total - complete

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Resumes", total)
            col2.metric("Complete", complete)
            col3.metric("Needs Review", review)

            ai_done = len(
                df[
                    df["AI Status"].astype(str).str.startswith(
                        "Completed"
                    )
                ]
            )
            col4.metric("AI Verified", ai_done)

            st.divider()

            # -------------------------------------------------
            # Main table
            # -------------------------------------------------

            st.subheader("📊 Candidate Summary")

            summary_cols = [
                "Resume",
                "Candidate Name",
                "Graduation Degree",
                "Graduation Year",
                "Graduation Confidence",
                "Post Graduation Degree",
                "Post Graduation Year",
                "PG Confidence",
                "PhD",
                "PhD Year",
                "PhD Confidence",
                "Rule Total Experience Years",
                "AI Total Experience Years",
                "Rule SCI/SCIE Count",
                "AI SCI/SCIE Count",
                "Status",
            ]

            summary_cols = [
                c for c in summary_cols if c in df.columns
            ]

            st.dataframe(
                df[summary_cols],
                use_container_width=True,
                hide_index=True,
            )

            # -------------------------------------------------
            # Verification
            # -------------------------------------------------

            st.divider()
            st.subheader("🔎 Evidence / Verification")

            evidence_cols = [
                "Resume",
                "Candidate Name",
                "Graduation Year",
                "Graduation Evidence",
                "Post Graduation Year",
                "Post Graduation Evidence",
                "PhD Year",
                "PhD Evidence",
                "Rule SCI/SCIE Count",
                "Rule SCI/SCIE Evidence",
                "AI SCI/SCIE Count",
                "AI SCI/SCIE Evidence",
                "Status",
                "Remarks",
            ]

            evidence_cols = [
                c for c in evidence_cols if c in df.columns
            ]

            st.dataframe(
                df[evidence_cols],
                use_container_width=True,
                hide_index=True,
            )

            # -------------------------------------------------
            # Experience
            # -------------------------------------------------

            st.divider()
            st.subheader("💼 Experience Analysis")

            exp_cols = [
                "Resume",
                "Candidate Name",
                "Rule Total Experience Years",
                "Rule Experience Months",
                "AI Total Experience Years",
                "AI Academic Experience Years",
                "AI Industry Experience Years",
                "Experience Method",
            ]

            exp_cols = [
                c for c in exp_cols if c in df.columns
            ]

            st.dataframe(
                df[exp_cols],
                use_container_width=True,
                hide_index=True,
            )

            # -------------------------------------------------
            # Publication Analysis
            # -------------------------------------------------

            st.divider()
            st.subheader("📚 SCI / SCIE Publication Analysis")

            pub_cols = [
                "Resume",
                "Candidate Name",
                "Rule SCI/SCIE Count",
                "AI SCI/SCIE Count",
                "AI SCI Count",
                "AI SCIE Count",
                "AI Publication Confidence",
                "AI SCI/SCIE Evidence",
            ]

            pub_cols = [
                c for c in pub_cols if c in df.columns
            ]

            st.dataframe(
                df[pub_cols],
                use_container_width=True,
                hide_index=True,
            )

            # -------------------------------------------------
            # Excel
            # -------------------------------------------------

            output_file = os.path.join(
                OUTPUT_FOLDER,
                "faculty_cv_intelligence_v3.xlsx",
            )

            with pd.ExcelWriter(
                output_file,
                engine="openpyxl",
            ) as writer:

                df.to_excel(
                    writer,
                    sheet_name="Candidate Summary",
                    index=False,
                )

                if "Graduation Evidence" in df.columns:
                    df[
                        evidence_cols
                    ].to_excel(
                        writer,
                        sheet_name="Verification",
                        index=False,
                    )

                df[
                    exp_cols
                ].to_excel(
                    writer,
                    sheet_name="Experience",
                    index=False,
                )

                df[
                    pub_cols
                ].to_excel(
                    writer,
                    sheet_name="SCI_SCIE",
                    index=False,
                )

            st.success(
                f"Excel report created: {output_file}"
            )

            with open(output_file, "rb") as f:
                st.download_button(
                    label="⬇️ Download Excel Report",
                    data=f,
                    file_name="faculty_cv_intelligence_v3.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                )

st.divider()

st.caption(
    "Version 3 uses deterministic extraction first and AI as a "
    "second-pass verifier. AI should not be treated as a substitute "
    "for human verification in recruitment decisions."
)
