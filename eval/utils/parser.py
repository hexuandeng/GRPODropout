import re
from typing import Any, Dict, Optional, Tuple


CHOICE_DATASETS = {"arc_challenge", "mmlu_pro", "gpqa_diamond", "supergpqa"}
SHORT_NUMERIC_DATASETS = {"aime24", "aime2025", "amc23"}


def strip_string(string):
    if not string:
        return ""
    string = str(string).strip()
    string = string.replace(r"\dfrac", r"\frac")
    string = string.replace(r"\tfrac", r"\frac")
    return string


def _available_choice_labels(data_name: str = None) -> str:
    if data_name in {"mmlu_pro", "supergpqa"}:
        return "ABCDEFGHIJ"
    if data_name == "arc_challenge":
        return "ABCDE12345"
    return "ABCDE"


def _answer_search_texts(response: str):
    response = "" if response is None else str(response)
    texts = []
    if "</think>" in response:
        final_part = response.rsplit("</think>", 1)[-1].strip()
        if final_part:
            texts.append(final_part)
    tail = response[-6000:].strip()
    if tail and tail not in texts:
        texts.append(tail)
    full = response.strip()
    if full and full not in texts:
        texts.append(full)
    return texts


def _clean_choice(response: str, data_name: str = None) -> Optional[str]:
    response = "" if response is None else str(response).strip()
    labels = _available_choice_labels(data_name)
    label_class = re.escape(labels)

    explicit_patterns = [
        rf"\\boxed\s*\{{?\s*([{label_class}{label_class.lower()}])\s*\}}?",
        rf"(?:final\s+answer|answer|correct\s+answer|correct\s+option|option|choice)\s*(?:is|:|=)?\s*\(?\s*([{label_class}{label_class.lower()}])\s*\)?\b",
        rf"(?:choose|select|pick)\s*\(?\s*([{label_class}{label_class.lower()}])\s*\)?\b",
        rf"\(([{label_class}{label_class.lower()}])\)\s*(?:is\s+)?(?:correct|right)\b",
        rf"\b([{label_class}{label_class.lower()}])\b\s*(?:is\s+)?(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\b",
    ]

    for text in _answer_search_texts(response):
        matches = []
        for pattern in explicit_patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines[-6:]):
            line = line.strip().strip(".")
            line_match = re.search(rf"^\(?([{label_class}{label_class.lower()}])\)?$", line)
            if line_match:
                return line_match.group(1).upper()

        token_match = re.search(rf"(?:^|\s)\(?([{label_class}{label_class.lower()}])\)?\s*$", text.strip().strip("."))
        if token_match:
            return token_match.group(1).upper()

    return None


def _format_choices(example: Dict[str, Any]) -> str:
    lines = []
    for choice in example.get("choices", []):
        label = str(choice.get("label", "")).strip().upper()
        text = str(choice.get("text", "")).strip()
        if label and text:
            lines.append(f"{label}. {text}")
    return "\n".join(lines)


def parse_question(example: Dict[str, Any], data_name=None) -> str:
    if data_name in CHOICE_DATASETS and example.get("choices"):
        question = str(example.get("question", "")).strip()
        choices = _format_choices(example)
        return f"{question}\n\n{choices}".strip()

    for key in ["question", "problem", "input", "content", "Question"]:
        if key in example and example[key]:
            return str(example[key]).strip()
    return ""


def parse_ground_truth(example: Dict[str, Any], data_name=None) -> Tuple[Optional[str], Optional[str]]:
    if data_name in CHOICE_DATASETS:
        for key in ["answer", "answerkey", "answer_key", "target", "label"]:
            if key in example and example[key]:
                return None, str(example[key]).strip().upper()
        return None, None

    target = None
    for key in ["answer", "solution", "output", "target", "Answer", "final_answer"]:
        if key in example and example[key]:
            target = str(example[key])
            break

    if target is None:
        return None, None

    if "####" in target:
        parts = target.split("####")
        return parts[0].strip(), parts[-1].strip()

    return None, target


def _extract_last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx + 6
    while i < len(text) and text[i].isspace():
        i += 1

    if i >= len(text) or text[i] != "{":
        return None

    brace_count = 0
    content_end = -1
    for j in range(i, len(text)):
        if text[j] == "{":
            brace_count += 1
        elif text[j] == "}":
            brace_count -= 1
            if brace_count == 0:
                content_end = j
                break

    if content_end != -1:
        return text[i + 1 : content_end]
    return None


def _strip_latex_delimiters(text):
    text = "" if text is None else str(text).strip()
    text = text.strip("$").strip()
    text = re.sub(r"^\\\((.*)\\\)$", r"\1", text, flags=re.DOTALL).strip()
    text = re.sub(r"^\\\[(.*)\\\]$", r"\1", text, flags=re.DOTALL).strip()
    return text


def normalize_short_numeric_answer(text):
    text = "" if text is None else str(text).strip()
    text = text.replace(",", "")
    text = _strip_latex_delimiters(text)
    text = text.rstrip(".").strip()
    text = _strip_latex_delimiters(text)
    text = re.sub(r"(?:\^\{?\\+circ\}?|\\+degree|\u00b0|degrees?)\s*$", "", text, flags=re.IGNORECASE).strip()
    text = text.rstrip(".").strip()
    text = _strip_latex_delimiters(text)

    exact = re.fullmatch(r"-?\d+(?:\.0+)?", text)
    if exact:
        value = exact.group(0)
        if value.endswith(".0"):
            value = value[:-2]
        try:
            return str(int(value))
        except ValueError:
            return value
    return ""


def _clean_short_numeric_answer(text):
    return normalize_short_numeric_answer(text)


def _clean_math_answer(text):
    text = _strip_latex_delimiters(text)
    text = re.sub(r"^\s*\\boxed\s*\{(.*)\}\s*$", r"\1", text, flags=re.DOTALL).strip()
    text = text.rstrip(".").strip()
    return text


def _extract_unclosed_short_box(response):
    pattern = (
        r"\\boxed\s*\{\s*\$?\s*"
        r"(-?\d+(?:,\d{3})*(?:\.0+)?)"
        r"(?:\s*(?:\^\{?\\+circ\}?|\\+degree|\u00b0|degrees?))?"
        r"\s*\$?\s*$"
    )
    match = re.search(pattern, response, flags=re.IGNORECASE)
    return _clean_short_numeric_answer(match.group(1)) if match else ""


def _extract_short_numeric_by_marker(text):
    numeric = r"(-?\d+(?:,\d{3})*(?:\.0+)?)"
    unit = r"(?:\s*(?:\^\{?\\+circ\}?|\\+degree|\u00b0|degrees?))?"
    marker_patterns = [
        rf"(?i)(?:final\s+answer|answer)\s*[:=]\s*\$?\s*{numeric}{unit}\s*\$?\s*\.?\s*$",
        rf"(?i)(?:final\s+answer|answer)\s+is\s+\$?\s*{numeric}{unit}\s*\$?\s*\.?\s*$",
        rf"(?i)the answer is\s*[:\s]*\$?\s*{numeric}{unit}\s*\$?\s*\.?\s*$",
    ]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-8:]):
        for pattern in marker_patterns:
            match = re.search(pattern, line)
            if match:
                cleaned = _clean_short_numeric_answer(match.group(1))
                if cleaned:
                    return cleaned

    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if re.fullmatch(r"(?i)#+?\s*(?:final\s+answer|answer|final\s+result)\s*:?\s*", line):
            for next_line in lines[idx + 1 : idx + 4]:
                cleaned = _clean_short_numeric_answer(next_line)
                if cleaned:
                    return cleaned
    return ""


def _extract_math_by_marker(text):
    marker_patterns = [
        r"(?i)(?:final\s+answer|final\s+result|answer)\s*[:=]\s*(.+)$",
        r"(?i)(?:final\s+answer|answer)\s+is\s+(.+)$",
        r"(?i)(?:therefore|thus|hence|so)[,:]?\s+(?:the\s+)?answer\s+is\s+(.+)$",
    ]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-8:]):
        for pattern in marker_patterns:
            match = re.search(pattern, line)
            if match:
                potential_ans = _clean_math_answer(match.group(1))
                if 0 < len(potential_ans) <= 200:
                    return potential_ans

    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if re.fullmatch(r"(?i)#+?\s*(?:final\s+answer|answer|final\s+result)\s*:?\s*", line):
            for next_line in lines[idx + 1 : idx + 4]:
                potential_ans = _clean_math_answer(next_line)
                if 0 < len(potential_ans) <= 200:
                    return potential_ans
    return ""


def extract_answer(response: str, data_name: str = None) -> str:
    if not response:
        return ""

    response = str(response).strip()
    is_short_numeric = data_name in SHORT_NUMERIC_DATASETS

    if data_name in CHOICE_DATASETS or data_name == "gaokao":
        return _clean_choice(response, data_name) or ""

    if "\\boxed" in response:
        ans = _extract_last_boxed(response)
        if ans:
            return _clean_short_numeric_answer(ans) if is_short_numeric else _clean_math_answer(ans)
        if is_short_numeric:
            unclosed = _extract_unclosed_short_box(response)
            if unclosed:
                return unclosed

    if "####" in response:
        ans = response.split("####")[-1].strip()
        return _clean_short_numeric_answer(ans) if is_short_numeric else _clean_math_answer(ans)

    if is_short_numeric:
        for text in _answer_search_texts(response):
            extracted = _extract_short_numeric_by_marker(text)
            if extracted:
                return extracted
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            for line in reversed(lines[-3:]):
                cleaned = _clean_short_numeric_answer(line)
                if cleaned:
                    return cleaned
        return ""

    for text in _answer_search_texts(response):
        extracted = _extract_math_by_marker(text)
        if extracted:
            return extracted
    return ""


def run_execute(executor, result, prompt_type, data_name, execute=False):
    return extract_answer(result, data_name)
