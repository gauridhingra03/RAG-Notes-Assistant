# src/generate.py

import json
import re
from groq import Groq, APIStatusError
from src.config import GROQ_API_KEY, LLM_MODEL

client = Groq(api_key=GROQ_API_KEY)


# ---------- Shared helpers ----------
def _call_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 2048) -> tuple[str, str]:
    """Returns (content, finish_reason). 'length' = truncated, 'error' = rate-limit/too-large."""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content, response.choices[0].finish_reason
    except APIStatusError as e:
        if e.status_code in (413, 429):
            return "", "error"
        raise


def _limit_context(chunks: list[dict], max_chars: int = 9000) -> str:
    context = ""
    for c in chunks:
        if len(context) + len(c["text"]) > max_chars:
            break
        context += c["text"] + "\n\n"
    return context.strip()


def _extract_json(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No JSON array found in LLM response.")


def _is_similar(text_a: str, text_b: str, threshold: float = 0.6) -> bool:
    """Word-overlap based similarity check — catches near-duplicates
    like 'purpose of X' vs 'purpose of using X'."""
    stopwords = {"the", "a", "an", "is", "of", "in", "to", "what", "how", "using"}
    words_a = set(re.findall(r"\w+", text_a.lower())) - stopwords
    words_b = set(re.findall(r"\w+", text_b.lower())) - stopwords
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
    return overlap >= threshold


def _generate_batched_items(context: str, prompt_fn, key_field: str, target_count: int,
                             max_rounds: int = 4, temperature: float = 0.7) -> list[dict]:
    """Tries to get everything in ONE call first (fast path). Makes additional
    calls for any shortfall. If still short by the final try, allows near-duplicates
    so the requested count is always met."""
    collected = []

    for round_num in range(max_rounds):
        if len(collected) >= target_count:
            break
        remaining = target_count - len(collected)
        recent = [item[key_field] for item in collected[-6:]]
        prompt = prompt_fn(context, remaining, recent)

        raw, finish_reason = _call_llm(prompt, temperature=temperature, max_tokens=2000)
        is_last_round = (round_num == max_rounds - 1)

        if finish_reason == "error":
            continue
        try:
            batch = _extract_json(raw)
        except Exception:
            continue

        for item in batch:
            if is_last_round or not any(_is_similar(item[key_field], existing[key_field]) for existing in collected):
                collected.append(item)
                if len(collected) >= target_count:
                    break

    # Safety net: agar 3 rounds ke baad bhi shortfall hai (jaise koi call rate-limit/parsing
    # error se fail hui thi), ek aakhri extra call karo — dedup skip taaki count guaranteed mile
    if len(collected) < target_count:
        remaining = target_count - len(collected)
        recent = [item[key_field] for item in collected[-6:]]
        prompt = prompt_fn(context, remaining, recent)
        raw, finish_reason = _call_llm(prompt, temperature=temperature, max_tokens=2000)
        if finish_reason != "error":
            try:
                batch = _extract_json(raw)
                for item in batch:
                    collected.append(item)
                    if len(collected) >= target_count:
                        break
            except Exception:
                pass

    return collected[:target_count]


def group_chunks_by_pages(chunks: list[dict], pages_per_section: int = 20) -> list[dict]:
    """Chunks ko page-range sections mein group karta hai.
    Last section ka end actual last page pe cap hota hai (e.g. Pages 21-36, not 21-40)."""
    if not chunks:
        return []

    max_page = max(c.get("page", 1) for c in chunks)
    sections = {}
    for chunk in chunks:
        page_num = chunk.get("page", 1)
        section_idx = (page_num - 1) // pages_per_section
        sections.setdefault(section_idx, []).append(chunk)

    result = []
    for idx in sorted(sections.keys()):
        start_page = idx * pages_per_section + 1
        end_page = min((idx + 1) * pages_per_section, max_page)
        result.append({"label": f"Pages {start_page}-{end_page}", "chunks": sections[idx]})
    return result


# ---------- Feature 1: Q&A ----------
def build_qa_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join([f"[Source {i+1}, Page {c.get('page', '?')}]: {c['text']}" for i, c in enumerate(chunks)])
    return f"""You are a study assistant. Answer the question using only the context below.
Give a complete, detailed explanation — synthesize information across all sources into ONE coherent answer, don't describe each source separately.
IMPORTANT: You MUST end your answer with the page number(s) the information came from, in the format "(Page 2)" or "(Pages 2, 5)" — this is required even if all the information comes from a single page. Never mention "Source" numbers.
If the answer is not in the context, say "This information is not available in the notes" — do not make up information.

Context:
{context}

Question: {query}

Answer:"""


def generate_answer(query: str, chunks: list[dict]) -> str:
    prompt = build_qa_prompt(query, chunks)
    text, finish_reason = _call_llm(prompt, temperature=0.2, max_tokens=3000)
    if finish_reason == "error":
        return "⚠️ The request was too large or the rate limit was hit. Try asking a shorter/simpler question."
    return text


# ---------- Feature 2: Summarize ----------
def generate_summary(chunks: list[dict], length: str = "medium") -> str:
    context = _limit_context(chunks)
    length_map = {
        "short": "in about 100 words",
        "medium": "in about 250 words",
        "detailed": "in about 500 words, covering all major sections"
    }
    prompt = f"""You are a study assistant. Summarize the following document content {length_map.get(length, 'in about 250 words')}.
Focus on the key ideas, main arguments, and important technical details.
Mention each point only ONCE — if you use headers/sections, make sure no idea, term, or detail is repeated across multiple sections or in a "key ideas" recap at the end.
If the document contains mathematical formulas or equations, preserve them exactly as written (do not paraphrase or drop them) and include them at the relevant point in the summary.
Stay close to the requested word count. Do not add information not present in the text.

Document content:
{context}

Summary:"""
    text, finish_reason = _call_llm(prompt, temperature=0.3, max_tokens=3000)
    if finish_reason == "error":
        return "⚠️ The request was too large or the rate limit was hit. Try a shorter summary length."
    return text


# ---------- Feature 3: Practice Questions ----------
def generate_practice_questions(chunks: list[dict], num_questions: int = 5, seed_note: str = "") -> list[dict]:
    context = _limit_context(chunks)

    def prompt_fn(ctx, n, recent):
        avoid = f"Avoid repeating these topics: {'; '.join(recent)}." if recent else ""
        seed_line = f"({seed_note})" if seed_note else ""
        return f"""You are a study assistant. Based on the following document content, generate {n} practice questions with answers.
Mix conceptual ("why/how") and factual ("what is") questions. Cover DIFFERENT concepts — don't ask about the same idea twice in different words.
Each answer must be a complete, self-contained explanation (2-3 full sentences, ~40-60 words) — detailed enough to actually teach the concept, not just a one-liner.
{avoid}
{seed_line}

Return ONLY a valid JSON array, no extra text, no markdown fences:
[{{"question": "...", "answer": "..."}}]

Document content:
{ctx}
"""

    items = _generate_batched_items(context, prompt_fn, key_field="question",
                                     target_count=num_questions, temperature=0.7)
    if not items:
        return [{"question": "Error parsing questions", "answer": "LLM request kept failing — try again or use a smaller section."}]
    return items


# ---------- Feature 4: Flashcards ----------
def generate_flashcards(chunks: list[dict], num_cards: int = 8, seed_note: str = "") -> list[dict]:
    context = _limit_context(chunks)

    def prompt_fn(ctx, n, recent):
        avoid = f"Avoid repeating these terms: {'; '.join(recent)}." if recent else ""
        seed_line = f"({seed_note})" if seed_note else ""
        return f"""You are a study assistant. Based on the following document content, generate {n} flashcards covering important terms and concepts.
Each definition must be SPECIFIC to how the term is used in THIS document (mention concrete details, not a generic textbook definition), and substantial enough to be useful — 1-2 full sentences, ~25-40 words. Never a one-word or overly short answer.
{avoid}
{seed_line}

Return ONLY a valid JSON array, no extra text, no markdown fences:
[{{"term": "...", "definition": "..."}}]

Document content:
{ctx}
"""

    items = _generate_batched_items(context, prompt_fn, key_field="term",
                                     target_count=num_cards, temperature=0.6)
    if not items:
        return [{"term": "Error parsing flashcards", "definition": "LLM request kept failing — try again or use a smaller section."}]
    return items