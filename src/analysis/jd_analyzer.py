import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.client import get_llm


SYSTEM_PROMPT = """
You are a job-description analysis component.

Extract only information supported by the provided job description.

Return valid JSON only with this schema:

{
  "job_title": "string",
  "must_have_skills": ["string"],
  "nice_to_have_skills": ["string"],
  "responsibilities": ["string"],
  "experience_requirements": ["string"],
  "education_requirements": ["string"],
  "languages": ["string"],
  "keywords": ["string"]
}

Rules:
- Do not invent requirements.
- Keep technology names exactly recognizable.
- Separate mandatory from optional requirements when the wording allows it.
- keywords should contain concise terms useful for candidate-profile retrieval.
"""


def analyze_jd(jd_text: str) -> Dict[str, Any]:
    if not jd_text.strip():
        raise ValueError("JD text cannot be empty.")

    llm = get_llm(temperature=0.0)

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=jd_text),
        ]
    )

    content = response.content.strip()

    # Some models may wrap JSON in Markdown fences.
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    return json.loads(content)


if __name__ == "__main__":
    sample_jd = """
    We are looking for a Working Student in AI Engineering.

    You will develop LLM-based prototypes, automate workflows,
    evaluate AI systems and integrate internal tools through APIs.

    Required:
    - Strong Python skills
    - Experience with machine learning or NLP
    - Enrolled at a university

    Nice to have:
    - LangChain or LangGraph
    - RAG systems
    - Docker
    """

    result = analyze_jd(sample_jd)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )