from typing import Any, Dict, List


MAX_QUERIES = 12


def _clean_items(
    values: Any,
) -> List[str]:
    if not isinstance(
        values,
        list,
    ):
        return []

    cleaned = []

    for value in values:
        if not isinstance(
            value,
            str,
        ):
            continue

        value = value.strip()

        if value:
            cleaned.append(
                value
            )

    return cleaned


def _deduplicate(
    values: List[str],
) -> List[str]:
    seen = set()
    result = []

    for value in values:
        normalized = (
            value.lower()
            .strip()
        )

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        result.append(
            value
        )

    return result


def _build_category_queries(
    jd_analysis: Dict[str, Any],
) -> Dict[str, List[str]]:
    """
    Build retrieval queries grouped by JD category.

    Keeping categories separate allows the query builder
    to guarantee coverage instead of letting early
    categories consume the entire query budget.
    """

    responsibilities = (
        _clean_items(
            jd_analysis.get(
                "responsibilities"
            )
        )
    )

    must_have_skills = (
        _clean_items(
            jd_analysis.get(
                "must_have_skills"
            )
        )
    )

    experience_requirements = (
        _clean_items(
            jd_analysis.get(
                "experience_requirements"
            )
        )
    )

    nice_to_have_skills = (
        _clean_items(
            jd_analysis.get(
                "nice_to_have_skills"
            )
        )
    )

    keywords = (
        _clean_items(
            jd_analysis.get(
                "keywords"
            )
        )
    )

    return {
        "responsibilities": (
            responsibilities
        ),

        "must_have": [
            f"experience using {skill}"
            for skill
            in must_have_skills
        ],

        "experience": (
            experience_requirements
        ),

        "nice_to_have": [
            f"experience using {skill}"
            for skill
            in nice_to_have_skills
        ],

        "keywords": [
            f"experience with {keyword}"
            for keyword
            in keywords
        ],
    }


def build_retrieval_queries(
    jd_analysis: Dict[str, Any],
    max_queries: int = MAX_QUERIES,
) -> List[str]:
    """
    Build retrieval queries with category-aware coverage.

    Strategy:

    1. Give every important JD category at least one
       retrieval query when possible.

    2. Fill remaining capacity using the remaining
       requirements.

    3. Use generic JD keywords only after structured
       requirements have been covered.

    This prevents nice-to-have skills from being silently
    dropped simply because responsibilities and must-have
    skills appeared earlier in the JD.
    """

    if max_queries <= 0:
        return []

    categories = (
        _build_category_queries(
            jd_analysis
        )
    )

    queries = []

    # -------------------------------------------------
    # Pass 1:
    # Guarantee category coverage.
    # -------------------------------------------------

    coverage_order = [
        "responsibilities",
        "must_have",
        "experience",
        "nice_to_have",
    ]

    positions = {
        category: 0
        for category
        in coverage_order
    }

    for category in coverage_order:
        items = categories[
            category
        ]

        if (
            items
            and len(queries)
            < max_queries
        ):
            queries.append(
                items[0]
            )

            positions[
                category
            ] = 1

    # -------------------------------------------------
    # Pass 2:
    # Fill remaining slots with structured
    # requirements using round-robin coverage.
    #
    # Must-have and responsibilities receive priority,
    # but nice-to-have requirements continue to receive
    # representation.
    # -------------------------------------------------

    fill_order = [
        "must_have",
        "responsibilities",
        "nice_to_have",
        "experience",
    ]

    while len(queries) < max_queries:
        added_any = False

        for category in fill_order:
            items = categories[
                category
            ]

            position = positions[
                category
            ]

            if (
                position
                >= len(items)
            ):
                continue

            queries.append(
                items[position]
            )

            positions[
                category
            ] += 1

            added_any = True

            if (
                len(queries)
                >= max_queries
            ):
                break

        if not added_any:
            break

    # -------------------------------------------------
    # Pass 3:
    # Only use broad keywords if structured
    # requirements did not fill the query budget.
    # -------------------------------------------------

    if len(queries) < max_queries:
        for query in categories[
            "keywords"
        ]:
            queries.append(
                query
            )

            if (
                len(queries)
                >= max_queries
            ):
                break

    queries = _deduplicate(
        queries
    )

    return queries[
        :max_queries
    ]


if __name__ == "__main__":
    example_analysis = {
        "job_title": (
            "Working Student AI Engineer"
        ),

        "must_have_skills": [
            "Python",
            "machine learning",
            "NLP",
            "RAG",
        ],

        "nice_to_have_skills": [
            "Docker",
            "LangGraph",
        ],

        "responsibilities": [
            (
                "Develop and evaluate "
                "LLM-based applications"
            ),
            (
                "Build retrieval-augmented "
                "generation workflows"
            ),
            (
                "Analyze model failures and "
                "improve system reliability"
            ),
        ],

        "experience_requirements": [
            (
                "hands-on experience with "
                "machine learning projects"
            )
        ],

        "education_requirements": [],

        "languages": [],

        "keywords": [
            "Python",
            "LLM",
            "RAG",
            "Docker",
            "evaluation",
        ],
    }

    queries = (
        build_retrieval_queries(
            example_analysis
        )
    )

    print(
        "RETRIEVAL QUERIES"
    )

    for index, query in enumerate(
        queries,
        start=1,
    ):
        print(
            f"{index}. {query}"
        )