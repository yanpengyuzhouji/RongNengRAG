"""Cross-domain retrieval orchestration independent of storage backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class CrossDomainResponse:
    query: str
    query_type: str
    domain: Optional[str]
    results: List[Any]
    total_candidates: int
    elapsed_ms: float
    context: str
    context_domain1: str
    context_domain2: str
    domain_names: List[str] = field(default_factory=list)


def apply_domain_override(analyzed_query: Any, domain_filter: Optional[str], builder: Any) -> Any:
    """Apply an explicit domain even when automatic analysis chose another one."""
    if domain_filter:
        analyzed_query.domain = domain_filter
        analyzed_query.filter_expr = builder(analyzed_query)
    return analyzed_query


def retrieve_cross_domain(retriever: Any, query: str, top_k: int) -> CrossDomainResponse:
    """Search exactly two distinct domains and build prompt-ready contexts."""
    responses = retriever.search_cross_domain(query, top_k=top_k)
    selected = list(responses.items())[:2]
    while len(selected) < 2:
        selected.append((f"未指定域{len(selected) + 1}", None))

    contexts: list[str] = []
    combined_results: list[Any] = []
    total_candidates = 0
    elapsed_ms = 0.0
    names: list[str] = []
    for domain, response in selected:
        names.append(domain)
        results = list(getattr(response, "results", []) or [])
        combined_results.extend(results)
        total_candidates += int(getattr(response, "total_candidates", 0) or 0)
        elapsed_ms += float(getattr(response, "elapsed_ms", 0.0) or 0.0)
        formatted = retriever.format_context_for_llm(results, max_chunks=top_k)
        if not formatted:
            formatted = "（该专业域未检索到可用资料）"
        contexts.append(f"【专业域：{domain}】\n{formatted}")

    return CrossDomainResponse(
        query=query,
        query_type="cross_domain_comparison",
        domain=" / ".join(names),
        results=combined_results,
        total_candidates=total_candidates,
        elapsed_ms=elapsed_ms,
        context="\n\n".join(contexts),
        context_domain1=contexts[0],
        context_domain2=contexts[1],
        domain_names=names,
    )
