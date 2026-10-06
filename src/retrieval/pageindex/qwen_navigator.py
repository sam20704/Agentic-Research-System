"""Phase 3.2 Qwen-based PageIndex tree navigator.

This module provides a query-time LLM navigator over an already-built
PageIndex structural tree.

Important design constraints:

    - No embeddings are generated.
    - No vector database is used.
    - No BM25 or lexical retrieval is used.
    - The model may only select node IDs that already exist in the tree.
    - The existing deterministic PageIndexNavigator remains unchanged.
    - This component is for Phase 3.2 evaluation only.

The evaluator can therefore compare:

    PageIndexNavigator
        vs
    QwenPageIndexNavigator

using exactly the same tree and ground-truth cases.

The Qwen implementation uses Ollama's native /api/chat endpoint rather
than Ollama's OpenAI-compatible endpoint. This is intentional: Qwen 3.5
may place its answer in the OpenAI-compatible ``message.reasoning`` field
while returning an empty ``message.content``. The native Ollama API,
with ``think=False`` and structured JSON output, returns the answer in
``message.content``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

import requests


@dataclass(frozen=True)
class QwenNavigationCandidate:
    """One node selected by the Qwen navigator."""

    node: dict[str, Any]
    rank: int
    path: tuple[str, ...]


class QwenPageIndexNavigator:
    """Use a local Qwen model to navigate an existing PageIndex tree.

    The model is presented with the structural tree as a flat list containing:

        node_id
        title
        hierarchy path
        page range

    It must return only node IDs from that supplied list.

    The returned node IDs are validated against the actual tree before they
    become navigation candidates.

    Ollama's native /api/chat endpoint is used so that Qwen's thinking mode
    can be explicitly disabled and JSON output can be requested directly.
    """

    def __init__(
        self,
        structure: dict[str, Any] | list[dict[str, Any]],
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        timeout: float = 120.0,
    ) -> None:
        if isinstance(structure, dict):
            nodes = structure.get("structure")
        else:
            nodes = structure

        if not isinstance(nodes, list):
            raise ValueError(
                "PageIndex structure must contain a list of nodes"
            )

        if not model or not model.strip():
            raise ValueError("model must not be empty")

        if temperature < 0:
            raise ValueError("temperature must be >= 0")

        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")

        if timeout <= 0:
            raise ValueError("timeout must be > 0")

        self.max_batch_size = 30

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

        self.base_url = base_url.rstrip("/")

        self._nodes: list[dict[str, Any]] = []
        self._paths: dict[str, tuple[str, ...]] = {}

        self._index_nodes(nodes)

    def navigate(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[QwenNavigationCandidate]:
        """Ask Qwen to select the most relevant existing PageIndex nodes."""

        if not query or not query.strip():
            raise ValueError("query must not be empty")

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        if len(self._nodes) <= self.max_batch_size:
            selected_ids = self._query_nodes_batch(
                candidate_nodes=self._nodes,
                query=query,
                top_k=top_k,
            )
        else:
            batch_candidate_ids: list[str] = []

            for i in range(0, len(self._nodes), self.max_batch_size):
                batch_nodes = self._nodes[i : i + self.max_batch_size]
                batch_ids = self._query_nodes_batch(
                    candidate_nodes=batch_nodes,
                    query=query,
                    top_k=top_k,
                )

                for node_id in batch_ids:
                    if node_id not in batch_candidate_ids:
                        batch_candidate_ids.append(node_id)

            if len(batch_candidate_ids) <= top_k:
                selected_ids = batch_candidate_ids
            else:
                candidate_set = set(batch_candidate_ids)
                consolidated_nodes = [
                    node
                    for node in self._nodes
                    if str(node["node_id"]) in candidate_set
                ]
                selected_ids = self._query_nodes_batch(
                    candidate_nodes=consolidated_nodes,
                    query=query,
                    top_k=top_k,
                )

        results: list[QwenNavigationCandidate] = []

        node_lookup = {
            str(node["node_id"]): node
            for node in self._nodes
        }

        for rank, node_id in enumerate(
            selected_ids[:top_k],
            start=1,
        ):
            node = node_lookup[node_id]

            results.append(
                QwenNavigationCandidate(
                    node=node,
                    rank=rank,
                    path=self._paths[node_id],
                )
            )

        return results

    def _query_nodes_batch(
        self,
        candidate_nodes: list[dict[str, Any]],
        query: str,
        top_k: int,
    ) -> list[str]:
        """Execute a single bounded Qwen navigation pass over candidate nodes."""

        if not candidate_nodes:
            return []

        prompt = self._build_prompt(
            candidate_nodes=candidate_nodes,
            query=query,
            top_k=top_k,
        )

        response_schema = {
            "type": "object",
            "properties": {
                "node_ids": {
                    "type": "array",
                    "maxItems": top_k,
                    "items": {
                        "type": "string",
                        "enum": [
                            str(node["node_id"])
                            for node in candidate_nodes
                        ],
                    },
                }
            },
            "required": ["node_ids"],
            "additionalProperties": False,
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "think": False,
            "format": response_schema,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
            },
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                "Qwen/Ollama navigation request failed: "
                f"HTTP {response.status_code}: {response.text}"
            ) from exc

        try:
            response_data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Qwen/Ollama navigation returned a non-JSON HTTP response"
            ) from exc

        message = response_data.get("message")

        if not isinstance(message, dict):
            raise RuntimeError(
                "Qwen/Ollama navigation response is missing "
                "a valid 'message' object"
            )

        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "Qwen navigator returned an empty response"
            )

        payload_data = self._parse_response(content)

        node_ids = payload_data.get("node_ids")

        if not isinstance(node_ids, list):
            raise ValueError(
                "Qwen navigator response must contain a "
                "'node_ids' list"
            )

        valid_candidate_ids = {
            str(node["node_id"]) for node in candidate_nodes
        }

        selected_ids: list[str] = []

        for value in node_ids:
            node_id = str(value)

            if node_id not in self._paths:
                raise ValueError(
                    "Qwen navigator returned an unknown node_id: "
                    f"{node_id}"
                )

            if node_id in valid_candidate_ids and node_id not in selected_ids:
                selected_ids.append(node_id)

            if len(selected_ids) >= top_k:
                break

        return selected_ids

    def _index_nodes(
        self,
        nodes: Iterable[dict[str, Any]],
        parent_path: tuple[str, ...] = (),
    ) -> None:
        """Flatten the tree while preserving hierarchy paths."""

        for node in nodes:
            if not isinstance(node, dict):
                raise ValueError(
                    "PageIndex node must be a JSON object"
                )

            required_fields = (
                "node_id",
                "title",
                "start_index",
                "end_index",
            )

            for field in required_fields:
                if field not in node:
                    raise ValueError(
                        "PageIndex node is missing required field "
                        f"'{field}'"
                    )

            node_id = str(node["node_id"])
            title = str(node.get("title") or "")

            if node_id in self._paths:
                raise ValueError(
                    f"Duplicate PageIndex node_id: {node_id}"
                )

            path = parent_path + (title,)

            self._nodes.append(node)
            self._paths[node_id] = path

            children = node.get("nodes") or []

            if not isinstance(children, list):
                raise ValueError(
                    f"Invalid children for PageIndex node {node_id}"
                )

            self._index_nodes(
                children,
                path,
            )

    @staticmethod
    def _system_prompt() -> str:
        return """
You are a structural navigation agent for a PageIndex document tree.

Your job is ONLY to select the existing PageIndex node IDs that are most
relevant to the user's query.

You are NOT answering the user's question.

Rules:

1. You may ONLY return node IDs supplied in the document tree.
2. Never invent a node ID.
3. Never modify a node ID.
4. Never create a new node.
5. Use the hierarchy paths and section titles to reason about relevance.
6. For semantic queries, infer which existing section best corresponds to
   the meaning of the query.
7. For multi-hop queries, select all relevant existing sections, in the
   order that best supports the query.
8. Prefer specific child sections over broad parent sections when appropriate.
9. Return at most the requested number of nodes.
10. Return valid JSON only.
11. Do not provide an explanation.
12. Do not use Markdown.
13. Do not include any fields other than node_ids.

The output schema supplied by the API is authoritative.
""".strip()

    def _build_prompt(
        self,
        *,
        candidate_nodes: list[dict[str, Any]],
        query: str,
        top_k: int,
    ) -> str:
        tree_lines: list[str] = []

        for node in candidate_nodes:
            node_id = str(node["node_id"])
            title = str(node["title"])
            start = node["start_index"]
            end = node["end_index"]
            path = " > ".join(self._paths[node_id])

            tree_lines.append(
                f"- node_id={node_id} | "
                f"title={title} | "
                f"path={path} | "
                f"pages={start}-{end}"
            )

        return (
            "USER QUERY:\n"
            f"{query.strip()}\n\n"
            f"SELECT UP TO {top_k} NODES.\n\n"
            "AVAILABLE PAGEINDEX NODES:\n"
            + "\n".join(tree_lines)
            + "\n\n"
            "Return only the JSON object containing node_ids."
        )

    @staticmethod
    def _parse_response(content: str) -> dict[str, Any]:
        """Parse and validate a JSON response from the model."""

        text = content.strip()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(
                r"\{.*\}",
                text,
                flags=re.DOTALL,
            )

            if not match:
                raise ValueError(
                    "Qwen navigator returned invalid JSON"
                )

            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Qwen navigator returned invalid JSON"
                ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "Qwen navigator response must be a JSON object"
            )

        return payload


__all__ = [
    "QwenNavigationCandidate",
    "QwenPageIndexNavigator",
]