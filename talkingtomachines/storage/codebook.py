"""
Codebook generator

Produces a ``codebook.json`` documenting all field definitions, response
options, and data types for a compiled experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from talkingtomachines.compiler.cep_schema import CompiledExperiment


def generate_codebook(cep: CompiledExperiment) -> dict[str, Any]:
    """Generate a codebook dictionary from a Compiled Experiment Package.

    The codebook documents all field definitions, response options, data types,
    prompt definitions, and profile columns for a compiled experiment. Fields
    are grouped by scope into logical tables (session, agent, group, responses).

    Args:
        cep: The compiled experiment package containing field definitions,
            prompts, and profile information.

    Returns:
        A structured dictionary with keys ``experiment_id``, ``cep_hash``,
        ``schema_version``, ``tables``, ``prompts``, and ``profile_columns``.
    """
    codebook: dict[str, Any] = {
        "experiment_id": cep.experiment_id,
        "cep_hash": cep.cep_hash,
        "schema_version": cep.schema_version,
        "tables": {},
    }

    # Group fields by scope → table
    scope_to_table = {
        "Session": "session_table",
        "Agent": "agent_table",
        "Group": "group_table",
        "Player": "player",
    }

    for field_dict in cep.fields:
        scope = field_dict.get("field_class", "Player")
        table = scope_to_table.get(scope, "player")
        module = field_dict.get("module", "")
        name = field_dict.get("name", "")
        full_name = f"{module}.{name}" if module else name

        entry: dict[str, Any] = {
            "field": name,
            "module": module,
            "scope": scope,
            "type": field_dict.get("type", "text"),
            "response_options": field_dict.get("response_options"),
            "response_options_intro": field_dict.get("response_options_intro", ""),
            "validate": field_dict.get("validate", False),
            "format_response": field_dict.get("format_response", False),
            "generate_speculation_score": field_dict.get(
                "generate_speculation_score", False
            ),
            "randomise_options_order": field_dict.get("randomise_options_order", False),
        }
        codebook["tables"].setdefault(table, {})[full_name] = entry

    # Prompt definitions
    codebook["prompts"] = {}
    for module, prompt_list in cep.prompts.items():
        codebook["prompts"][module] = [
            {
                "module": module,
                "sequence": p.get("prompt_sequence"),
                "type": p.get("type"),
                "is_displayed": p.get("is_displayed"),
                "field_class": p.get("field_class"),
                "field_name": p.get("field_name"),
                "llm_text": p.get("llm_text", ""),
                "human_text": p.get("human_text", ""),
                "is_adapted": p.get("is_adapted", False),
                "rag_vector_store_id": p.get("rag_vector_store_id", ""),
            }
            for p in prompt_list
        ]

    # Profile columns
    profiles = cep.profiles or {}
    codebook["profile_columns"] = profiles.get("short_names", [])

    return codebook


def save_codebook(cep: CompiledExperiment, output_dir: str | Path) -> str:
    """Save a codebook as ``codebook.json`` in the specified directory.

    Args:
        cep: The compiled experiment package to generate the codebook from.
        output_dir: Directory where ``codebook.json`` will be written.

    Returns:
        The absolute file path of the saved ``codebook.json``.
    """
    cb = generate_codebook(cep)
    path = Path(output_dir) / "codebook.json"
    path.write_text(
        json.dumps(cb, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    return str(path)
