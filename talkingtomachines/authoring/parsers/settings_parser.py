"""Settings worksheet parser.

Converts the two-column (name / value) Settings sheet into a
``SettingsConfig`` dataclass. Unknown keys are accepted with a warning
and type coercion is applied per field.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import pandas as pd

from talkingtomachines.core.models import SettingsConfig

logger = logging.getLogger(__name__)

# Expected settings keys and their Python types
_FIELD_TYPES: dict[str, type] = {
    "EXPERIMENT_ID": str,
    "MODEL_NAME": str,
    "HF_INFERENCE_ENDPOINT": str,
    "TEMPERATURE": float,
    "RANDOM_SEED": int,
    "PROFILE_FIELDS": str,
    "BUILD_PROFILE_QA": bool,
    "BUILD_PROFILE_BACKSTORIES": bool,
    "ASSIGN_MANUALLY": str,
    "NUM_AGENTS_PER_SESSION": int,
    "MODULE_SEQUENCE": str,  # parsed further below
    "CONTEXT_OVERFLOW_POLICY": str,  # terminate | summarize | truncate
}

_OVERFLOW_POLICIES = {"terminate", "summarize", "truncate"}

_REQUIRED = {
    "EXPERIMENT_ID",
    "MODEL_NAME",
    "RANDOM_SEED",
    "NUM_AGENTS_PER_SESSION",
    "MODULE_SEQUENCE",
}

_BOOL_MAP = {
    "true": True,
    "false": False,
    "1": True,
    "0": False,
    "yes": True,
    "no": False,
}


def _coerce(key: str, raw_value: Any) -> Any:
    """Apply type coercion for a known settings key.

    Looks up the expected Python type in ``_FIELD_TYPES`` and attempts
    to cast *raw_value* accordingly. Boolean values are resolved via
    ``_BOOL_MAP`` for common string representations.

    Args:
        key: Upper-cased settings key (e.g. ``"RANDOM_SEED"``).
        raw_value: The raw cell value read from the worksheet.

    Returns:
        The coerced value, or ``None`` if the field is optional and
        empty.

    Raises:
        ValueError: If a required field is empty or if the value
            cannot be cast to the expected type.
    """
    dtype = _FIELD_TYPES.get(key)
    if dtype is None:
        return raw_value
    if pd.isna(raw_value) or raw_value == "":
        if key in _REQUIRED:
            raise ValueError(f"Settings: required field '{key}' is missing or empty.")
        return None

    if dtype is bool:
        if isinstance(raw_value, bool):
            return raw_value
        return _BOOL_MAP.get(str(raw_value).strip().lower(), bool(raw_value))

    try:
        return dtype(raw_value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Settings: cannot coerce '{key}' value '{raw_value}' to {dtype.__name__}: {exc}"
        ) from exc


def parse_settings(df: pd.DataFrame) -> SettingsConfig:
    """Parse a two-column DataFrame into a ``SettingsConfig``.

    The first column is treated as the setting name and the second as
    the value, regardless of header labels. Required fields are
    enforced and ``MODULE_SEQUENCE`` is split into an ordered list.

    Args:
        df: A DataFrame with at least two columns where column 0
            contains setting names and column 1 contains their values.

    Returns:
        A fully populated ``SettingsConfig`` instance.

    Raises:
        ValueError: If the sheet has fewer than two columns, a
            required field is missing, or ``CONTEXT_OVERFLOW_POLICY``
            contains an invalid value.
    """
    if df.shape[1] < 2:
        raise ValueError("Settings sheet must have at least two columns.")

    # Normalise: use first two columns regardless of header names
    name_col = df.columns[0]
    value_col = df.columns[1]

    settings_raw: dict[str, Any] = {}
    for _, row in df.iterrows():
        key = str(row[name_col]).strip().upper()
        value = row[value_col]
        if not key or key == "NAN":
            continue
        if key not in _FIELD_TYPES:
            warnings.warn(
                f"Settings: unknown key '{key}' — accepted but ignored.", stacklevel=2
            )
            continue
        settings_raw[key] = _coerce(key, value)

    # Check required fields
    missing = _REQUIRED - settings_raw.keys()
    if missing:
        raise ValueError(f"Settings: missing required field(s): {sorted(missing)}")

    # Parse MODULE_SEQUENCE from comma-separated string
    module_seq_raw = settings_raw.get("MODULE_SEQUENCE", "")
    module_sequence = (
        [t.strip() for t in str(module_seq_raw).split(",") if t.strip()]
        if module_seq_raw
        else []
    )

    # Validate CONTEXT_OVERFLOW_POLICY
    overflow_policy = (
        str(settings_raw.get("CONTEXT_OVERFLOW_POLICY") or "terminate").strip().lower()
    )
    if overflow_policy not in _OVERFLOW_POLICIES:
        raise ValueError(
            f"Settings: CONTEXT_OVERFLOW_POLICY must be one of {sorted(_OVERFLOW_POLICIES)}, "
            f"got '{overflow_policy}'."
        )

    # Validate ASSIGN_MANUALLY tokens
    assign_manually_raw = str(settings_raw.get("ASSIGN_MANUALLY") or "").strip()
    _VALID_MANUAL_TOKENS = {"treatment", "group"}
    if assign_manually_raw:
        tokens = {
            t.strip().lower() for t in assign_manually_raw.split(",") if t.strip()
        }
        invalid = tokens - _VALID_MANUAL_TOKENS
        if invalid:
            raise ValueError(
                f"Settings: ASSIGN_MANUALLY contains unrecognised token(s): {sorted(invalid)}. "
                f"Expected 'Treatment', 'Group', or 'Treatment, Group'."
            )

    return SettingsConfig(
        experiment_id=settings_raw["EXPERIMENT_ID"],
        model_name=settings_raw["MODEL_NAME"],
        hf_inference_endpoint=settings_raw.get("HF_INFERENCE_ENDPOINT") or "",
        temperature=settings_raw.get("TEMPERATURE", 0.0) or 0.0,
        random_seed=settings_raw["RANDOM_SEED"],
        profile_fields=settings_raw.get("PROFILE_FIELDS") or "ALL",
        build_profile_qa=settings_raw.get("BUILD_PROFILE_QA", False) or False,
        build_profile_backstories=settings_raw.get("BUILD_PROFILE_BACKSTORIES", False)
        or False,
        assign_manually=assign_manually_raw,
        num_agents_per_session=settings_raw["NUM_AGENTS_PER_SESSION"],
        module_sequence=module_sequence,
        context_overflow_policy=overflow_policy,
    )
