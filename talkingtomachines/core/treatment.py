"""Treatment definition and parsing utilities.

Provides the ``TreatmentDefinition`` dataclass and helpers for parsing
treatment specifications from the compiled experiment package. This module
acts as the canonical location for treatment-related logic, separating it
from the broader ``RandomisationEngine``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TreatmentDefinition:
    """
    Represents a single treatment arm.

    Attributes:
        label:       Short identifier (e.g. "control", "treatment_A").
        description: Human-readable description shown in profiles/prompts.
        attributes:  Arbitrary key-value attributes (passed in via Jinja context as
                     ``{{ treatment.attribute_name }}``).
    """

    label: str
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise the treatment definition to a plain dict.

        Returns:
            A dict with keys ``label``, ``description``, and ``attributes``.
        """
        return {
            "label": self.label,
            "description": self.description,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TreatmentDefinition":
        """Construct a ``TreatmentDefinition`` from a plain dict.

        Args:
            data: Dict containing at minimum a ``label`` key, and
                optionally ``description`` and ``attributes``.

        Returns:
            A new ``TreatmentDefinition`` instance.
        """
        return cls(
            label=data["label"],
            description=data.get("description", ""),
            attributes=data.get("attributes", {}),
        )

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to ``attributes`` dict entries.

        Args:
            name: The attribute name to look up.

        Returns:
            The value stored in ``self.attributes[name]``.

        Raises:
            AttributeError: If *name* is not found in the ``attributes``
                dict.
        """
        attrs = object.__getattribute__(self, "attributes")
        if name in attrs:
            return attrs[name]
        raise AttributeError(f"TreatmentDefinition has no attribute '{name}'")


def parse_treatments_from_settings(settings: dict) -> dict[str, TreatmentDefinition]:
    """Parse treatment definitions from a settings dict produced by the compiler.

    Supports two formats: a list of treatment dicts (preferred) and a legacy
    dict format where keys are labels and values are description strings or
    nested dicts.

    Expected list format in settings::

        treatments:
          - label: "control"
            description: "Control group"
            attributes: {}
          - label: "treatment_A"
            description: "Receives information"
            attributes: {info_type: "full"}

    Args:
        settings: The experiment settings dict, expected to contain a
            ``treatments`` key.

    Returns:
        A dict mapping treatment labels to ``TreatmentDefinition``
        instances. Returns an empty dict if no treatments are found.
    """
    raw_treatments = settings.get("treatments", [])
    result: dict[str, TreatmentDefinition] = {}

    if isinstance(raw_treatments, list):
        for item in raw_treatments:
            if isinstance(item, dict) and "label" in item:
                td = TreatmentDefinition.from_dict(item)
                result[td.label] = td
    elif isinstance(raw_treatments, dict):
        # Support legacy dict format: {label: description_str}
        for label, val in raw_treatments.items():
            if isinstance(val, str):
                result[label] = TreatmentDefinition(label=label, description=val)
            elif isinstance(val, dict):
                result[label] = TreatmentDefinition(
                    label=label,
                    description=val.get("description", ""),
                    attributes={k: v for k, v in val.items() if k != "description"},
                )

    return result
