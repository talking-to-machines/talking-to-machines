"""Conjoint design generator for attribute-level experiments.

Generates attribute-level conjoint profiles using a deterministic seed
derived from ``player_id + round_number``, ensuring reproducibility
across runs. Profiles are formatted as a markdown comparison table
suitable for injection into LLM prompts.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any


def _derive_seed(player_id: str, round_number: int) -> int:
    """Derive a deterministic integer seed from a player ID and round number.

    Uses SHA-256 hashing to produce a reproducible seed so that the same
    player/round combination always yields the same conjoint profile.

    Args:
        player_id: Unique identifier for the player (or a composite key
            that includes a profile suffix).
        round_number: The current round number in the experiment.

    Returns:
        A non-negative integer suitable for seeding a random generator.
    """
    key = f"{player_id}_{round_number}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], byteorder="big")


class ConjointDesigner:
    """
    Generates conjoint attribute profiles deterministically for each agent/round.

    Usage::

        designer = ConjointDesigner(attributes={
            "party": ["Democrat", "Republican"],
            "age": [30, 45, 60],
            "gender": ["Male", "Female"],
        })
        table = designer.generate_table(player_id="run_a1", round_number=2, num_profiles=2)
        # Returns a formatted markdown table string for injection into LLM prompts
    """

    def __init__(self, attributes: dict[str, list[Any]]):
        """
        Args:
            attributes: mapping of attribute name → list of possible levels.
        """
        self._attributes = attributes

    def generate_profile(
        self, player_id: str, round_number: int, profile_number: int = 1
    ) -> dict[str, Any]:
        """Generate a single conjoint profile.

        Args:
            player_id: Unique identifier for the player.
            round_number: The current round number.
            profile_number: Index of the profile within a multi-profile
                comparison (1-based). Defaults to ``1``.

        Returns:
            A dict mapping each attribute name to its randomly selected
            level for this profile.
        """
        seed = _derive_seed(f"{player_id}_profile_{profile_number}", round_number)
        rng = random.Random(seed)
        return {attr: rng.choice(levels) for attr, levels in self._attributes.items()}

    def generate_table(
        self,
        player_id: str,
        round_number: int,
        num_profiles: int = 2,
    ) -> str:
        """Generate multiple conjoint profiles and format as a markdown table.

        Args:
            player_id: Unique identifier for the player.
            round_number: The current round number.
            num_profiles: Number of profiles to generate for side-by-side
                comparison. Defaults to ``2``.

        Returns:
            A markdown-formatted table string with one row per attribute
            and one column per profile, suitable for injection into an
            LLM prompt.
        """
        profiles = [
            self.generate_profile(player_id, round_number, i + 1)
            for i in range(num_profiles)
        ]

        # Build markdown table
        headers = ["Attribute"] + [f"Profile {i + 1}" for i in range(num_profiles)]
        rows = []
        for attr in self._attributes:
            row = [attr] + [str(p[attr]) for p in profiles]
            rows.append(row)

        col_widths = [
            max(len(headers[i]), max(len(r[i]) for r in rows))
            for i in range(len(headers))
        ]

        def fmt_row(cells: list[str]) -> str:
            """Format a list of cell strings into a padded markdown table row."""
            return (
                "| "
                + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))
                + " |"
            )

        separator = "| " + " | ".join("-" * w for w in col_widths) + " |"

        lines = [fmt_row(headers), separator] + [fmt_row(r) for r in rows]
        return "\n".join(lines)
