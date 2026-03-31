"""
Randomisation Engine

Implements all treatment and group assignment strategies as a first-class,
logged, replayable component.

Child seeds are derived deterministically using HMAC-SHA256:
    child_seed = HMAC-SHA256(global_seed, path_components)

Every randomisation event is logged with:
    event_type, seed_used, inputs, realized_output, timestamp
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


@dataclass
class RandomisationEvent:
    """A single logged randomisation event for auditability and replay.

    Attributes:
        event_type: Category of the randomisation event (e.g.
            ``"treatment_assignment"``, ``"group_assignment"``,
            ``"turn_order_shuffle"``, ``"options_shuffle"``).
        seed_used: The derived seed used for this specific event.
        inputs: Dictionary of input parameters that were passed to
            the randomisation function.
        realized_output: The concrete output produced by the
            randomisation (e.g. assignment mapping, shuffled list).
        timestamp: ISO-8601 UTC timestamp of when the event occurred.
    """

    event_type: str
    seed_used: int
    inputs: dict
    realized_output: Any
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        """Serialise the event to a plain dictionary.

        Returns:
            A dictionary containing all event fields.
        """
        return {
            "event_type": self.event_type,
            "seed_used": self.seed_used,
            "inputs": self.inputs,
            "realized_output": self.realized_output,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


def _derive_seed(global_seed: int, path: str) -> int:
    """Deterministically derive a child seed using HMAC-SHA256.

    Combines the global seed with a path string to produce a
    reproducible child seed. The first 8 bytes of the HMAC digest
    are converted to an integer.

    Args:
        global_seed: The experiment-level random seed.
        path: A dot-delimited path string that uniquely identifies
            the randomisation context (e.g.
            ``"turn_order.g1.round_1"``).

    Returns:
        A deterministic integer seed derived from the inputs.
    """
    key = str(global_seed).encode()
    msg = path.encode()
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    # Convert first 8 bytes to int
    return int.from_bytes(digest[:8], byteorder="big")


# ---------------------------------------------------------------------------
# Randomisation Engine
# ---------------------------------------------------------------------------


class RandomisationEngine:
    """Centralised, logged, replayable randomisation engine.

    All randomisation calls are deterministic given the same
    ``global_seed`` and the same sequence of calls. Every operation
    is recorded as a ``RandomisationEvent`` for auditability and
    replay.

    Attributes:
        _global_seed: The experiment-level random seed from which
            all child seeds are derived.
        _events: Accumulated list of randomisation events.
    """

    def __init__(self, global_seed: int) -> None:
        """Initialise the randomisation engine.

        Args:
            global_seed: The experiment-level random seed used to
                derive all child seeds via HMAC-SHA256.
        """
        self._global_seed = global_seed
        self._events: list[RandomisationEvent] = []

    @property
    def events(self) -> list[RandomisationEvent]:
        """Return a copy of all recorded randomisation events.

        Returns:
            A list of ``RandomisationEvent`` instances.
        """
        return list(self._events)

    def _rng(self, path: str) -> random.Random:
        """Create a seeded ``Random`` instance for a given path.

        Args:
            path: Dot-delimited path string identifying the
                randomisation context.

        Returns:
            A ``random.Random`` instance seeded deterministically.
        """
        seed = _derive_seed(self._global_seed, path)
        return random.Random(seed)

    def _log(self, event_type: str, seed: int, inputs: dict, output: Any) -> None:
        """Record a randomisation event.

        Args:
            event_type: Category label for the event.
            seed: The derived seed that was used.
            inputs: Dictionary of input parameters.
            output: The realised output of the randomisation.
        """
        self._events.append(
            RandomisationEvent(
                event_type=event_type,
                seed_used=seed,
                inputs=inputs,
                realized_output=output,
            )
        )

    # ------------------------------------------------------------------
    # Treatment assignment strategies
    # ------------------------------------------------------------------

    def assign_treatments(
        self,
        agent_ids: list[str],
        treatment_labels: list[str],
        strategy: str = "simple_random",
        path: str = "treatments",
        manual_assignments: Optional[dict[str, str]] = None,
        stratify_by: Optional[list] = None,
        block_size: Optional[int] = None,
    ) -> dict[str, str]:
        """Assign treatments to agents using the specified strategy.

        Supported strategies: ``"simple_random"``,
        ``"complete_random"``, ``"block_random"``, ``"stratified"``,
        ``"full_factorial"``, ``"wave"``, ``"manual"``.

        Args:
            agent_ids: List of agent identifiers to assign.
            treatment_labels: Available treatment condition labels.
            strategy: Assignment strategy name.
            path: Seed derivation path for deterministic
                randomisation.
            manual_assignments: Pre-defined assignments for the
                ``"manual"`` strategy (``{agent_id: label}``).
            stratify_by: Profile attribute values aligned with
                *agent_ids* for the ``"stratified"`` strategy.
            block_size: Number of agents per block for the
                ``"block_random"`` strategy.

        Returns:
            A dictionary mapping each agent ID to its assigned
            treatment label.
        """
        seed = _derive_seed(self._global_seed, path)
        rng = random.Random(seed)

        if strategy == "simple_random":
            result = self._simple_random(agent_ids, treatment_labels, rng)
        elif strategy == "complete_random":
            result = self._complete_random(agent_ids, treatment_labels, rng)
        elif strategy == "block_random":
            result = self._block_random(
                agent_ids, treatment_labels, rng, block_size or len(treatment_labels)
            )
        elif strategy == "stratified":
            result = self._stratified(
                agent_ids, treatment_labels, rng, stratify_by or []
            )
        elif strategy == "full_factorial":
            result = self._full_factorial(agent_ids, treatment_labels, rng)
        elif strategy == "wave":
            result = self._wave(agent_ids, treatment_labels, rng)
        elif strategy == "manual":
            result = manual_assignments or {}
        else:
            logger.warning(
                "Unknown treatment strategy '%s', falling back to simple_random.",
                strategy,
            )
            result = self._simple_random(agent_ids, treatment_labels, rng)

        self._log(
            "treatment_assignment",
            seed,
            {
                "strategy": strategy,
                "agent_ids": agent_ids,
                "treatment_labels": treatment_labels,
            },
            result,
        )
        return result

    def _simple_random(
        self, agent_ids: list[str], labels: list[str], rng: random.Random
    ) -> dict[str, str]:
        """Assign each agent independently with equal probability.

        Args:
            agent_ids: List of agent identifiers.
            labels: Available treatment labels.
            rng: Seeded random number generator.

        Returns:
            A dictionary mapping agent IDs to treatment labels.
        """
        return {aid: rng.choice(labels) for aid in agent_ids}

    def _complete_random(
        self, agent_ids: list[str], labels: list[str], rng: random.Random
    ) -> dict[str, str]:
        """Assign with balanced counts via round-robin on a shuffled list.

        Args:
            agent_ids: List of agent identifiers.
            labels: Available treatment labels.
            rng: Seeded random number generator.

        Returns:
            A dictionary mapping agent IDs to treatment labels with
            approximately equal counts per label.
        """
        shuffled = list(agent_ids)
        rng.shuffle(shuffled)
        result: dict[str, str] = {}
        for i, aid in enumerate(shuffled):
            result[aid] = labels[i % len(labels)]
        return result

    def _block_random(
        self,
        agent_ids: list[str],
        labels: list[str],
        rng: random.Random,
        block_size: int,
    ) -> dict[str, str]:
        """Assign treatments randomly within fixed-size blocks.

        Each block of agents receives a shuffled set of labels,
        ensuring local balance within every block.

        Args:
            agent_ids: List of agent identifiers.
            labels: Available treatment labels.
            rng: Seeded random number generator.
            block_size: Number of agents per block.

        Returns:
            A dictionary mapping agent IDs to treatment labels.
        """
        result: dict[str, str] = {}
        agents = list(agent_ids)
        for i in range(0, len(agents), block_size):
            block = agents[i : i + block_size]
            block_labels = [labels[j % len(labels)] for j in range(len(block))]
            rng.shuffle(block_labels)
            for aid, label in zip(block, block_labels):
                result[aid] = label
        return result

    def _stratified(
        self, agent_ids: list[str], labels: list[str], rng: random.Random, strata: list
    ) -> dict[str, str]:
        """Assign treatments balanced within each stratum.

        Agents are grouped by their stratum value, then assigned
        labels via round-robin within each group. Falls back to
        simple random if the strata length does not match
        *agent_ids*.

        Args:
            agent_ids: List of agent identifiers.
            labels: Available treatment labels.
            rng: Seeded random number generator.
            strata: Profile attribute values aligned with
                *agent_ids* for stratification.

        Returns:
            A dictionary mapping agent IDs to treatment labels.
        """
        if len(strata) != len(agent_ids):
            logger.warning(
                "Stratified assignment: strata length mismatch — falling back to simple_random."
            )
            return self._simple_random(agent_ids, labels, rng)

        # Group agents by stratum
        stratum_groups: dict[Any, list[str]] = {}
        for aid, stratum in zip(agent_ids, strata):
            stratum_groups.setdefault(stratum, []).append(aid)

        result: dict[str, str] = {}
        for stratum_ids in stratum_groups.values():
            rng.shuffle(stratum_ids)
            for i, aid in enumerate(stratum_ids):
                result[aid] = labels[i % len(labels)]
        return result

    def _full_factorial(
        self, agent_ids: list[str], labels: list[str], rng: random.Random
    ) -> dict[str, str]:
        """Assign all treatment combinations via full factorial design.

        Shuffles agents and assigns labels in round-robin order to
        cover all treatment conditions.

        Args:
            agent_ids: List of agent identifiers.
            labels: Available treatment labels.
            rng: Seeded random number generator.

        Returns:
            A dictionary mapping agent IDs to treatment labels.
        """
        result: dict[str, str] = {}
        shuffled = list(agent_ids)
        rng.shuffle(shuffled)
        for i, aid in enumerate(shuffled):
            result[aid] = labels[i % len(labels)]
        return result

    def _wave(
        self, agent_ids: list[str], labels: list[str], rng: random.Random
    ) -> dict[str, str]:
        """Assign treatments sequentially in waves.

        Agents are divided into equal-sized waves, with each wave
        receiving the next treatment label in order.

        Args:
            agent_ids: List of agent identifiers.
            labels: Available treatment labels.
            rng: Seeded random number generator (unused but kept
                for interface consistency).

        Returns:
            A dictionary mapping agent IDs to treatment labels.
        """
        result: dict[str, str] = {}
        wave_size = max(1, len(agent_ids) // len(labels))
        for i, aid in enumerate(agent_ids):
            wave_index = i // wave_size
            result[aid] = labels[wave_index % len(labels)]
        return result

    # ------------------------------------------------------------------
    # Group assignment strategies
    # ------------------------------------------------------------------

    def assign_groups(
        self,
        agent_ids: list[str],
        players_per_group: int,
        strategy: str = "random",
        path: str = "groups",
        previous_groups: Optional[dict] = None,
        stratify_by: Optional[list] = None,
        manual_groups: Optional[dict] = None,
    ) -> dict[str, list[str]]:
        """Assign agents to groups using the specified strategy.

        Supported strategies: ``"random"``, ``"keep"``, ``"swap"``,
        ``"stratified"``, ``"manual"``.

        Args:
            agent_ids: List of agent identifiers to assign.
            players_per_group: Target number of agents per group.
            strategy: Group formation strategy name.
            path: Seed derivation path for deterministic
                randomisation.
            previous_groups: Group assignments from the previous
                round, used by ``"keep"`` and ``"swap"`` strategies.
            stratify_by: Profile attribute values aligned with
                *agent_ids* for the ``"stratified"`` strategy.
            manual_groups: Pre-defined group assignments for the
                ``"manual"`` strategy.

        Returns:
            A dictionary mapping group labels to lists of agent IDs
            (``{group_label: [agent_id, ...]}``).
        """
        seed = _derive_seed(self._global_seed, path)
        rng = random.Random(seed)

        if strategy == "random":
            result = self._random_groups(agent_ids, players_per_group, rng)
        elif strategy == "keep":
            result = previous_groups or self._random_groups(
                agent_ids, players_per_group, rng
            )
        elif strategy == "swap":
            result = self._swap_groups(
                agent_ids, players_per_group, rng, previous_groups
            )
        elif strategy == "stratified":
            result = self._stratified_groups(
                agent_ids, players_per_group, rng, stratify_by or []
            )
        elif strategy == "manual":
            result = manual_groups or {}
        else:
            logger.warning(
                "Unknown group strategy '%s', falling back to random.", strategy
            )
            result = self._random_groups(agent_ids, players_per_group, rng)

        self._log(
            "group_assignment",
            seed,
            {
                "strategy": strategy,
                "agent_ids": agent_ids,
                "players_per_group": players_per_group,
            },
            result,
        )
        return result

    def _random_groups(
        self, agent_ids: list[str], size: int, rng: random.Random
    ) -> dict[str, list[str]]:
        """Form groups by shuffling agents and chunking into fixed sizes.

        Args:
            agent_ids: List of agent identifiers.
            size: Target number of agents per group.
            rng: Seeded random number generator.

        Returns:
            A dictionary mapping group labels (``"g1"``, ``"g2"``,
            ...) to lists of agent IDs.
        """
        shuffled = list(agent_ids)
        rng.shuffle(shuffled)
        groups: dict[str, list[str]] = {}
        for i, chunk_start in enumerate(range(0, len(shuffled), size)):
            chunk = shuffled[chunk_start : chunk_start + size]
            if chunk:
                groups[f"g{i + 1}"] = chunk
        return groups

    def _swap_groups(
        self,
        agent_ids: list[str],
        size: int,
        rng: random.Random,
        previous: Optional[dict],
    ) -> dict[str, list[str]]:
        """Re-form groups by swapping approximately 50% of members.

        Retains half of each previous group's members and
        redistributes the remaining agents randomly. Falls back to
        fully random groups if no previous assignments exist.

        Args:
            agent_ids: List of agent identifiers.
            size: Target number of agents per group.
            rng: Seeded random number generator.
            previous: Previous round's group assignments
                (``{group_label: [agent_id, ...]}``).

        Returns:
            A dictionary mapping group labels to lists of agent IDs.
        """
        if not previous:
            return self._random_groups(agent_ids, size, rng)
        # Simple implementation: re-randomise 50% of agents
        fixed_count = max(1, size // 2)
        new_groups: dict[str, list[str]] = {}
        agents_pool = list(agent_ids)
        rng.shuffle(agents_pool)
        for i, (gid, members) in enumerate(previous.items()):
            kept = members[:fixed_count]
            new_groups[gid] = kept
        # Distribute remaining agents
        assigned = {a for members in new_groups.values() for a in members}
        remaining = [a for a in agents_pool if a not in assigned]
        idx = 0
        for gid in new_groups:
            while len(new_groups[gid]) < size and idx < len(remaining):
                new_groups[gid].append(remaining[idx])
                idx += 1
        return new_groups

    def _stratified_groups(
        self,
        agent_ids: list[str],
        size: int,
        rng: random.Random,
        strata: list,
    ) -> dict[str, list[str]]:
        """Form groups balanced by a profile attribute.

        Agents are bucketed by stratum, shuffled within each bucket,
        then interleaved so that each group receives a mix of strata.
        Falls back to random groups if the strata length does not
        match *agent_ids*.

        Args:
            agent_ids: List of agent identifiers.
            size: Target number of agents per group.
            rng: Seeded random number generator.
            strata: Profile attribute values aligned with
                *agent_ids* for stratification.

        Returns:
            A dictionary mapping group labels to lists of agent IDs.
        """
        if len(strata) != len(agent_ids):
            return self._random_groups(agent_ids, size, rng)

        stratum_buckets: dict[Any, list[str]] = {}
        for aid, stratum in zip(agent_ids, strata):
            stratum_buckets.setdefault(stratum, []).append(aid)

        for bucket in stratum_buckets.values():
            rng.shuffle(bucket)

        # Interleave strata into groups
        interleaved: list[str] = []
        strata_lists = list(stratum_buckets.values())
        max_len = max(len(sl) for sl in strata_lists)
        for i in range(max_len):
            for sl in strata_lists:
                if i < len(sl):
                    interleaved.append(sl[i])

        groups: dict[str, list[str]] = {}
        for i, chunk_start in enumerate(range(0, len(interleaved), size)):
            chunk = interleaved[chunk_start : chunk_start + size]
            if chunk:
                groups[f"g{i + 1}"] = chunk
        return groups

    # ------------------------------------------------------------------
    # Turn order and option shuffling
    # ------------------------------------------------------------------

    def shuffle_turn_order(
        self,
        agent_instance_ids: list[str],
        group_id: str,
        round_number: int,
    ) -> list[str]:
        """Shuffle the speaking order within a group for a given round.

        Deterministic: the same inputs always produce the same order.

        Args:
            agent_instance_ids: List of agent instance identifiers
                in the group.
            group_id: The group identifier.
            round_number: The current round number.

        Returns:
            A new list of agent instance IDs in shuffled order.
        """
        path = f"turn_order.{group_id}.round_{round_number}"
        seed = _derive_seed(self._global_seed, path)
        rng = random.Random(seed)
        shuffled = list(agent_instance_ids)
        rng.shuffle(shuffled)
        self._log(
            "turn_order_shuffle",
            seed,
            {
                "group_id": group_id,
                "round_number": round_number,
                "original_order": agent_instance_ids,
            },
            shuffled,
        )
        return shuffled

    def shuffle_options(
        self,
        options: list,
        player_id: str,
        prompt_name: str,
    ) -> list:
        """Shuffle response options for a specific player and prompt.

        Deterministic: the same inputs always produce the same
        shuffle.

        Args:
            options: List of response options to shuffle.
            player_id: The player identifier requesting the options.
            prompt_name: Name of the prompt associated with the
                options.

        Returns:
            A new list with the options in shuffled order.
        """
        path = f"options.{player_id}.{prompt_name}"
        seed = _derive_seed(self._global_seed, path)
        rng = random.Random(seed)
        shuffled = list(options)
        rng.shuffle(shuffled)
        self._log(
            "options_shuffle",
            seed,
            {
                "player_id": player_id,
                "prompt_name": prompt_name,
                "original_options": options,
            },
            shuffled,
        )
        return shuffled
