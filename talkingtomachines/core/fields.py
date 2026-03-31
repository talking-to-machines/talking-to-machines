"""Thread-safe field state management for experiment runtime.

Provides a thread-safe key-value store (``ExperimentState``) that tracks
runtime state at four scopes: Session, Agent, Group, and Player. Each
scope is keyed by a module name and a field name.

Getter/setter API::

    state = ExperimentState()
    state.set_player(player_id, module, "decision", 5)
    val = state.get_player(player_id, module, "decision")

The ``build_jinja_context`` method assembles the full Jinja2 rendering
context for a given agent/player/group/session at a specific round.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from talkingtomachines.core.models import Agent, Player, Group, Session


class ExperimentState:
    """
    Thread-safe key-value store for all field scopes.

    Internal storage layout::

        _session[module][name]                       → value
        _agent[agent_id][module][name]               → value
        _group[group_id][module][name]               → value
        _player[player_id][module][name]             → value
    """

    def __init__(self) -> None:
        """Initialise empty state stores and per-scope threading locks."""
        self._session: dict[str, dict[str, Any]] = {}
        self._agent: dict[str, dict[str, dict[str, Any]]] = {}
        self._group: dict[str, dict[str, dict[str, Any]]] = {}
        self._player: dict[str, dict[str, dict[str, Any]]] = {}

        # Per-scope locks
        self._session_lock = threading.Lock()
        self._agent_lock = threading.Lock()
        self._group_lock = threading.Lock()
        self._player_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Session scope
    # ------------------------------------------------------------------

    def set_session(self, module: str, name: str, value: Any) -> None:
        """Store a value at session scope.

        Args:
            module: Module identifier (e.g. ``"pgg"``).
            name: Field name within the module.
            value: The value to store.
        """
        with self._session_lock:
            self._session.setdefault(module, {})[name] = value

    def get_session(self, module: str, name: str, default: Any = None) -> Any:
        """Retrieve a session-scoped value.

        Args:
            module: Module identifier.
            name: Field name within the module.
            default: Value returned when the field is not set.

        Returns:
            The stored value, or *default* if not found.
        """
        with self._session_lock:
            return self._session.get(module, {}).get(name, default)

    def get_session_module(self, module: str) -> dict[str, Any]:
        """Return a snapshot of all session-scoped fields for a module.

        Args:
            module: Module identifier.

        Returns:
            A shallow copy of the field-name-to-value mapping for the
            given module. Returns an empty dict if no fields are stored.
        """
        with self._session_lock:
            return dict(self._session.get(module, {}))

    # ------------------------------------------------------------------
    # Agent scope
    # ------------------------------------------------------------------

    def set_agent(self, agent_id: str, module: str, name: str, value: Any) -> None:
        """Store a value at agent scope.

        Args:
            agent_id: Unique agent identifier.
            module: Module identifier.
            name: Field name within the module.
            value: The value to store.
        """
        with self._agent_lock:
            self._agent.setdefault(agent_id, {}).setdefault(module, {})[name] = value

    def get_agent(
        self, agent_id: str, module: str, name: str, default: Any = None
    ) -> Any:
        """Retrieve an agent-scoped value.

        Args:
            agent_id: Unique agent identifier.
            module: Module identifier.
            name: Field name within the module.
            default: Value returned when the field is not set.

        Returns:
            The stored value, or *default* if not found.
        """
        with self._agent_lock:
            return self._agent.get(agent_id, {}).get(module, {}).get(name, default)

    def get_agent_module(self, agent_id: str, module: str) -> dict[str, Any]:
        """Return a snapshot of all agent-scoped fields for a module.

        Args:
            agent_id: Unique agent identifier.
            module: Module identifier.

        Returns:
            A shallow copy of the field-name-to-value mapping.
        """
        with self._agent_lock:
            return dict(self._agent.get(agent_id, {}).get(module, {}))

    # ------------------------------------------------------------------
    # Group scope
    # ------------------------------------------------------------------

    def set_group(self, group_id: str, module: str, name: str, value: Any) -> None:
        """Store a value at group scope.

        Args:
            group_id: Unique group identifier.
            module: Module identifier.
            name: Field name within the module.
            value: The value to store.
        """
        with self._group_lock:
            self._group.setdefault(group_id, {}).setdefault(module, {})[name] = value

    def get_group(
        self, group_id: str, module: str, name: str, default: Any = None
    ) -> Any:
        """Retrieve a group-scoped value.

        Args:
            group_id: Unique group identifier.
            module: Module identifier.
            name: Field name within the module.
            default: Value returned when the field is not set.

        Returns:
            The stored value, or *default* if not found.
        """
        with self._group_lock:
            return self._group.get(group_id, {}).get(module, {}).get(name, default)

    def get_group_module(self, group_id: str, module: str) -> dict[str, Any]:
        """Return a snapshot of all group-scoped fields for a module.

        Args:
            group_id: Unique group identifier.
            module: Module identifier.

        Returns:
            A shallow copy of the field-name-to-value mapping.
        """
        with self._group_lock:
            return dict(self._group.get(group_id, {}).get(module, {}))

    # ------------------------------------------------------------------
    # Group scope — aggregation
    # ------------------------------------------------------------------

    def accumulate_group(
        self, group_id: str, module: str, name: str, player_id: str, value: Any
    ) -> None:
        """Accumulate a per-player value into a group-scoped dict keyed by player ID.

        If the field does not yet exist or is not a dict, it is
        initialised as an empty dict before the value is inserted.

        Args:
            group_id: Unique group identifier.
            module: Module identifier.
            name: Field name that will hold the aggregated dict.
            player_id: Key under which the value is stored.
            value: The per-player value to record.
        """
        with self._group_lock:
            bucket = self._group.setdefault(group_id, {}).setdefault(module, {})
            if name not in bucket or not isinstance(bucket[name], dict):
                bucket[name] = {}
            bucket[name][player_id] = value

    # ------------------------------------------------------------------
    # Session scope — aggregation
    # ------------------------------------------------------------------

    def accumulate_session(
        self, module: str, name: str, player_id: str, value: Any
    ) -> None:
        """Accumulate a per-player value into a session-scoped dict keyed by player ID.

        If the field does not yet exist or is not a dict, it is
        initialised as an empty dict before the value is inserted.

        Args:
            module: Module identifier.
            name: Field name that will hold the aggregated dict.
            player_id: Key under which the value is stored.
            value: The per-player value to record.
        """
        with self._session_lock:
            bucket = self._session.setdefault(module, {})
            if name not in bucket or not isinstance(bucket[name], dict):
                bucket[name] = {}
            bucket[name][player_id] = value

    # ------------------------------------------------------------------
    # Player scope
    # ------------------------------------------------------------------

    def set_player(self, player_id: str, module: str, name: str, value: Any) -> None:
        """Store a value at player scope.

        Args:
            player_id: Unique player identifier.
            module: Module identifier.
            name: Field name within the module.
            value: The value to store.
        """
        with self._player_lock:
            self._player.setdefault(player_id, {}).setdefault(module, {})[name] = value

    def get_player(
        self, player_id: str, module: str, name: str, default: Any = None
    ) -> Any:
        """Retrieve a player-scoped value.

        Args:
            player_id: Unique player identifier.
            module: Module identifier.
            name: Field name within the module.
            default: Value returned when the field is not set.

        Returns:
            The stored value, or *default* if not found.
        """
        with self._player_lock:
            return self._player.get(player_id, {}).get(module, {}).get(name, default)

    def get_player_module(self, player_id: str, module: str) -> dict[str, Any]:
        """Return a snapshot of all player-scoped fields for a module.

        Args:
            player_id: Unique player identifier.
            module: Module identifier.

        Returns:
            A shallow copy of the field-name-to-value mapping.
        """
        with self._player_lock:
            return dict(self._player.get(player_id, {}).get(module, {}))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize all scoped state to a plain dict.

        Returns:
            A dictionary with keys ``"session"``, ``"agent"``, ``"group"``,
            and ``"player"``, each containing the nested field-value
            mappings for that scope.
        """
        with self._session_lock, self._agent_lock, self._group_lock, self._player_lock:
            return {
                "session": self._session,
                "agent": self._agent,
                "group": self._group,
                "player": self._player,
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentState":
        """Reconstruct an ``ExperimentState`` from a serialized dict.

        Args:
            data: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            A new ``ExperimentState`` with all scoped values restored.
        """
        state = cls()
        state._session = data.get("session", {})
        state._agent = data.get("agent", {})
        state._group = data.get("group", {})
        state._player = data.get("player", {})
        return state

    # ------------------------------------------------------------------
    # Jinja2 context construction
    # ------------------------------------------------------------------

    def build_jinja_context(
        self,
        agent: Agent,
        player: Player,
        group: Group,
        session: Session,
        module: str,
        round_number: int,
        constants: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Assemble the full Jinja2 rendering context for a given turn.

        Merges profile information, runtime state from all four scopes,
        constants, and treatment metadata into a single dict that is
        passed as ``**context`` to a Jinja2 ``SandboxedEnvironment``.

        Args:
            agent: The ``Agent`` model instance for the current turn.
            player: The ``Player`` model instance for the current turn.
            group: The ``Group`` the player belongs to this round.
            session: The ``Session`` model instance.
            module: Module identifier used to look up scoped fields.
            round_number: The current round number (1-based).
            constants: Optional nested dict of module-level constants,
                exposed in templates as ``C.<module>.<name>``.

        Returns:
            A dict suitable for Jinja2 template rendering, containing
            namespaced keys ``player``, ``agent``, ``group``,
            ``session``, ``C``, ``treatment``, ``round_number``,
            ``group_id``, ``session_id``, and ``run_id``.
        """
        # Profile fields accessible as ``player.<field>``
        player_profile: dict[str, Any] = dict(agent.profile_info)

        # Runtime player state
        player_state = self.get_player_module(player.player_id, module)
        player_profile.update(player_state)

        # Built-in player model attributes
        player_profile.setdefault("player_id", player.player_id)
        player_profile.setdefault("agent_id", agent.agent_id)
        player_profile.setdefault("agent_instance_id", player.agent_instance_id)
        player_profile.setdefault("treatment", agent.treatment_label)

        # Agent-level fields
        agent_fields = self.get_agent_module(agent.agent_id, module)
        # Built-in agent model attributes
        agent_fields.setdefault("agent_id", agent.agent_id)
        agent_fields.setdefault("agent_instance_id", agent.agent_instance_id)
        agent_fields.setdefault("treatment", agent.treatment_label)

        # Group-level fields
        group_fields = self.get_group_module(group.group_id, module)
        # Built-in group model attributes
        group_fields.setdefault("group_id", group.group_id)
        group_fields.setdefault("num_players", len(group.players))

        # Session-level fields
        session_fields = self.get_session_module(module)
        # Built-in session model attributes
        session_fields.setdefault("session_id", session.session_id)
        session_fields.setdefault("run_id", session.run_id)
        session_fields.setdefault("experiment_id", session.experiment_id)

        ctx: dict[str, Any] = {
            "round_number": round_number,
            "group_id": group.group_id,
            "session_id": session.session_id,
            "run_id": session.run_id,
            # Namespaced access: player.<field>
            "player": _Namespace(player_profile),
            # Agent-level
            "agent": _Namespace(agent_fields),
            # Group-level
            "group": _Namespace(group_fields),
            # Session-level
            "session": _Namespace(session_fields),
        }

        # Module-level constant access: C.<module>.<name>
        if constants:
            ctx["C"] = _DeepNamespace(constants)

        # Treatment / role shortcut
        ctx["treatment"] = agent.treatment_label

        return ctx


class _Namespace:
    """Simple attribute-access wrapper over a dict for Jinja2 templates.

    Allows Jinja2 expressions like ``player.decision`` to resolve to
    dictionary lookups.

    Args:
        data: The underlying dict whose values are exposed as attributes.
    """

    def __init__(self, data: dict):
        """Initialise the namespace with the given data dict.

        Args:
            data: Dict whose keys become attribute names.
        """
        self._data = data

    def __getattr__(self, name: str) -> Any:
        """Look up *name* in the underlying dict.

        Args:
            name: Attribute name to resolve.

        Returns:
            The value associated with *name*.

        Raises:
            AttributeError: If *name* is not present in the dict.
        """
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"No field '{name}'")

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return f"Namespace({self._data})"


class _DeepNamespace:
    """Two-level attribute-access namespace for constants (``C.<module>.<name>``).

    Wraps a dict-of-dicts so that the outer keys are resolved as
    attributes returning ``_Namespace`` instances for the inner dicts.

    Args:
        data: A dict mapping module names to dicts of constant values.
    """

    def __init__(self, data: dict):
        """Initialise with nested dicts wrapped as ``_Namespace`` objects.

        Args:
            data: Mapping of module names to field-value dicts.
        """
        self._data = {module: _Namespace(vals) for module, vals in data.items()}

    def __getattr__(self, name: str) -> Any:
        """Resolve a module name to its ``_Namespace``.

        Args:
            name: Module name to look up.

        Returns:
            A ``_Namespace`` instance for the requested module.

        Raises:
            AttributeError: If no constants exist for the given module.
        """
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"No constant module '{name}'")
