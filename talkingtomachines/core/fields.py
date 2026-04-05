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

        _session[module][name]                                → value
        _agent[agent_id][module][round_number][name]          → value
        _group[group_id][module][name]                        → value
        _player[player_id][module][name]                      → value
    """

    def __init__(self) -> None:
        """Initialise empty state stores and per-scope threading locks."""
        self._session: dict[str, dict[str, Any]] = {}
        # Agent scope: agent_id → module → round_number → field_name → value
        self._agent: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
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

    def set_agent(
        self, agent_id: str, module: str, name: str, value: Any, round_number: int = 1
    ) -> None:
        """Store a value at agent scope, indexed by module and round.

        Args:
            agent_id: Unique agent identifier.
            module: Module identifier.
            name: Field name within the module/round.
            value: The value to store.
            round_number: Round number (1-based). Defaults to 1.
        """
        with self._agent_lock:
            (
                self._agent.setdefault(agent_id, {})
                .setdefault(module, {})
                .setdefault(round_number, {})
            )[name] = value

    def get_agent(
        self,
        agent_id: str,
        module: str,
        name: str,
        round_number: int = 1,
        default: Any = None,
    ) -> Any:
        """Retrieve an agent-scoped value for a specific module and round.

        Args:
            agent_id: Unique agent identifier.
            module: Module identifier.
            name: Field name within the module/round.
            round_number: Round number (1-based). Defaults to 1.
            default: Value returned when the field is not set.

        Returns:
            The stored value, or *default* if not found.
        """
        with self._agent_lock:
            return (
                self._agent.get(agent_id, {})
                .get(module, {})
                .get(round_number, {})
                .get(name, default)
            )

    def get_agent_round(
        self, agent_id: str, module: str, round_number: int
    ) -> dict[str, Any]:
        """Return a snapshot of all agent-scoped fields for a specific round.

        Args:
            agent_id: Unique agent identifier.
            module: Module identifier.
            round_number: Round number (1-based).

        Returns:
            A shallow copy of the field-name-to-value mapping for that round.
        """
        with self._agent_lock:
            return dict(
                self._agent.get(agent_id, {}).get(module, {}).get(round_number, {})
            )

    def get_agent_module(self, agent_id: str, module: str) -> dict[int, dict[str, Any]]:
        """Return a snapshot of all agent-scoped fields for a module across rounds.

        Args:
            agent_id: Unique agent identifier.
            module: Module identifier.

        Returns:
            A dict mapping round_number to field-name-to-value dicts.
        """
        with self._agent_lock:
            module_data = self._agent.get(agent_id, {}).get(module, {})
            return {rnd: dict(fields) for rnd, fields in module_data.items()}

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

        Note:
            Agent scope round numbers (int keys) are converted to strings
            for JSON compatibility during serialization.
        """
        with self._session_lock, self._agent_lock, self._group_lock, self._player_lock:
            # Convert agent int round keys to strings for JSON compat
            agent_serialized: dict = {}
            for aid, modules in self._agent.items():
                agent_serialized[aid] = {}
                for mod, rounds in modules.items():
                    agent_serialized[aid][mod] = {
                        str(rnd): dict(fields) for rnd, fields in rounds.items()
                    }
            return {
                "session": self._session,
                "agent": agent_serialized,
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
        # Convert agent string round keys back to ints
        raw_agent = data.get("agent", {})
        for aid, modules in raw_agent.items():
            state._agent[aid] = {}
            for mod, rounds in modules.items():
                state._agent[aid][mod] = {}
                for rnd_str, fields in rounds.items():
                    try:
                        rnd = int(rnd_str)
                    except (ValueError, TypeError):
                        rnd = rnd_str
                    state._agent[aid][mod][rnd] = fields
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
            ``session``, ``C``, ``round_number``, ``group_id``,
            ``session_id``, and ``run_id``.
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

        # Agent-level fields: module → round → field namespace
        agent_all = self._agent.get(agent.agent_id, {})
        agent_ns: dict[str, Any] = {}
        for mod, rounds_data in agent_all.items():
            agent_ns[mod] = _RoundedModuleNamespace(rounds_data)

        # Current round's agent-scoped values as flat attributes
        # (Manual_ Agent entries accessible as agent.<field>)
        agent_round_data = self.get_agent_round(agent.agent_id, module, round_number)
        for k, v in agent_round_data.items():
            agent_ns.setdefault(k, v)

        # Profile fields as flat agent attributes: agent.age, agent.gender, etc.
        for k, v in agent.profile_info.items():
            agent_ns.setdefault(k, v)

        # Built-in flat attributes (always override)
        agent_ns["agent_id"] = agent.agent_id
        agent_ns["agent_instance_id"] = agent.agent_instance_id

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
            # Agent-level: agent.module[round].field or agent.agent_id
            "agent": _Namespace(agent_ns),
            # Group-level
            "group": _Namespace(group_fields),
            # Session-level
            "session": _Namespace(session_fields),
        }

        # Module-level constant access: C.<module>.<name>
        if constants:
            ctx["C"] = _DeepNamespace(constants)

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


class _RoundedModuleNamespace:
    """Bracket-access namespace for round-indexed agent data.

    Wraps ``{round_number: {field_name: value}}`` so that
    ``agent.module[1].field`` resolves correctly in Jinja2 templates.

    Args:
        data: A dict mapping round numbers (int) to field-value dicts.
    """

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key: Any) -> "_Namespace":
        round_data = self._data.get(int(key), {})
        return _Namespace(round_data)

    def __repr__(self) -> str:
        return f"RoundedModuleNamespace({list(self._data.keys())})"


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
