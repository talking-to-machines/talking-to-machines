"""Core object model for the T2M platform.

Defines canonical dataclasses for every entity in the oTree-style
hierarchy::

    ExperimentConfig -> Session -> Module -> Subsession -> Group -> Agent / Player

Static configuration models are produced by the compiler; runtime
models are instantiated during experiment execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Static configuration produced by the compiler
# ---------------------------------------------------------------------------


@dataclass
class SettingsConfig:
    """Parsed contents of the Settings worksheet.

    Attributes:
        experiment_id: Unique identifier for the experiment.
        model_name: Name of the LLM to use (e.g. ``"gpt-4"``).
        hf_inference_endpoint: Optional Hugging Face inference endpoint URL.
        temperature: Sampling temperature for LLM calls.
        random_seed: Seed for reproducible randomisation.
        profile_fields: Comma-separated profile fields or ``"ALL"``.
        build_profile_qa: Whether to generate profile Q&A pairs.
        build_profile_backstories: Whether to generate agent backstories.
        assign_manually: Comma-separated list indicating which
            assignments are manual: ``"Treatment"``, ``"Group"``, or
            ``"Treatment, Group"``. Empty means all assignments are
            random at session level.
        num_agents_per_session: Number of agents in each session.
        module_sequence: Ordered list of module names to execute.
        context_overflow_policy: Strategy when the context window is
            exceeded (``"terminate"``, ``"summarize"``, or ``"truncate"``).
    """

    experiment_id: str
    model_name: str
    hf_inference_endpoint: str = ""
    temperature: float = 0.0
    random_seed: int = 0
    profile_fields: str = "ALL"
    build_profile_qa: bool = False
    build_profile_backstories: bool = False
    assign_manually: str = ""
    num_agents_per_session: int = 1
    module_sequence: list[str] = field(default_factory=list)
    context_overflow_policy: str = "terminate"

    def to_dict(self) -> dict:
        """Serialise the settings to a plain dictionary.

        Returns:
            A dictionary containing all settings fields.
        """
        return {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "hf_inference_endpoint": self.hf_inference_endpoint,
            "temperature": self.temperature,
            "random_seed": self.random_seed,
            "profile_fields": self.profile_fields,
            "build_profile_qa": self.build_profile_qa,
            "build_profile_backstories": self.build_profile_backstories,
            "assign_manually": self.assign_manually,
            "num_agents_per_session": self.num_agents_per_session,
            "module_sequence": self.module_sequence,
            "context_overflow_policy": self.context_overflow_policy,
        }


@dataclass
class FieldDefinition:
    """A single field entry from the Fields worksheet.

    Attributes:
        field_class: Hierarchy level the field belongs to
            (``Session``, ``Subsession``, ``Agent``, ``Group``, or
            ``Player``).
        module: Module name this field is associated with.
        name: Field name used as a variable identifier.
        type: Data type (``integer``, ``float``, ``text``,
            ``category``, or ``boolean``).
        response_options: Allowed response values (list, dict, tuple,
            or ``None`` for free-form).
        response_options_intro: Introductory text displayed before
            the response options.
        randomise_options_order: Whether to shuffle response options
            before presenting them.
        validate: Whether to validate the agent's response against
            the allowed options.
        generate_speculation_score: Whether to generate a speculation
            score for this field.
        format_response: Whether to apply formatting to the response.
    """

    field_class: str
    module: str
    name: str
    type: str
    response_options: Any = None
    response_options_intro: str = ""
    randomise_options_order: bool = False
    validate: bool = False
    generate_speculation_score: bool = False
    format_response: bool = False

    def to_dict(self) -> dict:
        """Serialise the field definition to a plain dictionary.

        Returns:
            A dictionary containing all field attributes.
        """
        return {
            "field_class": self.field_class,
            "module": self.module,
            "name": self.name,
            "type": self.type,
            "response_options": self.response_options,
            "response_options_intro": self.response_options_intro,
            "randomise_options_order": self.randomise_options_order,
            "validate": self.validate,
            "generate_speculation_score": self.generate_speculation_score,
            "format_response": self.format_response,
        }


@dataclass
class PromptDefinition:
    """A single prompt entry from the Prompts worksheet.

    Attributes:
        module: Module name this prompt belongs to.
        prompt_sequence: Ordering index within the module.
        type: Prompt category (``CONTEXT``, ``DISCUSSION``,
            ``PUBLIC_QUESTION``, ``PRIVATE_QUESTION``, or
            ``FACILITATOR``).
        is_displayed: Optional Jinja condition controlling whether
            the prompt is shown.
        is_adapted: Whether the prompt text is adapted at runtime.
        human_text: Text shown to human participants (if any).
        llm_text: Jinja template text sent to the LLM.
        field_class: Hierarchy level for field-linked prompts, or
            ``None``.
        field_name: Name of the linked field, or ``None``.
        rag_vector_store_id: Optional vector store ID for
            retrieval-augmented generation.
    """

    module: str
    prompt_sequence: int
    type: str
    is_displayed: Optional[str] = None
    is_adapted: bool = False
    human_text: str = ""
    llm_text: str = ""
    field_class: Optional[str] = None
    field_name: Optional[str] = None
    rag_vector_store_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialise the prompt definition to a plain dictionary.

        Returns:
            A dictionary containing all prompt attributes.
        """
        return {
            "module": self.module,
            "prompt_sequence": self.prompt_sequence,
            "type": self.type,
            "is_displayed": self.is_displayed,
            "is_adapted": self.is_adapted,
            "human_text": self.human_text,
            "llm_text": self.llm_text,
            "field_class": self.field_class,
            "field_name": self.field_name,
            "rag_vector_store_id": self.rag_vector_store_id,
        }


@dataclass
class FacilitatorFunction:
    """A facilitator function from the Facilitator worksheet.

    Attributes:
        name: Function identifier (e.g. ``"assign_treatment"``).
        definition: Natural-language or built-in definition text.
        args: Optional keyword arguments parsed from the worksheet.
    """

    name: str
    definition: str
    args: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise the facilitator function to a plain dictionary.

        Returns:
            A dictionary with ``name``, ``definition``, and ``args``.
        """
        return {"name": self.name, "definition": self.definition, "args": self.args}


@dataclass
class ExperimentConfig:
    """Static definition of an experiment produced by the compiler.

    Contains the fully-parsed and validated contents of all worksheets.
    This is not a runtime object; it serves as the blueprint that the
    orchestrator uses to instantiate sessions.

    Attributes:
        settings: Parsed Settings worksheet.
        constants: Nested dictionary ``{module: {name: value}}`` from
            the C worksheet.
        fields: Ordered list of field definitions from the Fields
            worksheet.
        facilitator_functions: List of facilitator functions from the
            Facilitator worksheet.
        prompts: Dictionary ``{module: [PromptDefinition]}`` from the
            Prompts worksheet.
        profiles: Profile data with keys ``"short_names"``,
            ``"full_names"``, and ``"rows"``.
        config_hash: SHA-256 hash of the compiled configuration for
            reproducibility tracking.
    """

    settings: SettingsConfig
    constants: dict[str, dict[str, Any]]
    fields: list[FieldDefinition]
    facilitator_functions: list[FacilitatorFunction]
    prompts: dict[str, list[PromptDefinition]]
    profiles: dict
    config_hash: str = ""

    def to_dict(self) -> dict:
        """Serialise the full experiment configuration to a plain dictionary.

        Returns:
            A nested dictionary suitable for JSON serialisation.
        """
        return {
            "settings": self.settings.to_dict(),
            "constants": self.constants,
            "fields": [f.to_dict() for f in self.fields],
            "facilitator_functions": [f.to_dict() for f in self.facilitator_functions],
            "prompts": {
                module: [p.to_dict() for p in ps] for module, ps in self.prompts.items()
            },
            "profiles": self.profiles,
            "config_hash": self.config_hash,
        }


# ---------------------------------------------------------------------------
# Treatment / Role / Constant — backward-compatible with management/experiment.py
# ---------------------------------------------------------------------------


class Treatment:
    """Treatment condition with dynamically assigned attributes.

    All keyword arguments are set as instance attributes, allowing
    flexible treatment definitions from the worksheet.

    Args:
        **kwargs: Arbitrary keyword arguments. The ``description``
            key is extracted separately; all others become attributes.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.description = kwargs.pop("description", "")
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        """Return a string representation showing all attributes."""
        return f"Treatment({self.__dict__})"

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary.

        Returns:
            A shallow copy of the instance's attribute dictionary.
        """
        return self.__dict__.copy()


class Role:
    """Role definition with dynamically assigned attributes.

    All keyword arguments are set as instance attributes, allowing
    flexible role definitions from the worksheet.

    Args:
        **kwargs: Arbitrary keyword arguments. The ``description``
            key is extracted separately; all others become attributes.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.description = kwargs.pop("description", "")
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        """Return a string representation showing all attributes."""
        return f"Role({self.__dict__})"

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary.

        Returns:
            A shallow copy of the instance's attribute dictionary.
        """
        return self.__dict__.copy()


class Constant:
    """Experiment constant with dynamically assigned attributes.

    All keyword arguments are set as instance attributes.

    Args:
        **kwargs: Arbitrary keyword arguments that become instance
            attributes.
    """

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        """Return a string representation showing all attributes."""
        return f"Constant({self.__dict__})"

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary.

        Returns:
            A shallow copy of the instance's attribute dictionary.
        """
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Runtime entities
# ---------------------------------------------------------------------------


@dataclass
class Agent:
    """Persistent identity across all subsessions within a session.

    The ``agent_id`` is profile-linked and stable across runs
    (``{experiment_id}_a{profile_ID}``), while
    ``agent_instance_id`` is run-specific
    (``{run_id}_{agent_id}``).

    Attributes:
        agent_id: Profile-linked identifier, stable across runs.
        agent_instance_id: Run-specific instance identifier.
        profile_info: Dictionary of profile characteristics for
            this agent.
        treatment_label: Treatment condition label assigned to
            this agent.
        is_human: Whether this agent represents a human participant.
        message_history: Session-wide conversation history containing
            all messages from groups this agent has participated in.
            Persists across rounds and modules.
        state: Mutable state dictionary that persists across
            subsessions.
    """

    agent_id: str
    agent_instance_id: str
    profile_info: dict = field(default_factory=dict)
    treatment_label: str = ""
    is_human: bool = False
    message_history: list[dict] = field(default_factory=list)
    state: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise the agent to a plain dictionary.

        Returns:
            A dictionary containing all agent attributes.
        """
        return {
            "agent_id": self.agent_id,
            "agent_instance_id": self.agent_instance_id,
            "profile_info": self.profile_info,
            "treatment_label": self.treatment_label,
            "is_human": self.is_human,
            "message_history": self.message_history,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
        """Reconstruct an Agent from a plain dictionary.

        Args:
            data: Dictionary previously produced by ``to_dict``.

        Returns:
            A new ``Agent`` instance.
        """
        return cls(
            agent_id=data["agent_id"],
            agent_instance_id=data["agent_instance_id"],
            profile_info=data.get("profile_info", {}),
            treatment_label=data.get("treatment_label", ""),
            is_human=data.get("is_human", False),
            message_history=data.get("message_history", []),
            state=data.get("state", {}),
        )


@dataclass
class Player:
    """Runtime per-round instance of an agent within a group.

    The ``player_id`` is derived as
    ``{agent_instance_id}_{subsession_id}``. State is scoped to one
    subsession/group (e.g. current round decision, belief).

    Attributes:
        player_id: Unique identifier for this player instance.
        agent_instance_id: Back-reference to the parent agent
            instance.
        agent_id: Back-reference to the stable agent identity.
        group_id: Identifier of the group this player belongs to.
        message_history: Visibility-filtered conversation history
            for this player.
        state: Mutable state dictionary scoped to this subsession.
    """

    player_id: str
    agent_instance_id: str
    agent_id: str
    group_id: str
    message_history: list[dict] = field(default_factory=list)
    state: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise the player to a plain dictionary.

        Returns:
            A dictionary containing all player attributes.
        """
        return {
            "player_id": self.player_id,
            "agent_instance_id": self.agent_instance_id,
            "agent_id": self.agent_id,
            "group_id": self.group_id,
            "message_history": self.message_history,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Player":
        """Reconstruct a Player from a plain dictionary.

        Args:
            data: Dictionary previously produced by ``to_dict``.

        Returns:
            A new ``Player`` instance.
        """
        return cls(
            player_id=data["player_id"],
            agent_instance_id=data["agent_instance_id"],
            agent_id=data["agent_id"],
            group_id=data["group_id"],
            message_history=data.get("message_history", []),
            state=data.get("state", {}),
        )


@dataclass
class Group:
    """Interaction unit for one subsession.

    Contains the players participating in this group and the speaking
    order. Conversation history is stored on each ``Agent`` instance
    (session-wide) rather than on the group.

    Attributes:
        group_id: Unique identifier for this group.
        subsession_id: Back-reference to the parent subsession.
        players: List of players in this group.
        turn_order: Ordered list of ``agent_instance_id`` values
            defining the speaking sequence.
        state: Mutable state dictionary scoped to this group.
    """

    group_id: str
    subsession_id: str
    players: list[Player] = field(default_factory=list)
    turn_order: list[str] = field(default_factory=list)
    state: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise the group to a plain dictionary.

        Returns:
            A nested dictionary including serialised players.
        """
        return {
            "group_id": self.group_id,
            "subsession_id": self.subsession_id,
            "players": [p.to_dict() for p in self.players],
            "turn_order": self.turn_order,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Group":
        """Reconstruct a Group from a plain dictionary.

        Args:
            data: Dictionary previously produced by ``to_dict``.

        Returns:
            A new ``Group`` instance with nested ``Player`` objects.
        """
        return cls(
            group_id=data["group_id"],
            subsession_id=data["subsession_id"],
            players=[Player.from_dict(p) for p in data.get("players", [])],
            turn_order=data.get("turn_order", []),
            state=data.get("state", {}),
        )


@dataclass
class Subsession:
    """One round within a module.

    Attributes:
        subsession_id: Unique identifier for this subsession.
        module_id: Back-reference to the parent module.
        round_number: One-based round index within the module.
        groups: List of groups active in this subsession.
        state: Mutable state dictionary scoped to this subsession.
    """

    subsession_id: str
    module_id: str
    round_number: int
    groups: list[Group] = field(default_factory=list)
    state: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise the subsession to a plain dictionary.

        Returns:
            A nested dictionary including serialised groups.
        """
        return {
            "subsession_id": self.subsession_id,
            "module_id": self.module_id,
            "round_number": self.round_number,
            "groups": [g.to_dict() for g in self.groups],
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Subsession":
        """Reconstruct a Subsession from a plain dictionary.

        Args:
            data: Dictionary previously produced by ``to_dict``.

        Returns:
            A new ``Subsession`` instance with nested ``Group`` objects.
        """
        return cls(
            subsession_id=data["subsession_id"],
            module_id=data["module_id"],
            round_number=data["round_number"],
            groups=[Group.from_dict(g) for g in data.get("groups", [])],
            state=data.get("state", {}),
        )


@dataclass
class Module:
    """One module block within a session.

    Maps to a single entry in the ``MODULE_SEQUENCE`` setting and
    contains one or more subsessions (rounds).

    Attributes:
        module_id: Unique identifier for this module.
        session_id: Back-reference to the parent session.
        module_name: Name of the module this block executes.
        subsessions: Ordered list of subsessions (rounds) in this
            module.
        state: Mutable state dictionary scoped to this module.
    """

    module_id: str
    session_id: str
    module_name: str
    subsessions: list[Subsession] = field(default_factory=list)
    state: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise the module to a plain dictionary.

        Returns:
            A nested dictionary including serialised subsessions.
        """
        return {
            "module_id": self.module_id,
            "session_id": self.session_id,
            "module_name": self.module_name,
            "subsessions": [s.to_dict() for s in self.subsessions],
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Module":
        """Reconstruct a Module from a plain dictionary.

        Args:
            data: Dictionary previously produced by ``to_dict``.

        Returns:
            A new ``Module`` instance with nested ``Subsession``
            objects.
        """
        return cls(
            module_id=data["module_id"],
            session_id=data["session_id"],
            module_name=data["module_name"],
            subsessions=[Subsession.from_dict(s) for s in data.get("subsessions", [])],
            state=data.get("state", {}),
        )


@dataclass
class Session:
    """One complete run execution.

    Contains all modules and all agents participating in the
    session.

    Attributes:
        session_id: Unique identifier for this session.
        run_id: Identifier for the specific run execution.
        experiment_id: Back-reference to the parent experiment.
        cep_hash: Hash of the compiled experiment package used for
            this session.
        agents: List of agents participating in this session.
        modules: Ordered list of modules to execute.
        state: Mutable state dictionary scoped to this session.
    """

    session_id: str
    run_id: str
    experiment_id: str
    cep_hash: str = ""
    agents: list[Agent] = field(default_factory=list)
    modules: list[Module] = field(default_factory=list)
    state: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise the session to a plain dictionary.

        Returns:
            A nested dictionary including serialised agents and
            modules.
        """
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "cep_hash": self.cep_hash,
            "agents": [a.to_dict() for a in self.agents],
            "modules": [m.to_dict() for m in self.modules],
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Reconstruct a Session from a plain dictionary.

        Args:
            data: Dictionary previously produced by ``to_dict``.

        Returns:
            A new ``Session`` instance with nested ``Agent`` and
            ``Module`` objects.
        """
        return cls(
            session_id=data["session_id"],
            run_id=data["run_id"],
            experiment_id=data["experiment_id"],
            cep_hash=data.get("cep_hash", ""),
            agents=[Agent.from_dict(a) for a in data.get("agents", [])],
            modules=[Module.from_dict(m) for m in data.get("modules", [])],
            state=data.get("state", {}),
        )
