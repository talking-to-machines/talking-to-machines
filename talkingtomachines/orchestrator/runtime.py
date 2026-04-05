"""Experiment runtime engine for executing compiled experiments.

Replaces the monolithic ``run_session()`` with a modular, hierarchical
execution engine that supports parallel group execution, checkpointing,
guardrails, and multiple prompt types (facilitator, context, discussion,
public questions, private questions).

Runtime loop::

    ExperimentRuntime.run(session, cep):
      initialize_session()
      for each module in session:
        for each subsession in module:
          form_groups()
          for each group in subsession:
            for each prompt in module (filtered by is_displayed):
              if FACILITATOR      -> execute_facilitator()
              if CONTEXT          -> broadcast_context()
              if DISCUSSION       -> run_discussion_turn()
              if PUBLIC_QUESTION  -> run_question_turn(visibility=group_only)
              if PRIVATE_QUESTION -> run_question_turn(visibility=private)
            evaluate_stop_condition()
          # end_round/continue scoped to group; only end_session propagates
          if stop == "end_session" -> break
      finalize_session()
"""

from __future__ import annotations

import logging
import random
import threading
import time
from pathlib import Path
from typing import Optional

from talkingtomachines.core.models import (
    Agent,
    Player,
    Group,
    Module,
    Session,
    Subsession,
    PromptDefinition,
    FieldDefinition,
)
from talkingtomachines.core.fields import ExperimentState
from talkingtomachines.core.flow import FlowEvaluator
from talkingtomachines.core.facilitator import FacilitatorEngine
from talkingtomachines.core.randomisation import RandomisationEngine
from talkingtomachines.core.visibility import (
    make_message,
    VISIBILITY_GROUP_ONLY,
    VISIBILITY_PRIVATE,
    VISIBILITY_FACILITATOR,
)
from talkingtomachines.agents.synthetic_subject import ConversationalSyntheticSubject
from talkingtomachines.compiler.cep_schema import CompiledExperiment
from talkingtomachines.compiler.id_generator import (
    make_session_id,
    make_module_id,
    make_subsession_id,
    make_group_id,
    make_agent_id,
    make_agent_instance_id,
    make_player_id,
    make_turn_id,
)
from talkingtomachines.orchestrator.checkpointing import Checkpointer
from talkingtomachines.orchestrator.guardrails import Guardrails
from talkingtomachines.orchestrator.parallel import (
    run_groups_parallel,
    run_private_questions_parallel,
)
from talkingtomachines.gateway.router import LLMRouter
from talkingtomachines.gateway.cost_tracker import CostTracker
from talkingtomachines.storage.event_log import EventLogger

logger = logging.getLogger(__name__)


class ExperimentRuntime:
    """Executes a compiled experiment for a single session.

    Orchestrates the full lifecycle of an experiment session, including
    agent construction, module/subsession iteration, group formation,
    prompt dispatch, checkpointing, and finalization.

    Attributes:
        _cep: The compiled experiment protocol containing all configuration.
        _output_dir: Directory where session artifacts are written.
        _cost_tracker: Tracks cumulative LLM API costs.
        _router: Routes LLM calls to the configured provider.
        _context_guard: Enforces context window limits per model.
        _rng_engine: Seeded randomisation engine for reproducibility.
        _state: Mutable experiment state shared across the session.
        _flow: Evaluator for flow-control expressions (is_displayed, stop conditions).
        _guardrails: Anomaly detection and safety checks.
        _checkpointer: Saves/loads session state at subsession boundaries.
        _event_logger: Writes structured trace events to a JSONL file.
        _turn_counter: Global turn counter within the session.
        _turn_counter_lock: Thread lock protecting the turn counter.
        _jinja_env: Sandboxed Jinja2 environment for template rendering.

    Usage::

        cep = CompiledExperiment.load("compiled_experiment.json")
        runtime = ExperimentRuntime(cep, output_dir="experiment_results/run_001")
        session = runtime.run(session_number=1, test_mode=False)
    """

    def __init__(
        self,
        cep: CompiledExperiment,
        output_dir: str | Path,
        budget_cap_usd: float = 0.0,
    ) -> None:
        """Initialize the experiment runtime.

        Args:
            cep: Compiled experiment protocol containing settings, profiles,
                prompts, assignment plans, and facilitator definitions.
            output_dir: Filesystem path where session output, checkpoints,
                and trace logs are written. Created if it does not exist.
            budget_cap_usd: Maximum spend in USD for this session. A value
                of 0.0 means unlimited.
        """
        self._cep = cep
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._cost_tracker = CostTracker(budget_cap_usd=budget_cap_usd)
        self._router = LLMRouter(
            cost_tracker=self._cost_tracker,
            hf_inference_endpoint=cep.settings.get("hf_inference_endpoint", ""),
        )

        # Context window guard
        from talkingtomachines.gateway.context_guard import ContextGuard

        self._context_guard = ContextGuard(
            model_name=cep.settings.get("model_name", ""),
            policy=cep.settings.get("context_overflow_policy", "terminate"),
        )

        self._rng_engine = RandomisationEngine(
            global_seed=cep.settings.get("random_seed", 0)
        )
        self._state = ExperimentState()
        self._flow = FlowEvaluator()
        self._guardrails = Guardrails()
        self._checkpointer = Checkpointer(self._output_dir)
        self._event_logger = EventLogger(self._output_dir / "traces.jsonl")

        # Turn counter (global within session) — protected by lock for parallel groups
        self._turn_counter: int = 0
        self._turn_counter_lock = threading.Lock()

        # Shared Jinja2 environment for context broadcast rendering
        from jinja2.sandbox import SandboxedEnvironment

        self._jinja_env = SandboxedEnvironment()

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> "ExperimentState":
        """Read-only access to the experiment state."""
        return self._state

    @property
    def total_cost_usd(self) -> float:
        """Total accumulated cost in USD."""
        return self._cost_tracker.total_usd

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        session_number: int = 1,
        test_mode: bool = False,
        resume_session: Optional[Session] = None,
        on_module_complete: Optional[callable] = None,
    ) -> Session:
        """
        Execute the full experiment session.

        Args:
            session_number:  Numeric index of this session (for ID derivation).
            test_mode:       If True, run only one group per module.
            resume_session:  If provided, resume from this checkpoint. Already-completed
                             modules (those with subsessions) are skipped.
            on_module_complete: Optional callback invoked after each module
                completes. Called with ``(module_index, module_name, total_modules)``
                to support progress reporting.

        Returns:
            The completed ``Session`` object.
        """
        cep = self._cep

        if resume_session is not None:
            session = resume_session
            session_id = session.session_id
            completed_module_names = {
                m.module_name for m in session.modules if m.subsessions
            }
            logger.info(
                "Resuming session %s — %d module(s) already completed.",
                session_id,
                len(completed_module_names),
            )
        else:
            session_id = make_session_id(cep.run_id, session_number)
            session = Session(
                session_id=session_id,
                run_id=cep.run_id,
                experiment_id=cep.experiment_id,
                cep_hash=cep.cep_hash,
            )
            completed_module_names = set()

        # Build agents from profiles
        profiles = cep.profiles
        profile_rows = profiles.get("rows", []) if profiles else []
        id_col = profiles.get("id_column", "ID") if profiles else "ID"
        assignment_plan = cep.assignment_plan

        agents: list[Agent] = []
        for profile_row in profile_rows:
            pid = profile_row.get(id_col)
            if pid is None:
                continue
            agent_id = make_agent_id(cep.experiment_id, pid)
            agent_instance_id = make_agent_instance_id(cep.run_id, agent_id)
            agent = Agent(
                agent_id=agent_id,
                agent_instance_id=agent_instance_id,
                profile_info=profile_row,
            )
            agents.append(agent)

        session.agents = agents

        # Initialize session via creating_session facilitator (skip if resuming)
        if resume_session is None:
            self._initialize_session(session, cep)

        stop_signal = "continue"
        # In test mode, lock to the first group's agents across all modules
        test_group_agent_ids: set[str] | None = None

        # Execute modules
        for module_idx, module_name in enumerate(cep.module_sequence):
            if stop_signal == "end_session":
                break

            # Skip already-completed modules when resuming
            if module_name in completed_module_names:
                logger.info("Skipping already-completed module: %s", module_name)
                continue

            module_id = make_module_id(session_id, module_name)
            module = Module(
                module_id=module_id,
                session_id=session_id,
                module_name=module_name,
            )
            session.modules.append(module)

            module_consts = cep.constants.get(module_name, {})
            max_rounds = int(module_consts.get("MAX_NUM_ROUNDS", 1))

            prompts = [PromptDefinition(**p) for p in cep.prompts.get(module_name, [])]

            for round_num in range(1, max_rounds + 1):
                if stop_signal == "end_session":
                    break

                subsession_id = make_subsession_id(module_id, round_num)
                subsession = Subsession(
                    subsession_id=subsession_id,
                    module_id=module_id,
                    round_number=round_num,
                )
                module.subsessions.append(subsession)

                # Apply manual variables from the Manual_ sheet
                self._apply_manual_variables(
                    assignment_plan, agents, session, module_name, round_num, cep
                )

                # Form groups: use manual groups if defined, else sequential
                precomputed_groups = assignment_plan.get_manual_groups(
                    module_name, round_num
                )
                if not precomputed_groups:
                    # Default sequential grouping by Profile order
                    ppg = int(module_consts.get("PLAYERS_PER_GROUP", len(agents) or 1))
                    precomputed_groups = {}
                    for i in range(0, len(agents), ppg):
                        chunk = agents[i : i + ppg]
                        if chunk:
                            g_label = f"g{i // ppg + 1}"
                            precomputed_groups[g_label] = [a.agent_id for a in chunk]

                groups: list[Group] = []
                for g_idx, (g_label, member_agent_ids) in enumerate(
                    precomputed_groups.items()
                ):
                    group_id = make_group_id(subsession_id, g_idx + 1)
                    group_agents = [a for a in agents if a.agent_id in member_agent_ids]
                    players: list[Player] = []
                    for ga in group_agents:
                        player_id = make_player_id(ga.agent_instance_id, subsession_id)
                        player = Player(
                            player_id=player_id,
                            agent_instance_id=ga.agent_instance_id,
                            agent_id=ga.agent_id,
                            group_id=group_id,
                        )
                        players.append(player)
                    group = Group(
                        group_id=group_id,
                        subsession_id=subsession_id,
                        players=players,
                        turn_order=[ga.agent_instance_id for ga in group_agents],
                    )
                    groups.append(group)
                    subsession.groups.append(group)

                # Apply Player/Group Manual_ entries to proper scopes
                # now that player_id and group_id exist
                self._apply_scoped_manual_variables(
                    assignment_plan, agents, groups, module_name, round_num
                )

                if test_mode and groups:
                    if test_group_agent_ids is None:
                        # First module: capture agents from the first group
                        test_group_agent_ids = {p.agent_id for p in groups[0].players}
                        groups = groups[:1]
                    else:
                        # Subsequent modules: pick the group whose members
                        # best overlap with the locked test group agents
                        best = max(
                            groups,
                            key=lambda g: len(
                                {p.agent_id for p in g.players} & test_group_agent_ids
                            ),
                        )
                        groups = [best]

                # Build agent map once per subsession (shared across groups)
                agents_map = {a.agent_instance_id: a for a in agents}

                # Execute groups — parallel if max_group_workers > 1, else sequential
                max_group_workers = int(cep.settings.get("max_group_workers", 1))
                if max_group_workers > 1 and len(groups) > 1:

                    def _group_wrapper(grp: Group) -> str:
                        return self._run_group(
                            group=grp,
                            session=session,
                            agents_map=agents_map,
                            module_name=module_name,
                            round_number=round_num,
                            prompts=prompts,
                            cep=cep,
                        )

                    stop_signals = run_groups_parallel(
                        _group_wrapper,
                        [(grp,) for grp in groups],
                        max_workers=max_group_workers,
                    )
                    # Only propagate end_session across groups;
                    # end_round and continue are scoped to individual groups.
                    stop_signal = (
                        "end_session" if "end_session" in stop_signals else "continue"
                    )
                else:
                    for group in groups:
                        group_signal = self._run_group(
                            group=group,
                            session=session,
                            agents_map=agents_map,
                            module_name=module_name,
                            round_number=round_num,
                            prompts=prompts,
                            cep=cep,
                        )
                        # Only propagate end_session across groups;
                        # end_round and continue are scoped to individual groups.
                        if group_signal == "end_session":
                            stop_signal = "end_session"
                            break

                # Archive player-scoped fields to agent scope for cross-round access
                for group in groups:
                    for player in group.players:
                        player_fields = self._state.get_player_module(
                            player.player_id, module_name
                        )
                        for fname, fval in player_fields.items():
                            self._state.set_agent(
                                player.agent_id,
                                module_name,
                                fname,
                                fval,
                                round_number=round_num,
                            )

                # Checkpoint at subsession boundary
                self._checkpointer.save(session, self._state)

            # Notify progress after each module completes
            if on_module_complete is not None:
                on_module_complete(module_idx, module_name, len(cep.module_sequence))

        self._finalize_session(session)
        return session

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _initialize_session(self, session: Session, cep: CompiledExperiment) -> None:
        """Log a ``session_start`` event.

        Args:
            session: The newly created session object.
            cep: Compiled experiment package.
        """
        self._event_logger.log(
            "session_start",
            run_id=cep.run_id,
            session_id=session.session_id,
            data={"experiment_id": cep.experiment_id},
        )

    def _apply_manual_variables(
        self,
        assignment_plan,
        agents: list[Agent],
        session: Session,
        module_name: str,
        round_number: int,
        cep: "CompiledExperiment",
    ) -> None:
        """Apply manual variable assignments from the ``Manual_`` sheet.

        At the start of each round, iterates through the manual
        variables and sets values on the appropriate scope
        (Session, Agent, Player, Group) via ``ExperimentState``.
        Values are then automatically accessible in Jinja templates
        via dot notation.

        Args:
            assignment_plan: The assignment plan from the CEP.
            agents: List of agents in the session.
            session: The current session.
            module_name: Name of the current module.
            round_number: Current round number (1-based).
            cep: Compiled experiment package.
        """
        # Build profile_id → agent map.
        # agent_id equals the raw profile ID (from make_agent_id), so
        # register both the string and numeric forms.
        agent_by_pid: dict = {}
        for agent in agents:
            agent_by_pid[agent.agent_id] = agent
            try:
                agent_by_pid[int(agent.agent_id)] = agent
            except (ValueError, TypeError):
                pass

        for entry in assignment_plan.manual_variables:
            entry_module = entry.get("module", "")
            entry_round = entry.get("round_number", "")
            entry_class = entry.get("class", "")
            entry_name = entry.get("name", "")
            entry_value = entry.get("value", "")
            entry_pid = entry.get("profile_id", "")

            # Check if this entry applies to the current module/round
            if entry_module and entry_module != module_name:
                continue
            if entry_round != "" and entry_round != round_number:
                try:
                    if int(entry_round) != round_number:
                        continue
                except (ValueError, TypeError):
                    continue

            if entry_class == "Session":
                self._state.set_session(module_name, entry_name, entry_value)
            elif entry_class == "Agent":
                agent = agent_by_pid.get(entry_pid)
                if agent:
                    self._state.set_agent(
                        agent.agent_id,
                        module_name,
                        entry_name,
                        entry_value,
                        round_number=round_number,
                    )
            elif entry_class == "Player":
                agent = agent_by_pid.get(entry_pid)
                if agent:
                    # Player ID is constructed at group formation time;
                    # set on agent scope so it flows to the player context
                    self._state.set_agent(
                        agent.agent_id,
                        module_name,
                        entry_name,
                        entry_value,
                        round_number=round_number,
                    )
            elif entry_class == "Group":
                # Group-level variables (not id_in_subsession, which is
                # handled by group formation) are set on agent scope and
                # will be resolved at group execution time
                agent = agent_by_pid.get(entry_pid)
                if agent:
                    self._state.set_agent(
                        agent.agent_id,
                        module_name,
                        entry_name,
                        entry_value,
                        round_number=round_number,
                    )

    def _apply_scoped_manual_variables(
        self,
        assignment_plan,
        agents: list[Agent],
        groups: list[Group],
        module_name: str,
        round_number: int,
    ) -> None:
        """Apply Player/Group ``Manual_`` entries to their proper scopes.

        Called after group formation, when ``player_id`` and ``group_id``
        are known.  The existing ``_apply_manual_variables`` stores
        Player/Group entries on agent scope (for cross-round access);
        this method additionally sets them on player/group scope so that
        ``{{ player.<field> }}`` and ``{{ group.<field> }}`` resolve
        correctly in Jinja templates.

        Args:
            assignment_plan: The assignment plan from the CEP.
            agents: List of agents in the session.
            groups: List of groups formed for this round.
            module_name: Name of the current module.
            round_number: Current round number (1-based).
        """
        # Build agent_id → player/group mapping
        agent_to_player: dict[str, Player] = {}
        agent_to_group: dict[str, Group] = {}
        for group in groups:
            for player in group.players:
                agent_to_player[player.agent_id] = player
                agent_to_group[player.agent_id] = group

        # Build profile_id → agent map (same logic as _apply_manual_variables)
        agent_by_pid: dict = {}
        for agent in agents:
            agent_by_pid[agent.agent_id] = agent
            try:
                agent_by_pid[int(agent.agent_id)] = agent
            except (ValueError, TypeError):
                pass

        for entry in assignment_plan.manual_variables:
            entry_class = entry.get("class", "")
            if entry_class not in ("Player", "Group"):
                continue

            entry_module = entry.get("module", "")
            entry_round = entry.get("round_number", "")
            if entry_module and entry_module != module_name:
                continue
            if entry_round != "" and entry_round != round_number:
                try:
                    if int(entry_round) != round_number:
                        continue
                except (ValueError, TypeError):
                    continue

            entry_pid = entry.get("profile_id", "")
            agent = agent_by_pid.get(entry_pid)
            if not agent:
                continue

            entry_name = entry.get("name", "")
            entry_value = entry.get("value", "")

            if entry_class == "Player":
                player = agent_to_player.get(agent.agent_id)
                if player:
                    self._state.set_player(
                        player.player_id, module_name, entry_name, entry_value
                    )
            elif entry_class == "Group":
                grp = agent_to_group.get(agent.agent_id)
                if grp:
                    self._state.set_group(
                        grp.group_id, module_name, entry_name, entry_value
                    )

    def _finalize_session(self, session: Session) -> None:
        """Save the final checkpoint and log a session-end event.

        Args:
            session: The completed session to finalize.
        """
        self._checkpointer.save(session, self._state)
        self._event_logger.log(
            "session_end",
            run_id=self._cep.run_id,
            session_id=session.session_id,
            data={"total_cost_usd": self._cost_tracker.total_usd},
        )
        logger.info(
            "Session %s complete. Total cost: $%.4f USD",
            session.session_id,
            self._cost_tracker.total_usd,
        )

    # ------------------------------------------------------------------
    # Group execution
    # ------------------------------------------------------------------

    def _run_group(
        self,
        group: Group,
        session: Session,
        agents_map: dict[str, Agent],
        module_name: str,
        round_number: int,
        prompts: list[PromptDefinition],
        cep: CompiledExperiment,
    ) -> str:
        """Execute all prompts for a single group within a subsession.

        Iterates through each prompt in the module, dispatching to the
        appropriate handler based on prompt type (FACILITATOR, CONTEXT,
        DISCUSSION, PUBLIC_QUESTION, PRIVATE_QUESTION). Builds synthetic
        subjects for each player and evaluates flow-control expressions.

        Args:
            group: The group to execute prompts for.
            session: The parent session object.
            agents_map: Mapping from agent_instance_id to Agent objects.
            module_name: Name of the current module.
            round_number: Current round (subsession) number.
            prompts: Ordered list of prompt definitions for this module.
            cep: Compiled experiment for accessing settings and constants.

        Returns:
            A stop signal string: ``"continue"``, ``"end_round"``, or
            ``"end_session"``.
        """
        model_name = cep.settings.get("model_name", "")
        temperature = cep.settings.get("temperature", 0.0)
        profiles_meta = cep.profiles or {}

        # Build synthetic subjects for each player in the group
        subjects: dict[str, ConversationalSyntheticSubject] = {}
        for player in group.players:
            agent = agents_map.get(player.agent_instance_id)
            if agent is None:
                continue
            subject = ConversationalSyntheticSubject(
                agent=agent,
                player=player,
                group=group,
                session=session,
                state=self._state,
                router=self._router,
                model_name=model_name,
                temperature=temperature,
                profiles_meta=profiles_meta,
                build_profile_qa=cep.settings.get("build_profile_qa", False),
                build_profile_backstories=cep.settings.get(
                    "build_profile_backstories", False
                ),
                constants=cep.constants,
                context_guard=self._context_guard,
            )
            subjects[player.agent_instance_id] = subject

        # Build field definitions lookup once (not per-prompt)
        field_defs_by_key = {
            (f["module"], f["field_class"], f["name"]): FieldDefinition(**f)
            for f in cep.fields
        }

        # Build a representative Jinja context for is_displayed evaluation.
        # Uses the first player/agent in the group as a proxy so expressions
        # such as "player.treatment == 'T1'" or "agent.democrat == 1" work.
        if group.players:
            rep_player = group.players[0]
            rep_agent = agents_map.get(rep_player.agent_instance_id)
            if rep_agent:
                jinja_ctx = self._state.build_jinja_context(
                    agent=rep_agent,
                    player=rep_player,
                    group=group,
                    session=session,
                    module=module_name,
                    round_number=round_number,
                    constants=cep.constants,
                )
            else:
                jinja_ctx = {"round_number": round_number, "group_id": group.group_id}
        else:
            jinja_ctx = {"round_number": round_number, "group_id": group.group_id}

        def _is_player_specific(expr: str) -> bool:
            """Check if an ``is_displayed`` expression references player/agent attributes.

            Args:
                expr: A Jinja2 boolean expression string.

            Returns:
                ``True`` if the expression contains ``player.`` or ``agent.``.
            """
            if not expr:
                return False
            return "player." in expr or "agent." in expr

        def _player_jinja_ctx(player, agent):
            """Build a per-player Jinja context for ``is_displayed`` evaluation.

            Args:
                player: The ``Player`` instance.
                agent: The ``Agent`` instance.

            Returns:
                A Jinja2 context dict for the given player/agent.
            """
            return self._state.build_jinja_context(
                agent=agent,
                player=player,
                group=group,
                session=session,
                module=module_name,
                round_number=round_number,
                constants=cep.constants,
            )

        stop_signal = "continue"

        for prompt in prompts:
            # Evaluate is_displayed — per-player if expression references player/agent
            is_disp_expr = prompt.is_displayed
            per_player_filter = _is_player_specific(is_disp_expr)
            if not per_player_filter:
                # Non-player expression: evaluate once with representative context
                if not self._flow.is_displayed(is_disp_expr, jinja_ctx):
                    continue

            # Fetch field definition if applicable
            field_def = None
            if prompt.field_class and prompt.field_name:
                field_key = (module_name, prompt.field_class, prompt.field_name)
                field_def = field_defs_by_key.get(field_key)

            if prompt.type == "FACILITATOR":
                stop_signal = self._execute_facilitator_prompt(
                    prompt,
                    group,
                    agents_map,
                    cep,
                    module_name,
                    field_def,
                    jinja_ctx,
                    session=session,
                    round_number=round_number,
                    is_displayed_expr=is_disp_expr if per_player_filter else None,
                )
                if stop_signal in ("end_round", "end_session"):
                    break

            elif prompt.type == "CONTEXT":
                self._broadcast_context(
                    prompt,
                    group,
                    session,
                    module_name,
                    round_number,
                    agents_map,
                    cep,
                    is_displayed_expr=is_disp_expr if per_player_filter else None,
                )

            elif prompt.type == "DISCUSSION":
                # Sequential: each agent sees the previous agent's message before responding
                for agent_id in group.turn_order:
                    subject = subjects.get(agent_id)
                    if subject is None:
                        continue
                    player = next(
                        (p for p in group.players if p.agent_instance_id == agent_id),
                        None,
                    )
                    if player is None:
                        continue
                    if per_player_filter:
                        agent_obj_check = agents_map.get(agent_id)
                        if agent_obj_check and not self._flow.is_displayed(
                            is_disp_expr, _player_jinja_ctx(player, agent_obj_check)
                        ):
                            continue
                    t0 = time.perf_counter()
                    content, rendered_prompt = subject.respond(
                        module_name, round_number, prompt, field_def
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000
                    with self._turn_counter_lock:
                        self._turn_counter += 1
                        turn_id = make_turn_id(
                            group.group_id, self._turn_counter, agent_id
                        )
                    self._guardrails.check_response(
                        turn_id=turn_id,
                        agent_instance_id=agent_id,
                        content=content,
                        latency_ms=latency_ms,
                    )
                    self._event_logger.log(
                        "llm_call",
                        run_id=cep.run_id,
                        session_id=session.session_id,
                        group_id=group.group_id,
                        agent_instance_id=agent_id,
                        data={
                            "prompt_type": "DISCUSSION",
                            "module": module_name,
                            "round": round_number,
                            "turn_id": turn_id,
                            "latency_ms": round(latency_ms, 2),
                            "rendered_prompt": rendered_prompt,
                            "response": content,
                        },
                    )
                    agent_obj = agents_map.get(agent_id, Agent("", "", {}))
                    msg = make_message(
                        role="assistant",
                        content=content,
                        sender_agent_id=agent_obj.agent_id,
                        group_id=group.group_id,
                        treatment_label="",
                        visibility=VISIBILITY_GROUP_ONLY,
                        module=module_name,
                        round_number=round_number,
                    )
                    self._append_to_agent_histories(msg, group, agents_map)

            elif prompt.type == "PUBLIC_QUESTION":
                # Independent: each agent answers without seeing others' responses.
                # Responses are collected first, then appended to history so they
                # become publicly visible for subsequent prompts/rounds.
                pending_messages: list[dict] = []
                for agent_id in group.turn_order:
                    subject = subjects.get(agent_id)
                    if subject is None:
                        continue
                    if per_player_filter:
                        pq_player = next(
                            (
                                p
                                for p in group.players
                                if p.agent_instance_id == agent_id
                            ),
                            None,
                        )
                        agent_obj_check = agents_map.get(agent_id)
                        if (
                            pq_player
                            and agent_obj_check
                            and not self._flow.is_displayed(
                                is_disp_expr,
                                _player_jinja_ctx(pq_player, agent_obj_check),
                            )
                        ):
                            continue
                    t0 = time.perf_counter()
                    content, rendered_prompt = subject.respond(
                        module_name, round_number, prompt, field_def
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000
                    with self._turn_counter_lock:
                        self._turn_counter += 1
                        turn_id = make_turn_id(
                            group.group_id, self._turn_counter, agent_id
                        )
                    self._guardrails.check_response(
                        turn_id=turn_id,
                        agent_instance_id=agent_id,
                        content=content,
                        latency_ms=latency_ms,
                    )
                    self._event_logger.log(
                        "llm_call",
                        run_id=cep.run_id,
                        session_id=session.session_id,
                        group_id=group.group_id,
                        agent_instance_id=agent_id,
                        data={
                            "prompt_type": "PUBLIC_QUESTION",
                            "module": module_name,
                            "round": round_number,
                            "field_name": prompt.field_name,
                            "latency_ms": round(latency_ms, 2),
                            "rendered_prompt": rendered_prompt,
                            "response": content,
                        },
                    )
                    agent_obj = agents_map.get(agent_id, Agent("", "", {}))
                    pending_messages.append(
                        make_message(
                            role="assistant",
                            content=content,
                            sender_agent_id=agent_obj.agent_id,
                            group_id=group.group_id,
                            treatment_label="",
                            visibility=VISIBILITY_GROUP_ONLY,
                            module=module_name,
                            round_number=round_number,
                        )
                    )
                # Append all responses after everyone has answered
                for msg in pending_messages:
                    self._append_to_agent_histories(msg, group, agents_map)

            elif prompt.type == "PRIVATE_QUESTION":
                # Parallel: each agent's private answer is invisible to others
                def _respond_private(agent_id: str):
                    """Execute a private question for a single agent.

                    Args:
                        agent_id: The agent instance ID to respond.

                    Returns:
                        Tuple of (agent_id, raw_response, parsed_value, cost).
                    """
                    subj = subjects.get(agent_id)
                    if subj is None:
                        return agent_id, "", "", 0.0
                    if per_player_filter:
                        priv_player = next(
                            (
                                p
                                for p in group.players
                                if p.agent_instance_id == agent_id
                            ),
                            None,
                        )
                        agent_obj_check = agents_map.get(agent_id)
                        if (
                            priv_player
                            and agent_obj_check
                            and not self._flow.is_displayed(
                                is_disp_expr,
                                _player_jinja_ctx(priv_player, agent_obj_check),
                            )
                        ):
                            return agent_id, "", "", 0.0
                    t0 = time.perf_counter()
                    cnt, rp = subj.respond(module_name, round_number, prompt, field_def)
                    return agent_id, cnt, rp, (time.perf_counter() - t0) * 1000

                max_player_workers = int(cep.settings.get("max_player_workers", 1))
                if max_player_workers > 1 and len(group.turn_order) > 1:
                    raw_results = run_private_questions_parallel(
                        _respond_private,
                        [(aid,) for aid in group.turn_order],
                        max_workers=max_player_workers,
                    )
                else:
                    raw_results = [_respond_private(aid) for aid in group.turn_order]

                for agent_id, content, rendered_prompt, latency_ms in raw_results:
                    if not content:
                        continue
                    with self._turn_counter_lock:
                        self._turn_counter += 1
                        turn_id = make_turn_id(
                            group.group_id, self._turn_counter, agent_id
                        )
                    self._guardrails.check_response(
                        turn_id=turn_id,
                        agent_instance_id=agent_id,
                        content=content,
                        latency_ms=latency_ms,
                    )
                    self._event_logger.log(
                        "llm_call",
                        run_id=cep.run_id,
                        session_id=session.session_id,
                        group_id=group.group_id,
                        agent_instance_id=agent_id,
                        data={
                            "prompt_type": "PRIVATE_QUESTION",
                            "module": module_name,
                            "round": round_number,
                            "field_name": prompt.field_name,
                            "latency_ms": round(latency_ms, 2),
                            "rendered_prompt": rendered_prompt,
                            "response": content,
                        },
                    )
                    agent_obj = agents_map.get(agent_id, Agent("", "", {}))
                    msg = make_message(
                        role="assistant",
                        content=content,
                        sender_agent_id=agent_obj.agent_id,
                        group_id=group.group_id,
                        treatment_label="",
                        visibility=VISIBILITY_PRIVATE,
                        module=module_name,
                        round_number=round_number,
                    )
                    self._append_to_agent_histories(msg, group, agents_map)

        return stop_signal

    def _append_to_agent_histories(
        self,
        msg: dict,
        group: Group,
        agents_map: dict[str, Agent],
    ) -> None:
        """Append a message to every agent's session-wide message history.

        Each agent receives a copy of the message with the ``role`` field
        set relative to that agent: ``"assistant"`` if the agent is the
        sender, ``"user"`` otherwise. This ensures that when the history
        is replayed in the LLM context, only the agent's own prior
        responses appear as ``role="assistant"``.

        Args:
            msg: The standardised message dict to append.
            group: The group whose agents should receive the message.
            agents_map: Mapping from agent_instance_id to Agent objects.
        """
        sender = msg.get("sender_agent_id", "")
        for player in group.players:
            agent = agents_map.get(player.agent_instance_id)
            if agent is None:
                continue
            agent_msg = dict(msg)
            if sender == agent.agent_id:
                agent_msg["role"] = "assistant"
            else:
                agent_msg["role"] = "user"
            agent.message_history.append(agent_msg)

    def _broadcast_context(
        self,
        prompt: PromptDefinition,
        group: Group,
        session: Session,
        module_name: str,
        round_number: int,
        agents_map: dict[str, Agent],
        cep: CompiledExperiment,
        is_displayed_expr: Optional[str] = None,
    ) -> None:
        """Render a CONTEXT prompt per player and add to the agent's message history.

        The prompt text is rendered through the sandboxed Jinja2 environment
        once per player using that player's own attributes (profile, treatment,
        state). Each rendered message is tagged with ``VISIBILITY_PRIVATE`` so
        that each player sees only the version personalised for them.

        Args:
            prompt: The CONTEXT-type prompt definition to render.
            group: The target group receiving the context message.
            session: The parent session object.
            module_name: Name of the current module.
            round_number: Current round (subsession) number.
            agents_map: Mapping from agent_instance_id to Agent objects.
            cep: Compiled experiment for accessing constants.
            is_displayed_expr: Optional per-player is_displayed expression.
                If provided, evaluated per player to decide whether to
                broadcast the context to that player.
        """
        for player in group.players:
            agent = agents_map.get(player.agent_instance_id)
            if agent is None:
                continue
            if is_displayed_expr:
                player_ctx = self._state.build_jinja_context(
                    agent=agent,
                    player=player,
                    group=group,
                    session=session,
                    module=module_name,
                    round_number=round_number,
                    constants=cep.constants,
                )
                if not self._flow.is_displayed(is_displayed_expr, player_ctx):
                    continue
            jinja_ctx = self._state.build_jinja_context(
                agent=agent,
                player=player,
                group=group,
                session=session,
                module=module_name,
                round_number=round_number,
                constants=cep.constants,
            )
            try:
                rendered = self._jinja_env.from_string(prompt.llm_text).render(
                    **jinja_ctx
                )
            except Exception as exc:
                logger.warning(
                    "Jinja rendering error in CONTEXT prompt: %s — using raw text.", exc
                )
                rendered = prompt.llm_text

            msg = make_message(
                role="user",
                content=rendered,
                sender_agent_id=agent.agent_id,
                group_id=group.group_id,
                visibility=VISIBILITY_PRIVATE,
                module=module_name,
                round_number=round_number,
            )
            agent.message_history.append(msg)

            self._event_logger.log(
                "context_broadcast",
                run_id=cep.run_id,
                session_id=session.session_id,
                group_id=group.group_id,
                agent_instance_id=player.agent_instance_id,
                data={
                    "prompt_type": "CONTEXT",
                    "module": module_name,
                    "round": round_number,
                    "rendered_prompt": rendered,
                },
            )

    def _execute_facilitator_prompt(
        self,
        prompt: PromptDefinition,
        group: Group,
        agents_map: dict[str, Agent],
        cep: CompiledExperiment,
        module_name: str,
        field_def: Optional[FieldDefinition] = None,
        jinja_ctx: dict | None = None,
        session: Optional[Session] = None,
        round_number: int = 0,
        is_displayed_expr: Optional[str] = None,
    ) -> str:
        """Execute a FACILITATOR-type prompt via the FacilitatorEngine.

        Looks up the facilitator function by name from the CEP definitions.
        The ``llm_text`` of a FACILITATOR prompt contains the function name
        as defined in the Facilitator worksheet (e.g. ``"evaluate_group"``).

        If the prompt has a ``field_class`` and ``field_name`` (via
        *field_def*), the facilitator's response value is stored in
        ``ExperimentState`` at the appropriate scope (Session, Group).

        Args:
            prompt: The FACILITATOR-type prompt definition.
            group: The group in whose context the facilitator runs.
            agents_map: Mapping from agent_instance_id to Agent objects.
            cep: Compiled experiment containing facilitator function definitions.
            module_name: Name of the current module.
            field_def: Optional ``FieldDefinition`` describing where to
                store the facilitator's response value.
            jinja_ctx: Optional Jinja context dict merged into the facilitator
                execution context.
            session: Optional ``Session`` instance for context.
            round_number: Current round number (1-based).
            is_displayed_expr: Optional Jinja2 boolean expression for
                conditional display evaluation.

        Returns:
            A stop signal string: ``"continue"``, ``"end_round"``, or
            ``"end_session"``.
        """
        from talkingtomachines.core.models import FacilitatorFunction

        func_name = prompt.llm_text.strip()
        matched_fn = None
        for fn_dict in cep.facilitator_functions:
            if fn_dict.get("name") == func_name:
                matched_fn = FacilitatorFunction(**fn_dict)
                break
        if matched_fn is None:
            raise RuntimeError(
                f"Facilitator function '{func_name}' not found in Facilitator worksheet."
            )
        fn = matched_fn

        engine = FacilitatorEngine(
            state=self._state,
            randomisation=self._rng_engine,
            router=self._router,
            model_name=cep.settings.get("model_name", ""),
            temperature=0.0,
        )
        # Build facilitator's view: union of all agents' session-wide
        # histories in this group (deduplicated, preserving insertion order).
        # This gives the facilitator visibility of everything its group
        # members have experienced across all modules and rounds.
        seen_ids: set[int] = set()
        facilitator_messages: list[dict] = []
        for player in group.players:
            agent = agents_map.get(player.agent_instance_id)
            if agent is None:
                continue
            for msg in agent.message_history:
                msg_id = id(msg)
                if msg_id not in seen_ids:
                    seen_ids.add(msg_id)
                    facilitator_messages.append(msg)

        # Gather session-wide agent IDs for built-in facilitators
        all_session_agent_ids = (
            [a.agent_id for a in session.agents]
            if session and hasattr(session, "agents")
            else [p.agent_id for p in group.players]
        )

        # Read players_per_group from module constants
        module_consts = cep.constants.get(module_name, {})
        players_per_group = int(
            module_consts.get("PLAYERS_PER_GROUP", len(all_session_agent_ids) or 1)
        )

        # Build profile info map for stratified assignment
        agents_profile_info: dict[str, dict] = {}
        if session and hasattr(session, "agents"):
            for a in session.agents:
                agents_profile_info[a.agent_id] = a.profile_info or {}

        base_context = {
            "group_id": group.group_id,
            "agent_ids": [p.agent_id for p in group.players],
            "all_session_agent_ids": all_session_agent_ids,
            "players_per_group": players_per_group,
            "agents_profile_info": agents_profile_info,
            "facilitator_messages": facilitator_messages,
        }

        is_player_scoped = field_def and field_def.field_class == "Player"

        if is_player_scoped:
            # Player-scoped facilitator: run once per player with
            # player-specific Jinja context, storing the result in
            # each player's field.  The facilitator still sees the
            # full group conversation built above.
            result = ""
            for player in group.players:
                agent = agents_map.get(player.agent_instance_id)
                if agent is None:
                    continue
                if is_displayed_expr:
                    disp_ctx = self._state.build_jinja_context(
                        agent=agent,
                        player=player,
                        group=group,
                        session=session,
                        module=module_name,
                        round_number=round_number,
                        constants=cep.constants,
                    )
                    if not self._flow.is_displayed(is_displayed_expr, disp_ctx):
                        continue
                player_ctx = self._state.build_jinja_context(
                    agent=agent,
                    player=player,
                    group=group,
                    session=session,
                    module=module_name,
                    round_number=round_number,
                    constants=cep.constants,
                )
                player_context = dict(base_context)
                player_context.update(player_ctx)
                player_result, rendered_def = engine.execute(
                    fn,
                    player_context,
                    prompt_args=prompt.kwargs,
                )
                self._event_logger.log(
                    "facilitator_call",
                    run_id=cep.run_id,
                    session_id=session.session_id if session else "",
                    group_id=group.group_id,
                    agent_instance_id=player.agent_instance_id,
                    data={
                        "prompt_type": "FACILITATOR",
                        "prompt_name": prompt.field_name,
                        "module": module_name,
                        "round": round_number,
                        "rendered_prompt": rendered_def,
                        "response": player_result,
                    },
                )
                if player_result:
                    result = player_result  # keep last for stop signal
                    self._state.set_player(
                        player.player_id, module_name, field_def.name, player_result
                    )
                    msg = make_message(
                        role="user",
                        content=player_result,
                        sender_agent_id="facilitator",
                        group_id=group.group_id,
                        visibility=VISIBILITY_FACILITATOR,
                        module=module_name,
                        round_number=round_number,
                    )
                    self._append_to_agent_histories(msg, group, agents_map)
        else:
            # Session / Group / no-field: single group-wide execution
            context = dict(base_context)
            if jinja_ctx:
                context.update(jinja_ctx)
            result, rendered_def = engine.execute(
                fn, context, prompt_args=prompt.kwargs
            )
            self._event_logger.log(
                "facilitator_call",
                run_id=cep.run_id,
                session_id=session.session_id if session else "",
                group_id=group.group_id,
                data={
                    "prompt_type": "FACILITATOR",
                    "prompt_name": prompt.field_name,
                    "module": module_name,
                    "round": round_number,
                    "rendered_prompt": rendered_def,
                    "response": result,
                },
            )

            # Apply state changes from built-in facilitators
            if isinstance(result, dict):
                self._apply_builtin_facilitator_result(
                    result,
                    fn.name,
                    session,
                    group,
                    agents_map,
                    module_name,
                )
            else:
                if result:
                    msg = make_message(
                        role="user",
                        content=result,
                        sender_agent_id="facilitator",
                        group_id=group.group_id,
                        visibility=VISIBILITY_FACILITATOR,
                        module=module_name,
                        round_number=round_number,
                    )
                    self._append_to_agent_histories(msg, group, agents_map)

                if field_def and field_def.name and result:
                    if field_def.field_class == "Session":
                        self._state.set_session(module_name, field_def.name, result)
                    elif field_def.field_class == "Group":
                        self._state.set_group(
                            group.group_id, module_name, field_def.name, result
                        )

        # Check raw text for stop signal keywords
        stop_text = result if isinstance(result, str) else ""
        return self._flow.evaluate_stop_condition(stop_text)

    def _apply_builtin_facilitator_result(
        self,
        result: dict,
        func_name: str,
        session: Optional[Session],
        group: Group,
        agents_map: dict[str, Agent],
        module_name: str,
    ) -> None:
        """Apply state changes from a built-in facilitator result.

        For ``assign_groups``, rebuilds the current group's player
        roster from the new group assignments.

        Args:
            result: The dict returned by the built-in facilitator.
            func_name: Name of the facilitator function that produced the result.
            session: The current session (used to look up all agents).
            group: The group that invoked the facilitator.
            agents_map: Mapping from agent_instance_id to Agent objects.
            module_name: Name of the current module.
        """
        if func_name == "assign_groups" and "group_assignments" in result:
            new_groups = result["group_assignments"]
            logger.info(
                "Runtime assign_groups: rebuilt %d group(s) in module %s.",
                len(new_groups),
                module_name,
            )
