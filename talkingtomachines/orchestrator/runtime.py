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
            for each prompt in task (filtered by is_displayed):
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
        on_task_complete: Optional[callable] = None,
    ) -> Session:
        """
        Execute the full experiment session.

        Args:
            session_number:  Numeric index of this session (for ID derivation).
            test_mode:       If True, run only one group per task.
            resume_session:  If provided, resume from this checkpoint. Already-completed
                             modules (those with subsessions) are skipped.
            on_task_complete: Optional callback invoked after each task
                completes. Called with ``(task_index, task_name, total_tasks)``
                to support progress reporting.

        Returns:
            The completed ``Session`` object.
        """
        cep = self._cep

        if resume_session is not None:
            session = resume_session
            session_id = session.session_id
            completed_task_names = {
                m.task_name for m in session.modules if m.subsessions
            }
            logger.info(
                "Resuming session %s — %d module(s) already completed.",
                session_id,
                len(completed_task_names),
            )
        else:
            session_id = make_session_id(cep.run_id, session_number)
            session = Session(
                session_id=session_id,
                run_id=cep.run_id,
                experiment_id=cep.experiment_id,
                cep_hash=cep.cep_hash,
            )
            completed_task_names = set()

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
                treatment_label="",
            )
            agents.append(agent)

        session.agents = agents

        # Initialize session via creating_session facilitator (skip if resuming)
        if resume_session is None:
            self._initialize_session(session, cep)

        stop_signal = "continue"
        # In test mode, lock to the first group's agents across all tasks
        test_group_agent_ids: set[str] | None = None

        # Execute modules (tasks)
        for task_idx, task_name in enumerate(cep.task_sequence):
            if stop_signal == "end_session":
                break

            # Skip already-completed modules when resuming
            if task_name in completed_task_names:
                logger.info("Skipping already-completed module: %s", task_name)
                continue

            module_id = make_module_id(session_id, task_name)
            module = Module(
                module_id=module_id,
                session_id=session_id,
                task_name=task_name,
            )
            session.modules.append(module)

            task_consts = cep.constants.get(task_name, {})
            max_rounds = int(task_consts.get("MAX_NUM_ROUNDS", 1))

            prompts = [PromptDefinition(**p) for p in cep.prompts.get(task_name, [])]

            for round_num in range(1, max_rounds + 1):
                if stop_signal == "end_session":
                    break

                # Resolve treatment labels for this task/round
                for agent in agents:
                    agent.treatment_label = assignment_plan.get_treatment(
                        agent.agent_id,
                        task_name,
                        round_num,
                    )

                subsession_id = make_subsession_id(module_id, round_num)
                subsession = Subsession(
                    subsession_id=subsession_id,
                    module_id=module_id,
                    round_number=round_num,
                )
                module.subsessions.append(subsession)

                # Form groups for this round
                precomputed_groups = assignment_plan.group_assignments.get(
                    task_name, {}
                ).get(round_num, {})

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

                if test_mode and groups:
                    if test_group_agent_ids is None:
                        # First task: capture agents from the first group
                        test_group_agent_ids = {p.agent_id for p in groups[0].players}
                        groups = groups[:1]
                    else:
                        # Subsequent tasks: pick the group whose members
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
                            task_name=task_name,
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
                            task_name=task_name,
                            round_number=round_num,
                            prompts=prompts,
                            cep=cep,
                        )
                        # Only propagate end_session across groups;
                        # end_round and continue are scoped to individual groups.
                        if group_signal == "end_session":
                            stop_signal = "end_session"
                            break

                # Checkpoint at subsession boundary
                self._checkpointer.save(session, self._state)

            # Notify progress after each task completes
            if on_task_complete is not None:
                on_task_complete(task_idx, task_name, len(cep.task_sequence))

        self._finalize_session(session)
        return session

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _initialize_session(self, session: Session, cep: CompiledExperiment) -> None:
        """Run the ``creating_session`` facilitator to set up initial state.

        Logs a ``session_start`` event and executes the ``creating_session``
        facilitator function if one is defined in the CEP.

        Args:
            session: The newly created session object.
            cep: Compiled experiment containing facilitator function definitions.
        """
        self._event_logger.log(
            "session_start",
            run_id=cep.run_id,
            session_id=session.session_id,
            data={"experiment_id": cep.experiment_id},
        )
        for fn_dict in cep.facilitator_functions:
            from talkingtomachines.core.models import FacilitatorFunction

            fn = FacilitatorFunction(**fn_dict)
            if fn.name == "creating_session":
                self._event_logger.log(
                    "facilitator_call",
                    run_id=cep.run_id,
                    session_id=session.session_id,
                    data={"facilitator": fn.name},
                )
                engine = FacilitatorEngine(
                    state=self._state,
                    randomisation=self._rng_engine,
                    router=self._router,
                    model_name=cep.settings.get("model_name", ""),
                    temperature=0.0,
                )
                engine.execute(fn, context={"session_id": session.session_id})
                break

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
        task_name: str,
        round_number: int,
        prompts: list[PromptDefinition],
        cep: CompiledExperiment,
    ) -> str:
        """Execute all prompts for a single group within a subsession.

        Iterates through each prompt in the task, dispatching to the
        appropriate handler based on prompt type (FACILITATOR, CONTEXT,
        DISCUSSION, PUBLIC_QUESTION, PRIVATE_QUESTION). Builds synthetic
        subjects for each player and evaluates flow-control expressions.

        Args:
            group: The group to execute prompts for.
            session: The parent session object.
            agents_map: Mapping from agent_instance_id to Agent objects.
            task_name: Name of the current task/module.
            round_number: Current round (subsession) number.
            prompts: Ordered list of prompt definitions for this task.
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
        from talkingtomachines.core.models import FieldDefinition

        field_defs_by_key = {
            (f["task"], f["field_class"], f["name"]): FieldDefinition(**f)
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
                    task=task_name,
                    round_number=round_number,
                    constants=cep.constants,
                )
            else:
                jinja_ctx = {"round_number": round_number, "group_id": group.group_id}
        else:
            jinja_ctx = {"round_number": round_number, "group_id": group.group_id}

        stop_signal = "continue"

        for prompt in prompts:
            # Evaluate is_displayed
            if not self._flow.is_displayed(prompt.is_displayed, jinja_ctx):
                continue

            # Fetch field definition if applicable
            field_def = None
            if prompt.field_class and prompt.field_name:
                field_key = (task_name, prompt.field_class, prompt.field_name)
                field_def = field_defs_by_key.get(field_key)

            if prompt.type == "FACILITATOR":
                self._event_logger.log(
                    "facilitator_call",
                    run_id=cep.run_id,
                    session_id=session.session_id,
                    group_id=group.group_id,
                    data={
                        "prompt_name": prompt.field_name,
                        "task": task_name,
                        "round": round_number,
                    },
                )
                stop_signal = self._execute_facilitator_prompt(
                    prompt, group, agents_map, cep, task_name, field_def, jinja_ctx
                )
                if stop_signal in ("end_round", "end_session"):
                    break

            elif prompt.type == "CONTEXT":
                self._broadcast_context(
                    prompt, group, session, task_name, round_number, agents_map, cep
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
                    t0 = time.perf_counter()
                    content = subject.respond(
                        task_name, round_number, prompt, field_def
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
                            "task": task_name,
                            "round": round_number,
                            "turn_id": turn_id,
                            "latency_ms": round(latency_ms, 2),
                        },
                    )
                    msg = make_message(
                        role="assistant",
                        content=content,
                        sender_agent_instance_id=agent_id,
                        group_id=group.group_id,
                        treatment_label=agents_map.get(
                            agent_id, Agent("", "", {})
                        ).treatment_label,
                        visibility=VISIBILITY_GROUP_ONLY,
                        task=task_name,
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
                    t0 = time.perf_counter()
                    content = subject.respond(
                        task_name, round_number, prompt, field_def
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
                            "task": task_name,
                            "round": round_number,
                            "field_name": prompt.field_name,
                            "latency_ms": round(latency_ms, 2),
                        },
                    )
                    pending_messages.append(
                        make_message(
                            role="assistant",
                            content=content,
                            sender_agent_instance_id=agent_id,
                            group_id=group.group_id,
                            treatment_label=agents_map.get(
                                agent_id, Agent("", "", {})
                            ).treatment_label,
                            visibility=VISIBILITY_GROUP_ONLY,
                            task=task_name,
                            round_number=round_number,
                        )
                    )
                # Append all responses after everyone has answered
                for msg in pending_messages:
                    self._append_to_agent_histories(msg, group, agents_map)

            elif prompt.type == "PRIVATE_QUESTION":
                # Parallel: each agent's private answer is invisible to others
                def _respond_private(agent_id: str):
                    subj = subjects.get(agent_id)
                    if subj is None:
                        return agent_id, "", 0.0
                    t0 = time.perf_counter()
                    cnt = subj.respond(task_name, round_number, prompt, field_def)
                    return agent_id, cnt, (time.perf_counter() - t0) * 1000

                max_player_workers = int(cep.settings.get("max_player_workers", 1))
                if max_player_workers > 1 and len(group.turn_order) > 1:
                    raw_results = run_private_questions_parallel(
                        _respond_private,
                        [(aid,) for aid in group.turn_order],
                        max_workers=max_player_workers,
                    )
                else:
                    raw_results = [_respond_private(aid) for aid in group.turn_order]

                for agent_id, content, latency_ms in raw_results:
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
                            "task": task_name,
                            "round": round_number,
                            "field_name": prompt.field_name,
                            "latency_ms": round(latency_ms, 2),
                        },
                    )
                    msg = make_message(
                        role="assistant",
                        content=content,
                        sender_agent_instance_id=agent_id,
                        group_id=group.group_id,
                        treatment_label=agents_map.get(
                            agent_id, Agent("", "", {})
                        ).treatment_label,
                        visibility=VISIBILITY_PRIVATE,
                        task=task_name,
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
        sender = msg.get("sender_agent_instance_id", "")
        for player in group.players:
            agent = agents_map.get(player.agent_instance_id)
            if agent is None:
                continue
            agent_msg = dict(msg)
            if sender == agent.agent_instance_id:
                agent_msg["role"] = "assistant"
            else:
                agent_msg["role"] = "user"
            agent.message_history.append(agent_msg)

    def _broadcast_context(
        self,
        prompt: PromptDefinition,
        group: Group,
        session: Session,
        task_name: str,
        round_number: int,
        agents_map: dict[str, Agent],
        cep: CompiledExperiment,
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
            task_name: Name of the current task/module.
            round_number: Current round (subsession) number.
            agents_map: Mapping from agent_instance_id to Agent objects.
            cep: Compiled experiment for accessing constants.
        """
        for player in group.players:
            agent = agents_map.get(player.agent_instance_id)
            if agent is None:
                continue
            jinja_ctx = self._state.build_jinja_context(
                agent=agent,
                player=player,
                group=group,
                session=session,
                task=task_name,
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
                sender_agent_instance_id=player.agent_instance_id,
                group_id=group.group_id,
                visibility=VISIBILITY_PRIVATE,
                task=task_name,
                round_number=round_number,
            )
            agent.message_history.append(msg)

    def _execute_facilitator_prompt(
        self,
        prompt: PromptDefinition,
        group: Group,
        agents_map: dict[str, Agent],
        cep: CompiledExperiment,
        task_name: str,
        field_def: Optional[FieldDefinition] = None,
        jinja_ctx: dict | None = None,
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
            task_name: Name of the current task/module.
            field_def: Optional ``FieldDefinition`` describing where to
                store the facilitator's response value.
            jinja_ctx: Optional Jinja context dict merged into the facilitator
                execution context.

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
        # members have experienced across all tasks and rounds.
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

        context = {
            "group_id": group.group_id,
            "agent_ids": [p.agent_id for p in group.players],
            "facilitator_messages": facilitator_messages,
        }
        # Merge Jinja context so facilitator definitions can use {{ C.task.name }}, etc.
        if jinja_ctx:
            context.update(jinja_ctx)
        result = engine.execute(fn, context)

        # Append facilitator response to agent histories (facilitator-only visibility)
        if result:
            msg = make_message(
                role="user",
                content=result,
                sender_agent_instance_id="facilitator",
                group_id=group.group_id,
                visibility=VISIBILITY_FACILITATOR,
                task=task_name,
                round_number=0,  # facilitator not tied to a specific round
            )
            self._append_to_agent_histories(msg, group, agents_map)

        # Store facilitator response in ExperimentState if field is defined
        if field_def and field_def.name and result:
            if field_def.field_class == "Session":
                self._state.set_session(task_name, field_def.name, result)
            elif field_def.field_class == "Group":
                self._state.set_group(group.group_id, task_name, field_def.name, result)

        # Check raw text for stop signal keywords
        return self._flow.evaluate_stop_condition(result)
