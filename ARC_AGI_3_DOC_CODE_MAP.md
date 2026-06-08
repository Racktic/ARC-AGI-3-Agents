# ARC-AGI-3 — Docs ↔ Code Reference Map

A correspondence map between the official docs at <https://docs.arcprize.org/> and the actual
code in this repo (`agents/`) plus the installed toolkit SDK
(`.venv/lib/python3.12/site-packages/arc_agi`, `arcengine`). Built by reading the full doc
index (~50 pages) and verifying the headline claims against source.

> File:line anchors in §1–§5 were read directly. Template line numbers in §6 came from
> sub-agent exploration — spot-check before relying on them for edits.

---

## ⭐ Meta-finding (read this first)

**RESOLVED 2026-06-03 — local SDK upgraded `arc-agi 0.9.1 → 0.9.8` to match the docs.**
The agents repo was always current (local HEAD == upstream `main`); the only thing behind was our
frozen `uv.lock` (`arc-agi==0.9.1`). After `uv lock --upgrade-package arc-agi` + `uv sync` the
installed SDK now matches the live docs (verified in source):
- **Scoring** `arc_agi/scorecard.py:170-206`: `score = ((baseline/actions)**2)*100`, capped **115**
  (1.15×), per-game **weighted by 1-indexed level** — exactly the docs formula.
- **OperationMode** `base.py:50`: `COMPETITION` now present; `listen_and_serve` now exists.
- Installed: `arc-agi 0.9.8`, `arcengine 0.9.3`. Required collateral: `pillow 11.3.0 → 12.2.0`
  (arc-agi 0.9.8 requires `pillow>=12.1.1`), which forced a 1-line fix in
  `agents/templates/langgraph_thinking/vision.py` (added `from __future__ import annotations`
  because Pillow 12 removed `ImageDraw.Coords`). This is also a latent upstream bug: upstream
  `main` + latest `arc-agi` breaks identically without that fix.

> Historical (pre-upgrade) state, for reference: SDK 0.9.1 used linear `(baseline/actions)*100`
> capped 100, simple average, and lacked COMPETITION/listen_and_serve — i.e. it behaved like an
> older doc version. Any eval logs in `recordings/anthropic-opus-4-6-max-effort/` were produced
> under that older/online stack; their SDK-computed scores use the linear formula.

---

## 1. Core data model (`arcengine`) — verified

- `GameState` (4 values, not 3): `NOT_PLAYED, NOT_FINISHED, WIN, GAME_OVER`.
- `GameAction` (8): `RESET, ACTION1..ACTION5, ACTION6, ACTION7`. `ACTION6` = `ComplexAction`
  with `x,y ∈ [0,63]`; the rest are `SimpleAction` (`game_id` only). Docs label 1–4 as
  up/down/left/right, but that is a **generic keyboard hint, not the per-game effect** — true
  action semantics are hidden by design and must be discovered by trying.
- `FrameData`: `game_id, frame (list[level][64][64] ints 0-15), state, levels_completed (0-254),
  win_levels (0-254), action_input{id,data,reasoning}, guid, full_reset, available_actions[int]`.
  In 0.9.3 `score`→`levels_completed`, `win_score`→`win_levels`.
- `ActionInput.reasoning`: opaque client blob, **≤16 KB** (`MAX_REASONING_BYTES = 16*1024`),
  stored and echoed back verbatim in the next frame.

## 2. Agent loop (this repo) — verified

- `agents/agent.py` `Agent` (ABC): implement `is_done(frames, latest)` and
  `choose_action(frames, latest)`.
- `main()` (`agent.py:69-89`): `while not is_done(...) and action_counter <= MAX_ACTIONS`.
  Default `MAX_ACTIONS = 80` (`:22`); `Playback` overrides to 1e6.
- `do_action_request()` (`agent.py:133-142`): reads `action.reasoning`, wraps bare strings as
  `{"text": ...}`, calls `arc_env.step(action, data, reasoning)`.
- Registration (`agents/__init__.py:20-31`): auto-registers `Agent.__subclasses__()` by
  lowercased class name. `ReasoningAgent` needs a manual entry (it subclasses `ReasoningLLM`,
  not `Agent`). Recordings register as `Playback`.

## 3. Toolkit SDK ↔ REST (`arc_agi`) — verified

- `Arcade` (`base.py`): `make`, `open_scorecard`/`create_scorecard`, `close_scorecard`,
  `get_scorecard`, `get_environments`. Env config: `ARC_API_KEY`, `ARC_BASE_URL`
  (`https://three.arcprize.org`), `OPERATION_MODE`, `ENVIRONMENTS_DIR`, `RECORDINGS_DIR`.
  Auto-fetches an anonymous key if none set and not OFFLINE.
- `OperationMode` (`base.py:45-47`): NORMAL (local+API), ONLINE (API only), OFFLINE (local only).
  ONLINE → `RemoteEnvironmentWrapper`, otherwise `LocalEnvironmentWrapper`.
- REST (online): `GET /api/games`; `POST /api/cmd/RESET`; `POST /api/cmd/ACTION{1..7}`
  (ACTION6 adds `x,y`); `POST /api/scorecard/open` → `{card_id}`; `POST /api/scorecard/close`
  → summary; `GET /api/scorecard/{card_id}` and `.../{card_id}/{game_id}`. Header `X-API-Key`;
  session affinity via `AWSALB*` cookies (handled by `requests.Session`). Rate limit 600 RPM → 429.
- `remote_wrapper.step()` serializes `reasoning` to a JSON string; `local_wrapper.step()` keeps a dict.

## 4. Scorecard & scoring — verified

- `swarm.py`: opens one scorecard, one agent-thread per game, joins, closes. `main.py` prints
  `{ROOT_URL}/scorecards/{card_id}` (ONLINE only) and a SIGINT handler closes on Ctrl+C.
- Stale auto-close after 15 min (`STALE_MINUTES`, `scorecard.py`); leaderboard batches ~15 min.
- SDK per-level (`scorecard.py:125-126`): `(baseline/actions)*100`, `min(...,100)`; per-game
  (`:146`) simple mean; total = mean of games. (Official docs formula differs — see meta-finding.)

## 5. Recordings — verified

- New SDK path (`wrapper.py:86-90`): `recordings/{scorecard_id}/{game_id}-{guid}.jsonl` — **this is
  exactly the format of the eval logs in `recordings/anthropic-opus-4-6-max-effort/`**
  (e.g. `ar25-….json`), confirming those came from this SDK / the online API.
- Older `agents/recorder.py` path: `{prefix}.{guid}.recording.jsonl` (prefix = `game_id.agentname`),
  used by `Playback`. Each JSONL line = `{timestamp, data}`.

## 6. Agent templates (this repo)

- `llm_agents.py`: `LLM` (gpt-4o-mini, `DO_OBSERVATION=True` → 2 calls/turn, `MESSAGE_LIMIT=10`,
  function-calling), `FastLLM` (no observation, 1 call), `ReasoningLLM` (o4-mini, tools),
  `GuidedLLM` (o3, `REASONING_EFFORT=high`, **hard-codes LockSmith rules** = upper-bound, not
  a generalization result).
- `reasoning_agent.py`: renders grid → PNG (16-colour palette, zone labels), structured output
  with `hypothesis` + `aggregated_findings`, o4-mini, `MAX_ACTIONS=400`.
- `multimodal.py`: image input + prev/next frame diff, self-updating memory prompt.
- `openclaw_agent/`: server-side session memory via `x-openclaw-session-key`; per-run id;
  built-in memory/file tools referenced in its system prompt.
- `langgraph_*`, `smolagents.py`: framework integrations (LangGraph state graphs incl. SQLite
  persistence in `langgraph_thinking`; smolagents CodeAgent / Vision).
- `tracing.py`: optional AgentOps via `@trace_agent_session` (no-op if unconfigured).

## 7. Two ways to evaluate

- **This repo** (dev/research): `uv run main.py --agent=<name> [--game=<prefix,prefix>] [--tags=…]`.
  Omitting `--game` ⇒ swarm plays all games. `--game` is **prefix match**.
- **Official benchmarking** (separate repo `arc-agi-3-benchmarking`, BETA):
  `uv run python -m arcagi3.runner --check | --list-games | --list-models` and
  `--game_id … --config … --max_actions …`; saves scorecards server-side.
- Game discovery: `GET /api/games`; anonymous key sees a few games, a full key sees more.
  ⚠️ README/llms.txt examples `ls20`/`locksmith` are **stale**; real ids are hashed
  (`sc25-…, cn04-…, cd82-…, bp35-…, ar25-…`).

## Discrepancies at a glance (after the 0.9.8 upgrade)

| Topic | Docs (current) | Installed SDK 0.9.8 | Status |
|---|---|---|---|
| Per-level score | `(baseline/ai)^2`, cap 1.15× | `((baseline/actions)**2)*100`, cap 115 | ✅ aligned |
| Per-game score | level-weighted (1-indexed) | level-weighted (`scorecard.py:197-206`) | ✅ aligned |
| OperationMode | + COMPETITION | NORMAL/ONLINE/OFFLINE/COMPETITION | ✅ aligned |
| Toolkit | `listen_and_serve()` | present | ✅ aligned |
| GameState | often 3 listed | 4 (incl. `NOT_PLAYED`) | doc omission (unchanged) |
| ACTION6 range | 0-29 (one page) vs 0-63 | 0-63 enforced | doc conflict (unchanged) |
| Recording filename | older `{…}.recording.jsonl` | `{scorecard_id}/{game_id}-{guid}.jsonl` | doc lag (unchanged) |

## How to re-verify

- Open a cited line directly: `scorecard.py:125`, `base.py:45`, `wrapper.py:86`, `agent.py:22,133`.
- Live smoke test (free, no LLM): `uv run main.py --agent=random --game=<id> --tags=verify`
  and confirm a scorecard URL prints and `recordings/{card_id}/{game_id}-{guid}.jsonl` appears.
