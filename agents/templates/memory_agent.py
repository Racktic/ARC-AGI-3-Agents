"""
MemoryAgent — a memory/representation-ablation agent for ARC-AGI-3.

Scheme B (two Claude calls per turn):
  1) ACT:    input = system + MEMORY + CURRENT STATE  -> tool call -> one action.
  2) REFLECT/REWRITE: AFTER the action's result frame is visible (on the next
     turn's entry), update MEMORY from (action, before, after, changed?).

Independent variables (env switches):
  MEM_MODE  = append | rewrite   (how memory is updated)
  MEM_REPR  = matrix | image     (how the grid is shown to the model)

The ACT call never sees the rolling message history — only MEMORY + the
current frame — so MEMORY is the only carrier of past information.

Model = Claude via an Anthropic-compatible endpoint (default local proxy).
"""
import base64
import json
import os
import textwrap
from io import BytesIO
from typing import Any, Optional

import anthropic
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

# Standard ARC-AGI 16-colour palette (value -> RGB).
PALETTE = {
    0: (255, 255, 255), 1: (204, 204, 204), 2: (153, 153, 153), 3: (102, 102, 102),
    4: (51, 51, 51), 5: (0, 0, 0), 6: (229, 58, 163), 7: (255, 123, 204),
    8: (249, 60, 49), 9: (30, 147, 255), 10: (136, 216, 241), 11: (255, 220, 0),
    12: (255, 133, 27), 13: (146, 18, 49), 14: (79, 204, 48), 15: (163, 86, 214),
}


class MemoryAgent(Agent):
    MAX_ACTIONS: int = int(os.environ.get("MEM_MAX_ACTIONS", "60"))
    MODEL: str = os.environ.get("CLAUDE_MODEL", "claude-opus-4-6")
    MAX_MEMORY: int = int(os.environ.get("MEM_MAX_MEMORY", "0"))  # 0 = keep all
    MEM_MODE: str = os.environ.get("MEM_MODE", "append")          # append | rewrite
    MEM_REPR: str = os.environ.get("MEM_REPR", "matrix")          # matrix | image
    CELL: int = int(os.environ.get("MEM_CELL", "10"))            # px per grid cell (image mode)
    GRID_STEP: int = int(os.environ.get("MEM_GRID_STEP", "4"))   # gridline/label spacing (cells)
    # If 1, the action meanings are written INTO the prompt (study "given the
    # controls, can it plan & solve?" instead of "can it discover the controls?").
    TELL_ACTIONS: bool = os.environ.get("MEM_TELL_ACTIONS", "0") == "1"
    # MEM_PREDICT: 0=off, 1=action-effect prediction (check single-step mechanic),
    # 2=also GOAL/progress falsification. Single-step predictions mostly confirm the
    # already-correct action model (level 1 rarely fires); level 2 aims falsification at
    # the win-condition hypothesis via a hard "N turns, 0 level progress" counter +
    # contradiction check, forcing the agent to question the GOAL frame, not just the move.
    # 3=also STAGED escalation + a persistent "tried & ruled out" log, so a high
    # no-progress counter drives a coverage-based untried-approach sweep instead of
    # per-turn "radical rethink" thrashing (the failure seen at level 2).
    PREDICT_LEVEL: int = int(os.environ.get("MEM_PREDICT", "0") or "0")
    PREDICT: bool = PREDICT_LEVEL >= 1
    GOAL_CHECK: bool = PREDICT_LEVEL >= 2
    SYSTEMATIC: bool = PREDICT_LEVEL >= 3
    # If 1, feed an EXACT computed before/after diff (per-object bbox + rigid dx/dy) into
    # the memory-update step alongside the images, so the model is TOLD what moved instead
    # of having to eyeball it (image perception missed a 3-cell slide of a 45-cell block).
    SHOW_DIFF: bool = os.environ.get("MEM_DIFF", "0") == "1"
    HUMAN_ACTIONS = {
        "RESET": "start/restart the game",
        "ACTION1": "Move Up", "ACTION2": "Move Down",
        "ACTION3": "Move Left", "ACTION4": "Move Right",
        "ACTION5": "Perform action / interact",
        "ACTION6": "Click an object at coordinates x,y",
        "ACTION7": "Undo the last move",
    }

    _DOC_TEMPLATE = textwrap.dedent("""
        ## Observed action effects (state-dependent — re-test, don't write off)
        (none yet)

        ## Hypothesis about the goal / win condition
        (unknown)

        ## Current plan
        (probe each action to learn its effect)

        ## Uncertain / to test
        (everything)
    """).strip()

    _SYSTEM_TMPL = textwrap.dedent("""
        You are playing an unfamiliar turn-based grid game. The screen is a
        64x64 grid of cells, each one of 16 colours. The full action set is
        RESET and ACTION1–ACTION7 (ACTION6 is a click at x,y); there is no ACTION0
        or any other action. But NOT all of them are usable every turn — each turn
        only the actions in `available_actions` (shown to you, and enforced by the
        act tool's enum) can be taken.
        {action_clause}
        Your goal is to increase levels_completed and reach state=WIN with as few
        actions as possible. Any counter/bar drawn on the screen is NOT your
        score — ignore it as a goal. You are given MEMORY: insights you wrote on
        previous turns. You do NOT get the full history of past frames, so rely
        on MEMORY.

        IMPORTANT — action effects can still be STATE-DEPENDENT: the SAME action
        can do nothing in one state yet work in another (e.g. a move blocked by a
        wall here may succeed elsewhere). Don't conclude an action is permanently
        useless from a single no-op; hypothesize WHY and re-test as the state changes.
    """).strip()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.TELL_ACTIONS:
            desc = "\n".join(f"  {k}: {v}" for k, v in self.HUMAN_ACTIONS.items())
            clause = "Each action's effect is given below — use these to plan and solve:\n" + desc
        else:
            clause = ("The meaning of each action is NOT given; you must infer it "
                      "from how the screen changes.")
        self.SYSTEM = self._SYSTEM_TMPL.format(action_clause=clause)
        self.client = anthropic.Anthropic(
            base_url=os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:5678"),
            api_key=os.environ.get("ANTHROPIC_API_KEY", "anykey"),
            max_retries=int(os.environ.get("MEM_MAX_RETRIES", "8")),  # auto backoff on 429
            timeout=180.0,
        )
        self.memory: list[str] = []                 # append mode
        self.memory_doc: str = self._DOC_TEMPLATE   # rewrite mode
        self._prev_action: Optional[GameAction] = None
        self._prev_grid2d: Optional[list] = None
        self._prev_prediction: Optional[str] = None
        self._prev_goalpred: Optional[str] = None
        self._last_levels: int = 0          # for the hard no-progress counter
        self._stuck: int = 0                # turns since levels_completed last increased
        self._turn: int = 0
        # all outputs grouped under memruns/<MEM_RUN>/{io.log, images/}
        run = os.environ.get("MEM_RUN", "run")
        base = os.path.join(os.environ.get("MEM_OUT_ROOT", "memruns"), run)
        self._log_path = os.environ.get("MEM_LOG_FILE") or os.path.join(base, "io.log")
        self._img_dir = os.environ.get("MEM_IMG_DIR") or os.path.join(base, "images")
        self._call_no = 0
        if self._log_path:
            os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)
            open(self._log_path, "w").close()
        if self.MEM_REPR == "image":
            os.makedirs(self._img_dir, exist_ok=True)

    def _log(self, s: str) -> None:
        if not self._log_path:
            return
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    @property
    def name(self) -> str:
        mode = "memRW" if self.MEM_MODE == "rewrite" else "memAppend"
        return f"{super().name}.{self.MODEL.replace('/', '-')}.{mode}.{self.MEM_REPR}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    # ---------- representation ----------
    def _final_grid(self, latest_frame: FrameData) -> list:
        """The final settled grid of the frame (a frame may be an animation)."""
        return latest_frame.frame[-1] if latest_frame.frame else []

    def _grid_text(self, grid2d: list) -> str:
        if not grid2d:
            return "(no grid)"
        return "\n".join("  " + " ".join(str(v) for v in row) for row in grid2d)

    def _grid_diff_text(self, before2d: list, after2d: list) -> str:
        """Exact per-object diff of two grids (what changed, and rigid translations).

        Computed straight from the integer grids — the model is told to TRUST this over
        eyeballing the image, which can miss a moved object (e.g. a 45-cell block sliding
        3 cells reads as 'did not move')."""
        from collections import defaultdict
        if not before2d or not after2d or len(before2d) != len(after2d):
            return "(diff unavailable)"
        H = len(before2d); W = len(before2d[0]) if H else 0
        cb: dict[int, set] = defaultdict(set); ca: dict[int, set] = defaultdict(set)
        changed = 0
        for r in range(H):
            for c in range(W):
                b, a = before2d[r][c], after2d[r][c]
                cb[b].add((r, c)); ca[a].add((r, c))
                if b != a:
                    changed += 1
        if changed == 0:
            return ("COMPUTED DIFF: NOTHING changed — every one of the "
                    f"{H * W} cells is identical. The action was a true NO-OP.")

        def box(s: set) -> tuple:
            rs = [p[0] for p in s]; cs = [p[1] for p in s]
            return (min(cs), max(cs), min(rs), max(rs))

        lines = []
        for v in sorted(set(cb) | set(ca)):
            B, A = cb.get(v, set()), ca.get(v, set())
            if B == A:
                continue  # this value's cells did not change
            tag = f"value {v}"
            if not B:
                x0, x1, y0, y1 = box(A)
                lines.append(f"  {tag}: APPEARED, n={len(A)} at x[{x0}-{x1}] y[{y0}-{y1}]")
                continue
            if not A:
                lines.append(f"  {tag}: DISAPPEARED (was n={len(B)})")
                continue
            x0b, x1b, y0b, y1b = box(B); x0a, x1a, y0a, y1a = box(A)
            note = ""
            if len(B) == len(A):
                dx, dy = x0a - x0b, y0a - y0b
                if {(r + dy, c + dx) for (r, c) in B} == A:
                    note = f"  => MOVED rigidly dx={dx:+d} dy={dy:+d} (n={len(B)} unchanged)"
            if not note:
                note = f"  (n {len(B)}->{len(A)}, shape changed)"
            lines.append(f"  {tag}: x[{x0b}-{x1b}] y[{y0b}-{y1b}] -> "
                         f"x[{x0a}-{x1a}] y[{y0a}-{y1a}]{note}")
        return (f"COMPUTED DIFF (exact, from the raw grid — TRUST THIS over reading the image): "
                f"{changed} of {H * W} cells changed. Per-value change "
                f"(x=col 0-63, y=row 0-63; values with NO change are omitted; dx>0=right, "
                f"dy>0=down):\n" + "\n".join(lines))

    def _grid_image(self, grid2d: list) -> bytes:
        """Render a 64x64 grid to a colour PNG with coordinate gridlines/labels."""
        from PIL import Image, ImageDraw
        H = len(grid2d); W = len(grid2d[0]) if H else 0
        cell, m = self.CELL, 20
        img = Image.new("RGB", (m + W * cell, m + H * cell), (255, 255, 255))
        d = ImageDraw.Draw(img)
        for r in range(H):
            for c in range(W):
                col = PALETTE.get(int(grid2d[r][c]), (180, 180, 180))
                x0, y0 = m + c * cell, m + r * cell
                d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=col)
        # coordinate gridlines + labels every GRID_STEP cells (x=col, y=row)
        step = max(1, self.GRID_STEP)
        for k in range(0, W + 1, step):
            x = m + k * cell
            d.line([(x, m), (x, m + H * cell)], fill=(120, 120, 120))
            if k < W:
                d.text((x + 1, 5), str(k), fill=(0, 0, 0))
        for k in range(0, H + 1, step):
            y = m + k * cell
            d.line([(m, y), (m + W * cell, y)], fill=(120, 120, 120))
            if k < H:
                d.text((2, y + 1), str(k), fill=(0, 0, 0))
        buf = BytesIO(); img.save(buf, "PNG")
        return buf.getvalue()

    def _image_block(self, png: bytes) -> dict:
        return {"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": base64.b64encode(png).decode()}}

    def _save_img(self, png: bytes, label: str) -> str:
        path = os.path.join(self._img_dir, f"{label}.png")
        with open(path, "wb") as f:
            f.write(png)
        return path

    def _avail(self, latest_frame: FrameData) -> list[int]:
        out = []
        for a in latest_frame.available_actions or []:
            out.append(a.value if hasattr(a, "value") else int(a))
        return out

    # ---------- Claude calls ----------
    def _action_tool(self, avail: list[int]) -> dict:
        names = [f"ACTION{i}" for i in avail] or \
            ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]
        props: dict[str, Any] = {
            "action": {"type": "string", "enum": names},
            "reasoning": {"type": "string", "description": "one short sentence"},
        }
        if 6 in avail:
            props["x"] = {"type": "integer", "description": "0-63, for ACTION6"}
            props["y"] = {"type": "integer", "description": "0-63, for ACTION6"}
        required = ["action", "reasoning"]
        if self.PREDICT:
            props["prediction"] = {"type": "string", "description":
                "A SPECIFIC, checkable prediction of what THIS action will change on the "
                "screen, derived from your CURRENT world-model — concrete enough to verify "
                "next turn (e.g. 'the head at (50,48) moves left to (48,48) and the gray "
                "piece at (36,18) moves right to (38,18)'). It WILL be checked against reality."}
            required.append("prediction")
        if self.GOAL_CHECK:
            props["goal_progress_pred"] = {"type": "string", "description":
                "If your GOAL hypothesis + current plan are correct, what OBSERVABLE progress "
                "toward winning should appear, and by roughly when? (e.g. 'levels_completed "
                "should reach 1 within ~3 moves once the gray block overlaps the yellow target'). "
                "Be concrete — this is checked against levels_completed and the screen so a wrong "
                "GOAL frame gets falsified, not just a wrong move."}
            required.append("goal_progress_pred")
        return {
            "name": "act",
            "description": "Choose exactly one game action for this turn. "
                           "Only the actions listed in the enum are available now.",
            "input_schema": {"type": "object", "properties": props, "required": required},
        }

    def _memory_block(self) -> str:
        if self.MEM_MODE == "rewrite":
            return self.memory_doc
        if not self.memory:
            return "(empty — this is the first action)"
        return "\n".join(self.memory)

    _IMG_NOTE = ("(the current grid is the attached IMAGE; x = column 0-63 "
                 "left→right, y = row 0-63 top→bottom; light gridlines and "
                 "number labels mark every 8 cells)")

    def _act(self, latest_frame: FrameData) -> GameAction:
        avail = self._avail(latest_frame)
        tool = self._action_tool(avail)
        grid2d = self._final_grid(latest_frame)
        head = textwrap.dedent(f"""
            # MEMORY (insights you accumulated; no raw history is given)
            {self._memory_block()}

            # CURRENT STATE
            state: {latest_frame.state.name}
            levels_completed: {latest_frame.levels_completed}
            available_actions: {avail}
        """).strip()
        tail = ("# TURN\nPick exactly one action via the `act` tool (its enum "
                "lists the only actions available this turn). ACTION6 needs x,y in 0-63.")
        if self.PREDICT:
            tail += ("\nAlso fill `prediction`: under your CURRENT world-model, the exact, "
                     "checkable change this action should cause — it will be tested next turn "
                     "to confirm or REFUTE your model.")
        if self.GOAL_CHECK:
            tail += ("\nAlso fill `goal_progress_pred`: the observable PROGRESS toward winning "
                     "your goal hypothesis predicts (and by when). Many correct single moves with "
                     "NO progress toward the win is evidence your GOAL frame is wrong.")

        self._call_no += 1
        self._log("\n" + "#" * 100)
        self._log(f"# CALL {self._call_no}  [ACT]  turn~{self._turn + 1}  "
                  f"repr={self.MEM_REPR}  model={self.MODEL}")
        self._log("#" * 100)
        self._log("----- INPUT system -----\n" + self.SYSTEM)

        if self.MEM_REPR == "image":
            png = self._grid_image(grid2d)
            path = self._save_img(png, f"t{self._turn + 1}_act")
            body = head + "\n\n# CURRENT GRID\n" + self._IMG_NOTE + "\n\n" + tail
            content = [self._image_block(png), {"type": "text", "text": body}]
            self._log(f"----- INPUT image ----- [grid rendered -> {path}]")
            self._log("----- INPUT user -----\n" + body)
        else:
            body = head + "\n\n# CURRENT GRID\n" + self._grid_text(grid2d) + "\n\n" + tail
            content = body
            self._log("----- INPUT user -----\n" + body)
        self._log("----- INPUT tools -----\n" + json.dumps(tool) +
                  '\ntool_choice = {"type":"tool","name":"act"}')

        resp = self.client.messages.create(
            model=self.MODEL, max_tokens=1024, system=self.SYSTEM,
            tools=[tool], tool_choice={"type": "tool", "name": "act"},
            messages=[{"role": "user", "content": content}],
        )
        tu = next((b for b in resp.content if b.type == "tool_use"), None)
        self._log("===== OUTPUT (act) =====")
        self._log("tool_use input: " + (json.dumps(tu.input) if tu else "(none)"))
        self._log(f"usage: in={resp.usage.input_tokens} out={resp.usage.output_tokens}")
        if tu is None:
            return GameAction.ACTION5
        inp = tu.input
        name = inp.get("action", "ACTION5")
        action = GameAction.from_name(name)
        if name == "ACTION6":
            action.set_data({"x": int(inp.get("x", 0)), "y": int(inp.get("y", 0))})
        action.reasoning = {"reasoning": inp.get("reasoning", ""), "mode": self.MEM_MODE}
        if self.PREDICT:
            self._prev_prediction = inp.get("prediction", "")
        if self.GOAL_CHECK:
            self._prev_goalpred = inp.get("goal_progress_pred", "")
        return action

    def _changed(self, before2d: list, after2d: list) -> bool:
        return before2d != after2d

    def _update_memory(self, action: GameAction, before2d: list, latest_frame: FrameData) -> None:
        """REFLECT (append) or REWRITE the memory from the just-observed transition."""
        after2d = self._final_grid(latest_frame)
        changed = self._changed(before2d, after2d)
        data = action.action_data.model_dump()
        xy = f" at (x={data.get('x')}, y={data.get('y')})" if "x" in data else ""
        chg = "CHANGED" if changed else "did NOT change (no-op)"
        info = (f"You just took {action.name}{xy}. The screen {chg}. "
                f"levels_completed is now {latest_frame.levels_completed}, "
                f"state={latest_frame.state.name}.")

        # Hard no-progress counter (harness-computed, not the model's vibe)
        lv = latest_frame.levels_completed or 0
        if lv > self._last_levels:
            self._stuck = 0
            self._last_levels = lv
        else:
            self._stuck += 1

        # Prediction check block (predict→check→falsify)
        pred_block = ""
        if self.PREDICT and self._prev_prediction:
            pred_block = (
                "PREDICTION CHECK — last turn, under your current world-model, you predicted:\n"
                f'  "{self._prev_prediction}"\n'
                "Compare it to what ACTUALLY happened (shown below). FIRST state clearly whether\n"
                "the prediction HELD or FAILED. If it FAILED, decide: was it a one-off (the action\n"
                "was blocked by THIS particular state, so your rule may still hold) OR is your\n"
                "world-model WRONG? If your model is likely wrong — ESPECIALLY if predictions keep\n"
                "failing — you MUST adopt a fundamentally DIFFERENT hypothesis, NOT keep refining\n"
                "the same frame.\n\n")

        # Goal-level falsification block (level 2): aim the test at the win-condition.
        goal_block = ""
        if self.GOAL_CHECK:
            gp = f'  Earlier you claimed progress would look like:\n  "{self._prev_goalpred}"\n' \
                 if self._prev_goalpred else ""
            hdr = ("GOAL CHECK — the ONLY real reward is levels_completed.\n"
                   f"  levels_completed = {lv}. Turns since it last increased: {self._stuck}.\n"
                   f"{gp}")
            if not self.SYSTEMATIC:
                stage = (
                    "Your single-move predictions can keep HOLDING while you make ZERO progress to\n"
                    "the win — action-model fine, but GOAL/mechanic frame may be WRONG. After\n"
                    f"{self._stuck} turns with no level gained, do NOT just pick the next move inside\n"
                    "the same plan. Ask: does the win-condition really work the way I think? Does any\n"
                    "OBSERVED fact (pieces moving in lockstep, an action doing nothing) CONTRADICT my\n"
                    "framing? If stalled, change the GOAL hypothesis or try a qualitatively different\n"
                    "approach (a different action, target, or mechanic).\n\n")
            else:
                # Staged escalation keyed on the hard counter — turns panic into directed search.
                if self._stuck <= 5:
                    stage = ("Early: keep testing your current plan, but watch closely for ANY sign of\n"
                             "real progress toward the win (not just the piece moving).\n\n")
                elif self._stuck <= 15:
                    stage = (
                        "Stalling: single moves keep working but produce ZERO win-progress, so your\n"
                        "GOAL/mechanic frame is likely WRONG. State what OBSERVED fact contradicts it,\n"
                        "then switch to a DIFFERENT goal hypothesis or mechanic.\n\n")
                else:
                    stage = (
                        f"STUCK for {self._stuck} turns — stop 'rethinking' in the abstract. Do a\n"
                        "SYSTEMATIC sweep: consult your '## Tried & ruled out' list and choose ONE\n"
                        "CONCRETE approach you have NOT tried yet (a specific untried action, a\n"
                        "different target/object, ACTION5/ACTION6 on a specific spot, ACTION7 undo).\n"
                        "Name it and DO it. Never repeat an approach already on the ruled-out list.\n\n")
            goal_block = hdr + stage

        rewrite = self.MEM_MODE == "rewrite"
        kind = "REWRITE" if rewrite else "REFLECT"
        if rewrite:
            sections = [
                "## Observed action effects (state-dependent — re-test, don't write off)",
                "## Hypothesis about the goal / win condition",
                "## Current plan",
                "## Uncertain / to test",
            ]
            extra = ""
            if self.PREDICT:
                sections.insert(2, "## Prediction track-record (recent predictions and whether "
                                   "they HELD/FAILED; if the current hypothesis keeps FAILING "
                                   "predictions, REPLACE it with a different one)")
                extra = ("\n- Record whether last turn's prediction HELD or FAILED in the track-record. "
                         "If your current frame/hypothesis has FAILED several recent predictions, "
                         "REPLACE it with a fundamentally different hypothesis rather than refining it.")
            if self.GOAL_CHECK:
                sections.insert(len(sections) - 1, "## Goal-progress audit (turns stuck with 0 "
                                "level gain; is the GOAL/win-condition frame being falsified? what "
                                "qualitatively different approach to try if stalled)")
                extra += ("\n- In the goal-progress audit, note the no-progress counter. If it is high, "
                          "state plainly that the GOAL frame is suspect and commit to a DIFFERENT "
                          "approach this/next turn rather than continuing the stalled plan.")
            if self.SYSTEMATIC:
                sections.insert(len(sections) - 1, "## Tried & ruled out (a running list of DISTINCT "
                                "approaches you have attempted → their outcome; the search frontier "
                                "of what is still UNtried). Keep entries; never delete them.")
                extra += ("\n- MAINTAIN the 'Tried & ruled out' list: add the approach you just tried and "
                          "its outcome, and keep all prior entries. When stuck, your next move MUST be "
                          "an approach NOT already on this list — this is a coverage search, not a "
                          "vague 'rethink'.")
            instr = (
                "Rewrite your MEMORY into an UPDATED version, using EXACTLY these sections:\n"
                + "\n".join(sections) + "\n\n"
                "Integrate what this turn revealed. IMPORTANT:\n"
                "- Revise or DELETE anything that now seems wrong — do NOT merely append.\n"
                "- Record action effects CONDITIONALLY (what the action did in WHICH kind of "
                "state), never as a permanent global rule. If an action was a no-op, note the "
                "likely reason and keep it as a candidate to re-test — do NOT mark it useless."
                + extra +
                "\nKeep only what helps you play; be concise. Output ONLY the updated memory.")
            mem_hdr = f"# CURRENT MEMORY\n{self.memory_doc}\n\n"
        else:
            instr = textwrap.dedent(f"""
                Write ONE concise sentence capturing what you learned this turn.
                Phrase any action effect CONDITIONALLY (what {action.name} did in THIS
                kind of state), not as a permanent global rule. If nothing changed,
                hypothesize WHY (e.g. blocked / precondition unmet) rather than calling
                {action.name} useless. No preamble, just the insight.
            """).strip()
            mem_hdr = ""

        self._call_no += 1
        self._log("\n" + "#" * 100)
        self._log(f"# CALL {self._call_no}  [{kind}]  on {action.name}  repr={self.MEM_REPR}")
        self._log("#" * 100)
        self._log("----- INPUT system -----\n" + self.SYSTEM)

        diff_block = ("\n\n" + self._grid_diff_text(before2d, after2d)) if self.SHOW_DIFF else ""

        if self.MEM_REPR == "image":
            png_b = self._grid_image(before2d)
            png_a = self._grid_image(after2d)
            pb = self._save_img(png_b, f"t{self._turn}_before")
            pa = self._save_img(png_a, f"t{self._turn}_after")
            body = (mem_hdr + pred_block + goal_block + info + diff_block +
                    "\n\nTwo images are attached: IMAGE 1 = BEFORE, IMAGE 2 = AFTER. The COMPUTED "
                    "DIFF above is exact — use it to anchor what you see in the images.\n\n" + instr)
            content = [self._image_block(png_b), self._image_block(png_a),
                       {"type": "text", "text": body}]
            self._log(f"----- INPUT images ----- BEFORE={pb}  AFTER={pa}")
            self._log("----- INPUT user -----\n" + body)
        else:
            body = (mem_hdr + pred_block + goal_block + info + diff_block +
                    "\n\n# GRID BEFORE\n" + self._grid_text(before2d) +
                    "\n\n# GRID AFTER\n" + self._grid_text(after2d) + "\n\n" + instr)
            content = body
            self._log("----- INPUT user -----\n" + body)

        resp = self.client.messages.create(
            model=self.MODEL, max_tokens=800 if rewrite else 300,
            system=self.SYSTEM, messages=[{"role": "user", "content": content}],
        )
        txt = "".join(b.text for b in resp.content if b.type == "text").strip()
        self._log(f"===== OUTPUT ({kind}) =====\n" + txt)
        self._log(f"usage: in={resp.usage.input_tokens} out={resp.usage.output_tokens}")

        if rewrite:
            self.memory_doc = txt or self.memory_doc
        else:
            self.memory.append(f"[t{self._turn}] {action.name}: {txt.replace(chr(10), ' ')}")
            if self.MAX_MEMORY and len(self.memory) > self.MAX_MEMORY:
                self.memory = self.memory[-self.MAX_MEMORY:]

    # ---------- main per-turn entry ----------
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # Update MEMORY from the previous action's result (now visible).
        if self._prev_action is not None and self._prev_grid2d is not None:
            self._turn += 1
            try:
                self._update_memory(self._prev_action, self._prev_grid2d, latest_frame)
            except Exception as e:
                self._log(f"(memory update failed: {e})")

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
            action.reasoning = {"reasoning": "reset to (re)start"}
            self._prev_prediction = None
            self._prev_goalpred = None
        else:
            action = self._act(latest_frame)

        self._prev_action = action
        self._prev_grid2d = self._final_grid(latest_frame)
        return action
