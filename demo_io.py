"""Full, untruncated transparency dump: play a real game for N turns with the
repo's real `llm` agent + real OpenAI calls, and print EVERY message, the FULL
grid, the FULL action schema, and EVERY response verbatim. Nothing omitted."""
import os, json
from dotenv import load_dotenv
load_dotenv(".env.example"); load_dotenv(".env", override=True)

N_TURNS = int(os.environ.get("N_TURNS", "5"))  # how many game turns to play
GAME = os.environ.get("DEMO_GAME", "sc25-635fd71a")

import openai
from openai.resources.chat.completions import Completions
_orig = Completions.create

_call = {"n": 0}
def logged(self, **kw):
    _call["n"] += 1
    print("\n" + "#"*90)
    print(f"# ===== OPENAI REQUEST  (api call #{_call['n']})   model={kw.get('model')} =====")
    print("#"*90)
    def mg(m, key):
        # messages can be plain dicts OR pydantic response objects
        if isinstance(m, dict):
            return m.get(key)
        return getattr(m, key, None)
    def jdump(v):
        try:
            return json.dumps(v)
        except TypeError:
            return str(v.model_dump() if hasattr(v, "model_dump") else v)
    for i, m in enumerate(kw["messages"]):
        print(f"\n----- messages[{i}]  role={mg(m,'role')} -----")
        if mg(m, "content") is not None:
            print(mg(m, "content"))      # FULL content, no truncation
        if mg(m, "function_call"):
            print("function_call:", jdump(mg(m, "function_call")))
        if mg(m, "name"):
            print("(name field:", mg(m, "name"), ")")
        if mg(m, "tool_calls"):
            print("tool_calls:", jdump(mg(m, "tool_calls")))
    if "functions" in kw:
        print("\n----- functions (FULL action schema sent to model) -----")
        print(json.dumps(kw["functions"], indent=2))   # FULL, all descriptions
    if "tools" in kw:
        print("\n----- tools (FULL) -----")
        print(json.dumps(kw["tools"], indent=2))
    for k in ("function_call", "tool_choice", "reasoning_effort"):
        if k in kw:
            print(f"({k} = {kw[k]})")
    resp = _orig(self, **kw)
    msg = resp.choices[0].message
    print("\n" + "="*90)
    print(f"# ===== OPENAI RESPONSE (api call #{_call['n']}) =====")
    print("="*90)
    if msg.content:
        print("\n[assistant content]:")
        print(msg.content)               # FULL
    if getattr(msg, "function_call", None):
        print("\n[function_call CHOSEN]:", msg.function_call.name,
              "arguments=", msg.function_call.arguments)
    if getattr(msg, "tool_calls", None):
        print("\n[tool_calls CHOSEN]:", json.dumps([
            {"name": t.function.name, "arguments": t.function.arguments} for t in msg.tool_calls]))
    print("\n[usage]:", json.dumps(resp.usage.model_dump() if resp.usage else None))
    return resp

Completions.create = logged

from arc_agi import Arcade
from agents.templates.llm_agents import LLM

arc = Arcade()
card = arc.open_scorecard(tags=["io-demo-full"])
env = arc.make(GAME, scorecard_id=card)
agent = LLM(card_id=card, game_id=GAME, agent_name="llm",
            ROOT_URL="https://three.arcprize.org", record=False, arc_env=env)
# A single frame can contain MANY stacked grids (animation sub-frames); the
# agent dumps ALL of them as text, so one message can be ~280k tokens, which
# overflows gpt-4o-mini's 128k window. Use a 1M-context model so the FULL input
# fits and nothing is truncated. Override with DEMO_MODEL if you like.
agent.MODEL = os.environ.get("DEMO_MODEL", "gpt-4.1-mini")
agent.MESSAGE_LIMIT = int(os.environ.get("MSG_LIMIT", "10"))
print(f"(using model={agent.MODEL}, MESSAGE_LIMIT={agent.MESSAGE_LIMIT})")

# seed the loop with the current observation
latest = agent._convert_raw_frame_data(env.observation_space)

for turn in range(1, N_TURNS + 1):
    print("\n\n" + "@"*90)
    print(f"@@@@@  GAME TURN {turn}   (state before decision: {latest.state.name}, "
          f"levels_completed={latest.levels_completed})")
    print("@"*90)
    action = agent.choose_action(agent.frames, latest)   # may trigger 0, 1, or 2 API calls
    print(f"\n>>> TURN {turn}: agent will send action -> {action.name} "
          f"data={action.action_data.model_dump()}")
    frame = agent.take_action(action)
    agent.append_frame(frame)
    latest = frame
    print(f">>> TURN {turn} RESULT: state={frame.state.name} "
          f"levels_completed={frame.levels_completed} "
          f"available_actions={frame.available_actions}")

try:
    arc.close_scorecard(card)
except Exception as e:
    print("(scorecard close note:", e, ")")
