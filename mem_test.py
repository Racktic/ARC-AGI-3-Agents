"""Quick driver: run MemoryAgent (scheme B) for a few turns and dump the memory."""
import os
from dotenv import load_dotenv
load_dotenv(".env.example"); load_dotenv(".env", override=True)
os.environ.setdefault("ANTHROPIC_BASE_URL", "http://localhost:5678")
os.environ.setdefault("ANTHROPIC_API_KEY", "anykey")

from arc_agi import Arcade
from agents.templates.memory_agent import MemoryAgent

GAME = os.environ.get("DEMO_GAME", "sc25-635fd71a")
arc = Arcade()
card = arc.open_scorecard(tags=["mem-test"])
env = arc.make(GAME, scorecard_id=card)
agent = MemoryAgent(card_id=card, game_id=GAME, agent_name="memoryagent",
                    ROOT_URL="https://three.arcprize.org", record=False, arc_env=env)

latest = agent._convert_raw_frame_data(env.observation_space)
N = int(os.environ.get("MEM_MAX_ACTIONS", "8"))
done = 0
for t in range(1, N + 1):
    try:
        action = agent.choose_action(agent.frames, latest)
        r = action.reasoning if isinstance(action.reasoning, dict) else {"reasoning": action.reasoning}
        print(f"\n=== TURN {t}: chose {action.name} {action.action_data.model_dump().get('x','')},"
              f"{action.action_data.model_dump().get('y','')}  reason={r.get('reasoning','')[:80]}")
        frame = agent.take_action(action)
        agent.append_frame(frame)
        latest = frame
        done = t
        print(f"    -> state={frame.state.name} levels={frame.levels_completed}")
    except Exception as e:
        # don't lose the run (and its GIF) to a transient 429 / network blip
        print(f"\n!!! TURN {t} aborted: {type(e).__name__}: {e}")
        break

print("\n\n################  FINAL MEMORY  (mode=%s)  ################" % agent.MEM_MODE)
if agent.MEM_MODE == "rewrite":
    print(agent.memory_doc)
else:
    for m in agent.memory:
        print(m)
try:
    arc.close_scorecard(card)
except Exception:
    pass

# Always produce the per-turn action GIF for this run (even on partial/aborted runs).
import subprocess, sys
run = os.environ.get("MEM_RUN", "run")
print(f"\n[mem_test] {done}/{N} turns done; generating GIF for run '{run}' ...")
try:
    subprocess.run([sys.executable, "make_gif.py", run], check=False)
except Exception as e:
    print(f"(gif generation skipped: {e})")
