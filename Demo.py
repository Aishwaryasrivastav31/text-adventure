"""
Streamlit demo: play the text adventure yourself, or watch trained agents play it.

    pip install streamlit numpy scipy
    streamlit run demo/app.py
"""
import os, sys, random
import numpy as np
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from textworld_lite import TextWorld, LEXICONS            # noqa: E402
from agents import DRRN, GreedyAgent, RandomAgent          # noqa: E402
from train import make_vocab_texts                         # noqa: E402

st.set_page_config(page_title="Text Adventure RL demo", page_icon="🗝️", layout="wide")

LEX_LABEL = {
    "train": "Training words (kitchen, brass key, ...)",
    "new_names": "Unseen names (pantry, copper key, ...)",
    "paraphrase": "Unseen names + paraphrased sentences",
}
AGENTS = {
    "Match-head DRRN (proposed)": ("drrn_match_aug_s0", "word", True),
    "Word DRRN (standard)": ("drrn_word_s0", "word", False),
    "Char-trigram DRRN (standard)": ("drrn_ngram_s0", "ngram", False),
    "One-step greedy (sees the simulator)": ("greedy", None, None),
    "Random": ("random", None, None),
}


@st.cache_resource
def load_agent(name):
    file, feat, match = AGENTS[name]
    if file == "greedy":
        return GreedyAgent(0)
    if file == "random":
        return RandomAgent(0)
    ag = DRRN(feat, make_vocab_texts(), seed=0, match=match)
    w = np.load(os.path.join(HERE, "weights", file + ".npz"))
    for k in ag.params:
        ag.params[k][...] = w[k]
    return ag


def new_game(lexicon, seed):
    env = TextWorld(LEXICONS[lexicon])
    obs = env.reset(seed)
    st.session_state.update(env=env, obs=obs, log=[("start", obs, 0.0)], score=0.0, over=False,
                            won=False, lexicon=lexicon, seed=seed)


def do_step(cmd, q=None):
    env = st.session_state.env
    obs, r, done, info = env.step(cmd)
    st.session_state.score += r
    st.session_state.obs = obs
    st.session_state.log.append((cmd if q is None else f"{cmd}   (Q={q:.2f})", obs, r))
    if done:
        st.session_state.over = True
        st.session_state.won = bool(info["success"])


def agent_pick(agent, env, obs):
    cmds = env.admissible()
    if isinstance(agent, GreedyAgent):
        agent.bind(env)
        return agent.act(obs, cmds), None
    if isinstance(agent, DRRN):
        q = agent.q_values(obs, cmds)
        return cmds[int(np.argmax(q))], (cmds, q)
    return agent.act(obs, cmds), None


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🗝️ Settings")
lexicon = st.sidebar.selectbox("World vocabulary", list(LEX_LABEL), format_func=LEX_LABEL.get, index=1)
seed_in = st.sidebar.number_input("World seed", min_value=0, value=10_000_006, step=1)
mode = st.sidebar.radio("Mode", ["I play", "Watch an agent"])
agent_name = None
if mode == "Watch an agent":
    agent_name = st.sidebar.selectbox("Agent", list(AGENTS))
c1, c2 = st.sidebar.columns(2)
if c1.button("New game", use_container_width=True) or "env" not in st.session_state:
    new_game(lexicon, int(seed_in))
if c2.button("Random world", use_container_width=True):
    new_game(lexicon, random.randint(10_000_000, 10_900_000))
if st.session_state.get("lexicon") != lexicon:
    st.sidebar.info("Vocabulary changed: press **New game** to apply it.")

st.sidebar.markdown("---")
st.sidebar.caption("Goal: read the door text (e.g. *a locked brass door*), find the **matching** key, unlock the door, "
                   "take the treasure in the vault. You can carry 2 items, 50 steps. "
                   "Reward: +1 matching key, +1 unlock, +10 treasure, -0.05 per step.")

# ---------------------------------------------------------------------- main
st.title("Text adventure: can an RL agent read?")
st.caption("A DRRN agent trained with deep Q-learning, plus an exact-match 'copy' head for unseen words. "
           "Play it yourself, or watch the agents - try *Unseen names* with the standard agents vs the match-head agent.")

env, obs = st.session_state.env, st.session_state.obs
left, right = st.columns([3, 2])

with left:
    st.subheader(f"World {st.session_state.seed} · {LEX_LABEL[st.session_state.lexicon]}")
    st.info(obs)
    m1, m2, m3 = st.columns(3)
    m1.metric("Score", f"{st.session_state.score:.2f}")
    m2.metric("Steps used", f"{env.steps} / {env.MAX_STEPS}")
    m3.metric("Items carried", f"{len(env.inv)} / {env.CAPACITY}")

    if st.session_state.over:
        if st.session_state.won:
            st.success(f"🎉 Treasure taken! Final score {st.session_state.score:.2f}")
        else:
            st.error(f"Out of steps. Final score {st.session_state.score:.2f}")
    elif mode == "I play":
        st.markdown("**Choose a command:**")
        cmds = env.admissible()
        cols = st.columns(2)
        for i, c in enumerate(cmds):
            if cols[i % 2].button(c, key=f"cmd{env.steps}_{i}", use_container_width=True):
                do_step(c)
                st.rerun()
        with st.expander("Need a hint? Ask the match-head agent"):
            ag = load_agent("Match-head DRRN (proposed)")
            q = ag.q_values(obs, cmds)
            order = np.argsort(-q)
            st.write("Agent's ranking:", ", ".join(f"`{cmds[j]}` ({q[j]:.1f})" for j in order))
    else:
        agent = load_agent(agent_name)
        b1, b2 = st.columns(2)
        if b1.button("▶ One step", use_container_width=True):
            cmd, info = agent_pick(agent, env, obs)
            do_step(cmd, None if info is None else float(max(info[1])))
            st.rerun()
        if b2.button("⏩ Play whole episode", use_container_width=True):
            while not st.session_state.over:
                cmd, info = agent_pick(agent, st.session_state.env, st.session_state.obs)
                do_step(cmd, None if info is None else float(max(info[1])))
            st.rerun()
        if isinstance(agent, DRRN):
            cmds = env.admissible()
            q = agent.q_values(obs, cmds)
            st.markdown("**What the agent is thinking (Q-value of each command):**")
            st.bar_chart({"Q": {c: float(v) for c, v in zip(cmds, q)}}, horizontal=True)

with right:
    st.subheader("Game log")
    for i, (cmd, text, r) in enumerate(reversed(st.session_state.log)):
        if cmd == "start":
            st.markdown("**Start**")
        else:
            st.markdown(f"**> {cmd}**  \n`reward {r:+.2f}`")
        st.caption(text)

with st.expander("What am I looking at? (read this before an interview)"):
    st.markdown(
        "* The agent sees **only text**. It scores every admissible command with `Q(s, a)`.\n"
        "* **Standard DRRN** memorises word-specific behaviour: it plays well on the training words and fails on *unseen names* "
        "(unseen words all look like `<unk>`).\n"
        "* **Match-head DRRN** compares command words with state words by exact string identity and reads their context "
        "(e.g. *locked ... door*), so it can pick the key whose word also appears in the door text, even for new words.\n"
        "* Results over 10 seeds are in `README.md`; this demo shows single episodes (seed 0 models), which are illustrative, not evidence.")
