"""
Uncertainty Quantification demo (Streamlit).

Organised BY MODEL: pick a model, then pick one of the questions curated for
that model. Each question is a clean, self-contained example of one outcome —
confident & correct, uncertain & wrong, or confident & wrong — for THAT model.
The selected model is always the one whose answer is shown, so the outcome label
never contradicts what's on screen. Different models get different questions
(e.g. Claude Haiku is never genuinely uncertain & wrong, so it shows none).

You see the model's answer (with a ✓/✗ against the gold answer) and its
sampling-based uncertainty under a chosen method. A second view shows how the
sampled answers were grouped into semantic clusters to produce a Discrete
Semantic Entropy (SE) estimate — the most intuitive of the sampling methods.

Uncertainty methods (computed via the project's uncertainty_quantification_methods.py):
  - Semantic Entropy / Discrete SE: Kuhn, Gal & Farquhar, "Semantic Uncertainty",
    ICLR 2023; discrete variant in Farquhar et al., Nature 2024.
  - Kernel Language Entropy (KLE-full, KLE-heat): Nikitin, Kossen, Gal & Marttinen,
    "Kernel Language Entropy", NeurIPS 2024.
  - Chao-Shen: Chao & Shen, 2003 (coverage-adjusted entropy). Hybrid Chao-Shen
    (coverage from a hybrid semantic-alphabet-size estimate, max of Good-Turing and
    the U-EIGV spectral estimate): McCabe et al., "Estimating Semantic Alphabet Size
    for LLM Uncertainty Quantification", arXiv:2509.14478; the U-EIGV term itself
    follows https://openreview.net/pdf?id=DWkJCSxKU5.
  - Evidential Semantic Entropy: Kunitomo-Jacquin et al., EACL 2026 (implementation
    adapted from the `evsme` repository).

Run:  python -m streamlit run demo/app.py
"""
import html as _html
import json
import math
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Palette (validated data-viz reference instance, light surface)
# ---------------------------------------------------------------------------
GOOD = "#0ca30c"       # status: correct
CRITICAL = "#d03b3b"   # status: wrong
WARN = "#fab219"       # status: uncertain
BLUE = "#2a78d6"       # sequential slot (magnitude: uncertainty, cluster size)
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
# categorical slots (validated order) for colouring clusters; overflow -> gray.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

DATA_PATH = os.path.join(os.path.dirname(__file__), "demo_data.json")


@st.cache_data
def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def clean_gold(gold: str):
    """Gold answers are stored as ' / '-joined aliases; show the primary + count."""
    parts = [p.strip() for p in gold.split("/") if p.strip()]
    if not parts:
        return gold, 0
    primary = min(parts, key=len)
    return primary, len(parts) - 1


OUTCOME_STYLE = {
    "Confident & correct": ("✅", GOOD),
    "Uncertain & wrong":   ("⚠️", WARN),
    "Confident & wrong":   ("🚨", CRITICAL),
}


# ---------------------------------------------------------------------------
# Animated chat bubble ("chat gif"): a self-contained, looping HTML/CSS replay
# of one conversation — question → answer (✓/✗) → "Are you sure?" → revised
# answer, with the verdict visibly flipping. Rendered in an iframe so it carries
# its own styles and animates like a GIF without a rerun.
# ---------------------------------------------------------------------------
_CHAT_FONT = ("Inter, system-ui, -apple-system, 'Segoe UI', Roboto, "
              "'Helvetica Neue', Arial, sans-serif")


def _esc(s):
    return _html.escape((s or "").replace("\n", " ").strip())


def chat_animation_html(ex, challenge, uid, cycle=11.0):
    """Return a full HTML doc animating one revision conversation on a loop."""
    def pct(sec):
        return round(sec / cycle * 100, 2)

    # timeline (seconds): q in, assistant-1 row in (typing→answer), challenge in,
    # assistant-2 row in (typing→flipped answer), hold, reset.
    t_q, t_a1_row, t_a1_txt = 0.4, 1.4, 3.0
    t_q2, t_a2_row, t_a2_txt = 4.4, 5.4, 7.0
    t_hold_end = cycle - 1.1  # everything fades just before the loop restarts

    def bubble_kf(name, start):
        s0, s1, h0 = pct(start), pct(start + 0.35), pct(t_hold_end)
        return (f"@keyframes {name}{{"
                f"0%,{s0}%{{opacity:0;transform:translateY(10px);}}"
                f"{s1}%{{opacity:1;transform:translateY(0);}}"
                f"{h0}%{{opacity:1;transform:translateY(0);}}"
                f"100%{{opacity:0;transform:translateY(-6px);}}}}")

    def window_kf(name, show, hide):
        w0, w1 = pct(show), pct(hide)
        return (f"@keyframes {name}{{"
                f"0%,{max(w0-0.4,0)}%{{opacity:0;}}"
                f"{w0}%{{opacity:1;}}{w1}%{{opacity:1;}}"
                f"{min(w1+0.4,100)}%{{opacity:0;}}100%{{opacity:0;}}}}")

    def text_kf(name, start):
        s0, s1, h0 = pct(start), pct(start + 0.3), pct(t_hold_end)
        return (f"@keyframes {name}{{"
                f"0%,{s0}%{{opacity:0;}}{s1}%{{opacity:1;}}"
                f"{h0}%{{opacity:1;}}100%{{opacity:0;}}}}")

    a1_ok = ex["initial_correct"]
    a2_ok = ex["revised_correct"]

    def chip(ok):
        c = GOOD if ok else CRITICAL
        return (f"<span class='chip' style='background:{c}1f;color:{c};'>"
                f"{'✓ correct' if ok else '✗ wrong'}</span>")

    kf = "".join([
        bubble_kf(f"q{uid}", t_q),
        bubble_kf(f"a1{uid}", t_a1_row),
        bubble_kf(f"q2{uid}", t_q2),
        bubble_kf(f"a2{uid}", t_a2_row),
        window_kf(f"d1{uid}", t_a1_row, t_a1_txt),
        window_kf(f"d2{uid}", t_a2_row, t_a2_txt),
        text_kf(f"t1{uid}", t_a1_txt),
        text_kf(f"t2{uid}", t_a2_txt),
    ])

    kh = ex.get("kle_heat")
    meta = (f"{_esc(ex['model'])} · {_esc(ex['dataset'])}"
            + (f" · KLE-heat {kh:.2f}" if kh is not None else ""))

    return f"""<!doctype html><html><head><meta charset='utf-8'><style>
      *{{box-sizing:border-box;}}
      body{{margin:0;font-family:{_CHAT_FONT};background:transparent;}}
      .wrap{{background:linear-gradient(180deg,#ffffff, #fbfcfe);
        border:1px solid {GRID};border-radius:18px;padding:16px 16px 18px;
        box-shadow:0 1px 2px rgba(11,11,11,.04),0 10px 30px rgba(11,11,11,.06);}}
      .meta{{font-size:.74rem;color:{MUTED};font-weight:600;letter-spacing:.03em;
        text-transform:uppercase;margin:0 4px 12px;}}
      .row{{display:flex;align-items:flex-end;gap:8px;margin:9px 0;opacity:0;}}
      .row.user{{flex-direction:row-reverse;}}
      .ava{{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;
        justify-content:center;font-size:16px;flex:0 0 30px;
        background:#eef1f6;border:1px solid {GRID};}}
      .bubble{{max-width:78%;padding:10px 14px;border-radius:16px;font-size:.95rem;
        line-height:1.35;}}
      .bubble.asst{{background:#ffffff;border:1px solid {GRID};color:{INK};
        border-bottom-left-radius:5px;display:grid;
        box-shadow:0 1px 2px rgba(11,11,11,.04);}}
      .bubble.user{{background:{BLUE};color:#fff;border-bottom-right-radius:5px;}}
      .bubble.asst>.dots,.bubble.asst>.txt{{grid-area:1/1;}}
      .txt{{opacity:0;}}
      .chip{{display:inline-block;margin-left:8px;padding:1px 9px;border-radius:999px;
        font-size:.72rem;font-weight:700;vertical-align:middle;}}
      .dots{{display:flex;gap:5px;align-items:center;padding:3px 0;opacity:0;}}
      .dots span{{width:7px;height:7px;border-radius:50%;background:{MUTED};
        animation:blink 1.2s infinite ease-in-out;}}
      .dots span:nth-child(2){{animation-delay:.2s;}}
      .dots span:nth-child(3){{animation-delay:.4s;}}
      @keyframes blink{{0%,80%,100%{{opacity:.25;transform:translateY(0);}}
        40%{{opacity:1;transform:translateY(-3px);}}}}
      .r_q{{animation:q{uid} {cycle}s infinite;}}
      .r_a1{{animation:a1{uid} {cycle}s infinite;}}
      .r_q2{{animation:q2{uid} {cycle}s infinite;}}
      .r_a2{{animation:a2{uid} {cycle}s infinite;}}
      .w_d1{{animation:d1{uid} {cycle}s infinite;}}
      .w_d2{{animation:d2{uid} {cycle}s infinite;}}
      .w_t1{{animation:t1{uid} {cycle}s infinite;}}
      .w_t2{{animation:t2{uid} {cycle}s infinite;}}
      {kf}
    </style></head><body><div class='wrap'>
      <div class='meta'>{meta}</div>
      <div class='row user r_q'><div class='ava'>🧑</div>
        <div class='bubble user'>{_esc(ex['question'])}</div></div>
      <div class='row asst r_a1'><div class='ava'>🤖</div>
        <div class='bubble asst'>
          <div class='dots w_d1'><span></span><span></span><span></span></div>
          <div class='txt w_t1'>{_esc(ex['initial_answer'])} {chip(a1_ok)}</div>
        </div></div>
      <div class='row user r_q2'><div class='ava'>🧑</div>
        <div class='bubble user'>{_esc(challenge)}</div></div>
      <div class='row asst r_a2'><div class='ava'>🤖</div>
        <div class='bubble asst'>
          <div class='dots w_d2'><span></span><span></span><span></span></div>
          <div class='txt w_t2'>{_esc(ex['revised_answer'])} {chip(a2_ok)}</div>
        </div></div>
    </div></body></html>"""


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Uncertainty Quantification Demo",
                   page_icon="📊", layout="wide")

FONT = ("Inter, system-ui, -apple-system, 'Segoe UI', Roboto, "
        "'Helvetica Neue', Arial, sans-serif")

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] {{ font-family: {FONT}; }}
  .stApp {{
      background:
        radial-gradient(1200px 500px at 12% -8%, #eef4fd 0%, rgba(238,244,253,0) 55%),
        radial-gradient(1000px 460px at 100% 0%, #f3f7ee 0%, rgba(243,247,238,0) 50%),
        {SURFACE};
  }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 2.4rem; max-width: 1180px; }}

  h1, h2, h3, h4 {{ font-family: {FONT}; letter-spacing: -0.012em; color: {INK}; }}
  h3 {{ font-weight: 700; }}

  /* ---- hero ---- */
  .hero-title {{ font-size: 2.15rem; font-weight: 800; line-height: 1.08;
      letter-spacing: -0.02em; margin: 0 0 .35rem 0;
      background: linear-gradient(92deg, {INK} 0%, {BLUE} 78%);
      -webkit-background-clip: text; background-clip: text;
      -webkit-text-fill-color: transparent; }}
  .hero-sub {{ color: {INK_2}; font-size: 1.02rem; margin: 0 0 .2rem 0; max-width: 60ch; }}
  .hero-pills {{ margin-top: .7rem; }}
  .pill {{ display:inline-block; padding:4px 12px; border-radius:999px;
           font-size:.78rem; font-weight:600; margin-right:.4rem; margin-bottom:.35rem;
           background:#ffffff; border:1px solid {GRID}; color:{INK_2}; }}
  .pill b {{ color:{BLUE}; }}

  /* ---- badges & cards ---- */
  .badge {{ display:inline-block; padding:3px 12px; border-radius:999px;
            font-size:0.8rem; font-weight:600; }}
  .answer-box {{ border:1px solid {GRID}; border-radius:14px; padding:16px 20px;
                 background:#ffffff;
                 box-shadow: 0 1px 2px rgba(11,11,11,.04), 0 6px 20px rgba(11,11,11,.05); }}

  /* ---- meter ---- */
  .meter-track {{ background:{GRID}; border-radius:999px; height:12px; width:100%;
                  overflow:hidden; }}
  .meter-fill  {{ background:linear-gradient(90deg, #7fb0ec, {BLUE});
                  border-radius:999px; height:12px;
                  box-shadow: inset 0 0 0 1px rgba(255,255,255,.25); }}

  /* ---- kpi ---- */
  .kpi-num {{ font-size:2.4rem; font-weight:800; color:{INK}; line-height:1;
              font-variant-numeric: tabular-nums; letter-spacing:-0.02em; }}
  .kpi-lab {{ font-size:0.78rem; color:{MUTED}; text-transform:uppercase;
              letter-spacing:0.06em; font-weight:600; margin-bottom:.25rem; }}

  /* ---- sidebar ---- */
  section[data-testid="stSidebar"] {{ background:#ffffffcc;
      border-right:1px solid {GRID}; backdrop-filter: blur(6px); }}
  section[data-testid="stSidebar"] .stButton>button {{
      width:100%; text-align:left; border-radius:12px; border:1px solid {GRID};
      background:#ffffff; color:{INK}; font-weight:600; padding:.55rem .7rem;
      transition: all .12s ease; box-shadow: 0 1px 2px rgba(11,11,11,.03); }}
  section[data-testid="stSidebar"] .stButton>button:hover:enabled {{
      border-color:{BLUE}; color:{BLUE}; transform: translateX(2px);
      box-shadow: 0 2px 10px rgba(42,120,214,.14); }}

  /* ---- tabs ---- */
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
  .stTabs [data-baseweb="tab"] {{ border-radius: 10px 10px 0 0; }}
</style>
""", unsafe_allow_html=True)

data = load_data()
examples_by_model = data["examples_by_model"]
models = data["models"]
methods = data["methods"]
outcomes = data["outcomes"]
chat_examples = data.get("chat_examples", [])
challenge_prompt = data.get("challenge_prompt", "Are you sure? Please reconsider your answer.")
benchmark = data.get("benchmark")
revision = data.get("revision")
injection = data.get("injection")

st.markdown(
    "<div class='hero-title'>When does a language model actually know?</div>"
    "<div class='hero-sub'>Sampling-based uncertainty quantification for LLMs — and "
    "what a UQ score reveals about how a model behaves when you push back.</div>"
    "<div class='hero-pills'>"
    "<span class='pill'>5 models</span>"
    "<span class='pill'>7 UQ methods</span>"
    "<span class='pill'>3 datasets</span>"
    "<span class='pill'><b>“Are you sure?”</b> revision challenge</span>"
    "</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# View toggle (left panel): explorer vs the sycophancy chat demo
# ---------------------------------------------------------------------------
if "view" not in st.session_state:
    st.session_state.view = "explore"


def _set_view(v):
    st.session_state.view = v


# on_click callbacks run BEFORE the rerun, so the view is already updated when
# the rest of the script executes — no st.rerun() needed (more robust across
# Streamlit versions / Streamlit Cloud).
if st.session_state.view == "explore":
    st.sidebar.markdown("**The three findings**")
    st.sidebar.button("📊  ① Which methods to trust",
                      on_click=_set_view, args=("benchmark",))
    st.sidebar.button("🔁  ② Does uncertainty predict revision?",
                      on_click=_set_view, args=("revision",))
    st.sidebar.button("🧪  ③ Can the model use its own score?",
                      on_click=_set_view, args=("injection",))
    st.sidebar.markdown("---")
    st.sidebar.button("💬  Sycophancy chat demo",
                      on_click=_set_view, args=("chat",))
    st.sidebar.button("🔴  Live UQ (your own OpenRouter key)",
                      on_click=_set_view, args=("live",), disabled=True)
else:
    st.sidebar.button("←  Back to explorer",
                      on_click=_set_view, args=("explore",))

# ---------------------------------------------------------------------------
# Chat / sycophancy view
# ---------------------------------------------------------------------------
if st.session_state.view == "chat":
    st.subheader("“Are you sure?” — watch the model flip")
    st.markdown(
        "Each model answers a question (with its **KLE-heat** uncertainty), then "
        "gets one mild challenge — *“Are you sure? Please reconsider your answer.”* "
        "A well-calibrated model should hold a **confident** answer and only revise "
        "an **uncertain** one. These conversations **replay on a loop** — watch how "
        "often a ✓ turns into a ✗. That's **sycophancy**.")

    for i, c in enumerate(chat_examples):
        emoji, color = OUTCOME_STYLE[c["outcome"]]
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        gold_primary, _ = clean_gold(c["gold_answer"])
        flipped = c["initial_correct"] and not c["revised_correct"]
        st.markdown(
            f"<span class='badge' style='background:{color}22;color:{color};'>"
            f"{emoji} {c['outcome']}</span> "
            + (f"<span class='badge' style='background:{CRITICAL}1f;color:{CRITICAL};'>"
               f"↯ flipped ✓→✗</span> " if flipped else "")
            + f"<span style='color:{MUTED};font-size:0.85rem'>gold: "
              f"<b>{_esc(gold_primary)}</b></span>",
            unsafe_allow_html=True)

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.iframe(chat_animation_html(c, challenge_prompt, uid=i),
                      height=360)
        with right:
            st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
            st.info(c["takeaway"])
    st.stop()

# ---------------------------------------------------------------------------
# Live view — sample via OpenRouter (bring-your-own key) and compute UQ in real time
# ---------------------------------------------------------------------------
def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _openrouter_sample(client, model, question, n, temp, max_tokens):
    """Sample n answers via OpenRouter; return (answers, sequence log-likelihoods)."""
    prompt = ("Answer the question as briefly as possible — just the answer.\n"
              f"Question: {question}\nAnswer:")
    answers, lls = [], []
    prog = st.progress(0.0, "Sampling…")
    for i in range(n):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temp,
                logprobs=True,
                top_logprobs=1,
            )
            answers.append((r.choices[0].message.content or "").strip())
            ll = 0.0
            try:
                for tok in r.choices[0].logprobs.content:
                    ll += float(tok.logprob)
            except Exception:
                ll = float("nan")
        except Exception:
            answers.append("")
            ll = float("nan")
        lls.append(ll)
        prog.progress((i + 1) / (n + 1), f"Sampling… {i + 1}/{n}")
    prog.progress(1.0, "Done sampling")
    prog.empty()
    return answers, lls


def _openrouter_nli_matrix(client, model, question, answers):
    """One call: full ordered-pair NLI matrix (2=entails, 1=neutral, 0=contradicts)."""
    import json as _json
    listing = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(answers))
    prompt = (
        f"Question: {question}\nCandidate answers:\n{listing}\n\n"
        "For every ordered pair (i, j) with i != j (1-indexed), classify how answer i "
        "relates to answer j AS ANSWERS TO THE QUESTION: 2 = i entails or means the "
        "same as j, 0 = i contradicts or is incompatible with j, 1 = neutral. "
        'Return ONLY a JSON object whose keys are "i,j" and values are 0, 1 or 2, '
        "covering all ordered pairs.")
    n = len(answers)
    labels = {}
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=1024,
        )
        raw = _json.loads(r.choices[0].message.content or "{}")
        for kk, vv in raw.items():
            a, b = kk.replace(" ", "").split(",")
            labels[(int(a) - 1, int(b) - 1)] = int(vv)
    except Exception:
        pass
    for i in range(n):          # default any missing pair to neutral
        for j in range(n):
            if i != j:
                labels.setdefault((i, j), 1)
    return labels


OR_MODEL = "openrouter/free"

if st.session_state.view == "live":
    st.subheader("🔴 Live: sampling-based UQ on your own question")
    st.markdown(
        "Bring your own **OpenRouter API key** — it stays in this browser "
        "session only (never stored or logged). We sample a free model "
        "several times, cluster the answers by meaning, and compute the "
        "uncertainty live, exactly like the pre-computed examples.")
    with st.expander("How to get an OpenRouter API key (free)"):
        st.markdown("Create one at [openrouter.ai/keys](https://openrouter.ai/keys). "
                    "Free-tier models (`:free` suffix) require no credits.")

    key = st.text_input("OpenRouter API key", type="password",
                        placeholder="sk-or-…", help="Kept in session memory only.")
    suggested = [
        "Who wrote the novel Cider With Rosie?",
        "What is the capital city of Australia?",
        "Which English football club has H'Angus the Monkey as its mascot?",
        "How many hearts does an octopus have?",
        "Who painted the ceiling of the Sistine Chapel?",
    ]
    c1, c2 = st.columns([3, 1])
    with c1:
        pick = st.selectbox("Pick a question…", ["(type my own)"] + suggested)
        question = st.text_input("…or type your own",
                                 value="" if pick == "(type my own)" else pick)
        if not question and pick != "(type my own)":
            question = pick
    with c2:
        n_samples = st.slider("Samples", 5, 12, 10)
        temp = st.slider("Temperature", 0.3, 1.5, 1.0, 0.1)

    run = st.button("Run live UQ", type="primary",
                    disabled=not (key and question.strip()))

    if run:
        try:
            from openai import OpenAI as _OpenAI
        except Exception:
            st.error("`openai` isn't installed. Add it to demo/requirements.txt.")
            st.stop()
        import sys as _sys
        _sys.path.insert(0, _repo_root())
        try:
            import numpy as _np
            import networkx as _nx
            from uncertainty_quantification_methods import (
                get_semantic_ids, _unc_from_artifacts, discrete_semantic_entropy,
                build_weighted_graph)
        except Exception as ex:
            st.error(f"UQ dependencies missing ({ex}). Add numpy, scipy, networkx, "
                     "scikit-learn to demo/requirements.txt.")
            st.stop()

        try:
            client = _OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
            )
            with st.spinner("Calling OpenRouter…"):
                answers, lls = _openrouter_sample(client, OR_MODEL,
                                                  question, n_samples, temp, 64)
                labels = _openrouter_nli_matrix(client, OR_MODEL,
                                                question, answers)
        except Exception as ex:
            st.error(f"OpenRouter call failed: {ex}")
            st.stop()

        n = len(answers)
        has_lls = all(l == l for l in lls) and any(l != 0.0 for l in lls)
        lls_use = [0.0 if (l != l) else l for l in lls]  # NaN -> 0 fallback
        art = _unc_from_artifacts(answers, lls_use, labels, kle_t=0.3, alpha=0.5,
                                  log_likelihoods_eos=lls_use)
        sids = get_semantic_ids(answers, labels)
        dse_raw = discrete_semantic_entropy(sids)["unc_dse"]
        dse_norm = float(dse_raw / _np.log(n)) if n > 1 else 0.0
        metrics = {
            "Semantic Entropy": art["unc_semantic_entropy"] if has_lls else None,
            "Discrete SE": dse_norm,
            "Evidential SE": art["unc_ese"] if has_lls else None,
            "KLE-full": art["unc_kle_full"] if has_lls else None,
            "KLE-heat": art["unc_kle_heat"],
            "Chao-Shen": art["unc_chao_shen"],
            "Hybrid Chao-Shen": art["unc_hybrid_chao_shen"],
        }
        # clusters
        from collections import Counter as _Counter
        clusters = {}
        for idx, s in enumerate(sids):
            clusters.setdefault(int(s), []).append(answers[idx])
        ordered = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))

        def _rep(mem):
            nonempty = [m for m in mem if m.strip()]
            pool = nonempty if nonempty else mem
            return _Counter(pool).most_common(1)[0][0] if pool else ""

        cluster_list = [{"members": mem, "representative": _rep(mem),
                         "size": len(mem)} for _, mem in ordered]
        # graph
        G = build_weighted_graph(labels, n)
        G.add_nodes_from(range(n))
        pos = _nx.spring_layout(G, seed=42, weight="weight")
        rank = {c: i for i, (c, _) in enumerate(_Counter(sids).most_common())}
        gnodes = [{"x": float(pos[i][0]), "y": float(pos[i][1]),
                   "crank": rank[sids[i]], "answer": answers[i]} for i in range(n)]
        gedges = [{"s": int(u), "t": int(v), "w": float(d["weight"])}
                  for u, v, d in G.edges(data=True)]

        st.session_state["live_result"] = {
            "question": question, "answers": answers, "n": n,
            "metrics": metrics, "has_lls": has_lls,
            "clusters": cluster_list, "n_clusters": len(clusters),
            "modal": _Counter(a for a in answers if a.strip()).most_common(1)[0][0]
                     if any(a.strip() for a in answers) else "",
            "gnodes": gnodes, "gedges": gedges,
        }

    res = st.session_state.get("live_result")
    if res:
        st.markdown("---")
        st.markdown("#### Most frequent answer")
        modal_safe = _html.escape(res["modal"]) if res["modal"] else "<em style='color:#898781'>no answer returned</em>"
        st.markdown(
            f"<div class='answer-box'><span style='font-size:1.05rem'>"
            f"{modal_safe}</span></div>", unsafe_allow_html=True)
        st.caption(f"{res['n_clusters']} distinct answer clusters across "
                   f"{res['n']} samples. No gold answer here — this is uncertainty "
                   f"without ground truth.")

        method = st.selectbox("Uncertainty method", methods, index=methods.index("KLE-heat"))
        unc = res["metrics"].get(method)
        if unc is None:
            st.info(f"{method} needs token log-probabilities that weren't returned.")
        else:
            pct = int(round(unc * 100))
            st.markdown(
                f"<div class='kpi-lab'>{method} — uncertainty</div>"
                f"<div class='kpi-num'>{unc:.2f}</div>"
                f"<div class='meter-track'><div class='meter-fill' "
                f"style='width:{pct}%;'></div></div>", unsafe_allow_html=True)

        lt1, lt2, lt3 = st.tabs(["📋 All methods", "🔬 Clusters (Discrete SE)",
                                 "🕸️ Similarity graph (KLE-heat)"])
        with lt1:
            rows = [{"Method": m,
                     "Uncertainty": "—" if res["metrics"][m] is None else f"{res['metrics'][m]:.3f}"}
                    for m in methods]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            if not res["has_lls"]:
                st.caption("Log-prob-based methods (SE, KLE-full, Evidential SE) are "
                           "n/a — the model didn't return token log-probabilities.")

        with lt2:
            n = res["n"]
            sizes = [c["size"] for c in res["clusters"]]
            xlab = [f"C{i+1}" for i in range(len(res["clusters"]))]
            fig = go.Figure(go.Bar(x=xlab, y=sizes, marker_color=BLUE, text=sizes,
                                   textposition="outside", marker_line_width=0))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                              paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                              yaxis=dict(title="samples", gridcolor=GRID, dtick=1,
                                         color=INK_2), xaxis=dict(color=INK_2),
                              showlegend=False, font=dict(color=INK))
            st.plotly_chart(fig, width="stretch")
            for i, c in enumerate(res["clusters"]):
                with st.expander(f"C{i+1} · {c['representative'][:60]} — "
                                 f"{c['size']}/{n}"):
                    for mem in c["members"]:
                        st.markdown(f"- {mem or '*(empty)*'}")

        with lt3:
            nodes, edges = res["gnodes"], res["gedges"]
            figg = go.Figure()
            byw = {}
            for ed in edges:
                byw.setdefault(ed["w"], []).append(ed)
            for w in sorted(byw):
                xs, ys = [], []
                for ed in byw[w]:
                    a, b = nodes[ed["s"]], nodes[ed["t"]]
                    xs += [a["x"], b["x"], None]
                    ys += [a["y"], b["y"], None]
                figg.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                    line=dict(width=0.6 + 1.4 * w,
                              color=f"rgba(82,81,78,{0.18 + 0.22 * w:.2f})")))
            cols = [CAT[nd["crank"]] if nd["crank"] < len(CAT) else MUTED
                    for nd in nodes]
            figg.add_trace(go.Scatter(
                x=[nd["x"] for nd in nodes], y=[nd["y"] for nd in nodes],
                mode="markers", showlegend=False,
                hovertext=[nd["answer"][:70] for nd in nodes], hoverinfo="text",
                marker=dict(size=22, color=cols, line=dict(width=2, color=SURFACE))))
            figg.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                               paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                               xaxis=dict(visible=False),
                               yaxis=dict(visible=False, scaleanchor="x"),
                               font=dict(color=INK))
            st.plotly_chart(figg, width="stretch")
            st.caption("Connected blob → low uncertainty; scattered nodes → high.")
    st.stop()

# ---------------------------------------------------------------------------
# Benchmark view — correctness-AUROC per method across datasets
# ---------------------------------------------------------------------------
if st.session_state.view == "benchmark":
    if not benchmark:
        st.warning("Benchmark data isn't present in demo_data.json. Regenerate it "
                   "with `python demo/export_demo_data.py` and redeploy.")
        st.stop()
    bmethods = benchmark["methods"]
    bdatasets = benchmark["datasets"]
    bmodels = benchmark["models"]

    st.subheader("How the methods perform across datasets")
    st.markdown(
        "**AUROC** = how well a method's uncertainty score separates **incorrect** "
        "from correct answers (**0.5 = chance, 1.0 = perfect**). Higher is better. "
        "Only sampling-based methods are shown. ")

    bmodel = st.radio("Model", bmodels, horizontal=True, key="bench_model")
    mv = benchmark["values"][bmodel]

    # best method per dataset (column) among available methods
    best = {}
    for ds in bdatasets:
        avail = [(m, mv[m][ds][0]) for m in bmethods if mv[m]]
        best[ds] = max(avail, key=lambda t: t[1])[0] if avail else None

    z, hover = [], []
    for m in bmethods:
        zr, hr = [], []
        for ds in bdatasets:
            cell = mv[m][ds] if mv[m] else None
            if cell is None:
                zr.append(None)
                hr.append(f"{m} · {ds}<br>n/a (needs token probabilities)")
            else:
                mean, sd = cell[0], cell[1]
                zr.append(mean)
                hr.append(f"{m} · {ds}<br>AUROC {mean:.3f}"
                          + (f" ± {sd:.3f}" if sd is not None else " (single run)"))
        z.append(zr)
        hover.append(hr)

    fig_b = go.Figure(go.Heatmap(
        z=z, x=bdatasets, y=bmethods, text=hover, hoverinfo="text",
        colorscale=[[0.0, "#eaf2fd"], [0.5, "#6da7ec"], [1.0, "#16508f"]],
        zmin=0.5, zmax=0.9, xgap=3, ygap=3, hoverongaps=False,
        colorbar=dict(title="AUROC", thickness=12, outlinewidth=0,
                      tickfont=dict(color=INK_2))))
    for i, m in enumerate(bmethods):
        for ds in bdatasets:
            cell = mv[m][ds] if mv[m] else None
            if cell is None:
                txt, color = "n/a", MUTED
            else:
                v = cell[0]
                txt = f"{v:.2f}" + ("  ★" if best[ds] == m else "")
                color = "#ffffff" if v >= 0.74 else INK
            fig_b.add_annotation(x=ds, y=m, text=txt, showarrow=False,
                                 font=dict(color=color, size=13))
    fig_b.update_yaxes(autorange="reversed")
    fig_b.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        xaxis=dict(side="top", color=INK_2), yaxis=dict(color=INK),
        font=dict(color=INK, family="system-ui, -apple-system, Segoe UI, sans-serif"))
    st.plotly_chart(fig_b, width="stretch")
    st.caption("★ = best method for that dataset (mean over 3 runs; Claude Haiku "
               "is a single run). Hover for ± std.")

    st.info(benchmark["headline"])
    if any(mv[m] is None for m in bmethods):
        st.caption("ℹ️ " + benchmark["note"])
    st.stop()

# ---------------------------------------------------------------------------
# ② Revision view — does an uncertainty score predict revision under challenge,
#    and what does that revision do to accuracy? (sycophancy, quantified)
# ---------------------------------------------------------------------------
if st.session_state.view == "revision":
    if not revision:
        st.warning("Revision data isn't present in demo_data.json. Regenerate it "
                   "with `python demo/export_demo_data.py`.")
        st.stop()
    rmethods = revision["methods"]
    rdatasets = revision["datasets"]
    rmodels = revision["models"]
    stable = revision.get("stable_rate", 0.10)

    st.subheader("Does a UQ score predict whether the model backs down?")
    st.markdown(
        "Each model answers, then is challenged once — *“Are you sure? Please "
        "reconsider your answer.”* We ask two things: **(1)** does a high "
        "uncertainty score *predict* which answers get **revised** "
        "(revision-AUROC, 0.5 = chance), and **(2)** does revising actually "
        "**help**? Higher AUROC = uncertainty anticipates revision better.")

    rmodel = st.radio("Model", rmodels, horizontal=True, key="rev_model")
    mv = revision["auroc"][rmodel]
    acc = revision["accuracy"][rmodel]

    # ---- Panel A: revision-AUROC heatmap (method × dataset) -----------------
    st.markdown("##### ① Uncertainty vs. likelihood of revision — AUROC")
    z, hover, unstable = [], [], set()
    for ds in rdatasets:
        if acc.get(ds, {}).get("rev_rate", 1.0) < stable:
            unstable.add(ds)
    for m in rmethods:
        zr, hr = [], []
        for ds in rdatasets:
            cell = mv[m][ds] if mv[m] else None
            if cell is None:
                zr.append(None)
                hr.append(f"{m} · {ds}<br>n/a (needs token probabilities)")
            else:
                mean, sd = cell[0], cell[1]
                tag = "  ⚠ low revision rate" if ds in unstable else ""
                hr.append(f"{m} · {ds}<br>revision-AUROC {mean:.3f}"
                          + (f" ± {sd:.3f}" if sd is not None else " (single run)")
                          + tag)
                zr.append(mean)
        z.append(zr)
        hover.append(hr)

    fig_r = go.Figure(go.Heatmap(
        z=z, x=rdatasets, y=rmethods, text=hover, hoverinfo="text",
        colorscale=[[0.0, "#eaf2fd"], [0.5, "#6da7ec"], [1.0, "#16508f"]],
        zmin=0.5, zmax=0.9, xgap=3, ygap=3, hoverongaps=False,
        colorbar=dict(title="AUROC", thickness=12, outlinewidth=0,
                      tickfont=dict(color=INK_2))))
    for i, m in enumerate(rmethods):
        for ds in rdatasets:
            cell = mv[m][ds] if mv[m] else None
            if cell is None:
                txt, color = "n/a", MUTED
            else:
                v = cell[0]
                txt = f"{v:.2f}" + ("*" if ds in unstable else "")
                color = "#ffffff" if v >= 0.74 else INK
            fig_r.add_annotation(x=ds, y=m, text=txt, showarrow=False,
                                 font=dict(color=color, size=13))
    fig_r.update_yaxes(autorange="reversed")
    fig_r.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        xaxis=dict(side="top", color=INK_2), yaxis=dict(color=INK),
        font=dict(color=INK, family="system-ui, -apple-system, Segoe UI, sans-serif"))
    st.plotly_chart(fig_r, width="stretch")
    if unstable:
        st.caption("＊ = revision rate below "
                   f"{int(stable*100)}% (" + ", ".join(sorted(unstable)) +
                   ") — too few revisions for the AUROC to be stable; ignore these "
                   "columns. " + revision["note"])

    # ---- Panel B: accuracy before vs after (dumbbell) -----------------------
    st.markdown("##### ② …but does revising help? Accuracy before vs. after the challenge")
    fig_a = go.Figure()
    ys = list(rdatasets)
    for ds in ys:
        a = acc[ds]
        ai, ar = a["acc_init"], a["acc_rev"]
        col = GOOD if ar >= ai - 1e-9 else CRITICAL
        fig_a.add_trace(go.Scatter(
            x=[ai, ar], y=[ds, ds], mode="lines",
            line=dict(color=col, width=4), hoverinfo="skip", showlegend=False))
    # before / after markers
    fig_a.add_trace(go.Scatter(
        x=[acc[ds]["acc_init"] for ds in ys], y=ys, mode="markers",
        marker=dict(size=15, color=MUTED, line=dict(width=2, color=SURFACE)),
        name="before challenge",
        hovertext=[f"{ds}: initial accuracy {acc[ds]['acc_init']:.1%}" for ds in ys],
        hoverinfo="text"))
    fig_a.add_trace(go.Scatter(
        x=[acc[ds]["acc_rev"] for ds in ys], y=ys, mode="markers",
        marker=dict(size=15, color=INK, line=dict(width=2, color=SURFACE)),
        name="after challenge",
        hovertext=[f"{ds}: revised accuracy {acc[ds]['acc_rev']:.1%} "
                   f"(Δ {acc[ds]['acc_rev']-acc[ds]['acc_init']:+.1%})" for ds in ys],
        hoverinfo="text"))
    for ds in ys:
        a = acc[ds]
        d = a["acc_rev"] - a["acc_init"]
        fig_a.add_annotation(x=max(a["acc_init"], a["acc_rev"]), y=ds,
                             text=f"  Δ {d:+.1%}", showarrow=False, xanchor="left",
                             font=dict(color=(GOOD if d >= 0 else CRITICAL), size=12))
    fig_a.update_layout(
        height=260, margin=dict(l=10, r=70, t=10, b=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        xaxis=dict(title="accuracy", tickformat=".0%", gridcolor=GRID,
                   range=[min(min(acc[ds]["acc_init"], acc[ds]["acc_rev"]) for ds in ys)-0.08, 1.0],
                   color=INK_2),
        yaxis=dict(color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(color=INK, family="system-ui, -apple-system, Segoe UI, sans-serif"))
    st.plotly_chart(fig_a, width="stretch")
    cols = st.columns(len(ys))
    for c, ds in zip(cols, ys):
        a = acc[ds]
        with c:
            st.markdown(
                f"<div class='kpi-lab'>{ds} · revised {a['rev_rate']:.0%}</div>"
                f"<div style='font-size:0.95rem;color:{INK}'>"
                f"<span style='color:{GOOD}'>▲ helped {a['helped']:.0f}</span> · "
                f"<span style='color:{CRITICAL}'>▼ hurt {a['hurt']:.0f}</span></div>",
                unsafe_allow_html=True)
    st.caption("Grey = accuracy before the challenge, black = after. A red bar means "
               "the challenge **lowered** accuracy. Across almost every cell, being "
               "challenged hurts more answers than it helps — the model caves on "
               "answers it should have kept.")

    # ---- Panel C: signal survives on initially-correct answers --------------
    ac = [r for r in revision["auroc_correct"] if r[0] == rmodel]
    with st.expander("Is this just 'uncertain answers were already wrong'? — No "
                     "(revision-AUROC on initially-CORRECT answers only)"):
        st.markdown(
            "Restricted to answers that were **correct to begin with**, uncertainty "
            "*still* separates the ones the model later abandons (all 95% CIs exclude "
            "0.5). High uncertainty flags correct answers the model talks itself out "
            "of — a genuine fragility signal, not just a proxy for being wrong.")
        show = ac if ac else revision["auroc_correct"]
        rows = [{"Model": r[0], "Dataset": r[1], "Method": r[2],
                 "AUROC (correct-only)": f"{r[3][0]:.3f}",
                 "95% CI": f"[{r[3][1]:.3f}, {r[3][2]:.3f}]"} for r in show]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if not ac:
            st.caption(f"No initially-correct breakdown was reported for {rmodel} "
                       "(only the high-revision-rate cells were analysed); showing all.")

    st.info(revision["headline"])
    st.stop()

# ---------------------------------------------------------------------------
# ③ Injection view — does handing the model its own UQ score at challenge time
#    counter the sycophancy? (spoiler: no)
# ---------------------------------------------------------------------------
if st.session_state.view == "injection":
    if not injection:
        st.warning("Injection-ablation data isn't present in demo_data.json. "
                   "Regenerate it with `python demo/export_demo_data.py`.")
        st.stop()
    conds = injection["conditions"]           # (key, label, kind)
    imodels = injection["models"]
    idatasets = injection["datasets"]

    st.subheader("If we tell the model how uncertain it is, does it stop caving?")
    st.markdown(
        "We re-run the same challenge, but inject one extra clause. Two **true-score** "
        "conditions carry the model's *real* KLE-heat uncertainty (and the same number "
        "reframed as confidence). Each has a matched **control**: a random number, "
        "text that mentions uncertainty with no value, a neutral placebo clause, or a "
        "bare baseline. If the *number* mattered, the true-score bars would sit clearly "
        "above the controls.")

    c1, c2 = st.columns(2)
    with c1:
        imodel = st.radio("Model", imodels, horizontal=True, key="inj_model")
    with c2:
        idataset = st.radio("Dataset", idatasets, horizontal=True, key="inj_ds")
    dv = injection["delta"][imodel][idataset]

    xlabs = [lab for _, lab, _ in conds]
    vals = [dv[k][0] for k, _, _ in conds]
    errs = [dv[k][1] if dv[k][1] is not None else 0 for k, _, _ in conds]
    colors = [CRITICAL if kind == "true" else MUTED for _, _, kind in conds]

    fig_i = go.Figure(go.Bar(
        x=xlabs, y=vals, marker_color=colors,
        error_y=dict(type="data", array=errs, color=INK_2, thickness=1, width=4),
        text=[f"{v:+.1%}" for v in vals], textposition="outside",
        hovertext=[f"{lab}<br>Δaccuracy {dv[k][0]:+.3f} ± {dv[k][1]:.3f}"
                   for k, lab, _ in conds], hoverinfo="text", marker_line_width=0))
    fig_i.add_hline(y=0, line_width=1, line_color=INK_2)
    fig_i.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=90),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        yaxis=dict(title="Δ accuracy (revised − initial)", tickformat="+.0%",
                   gridcolor=GRID, zeroline=False, color=INK_2),
        xaxis=dict(color=INK_2, tickangle=-30),
        font=dict(color=INK, family="system-ui, -apple-system, Segoe UI, sans-serif"),
        showlegend=False)
    st.plotly_chart(fig_i, width="stretch")
    st.markdown(
        f"<span class='badge' style='background:{CRITICAL}22;color:{CRITICAL};'>"
        f"■ true score</span> "
        f"<span class='badge' style='background:{MUTED}22;color:{MUTED};'>"
        f"■ control</span>", unsafe_allow_html=True)
    st.caption(injection["note"])
    st.info(injection["headline"])
    st.stop()

st.caption("Pre-generated results. Pick a model, then a question curated for "
           "that model. The answer shown is always the selected model's, and "
           "each question is a clean example of one outcome.")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
model = st.sidebar.selectbox("Model", models, index=0, key="model_select")
method = st.sidebar.selectbox("Uncertainty method (sampling-based)", methods,
                              index=0, key="method_select")
st.sidebar.markdown("---")
st.sidebar.markdown("**Methods** all estimate uncertainty from repeated samples "
                    "of the model's answer, clustered by meaning. Higher = more "
                    "uncertain.")
st.sidebar.caption("Only sampling-based methods are shown (no verbalized / "
                   "prompted self-confidence).")

examples = examples_by_model[model]

# order examples by outcome (cc, uw, cw) then dataset, for a tidy picker
order = {o: i for i, o in enumerate(outcomes)}
examples = sorted(examples, key=lambda e: (order[e["outcome"]], e["dataset"]))


def q_label(e):
    emoji = OUTCOME_STYLE[e["outcome"]][0]
    q = e["question"]
    q = q if len(q) <= 70 else q[:69] + "…"
    return f"{emoji}  {e['outcome']} · {e['dataset']} — {q}"


# Use unique label strings as options (robust; avoids format_func edge cases).
q_labels = []
for i, ex in enumerate(examples):
    lab = q_label(ex)
    if lab in q_labels:
        lab = f"{lab}  ·#{i + 1}"
    q_labels.append(lab)

choice = st.radio("Question (curated for this model)", q_labels, index=0,
                      key=f"q_{model}")
e = examples[q_labels.index(choice)]

# note any outcomes this model never exhibits
present = {ex["outcome"] for ex in examples}
missing = [o for o in outcomes if o not in present]
if missing:
    st.caption(f"ℹ️ {model} has no “{', '.join(missing)}” example — it never "
               f"reaches that outcome under the strict thresholds.")

emoji, cat_color = OUTCOME_STYLE[e["outcome"]]
primary_gold, extra = clean_gold(e["gold_answer"])

# ---------------------------------------------------------------------------
# Question header — outcome is fixed for this (model, question), always matches.
# ---------------------------------------------------------------------------
st.markdown(
    f"<span class='badge' style='background:{cat_color}22;color:{cat_color};'>"
    f"{emoji} {e['outcome']}</span> "
    f"<span style='color:{MUTED};font-size:0.85rem'>{model} · {e['dataset']}</span>",
    unsafe_allow_html=True)

st.markdown(f"### {e['question']}")

if e.get("context"):
    with st.expander("Show context passage"):
        st.write(e["context"])

gold_txt = primary_gold + (f"  <span style='color:{MUTED}'>(+{extra} accepted "
                           f"variants)</span>" if extra else "")
st.markdown(f"**Gold answer:** {gold_txt}", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Answer + correctness + uncertainty
# ---------------------------------------------------------------------------
correct = e["correct"]
mark = "✓" if correct else "✗"
mark_color = GOOD if correct else CRITICAL
verdict = "Correct" if correct else "Wrong"
unc = e["metrics"].get(method)

left, right = st.columns([3, 2])

with left:
    st.markdown(f"#### {model}'s answer")
    ans_txt = e["answer"].strip()
    ans_html = (_html.escape(ans_txt).replace("\n", "<br>") if ans_txt
                else f"<em style='color:{MUTED}'>(no answer)</em>")
    st.markdown(
        f"<div class='answer-box'>"
        f"<div style='display:flex;align-items:center;gap:8px;"
        f"padding-bottom:10px;margin-bottom:10px;border-bottom:1px solid {GRID};'>"
        f"<span style='color:{mark_color};font-weight:700;font-size:1.3rem'>{mark}</span>"
        f"<span class='badge' style='background:{mark_color}22;color:{mark_color};'>"
        f"{verdict}</span></div>"
        f"<div style='font-size:1.05rem;color:{INK};line-height:1.4'>{ans_html}</div>"
        f"</div>",
        unsafe_allow_html=True)

with right:
    st.markdown(f"#### {method}")
    if unc is None:
        st.info("Not available for this item.")
    else:
        conf = 1.0 - unc
        pct = int(round(unc * 100))
        st.markdown(
            f"<div class='kpi-lab'>Uncertainty</div>"
            f"<div class='kpi-num'>{unc:.2f}</div>"
            f"<div class='meter-track'><div class='meter-fill' "
            f"style='width:{pct}%;'></div></div>"
            f"<div style='color:{INK_2};font-size:0.85rem;margin-top:6px'>"
            f"Confidence = {conf:.2f} &nbsp;·&nbsp; {e['n_clusters']} distinct "
            f"answer clusters across {e['n_samples']} samples</div>",
            unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Two views
# ---------------------------------------------------------------------------
tab_overview, tab_clusters, tab_graph = st.tabs(
    ["📋 All methods", "🔬 How the samples clustered — Discrete SE",
     "🕸️ Semantic similarity graph — KLE-heat"])

with tab_overview:
    st.markdown(f"Every sampling-based method scored on this question for "
                f"**{model}**. The selected method is highlighted.")
    rows = []
    for name in methods:
        v = e["metrics"].get(name)
        rows.append({
            "Method": name,
            "Uncertainty": "—" if v is None else f"{v:.3f}",
            "Confidence": "—" if v is None else f"{1 - v:.3f}",
        })

    def _hl(row):
        return ["background-color:#cde2fb" if row["Method"] == method else ""
                for _ in row]

    df = pd.DataFrame(rows)
    st.dataframe(df.style.apply(_hl, axis=1), hide_index=True, width="stretch")
    st.caption("All values normalized to 0–1 (1 = maximally uncertain). "
               "Methods weight cluster frequencies and answer likelihoods "
               "differently, so they don't always agree.")

with tab_clusters:
    st.markdown(
        "**Discrete Semantic Entropy** is the simplest sampling method: sample "
        "the model many times, group the answers that *mean the same thing* into "
        "clusters, then measure how spread out the samples are across clusters.")
    st.markdown(
        "- One dominant cluster → the model keeps saying the same thing → **low** "
        "uncertainty.\n"
        "- Samples scattered across many clusters → the model is unsure → **high** "
        "uncertainty.")

    clusters = e["clusters"]
    n = e["n_samples"]
    sizes = [c["size"] for c in clusters]
    reps = [c["representative"] for c in clusters]

    def trunc(s, k=42):
        s = s.replace("\n", " ").strip()
        return s if len(s) <= k else s[: k - 1] + "…"

    x_labels = [f"C{i+1}" for i in range(len(clusters))]
    hover = [f"{trunc(r, 70)}<br>size: {s}/{n}" for r, s in zip(reps, sizes)]

    fig = go.Figure(go.Bar(
        x=x_labels, y=sizes, marker_color=BLUE,
        text=sizes, textposition="outside",
        hovertext=hover, hoverinfo="text", marker_line_width=0,
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        yaxis=dict(title="samples in cluster", gridcolor=GRID, zeroline=False,
                   dtick=1, color=INK_2),
        xaxis=dict(color=INK_2),
        font=dict(color=INK, family="system-ui, -apple-system, Segoe UI, sans-serif"),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    probs = [s / n for s in sizes]
    H = -sum(p * math.log(p) for p in probs if p > 0)
    H_norm = H / math.log(n) if n > 1 else 0.0
    st.markdown(
        f"**Discrete SE** = −Σ pᵢ·ln(pᵢ) over cluster proportions "
        f"= **{H:.3f}** nats → normalized **{H_norm:.3f}** (÷ ln {n}).")

    st.markdown("##### The clusters")
    for i, c in enumerate(clusters):
        share = c["size"] / n
        with st.expander(f"C{i+1} · {trunc(c['representative'], 60)}  "
                         f"— {c['size']}/{n} samples ({share:.0%})"):
            for member in c["members"]:
                st.markdown(f"- {member.strip() or '*(empty)*'}")

with tab_graph:
    st.markdown(
        "**KLE-heat** doesn't make hard clusters. It builds a *semantic "
        "similarity graph* over the samples and diffuses a heat kernel across it, "
        "then reads uncertainty off how connected the graph is.")
    st.markdown(
        "- Each **node** is one sampled answer.\n"
        "- An **edge** links two answers the NLI model finds semantically "
        "compatible — thicker = stronger (mutual entailment = 2, one-way / "
        "neutral = 0.5–1.5, contradiction = no edge).\n"
        "- **One connected blob** → the model keeps agreeing with itself → "
        "**low** uncertainty. **Scattered / isolated** nodes → disagreement → "
        "**high** uncertainty.")

    g = e["graph"]
    nodes, edges = g["nodes"], g["edges"]

    def trunc_g(s, k=70):
        s = s.replace("\n", " ").strip()
        return s if len(s) <= k else s[: k - 1] + "…"

    fig_g = go.Figure()
    # edges grouped by weight so each bucket gets its own width / opacity
    by_w = {}
    for ed in edges:
        by_w.setdefault(ed["w"], []).append(ed)
    for w in sorted(by_w):
        xs, ys = [], []
        for ed in by_w[w]:
            a, b = nodes[ed["s"]], nodes[ed["t"]]
            xs += [a["x"], b["x"], None]
            ys += [a["y"], b["y"], None]
        fig_g.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(width=0.6 + 1.4 * w, color=f"rgba(82,81,78,{0.18 + 0.22 * w:.2f})"),
            hoverinfo="skip", showlegend=False))

    # nodes coloured by cluster (redundant with position + edges; gray overflow)
    node_colors = [CAT[nd["crank"]] if nd["crank"] < len(CAT) else MUTED
                   for nd in nodes]
    hover = [f"cluster {nd['cluster']}<br>{trunc_g(nd['answer'])}" for nd in nodes]
    fig_g.add_trace(go.Scatter(
        x=[nd["x"] for nd in nodes], y=[nd["y"] for nd in nodes],
        mode="markers", marker=dict(size=22, color=node_colors,
                                    line=dict(width=2, color=SURFACE)),
        hovertext=hover, hoverinfo="text", showlegend=False))
    fig_g.update_layout(
        height=440, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        xaxis=dict(visible=False), yaxis=dict(visible=False,
                                              scaleanchor="x", scaleratio=1),
        font=dict(color=INK, family="system-ui, -apple-system, Segoe UI, sans-serif"),
    )
    st.plotly_chart(fig_g, width="stretch")

    kh = e["metrics"].get("KLE-heat")
    n_edges = len(edges)
    max_edges = e["n_samples"] * (e["n_samples"] - 1) // 2
    kh_txt = "—" if kh is None else f"{kh:.2f}"
    st.markdown(
        f"**KLE-heat uncertainty = {kh_txt}** · {n_edges}/{max_edges} possible "
        f"semantic links present · {e['n_clusters']} colour"
        f"{'s' if e['n_clusters'] != 1 else ''} = hard clusters (shown only to "
        f"help read the graph; KLE-heat itself never commits to them).")
    st.caption("Node colour marks the hard semantic cluster purely as a reading "
               "aid — the clustering KLE-heat actually uses is the soft, weighted "
               "connectivity shown by the edges. Hover a node to see its answer.")