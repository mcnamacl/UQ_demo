"""
Uncertainty Quantification demo (Streamlit).

15 pre-generated questions (5 per dataset: SQuAD, SVAMP, TriviaQA) across five
models. Pick a dataset and a question; the model that cleanly demonstrates that
question's variant loads by default, but you can switch models to compare. For
each model you see its answer (with a ✓/✗ against the gold answer) and its
sampling-based uncertainty under a chosen method. A second view shows how the
sampled answers were grouped into semantic clusters to produce a Discrete
Semantic Entropy (SE) estimate — the most intuitive of the sampling methods.

Run:  python -m streamlit run demo/app.py
"""
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


CATEGORY_STYLE = {
    "Confident & correct": ("✅", GOOD),
    "Uncertain & wrong":   ("⚠️", WARN),
    "Confident & wrong":   ("🚨", CRITICAL),
}


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Uncertainty Quantification Demo",
                   page_icon="📊", layout="wide")

st.markdown(f"""
<style>
  .stApp {{ background: {SURFACE}; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:999px;
            font-size:0.8rem; font-weight:600; }}
  .answer-box {{ border:1px solid {GRID}; border-radius:10px; padding:14px 18px;
                 background:#ffffff; }}
  .meter-track {{ background:{GRID}; border-radius:999px; height:14px; width:100%; }}
  .meter-fill  {{ background:{BLUE}; border-radius:999px; height:14px; }}
  .kpi-num {{ font-size:2.2rem; font-weight:700; color:{INK}; line-height:1; }}
  .kpi-lab {{ font-size:0.8rem; color:{INK_2}; text-transform:uppercase;
              letter-spacing:0.04em; }}
</style>
""", unsafe_allow_html=True)

data = load_data()
all_questions = data["questions"]
models = data["models"]
methods = data["methods"]
datasets = data["datasets"]

st.title("Uncertainty Quantification — sampling-based methods")
st.caption("Pre-generated results across SQuAD, SVAMP and TriviaQA. Pick a "
           "dataset and question; the model that best illustrates it loads by "
           "default — switch models and methods to explore.")

# ---------------------------------------------------------------------------
# Sidebar: dataset (question picker lives in the main pane, before the model
# selector, so switching questions can reset the model to that question's anchor)
# ---------------------------------------------------------------------------
dataset = st.sidebar.radio("Dataset", datasets, index=0, key="dataset_select",
                           horizontal=True)
qs = [q for q in all_questions if q["dataset"] == dataset]

labels = [f"{CATEGORY_STYLE[q['category']][0]}  {q['question']}" for q in qs]
sel = st.radio("Question", range(len(qs)),
               format_func=lambda i: labels[i], index=0, key=f"q_{dataset}")
q = qs[sel]

# Reset the model selection to this question's anchor whenever the question
# changes; manual model changes then stick until the next question switch.
if st.session_state.get("_last_qid") != q["id"]:
    st.session_state["model_select"] = q["anchor_model"]
    st.session_state["_last_qid"] = q["id"]

# Now the rest of the sidebar controls.
model = st.sidebar.radio("Model", models, key="model_select")
method = st.sidebar.radio("Uncertainty method (sampling-based)", methods,
                          index=0, key="method_select")
st.sidebar.markdown("---")
st.sidebar.markdown("**Methods** all estimate uncertainty from repeated samples "
                    "of the model's answer, clustered by meaning. Higher = more "
                    "uncertain.")
st.sidebar.caption("Only sampling-based methods are shown (no verbalized / "
                   "prompted self-confidence).")

m = q["per_model"][model]
emoji, cat_color = CATEGORY_STYLE[q["category"]]
primary_gold, extra = clean_gold(q["gold_answer"])

# ---------------------------------------------------------------------------
# Question header
# ---------------------------------------------------------------------------
st.markdown(
    f"<span class='badge' style='background:{cat_color}22;color:{cat_color};'>"
    f"{emoji} {q['category']}</span> "
    f"<span style='color:{MUTED};font-size:0.85rem'>illustrated by "
    f"<b>{q['anchor_model']}</b> (loaded by default)</span>",
    unsafe_allow_html=True)

st.markdown(f"### {q['question']}")

if q.get("context"):
    with st.expander("Show context passage"):
        st.write(q["context"])

gold_txt = primary_gold + (f"  <span style='color:{MUTED}'>(+{extra} accepted "
                           f"variants)</span>" if extra else "")
st.markdown(f"**Gold answer:** {gold_txt}", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Answer + correctness + uncertainty (reflect the SELECTED model)
# ---------------------------------------------------------------------------
correct = m["correct"]
mark = "✓" if correct else "✗"
mark_color = GOOD if correct else CRITICAL
verdict = "Correct" if correct else "Wrong"
unc = m["metrics"].get(method)

left, right = st.columns([3, 2])

with left:
    st.markdown(f"#### {model}'s answer")
    ans_display = m["answer"].strip().replace("\n", "  \n") or "*(no answer)*"
    st.markdown(
        f"<div class='answer-box'>"
        f"<span style='color:{mark_color};font-weight:700;font-size:1.3rem'>{mark}</span> "
        f"<span class='badge' style='background:{mark_color}22;color:{mark_color};'>"
        f"{verdict}</span></div>",
        unsafe_allow_html=True)
    st.markdown(ans_display)

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
            f"Confidence = {conf:.2f} &nbsp;·&nbsp; {m['n_clusters']} distinct "
            f"answer clusters across {m['n_samples']} samples</div>",
            unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Two views
# ---------------------------------------------------------------------------
tab_overview, tab_clusters = st.tabs(
    ["📋 All methods (this model)", "🔬 How the samples clustered — Discrete SE"])

with tab_overview:
    st.markdown("Every sampling-based method scored on **this** question for "
                f"**{model}**. The selected method is highlighted.")
    rows = []
    for name in methods:
        v = m["metrics"].get(name)
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

    clusters = m["clusters"]
    n = m["n_samples"]
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