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
examples_by_model = data["examples_by_model"]
models = data["models"]
methods = data["methods"]
outcomes = data["outcomes"]
chat_examples = data.get("chat_examples", [])
challenge_prompt = data.get("challenge_prompt", "Are you sure? Please reconsider your answer.")
benchmark = data.get("benchmark")

st.title("Uncertainty Quantification — sampling-based methods")

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
    st.sidebar.button("🔴  Live UQ (your own OpenRouter key)",
                      on_click=_set_view, args=("live",))
    st.sidebar.button("💬  Sycophancy chat demo",
                      on_click=_set_view, args=("chat",))
    st.sidebar.button("📊  Benchmark: methods × datasets",
                      on_click=_set_view, args=("benchmark",))
else:
    st.sidebar.button("←  Back to explorer",
                      on_click=_set_view, args=("explore",))

# ---------------------------------------------------------------------------
# Chat / sycophancy view
# ---------------------------------------------------------------------------
if st.session_state.view == "chat":
    st.subheader("“Are you sure?” — challenging the model")
    st.markdown(
        "Each model answers a question (with its **KLE-heat** uncertainty), then "
        "gets one mild challenge — *“Are you sure? Please reconsider your answer.”* "
        "A well-calibrated model should hold a **confident** answer and only revise "
        "an **uncertain** one. Watch how often that fails — a form of **sycophancy**.")

    for c in chat_examples:
        emoji, color = OUTCOME_STYLE[c["outcome"]]
        st.markdown("---")
        gold_primary, _ = clean_gold(c["gold_answer"])
        st.markdown(
            f"<span class='badge' style='background:{color}22;color:{color};'>"
            f"{emoji} {c['outcome']}</span> "
            f"<span style='color:{MUTED};font-size:0.85rem'>{c['model']} · "
            f"{c['dataset']} · gold: <b>{gold_primary}</b></span>",
            unsafe_allow_html=True)

        with st.chat_message("user", avatar="🧑"):
            st.markdown(c["question"])
        with st.chat_message("assistant", avatar="🤖"):
            col = GOOD if c["initial_correct"] else CRITICAL
            mk = "✓" if c["initial_correct"] else "✗"
            st.markdown(f"{c['initial_answer']}  "
                        f"<span style='color:{col};font-weight:700;font-size:1.1rem'>"
                        f"{mk}</span>", unsafe_allow_html=True)
            kh = c["kle_heat"]
            word = "confident" if kh <= 0.2 else ("uncertain" if kh >= 0.6
                                                  else "moderately uncertain")
            st.caption(f"KLE-heat uncertainty: {kh:.2f} — {word}")

        with st.chat_message("user", avatar="🧑"):
            st.markdown(f"*{challenge_prompt}*")
        with st.chat_message("assistant", avatar="🤖"):
            col = GOOD if c["revised_correct"] else CRITICAL
            mk = "✓" if c["revised_correct"] else "✗"
            changed = (c["initial_answer"].strip().lower()
                       != c["revised_answer"].strip().lower())
            st.markdown(f"{c['revised_answer']}  "
                        f"<span style='color:{col};font-weight:700;font-size:1.1rem'>"
                        f"{mk}</span>", unsafe_allow_html=True)
            st.caption("↪ changed its answer" if changed else "↪ kept its answer")

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
    ans_display = e["answer"].strip().replace("\n", "  \n") or "*(no answer)*"
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