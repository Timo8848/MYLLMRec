"""
LLMRec Steam Demo — Gradio app.

Run locally:
    cd LLMRec/hf_demo
    pip install gradio
    python app.py
"""

import json
from pathlib import Path

import gradio as gr

DATA_FILE = Path(__file__).parent / "recommendations.json"

with open(DATA_FILE) as f:
    DATA = json.load(f)

STATS   = DATA["stats"]
USERS   = DATA["users"]
POP_TOP = DATA["pop_top20"]
N_USERS = STATS["n_users"]


# ── Rendering helpers ─────────────────────────────────────────────────────────

def genre_set(items: list[dict]) -> set[str]:
    out = set()
    for item in items:
        for g in item.get("genres", "").split(" | "):
            g = g.strip()
            if g:
                out.add(g)
    return out


def _tag_html(tags: list[str]) -> str:
    return " ".join(
        f'<span style="background:#2a2a2a;color:#bbb;padding:1px 6px;'
        f'border-radius:8px;font-size:11px;">{t}</span>'
        for t in tags[:4]
    )


def render_card(item: dict, shared_genres: set | None = None,
                highlight_hit: bool = False, rank: int | None = None) -> str:
    title  = item.get("title", "Unknown")
    genres = item.get("genres", "")
    tags   = item.get("tags", [])

    item_genres = {g.strip() for g in genres.split(" | ") if g.strip()}
    genre_match = bool(shared_genres and item_genres & shared_genres) and not highlight_hit

    border = "#4CAF50" if highlight_hit else ("#1976D2" if genre_match else "#383838")
    bg     = "#0d1f0d" if highlight_hit else ("#0d1226" if genre_match else "#1c1c1c")

    badges = ""
    if highlight_hit:
        badges += '<span style="background:#4CAF50;color:#fff;padding:1px 7px;border-radius:10px;font-size:11px;margin-left:6px;">✓ Hit</span>'
    if genre_match:
        badges += '<span style="background:#1976D2;color:#fff;padding:1px 7px;border-radius:10px;font-size:11px;margin-left:6px;">genre match</span>'

    rank_str = f'<span style="color:#666;font-size:12px;margin-right:4px;">#{rank}</span>' if rank else ""

    return (
        f'<div style="border:1px solid {border};border-radius:8px;'
        f'padding:10px 14px;margin:4px 0;background:{bg};">'
        f'<div style="font-weight:600;color:#e8e8e8;">{rank_str}{title}{badges}</div>'
        f'<div style="color:#999;font-size:12px;margin-top:2px;">{genres}</div>'
        f'<div style="margin-top:5px;">{_tag_html(tags)}</div>'
        f'</div>'
    )


def section(heading: str, cards_html: str) -> str:
    return (
        f'<div style="margin-bottom:8px;">'
        f'<div style="color:#aaa;font-size:13px;font-weight:600;'
        f'margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px;">{heading}</div>'
        f'{cards_html}</div>'
    )


# ── Main update callback ──────────────────────────────────────────────────────

def update(user_label: str):
    uid = user_label.split()[1]
    user = USERS[uid]

    history = user["history"]
    recs    = user["recommendations"][:10]
    test    = user["test_item"]
    n_hist  = user["history_total"]

    history_genres = genre_set(history)

    # History panel
    hist_cards = "".join(render_card(it) for it in history)
    hist_html  = section(
        f"Play history · {n_hist} games total, showing top {len(history)} by popularity",
        hist_cards,
    )

    # Recommendation panel
    rec_cards = "".join(
        render_card(it, shared_genres=history_genres,
                    highlight_hit=it.get("is_test_hit", False), rank=it["rank"])
        for it in recs
    )
    rec_html = section("LLMRec top-10 recommendations", rec_cards)

    # Test item + verdict panel
    if test:
        test_card  = render_card(test, shared_genres=history_genres)
        llm_hit    = user["llm_hit_at_20"]
        pop_hit    = user["pop_hit_at_20"]
        llm_color  = "#4CAF50" if llm_hit  else "#e57373"
        pop_color  = "#4CAF50" if pop_hit  else "#e57373"
        verdict = (
            f'<div style="margin-top:6px;font-size:13px;">'
            f'<span style="color:{llm_color};font-weight:600;">'
            f'{"✓ LLMRec found it in top-20" if llm_hit else "✗ LLMRec missed it"}'
            f'</span>'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;'
            f'<span style="color:{pop_color};">'
            f'{"✓ Pop@20 found it" if pop_hit else "✗ Pop@20 missed it"}'
            f'</span></div>'
        )
        test_html = section("Hidden test item (held-out ground truth)", test_card + verdict)
    else:
        test_html = ""

    return hist_html, rec_html, test_html


# ── Stats banner ──────────────────────────────────────────────────────────────

def stats_banner() -> str:
    llm = STATS["llmrec_recall_20"]
    pop = STATS["pop_recall_20"]
    lift = round(llm / pop, 1) if pop > 0 else "∞"
    return (
        f'<div style="background:#111827;border:1px solid #2a2a3a;border-radius:8px;'
        f'padding:12px 18px;font-size:13px;line-height:1.8;">'
        f'<b style="color:#d0d0d0;">Dataset</b> &nbsp;'
        f'<span style="color:#888;">700 users · 2,119 Steam games · ~41k interactions (k-core dense subgraph)</span><br>'
        f'<b style="color:#d0d0d0;">Model</b> &nbsp;'
        f'<span style="color:#888;">LightGCN + MiniLM-L6 text features + user profiles (BPR + prune loss)</span><br>'
        f'<b style="color:#d0d0d0;">LLMRec Recall@20</b> &nbsp;'
        f'<span style="color:#4CAF50;font-weight:700;">{llm:.4f}</span>'
        f'&emsp;'
        f'<b style="color:#d0d0d0;">Pop@20 baseline</b> &nbsp;'
        f'<span style="color:#e57373;">{pop:.4f}</span>'
        f'&emsp;'
        f'<span style="color:#FFD700;font-weight:700;">{lift}× lift over popularity</span><br>'
        f'<span style="color:#555;font-size:12px;">'
        f'🔵 Blue border = genre overlap with history &nbsp;·&nbsp; 🟢 Green + ✓ Hit = model correctly recommended the hidden test item'
        f'</span>'
        f'</div>'
    )


# ── Layout ────────────────────────────────────────────────────────────────────

user_options = [f"User {i}" for i in range(N_USERS)]

with gr.Blocks(title="LLMRec Steam Demo") as demo:

    gr.Markdown("# LLMRec · Steam Game Recommender Demo", elem_id="title")
    gr.Markdown(
        "Select any of the 700 demo users to see their play history and what the model recommends.\n\n"
        "Genre-matched recommendations are highlighted in **blue**; if the model's top-20 includes "
        "the held-out test item it appears in **green** with a ✓ badge.",
        elem_id="subtitle",
    )

    gr.HTML(stats_banner())

    with gr.Row():
        user_dd = gr.Dropdown(
            choices=user_options,
            value="User 42",
            label="Select a demo user",
            scale=1,
        )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            hist_out = gr.HTML(label="Play history")
        with gr.Column(scale=1):
            rec_out  = gr.HTML(label="Recommendations")

    test_out = gr.HTML(label="Test item")

    # Wire up
    user_dd.change(fn=update, inputs=user_dd, outputs=[hist_out, rec_out, test_out])

    # Seed the initial view
    demo.load(fn=lambda: update("User 42"), outputs=[hist_out, rec_out, test_out])


if __name__ == "__main__":
    demo.launch(
        share=False,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
        css="body{background:#0f0f0f} .gradio-container{max-width:1100px!important} #title{text-align:center;margin-bottom:0}",
    )
