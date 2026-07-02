"""
FloodSense-PK design system.

One central place for the dashboard's visual layer: global CSS (fonts, cards,
tabs, sidebar, inputs, chat), reusable HTML components (hero banner, metric
cards, callouts) and the shared Plotly template. Import and call
``inject_css()`` + ``setup_plotly_theme()`` once at app start; everything else
is pure helpers that return/render styled HTML.

Nothing in here touches data flow — it is UI only.
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── Design tokens ────────────────────────────────────────────────────────────
PALETTE = {
    "bg": "#0B0F19",
    "surface": "#131A2A",
    "surface_2": "#1A2336",
    "border": "rgba(255,255,255,0.08)",
    "text": "#E5EAF5",
    "muted": "#8A94AD",
    "primary": "#3B82F6",
    "cyan": "#22D3EE",
    "danger": "#EF4444",
    "warning": "#F59E0B",
    "success": "#10B981",
}

GRADIENT = f"linear-gradient(120deg, {PALETTE['primary']} 0%, {PALETTE['cyan']} 100%)"


# ── Global CSS ───────────────────────────────────────────────────────────────
def inject_css() -> None:
    """Inject the full design system. Call exactly once, at app start."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        /* ── Base ── */
        html, body, [data-testid="stAppViewContainer"] {{
            background: {PALETTE['bg']};
            font-family: 'Inter', sans-serif;
            color: {PALETTE['text']};
        }}
        h1, h2, h3, h4, h5 {{
            font-family: 'Space Grotesk', 'Inter', sans-serif !important;
            letter-spacing: -0.02em;
            color: {PALETTE['text']};
        }}
        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1600px;
        }}
        [data-testid="stHeader"] {{ background: transparent; }}

        /* ── Uppercase micro-label ── */
        .micro-label {{
            font-size: 11px;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: {PALETTE['muted']};
            font-weight: 600;
        }}

        /* ── Hero banner ── */
        .fs-hero {{
            border-radius: 16px;
            padding: 1.6rem 2rem;
            margin-bottom: 1.4rem;
            background:
                linear-gradient(rgba(11,15,25,0.72), rgba(11,15,25,0.72)),
                {GRADIENT};
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 18px 40px rgba(0,0,0,0.35);
        }}
        .fs-hero .fs-hero-top {{
            display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
        }}
        .fs-hero h1 {{
            margin: 0; font-size: 2.2rem; font-weight: 700; line-height: 1.1;
            background: {GRADIENT};
            -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .fs-hero .fs-sub {{
            margin: 0.45rem 0 0 0; color: #B9C4DE; font-size: 1rem;
        }}
        .fs-pill {{
            display: inline-flex; align-items: center; gap: 7px;
            font-size: 12px; font-weight: 600; letter-spacing: 0.6px;
            padding: 5px 14px; border-radius: 999px;
            background: rgba(16,185,129,0.12); color: {PALETTE['success']};
            border: 1px solid rgba(16,185,129,0.35);
        }}
        .fs-pill .dot {{
            width: 8px; height: 8px; border-radius: 50%;
            background: {PALETTE['success']};
            animation: fs-pulse 2s infinite;
        }}
        @keyframes fs-pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }}
            70% {{ box-shadow: 0 0 0 8px rgba(16,185,129,0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(16,185,129,0); }}
        }}
        .fs-badge {{
            display: inline-block; margin-top: 0.8rem; margin-right: 0.45rem;
            font-size: 11.5px; font-weight: 500;
            padding: 4px 12px; border-radius: 8px;
            background: rgba(255,255,255,0.06); color: #C6D0E8;
            border: 1px solid {PALETTE['border']};
        }}

        /* ── Tabs → pill segmented control ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            background: {PALETTE['surface']};
            border: 1px solid {PALETTE['border']};
            border-radius: 12px;
            padding: 5px;
            width: fit-content;
            max-width: 100%;
            overflow-x: auto;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 7px 18px;
            background: transparent;
            border: none;
            color: {PALETTE['muted']};
            font-weight: 500;
            font-size: 0.9rem;
            transition: background 0.15s ease, color 0.15s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background: rgba(255,255,255,0.05);
            color: {PALETTE['text']};
        }}
        .stTabs [aria-selected="true"] {{
            background: rgba(59,130,246,0.16) !important;
            color: #DBEAFE !important;
        }}
        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {{ display: none; }}

        /* ── Cards / metric cards ── */
        .fs-card {{
            background: {PALETTE['surface']};
            border: 1px solid {PALETTE['border']};
            border-radius: 14px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.22);
            height: 100%;
        }}
        .fs-metric-row {{
            display: flex; gap: 14px; flex-wrap: wrap; margin: 0.4rem 0 0.8rem 0;
        }}
        .fs-metric {{
            flex: 1 1 180px;
            background: {PALETTE['surface']};
            border: 1px solid {PALETTE['border']};
            border-radius: 14px;
            padding: 1rem 1.2rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.22);
            position: relative;
            overflow: hidden;
        }}
        .fs-metric::before {{
            content: ''; position: absolute; inset: 0 auto 0 0; width: 3px;
            background: var(--fs-accent, {PALETTE['primary']});
            border-radius: 3px 0 0 3px;
        }}
        .fs-metric .fs-m-label {{
            font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase;
            color: {PALETTE['muted']}; font-weight: 600;
            display: flex; align-items: center; gap: 7px;
        }}
        .fs-metric .fs-m-value {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.85rem; font-weight: 700; color: {PALETTE['text']};
            margin-top: 6px; line-height: 1.15;
        }}
        .fs-metric .fs-m-sub {{ font-size: 12px; color: {PALETTE['muted']}; margin-top: 3px; }}
        .fs-chip {{
            display: inline-block; margin-top: 8px;
            font-size: 12px; font-weight: 600;
            padding: 2px 10px; border-radius: 999px;
        }}
        .fs-chip.good {{
            background: rgba(16,185,129,0.13); color: {PALETTE['success']};
            border: 1px solid rgba(16,185,129,0.35);
        }}
        .fs-chip.bad {{
            background: rgba(239,68,68,0.13); color: {PALETTE['danger']};
            border: 1px solid rgba(239,68,68,0.35);
        }}
        .fs-chip.neutral {{
            background: rgba(255,255,255,0.06); color: {PALETTE['muted']};
            border: 1px solid {PALETTE['border']};
        }}

        /* ── Callouts ── */
        .fs-callout {{
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            margin: 0.5rem 0;
            font-size: 0.9rem;
            line-height: 1.55;
            border: 1px solid {PALETTE['border']};
            background: {PALETTE['surface']};
        }}
        .fs-callout .fs-c-title {{
            font-weight: 700; font-size: 0.82rem; letter-spacing: 0.8px;
            text-transform: uppercase; display: block; margin-bottom: 4px;
        }}
        .fs-callout.info    {{ border-left: 4px solid {PALETTE['primary']}; }}
        .fs-callout.info .fs-c-title    {{ color: {PALETTE['primary']}; }}
        .fs-callout.warning {{ border-left: 4px solid {PALETTE['warning']}; }}
        .fs-callout.warning .fs-c-title {{ color: {PALETTE['warning']}; }}
        .fs-callout.danger  {{ border-left: 4px solid {PALETTE['danger']}; }}
        .fs-callout.danger .fs-c-title  {{ color: {PALETTE['danger']}; }}
        .fs-callout.success {{ border-left: 4px solid {PALETTE['success']}; }}
        .fs-callout.success .fs-c-title {{ color: {PALETTE['success']}; }}

        /* ── Images: card treatment + caption bar ── */
        [data-testid="stImage"] {{
            background: {PALETTE['surface']};
            border: 1px solid {PALETTE['border']};
            border-radius: 14px;
            padding: 8px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.22);
        }}
        [data-testid="stImage"] img {{ border-radius: 8px; }}
        [data-testid="stImage"] [data-testid="stImageCaption"] {{
            background: {PALETTE['surface_2']};
            border-radius: 0 0 8px 8px;
            padding: 6px 10px;
            margin-top: 4px;
            color: {PALETTE['muted']};
            font-size: 12px;
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background: {PALETTE['surface']};
            border-right: 1px solid {PALETTE['border']};
        }}
        [data-testid="stSidebar"] .stMarkdown h3 {{
            font-size: 0.82rem; letter-spacing: 1.2px; text-transform: uppercase;
            color: {PALETTE['muted']};
        }}
        [data-testid="stSidebar"] hr {{ border-color: {PALETTE['border']}; }}

        /* ── Inputs ── */
        [data-testid="stTextInput"] input,
        [data-testid="stDateInput"] input,
        [data-testid="stSelectbox"] > div > div {{
            background: {PALETTE['surface_2']} !important;
            border: 1px solid {PALETTE['border']} !important;
            border-radius: 10px !important;
            color: {PALETTE['text']} !important;
        }}

        /* ── Buttons ── */
        .stButton > button {{
            border-radius: 10px;
            border: 1px solid {PALETTE['border']};
            background: {PALETTE['surface_2']};
            color: {PALETTE['text']};
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }}
        .stButton > button:hover {{
            border-color: rgba(59,130,246,0.5);
            color: #DBEAFE;
        }}
        .stButton > button[kind="primary"] {{
            background: {GRADIENT};
            border: none;
            font-weight: 700;
            letter-spacing: 0.4px;
            box-shadow: 0 8px 20px rgba(59,130,246,0.28);
        }}
        .stButton > button[kind="primary"]:hover {{
            transform: translateY(-1px);
            box-shadow: 0 12px 26px rgba(59,130,246,0.4);
        }}

        /* ── Native st.metric (used where cards aren't applied yet) ── */
        div[data-testid="stMetric"] {{
            background: {PALETTE['surface']};
            border: 1px solid {PALETTE['border']};
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.22);
        }}
        div[data-testid="stMetric"] label {{
            font-size: 11px !important; letter-spacing: 1.5px;
            text-transform: uppercase; color: {PALETTE['muted']} !important;
        }}

        /* ── Expander / dataframe / status containers ── */
        [data-testid="stExpander"] {{
            background: {PALETTE['surface']};
            border: 1px solid {PALETTE['border']};
            border-radius: 12px;
        }}

        /* ── Agent pipeline stepper ── */
        .agent-flow {{
            display: flex; align-items: stretch; gap: 0;
            margin: 0.5rem 0 1.25rem 0; flex-wrap: wrap;
        }}
        .agent-card {{
            flex: 1 1 0; min-width: 150px;
            background: {PALETTE['surface']};
            border: 1px solid {PALETTE['border']};
            border-radius: 14px;
            padding: 0.95rem 0.85rem;
            text-align: center; position: relative;
            box-shadow: 0 8px 24px rgba(0,0,0,0.22);
        }}
        .agent-card .agent-step {{
            display: inline-block;
            font-size: 10.5px; font-weight: 700; letter-spacing: 1.6px;
            color: #93C5FD;
            background: rgba(59,130,246,0.14);
            border: 1px solid rgba(59,130,246,0.3);
            padding: 2px 10px; border-radius: 999px;
        }}
        .agent-card .agent-name {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1rem; font-weight: 700; color: {PALETTE['text']};
            margin-top: 0.5rem;
        }}
        .agent-card .agent-role {{ font-size: 0.78rem; color: {PALETTE['muted']}; margin-top: 0.2rem; }}
        .agent-card .agent-status {{
            display: inline-flex; align-items: center; gap: 5px;
            margin-top: 0.55rem;
            font-size: 0.68rem; font-weight: 700; letter-spacing: 1px;
            padding: 3px 12px; border-radius: 999px;
            background: rgba(16,185,129,0.13); color: {PALETTE['success']};
            border: 1px solid rgba(16,185,129,0.35);
        }}
        .agent-card .agent-status::before {{ content: '✓'; font-weight: 900; }}
        .agent-arrow {{
            display: flex; align-items: center; justify-content: center;
            font-size: 1.15rem; color: {PALETTE['primary']}; padding: 0 7px;
        }}
        @media (max-width: 768px) {{
            .agent-flow {{ flex-direction: column; }}
            .agent-arrow {{ transform: rotate(90deg); padding: 2px 0; }}
            .fs-hero h1 {{ font-size: 1.6rem; }}
        }}

        /* ── Fused input stream chips ── */
        .stream-chip {{
            background: {PALETTE['surface']};
            border: 1px solid {PALETTE['border']};
            border-radius: 12px;
            padding: 0.75rem 0.95rem;
            height: 100%;
        }}
        .stream-chip .chip-label {{
            font-size: 11px; letter-spacing: 1.5px; color: {PALETTE['muted']};
            text-transform: uppercase; font-weight: 600;
        }}
        .stream-chip .chip-value {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.15rem; font-weight: 700; color: {PALETTE['text']}; margin-top: 2px;
        }}
        .stream-chip .chip-sub {{ font-size: 0.78rem; color: #93C5FD; margin-top: 2px; }}

        /* ── Chat (Knowledge Assistant) ── */
        .chat-hero {{
            padding: 1.15rem 1.4rem;
            border-radius: 16px;
            background:
                linear-gradient(rgba(11,15,25,0.78), rgba(11,15,25,0.78)),
                {GRADIENT};
            border: 1px solid rgba(255,255,255,0.12);
            margin-bottom: 1rem;
        }}
        .chat-hero .chat-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.3rem; font-weight: 700; color: {PALETTE['text']}; margin: 0;
        }}
        .chat-hero .chat-sub {{ font-size: 0.86rem; color: #B9C4DE; margin: 0.3rem 0 0 0; }}
        .kb-badge {{
            display: inline-block; margin-top: 0.6rem; margin-right: 0.4rem;
            font-size: 11.5px; font-weight: 600;
            padding: 3px 12px; border-radius: 999px;
            background: rgba(16,185,129,0.12); color: {PALETTE['success']};
            border: 1px solid rgba(16,185,129,0.35);
        }}
        .kb-badge.dim {{
            background: rgba(255,255,255,0.06); color: {PALETTE['muted']};
            border-color: {PALETTE['border']};
        }}
        [data-testid="stChatMessage"] {{
            background: {PALETTE['surface']};
            border: 1px solid {PALETTE['border']};
            border-radius: 14px;
            padding: 0.6rem 0.9rem;
        }}
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
            background: rgba(59,130,246,0.10);
            border-color: rgba(59,130,246,0.28);
        }}
        [data-testid="stChatInput"] {{
            border-radius: 12px;
            border: 1px solid {PALETTE['border']};
        }}

        /* ── Alert preview boxes (Response Agent) ── */
        .alert-box {{
            border-radius: 12px; padding: 1rem 1.1rem; height: 100%;
            border: 1px solid {PALETTE['border']};
            background: {PALETTE['surface']};
            font-size: 0.9rem; line-height: 1.6; color: {PALETTE['text']};
            white-space: pre-wrap;
            font-family: 'Inter', sans-serif;
        }}
        .alert-box.citizen   {{ border-left: 5px solid {PALETTE['warning']}; }}
        .alert-box.authority {{ border-left: 5px solid {PALETTE['primary']}; }}
        .alert-box .alert-tag {{
            font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
            text-transform: uppercase; color: {PALETTE['muted']}; display: block;
            margin-bottom: 0.45rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Plotly shared template ───────────────────────────────────────────────────
def setup_plotly_theme() -> None:
    """Register + activate the shared dark Plotly template for every chart."""
    pio.templates["floodsense"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=PALETTE["surface"],
            font={"family": "Inter, sans-serif", "color": PALETTE["text"], "size": 13},
            colorway=[
                PALETTE["primary"],
                PALETTE["cyan"],
                PALETTE["success"],
                PALETTE["warning"],
                PALETTE["danger"],
            ],
            xaxis={
                "gridcolor": "rgba(255,255,255,0.06)",
                "zerolinecolor": "rgba(255,255,255,0.10)",
            },
            yaxis={
                "gridcolor": "rgba(255,255,255,0.06)",
                "zerolinecolor": "rgba(255,255,255,0.10)",
            },
            margin={"l": 10, "r": 10, "t": 48, "b": 10},
            hoverlabel={"bgcolor": PALETTE["surface_2"], "font": {"family": "Inter"}},
        )
    )
    pio.templates.default = "floodsense"


# ── Components ───────────────────────────────────────────────────────────────
def hero(title: str, subtitle: str, status_text: str, badges: list[str]) -> None:
    """Render the gradient hero banner with live pill and tech badges."""
    badge_html = "".join(f"<span class='fs-badge'>{b}</span>" for b in badges)
    st.markdown(
        "<div class='fs-hero'>"
        "<div class='fs-hero-top'>"
        f"<h1>{title}</h1>"
        f"<span class='fs-pill'><span class='dot'></span>{status_text}</span>"
        "</div>"
        f"<p class='fs-sub'>{subtitle}</p>"
        f"{badge_html}"
        "</div>",
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: str,
    sub: str = "",
    trend: str = "",
    trend_dir: str = "neutral",
    accent: str = "",
) -> str:
    """Return one styled metric card. ``trend_dir``: good | bad | neutral."""
    accent = accent or PALETTE["primary"]
    chip = (
        f"<span class='fs-chip {trend_dir}'>{trend}</span>" if trend else ""
    )
    sub_html = f"<div class='fs-m-sub'>{sub}</div>" if sub else ""
    return (
        f"<div class='fs-metric' style='--fs-accent:{accent}'>"
        f"<div class='fs-m-label'>{label}</div>"
        f"<div class='fs-m-value'>{value}</div>"
        f"{sub_html}{chip}"
        "</div>"
    )


def metric_row(cards: list[str]) -> None:
    """Render a responsive flex row of ``metric_card`` HTML strings."""
    st.markdown(
        f"<div class='fs-metric-row'>{''.join(cards)}</div>",
        unsafe_allow_html=True,
    )


def callout(body: str, kind: str = "info", title: str = "") -> None:
    """Custom callout (kind: info | warning | danger | success)."""
    title_html = f"<span class='fs-c-title'>{title}</span>" if title else ""
    st.markdown(
        f"<div class='fs-callout {kind}'>{title_html}{body}</div>",
        unsafe_allow_html=True,
    )


def risk_gauge(score: float, max_value: float = 10.0, title: str = "Composite Risk") -> go.Figure:
    """Dark-themed Plotly gauge with green/amber/red zones and centered number."""
    color = (
        PALETTE["danger"] if score > 7 else
        PALETTE["warning"] if score > 4 else
        PALETTE["success"]
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={
                "suffix": f" / {max_value:g}",
                "font": {"size": 40, "color": PALETTE["text"], "family": "Space Grotesk"},
            },
            title={
                "text": f"<span style='font-size:0.8em;color:{PALETTE['muted']};"
                f"letter-spacing:1.5px'>{title.upper()}</span>",
            },
            gauge={
                "axis": {
                    "range": [0, max_value],
                    "tickcolor": PALETTE["muted"],
                    "tickfont": {"size": 11},
                },
                "bar": {"color": color, "thickness": 0.30},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 4], "color": "rgba(16,185,129,0.16)"},
                    {"range": [4, 7], "color": "rgba(245,158,11,0.16)"},
                    {"range": [7, max_value], "color": "rgba(239,68,68,0.16)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.85,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin={"l": 30, "r": 30, "t": 40, "b": 6},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
