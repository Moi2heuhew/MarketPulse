#!/usr/bin/env python3
"""
geo_engine — Real-time geopolitical context, auto-detected from live news.
===========================================================================
Instead of the user choosing a geopolitical scenario by hand, this module
reads REAL world-news feeds (RSS, free, no API key), runs FinBERT on the
headlines, detects the dominant events and transmission channels, and
returns a geopolitical climate index the composite score can consume.

Usage (inside app.py):
    from geo_engine import compute_geo_context
    geo = compute_geo_context(nlp)          # nlp = FinBERT pipeline
    geo["index"]        # climate in [-1, +1]  (-1 = high tension)
    geo["events"]       # auto-detected events, ranked
    geo["channels"]     # transmission channel intensities in [0, 1]
    geo["headlines"]    # the real headlines it read (auditable)

Install:  pip install feedparser requests
"""

import re
from collections import defaultdict

import feedparser
import requests

# ---------------------------------------------------------------------------
# Real news sources (RSS, free, no key). Mixed regions for balance.
# ---------------------------------------------------------------------------
FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://www.euronews.com/rss?level=theme&name=news",
    "https://news.un.org/feed/subscribe/en/news/topic/peace-and-security/feed/rss.xml",
]

# ---------------------------------------------------------------------------
# Taxonomies. Word-boundary regex only (avoids "ai" inside "said", etc.)
# ---------------------------------------------------------------------------
EVENT_TAXONOMY = {
    "energy_shock": [
        r"\boil price", r"\bbrent\b", r"\bopec\w*", r"\bgas suppl", r"\bembargo",
        r"\benergy crisis", r"\bpipeline", r"\bstrait of", r"\bsanction\w* on (oil|gas|energy)",
    ],
    "maritime_conflict": [
        r"\bblockade\b", r"\bstrait\b", r"\bshipping lane", r"\btanker",
        r"\bport strike", r"\bnaval\b", r"\bmaritime",
    ],
    "trade_war": [
        r"\btariff\w*", r"\btrade war", r"\bexport (control|ban|restriction)",
        r"\bimport ban", r"\bcustoms dut", r"\bchips? (ban|restriction|export)",
    ],
    "military_escalation": [
        r"\bmissile", r"\bairstrike", r"\bmilitary strike", r"\btroops?\b",
        r"\binvasion", r"\bescalation", r"\bnuclear", r"\bdrone attack",
    ],
    "political_instability": [
        r"\bcoup\b", r"\bcovert? d'etat", r"\bstate of emergency",
        r"\bgovernment collapse", r"\bcivil unrest", r"\bprotest\w* turn",
    ],
}

# Transmission channels (used to weight sector impact downstream)
CHANNEL_TAXONOMY = {
    "energy_freight": [r"\boil\b", r"\bgas\b", r"\bbrent\b", r"\bfreight", r"\bshipping cost",
                       r"\benergy", r"\bpipeline", r"\btanker", r"\bstrait"],
    "rates_inflation": [r"\binflation", r"\binterest rate", r"\bcentral bank", r"\byields?",
                        r"\bbond market", r"\bcurrenc(y|ies)"],
    "china_trade": [r"\bchina\b", r"\bchinese\b", r"\btariff", r"\bexport", r"\bimport",
                    r"\btrade", r"\bsemiconductor", r"\bchips?\b"],
    "defense_budget": [r"\bmilitary", r"\bdefense (budget|spending|minister)", r"\bnato\b",
                       r"\barms?", r"\bweapons"],
    "risk_appetite": [r"\bwar\b", r"\bconflict", r"\bcrisis", r"\battack", r"\bsanction",
                      r"\bescalat", r"\buncertainty", r"\brecession"],
}

# Global tension vocabulary (drives the climate index)
_TENSION_WORDS = [
    "war", "conflict", "crisis", "attack", "strike", "sanction",
    "embargo", "tariff", "escalat", "invasion", "missile", "coup",
    "blockade", "nuclear", "recession", "emergency",
]
TENSION_RE = re.compile("|".join(r"\b" + w for w in _TENSION_WORDS), re.I)

COMPILED_EVENTS = {k: [re.compile(p, re.I) for p in v] for k, v in EVENT_TAXONOMY.items()}
COMPILED_CHANNELS = {k: [re.compile(p, re.I) for p in v] for k, v in CHANNEL_TAXONOMY.items()}
WORD_RE = re.compile(r"[a-z']+")


def _tokens(text: str) -> set:
    return {t for t in WORD_RE.findall(text.lower()) if len(t) > 3}


def _near_duplicate(a: set, b: set, threshold: float = 0.6) -> bool:
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= threshold


def fetch_geo_headlines(max_per_feed: int = 25, timeout: int = 12) -> list:
    """Fetch real world news headlines from the RSS feeds, deduplicated."""
    headlines, seen = [], []
    for url in FEEDS:
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "MarketPulse/1.0"})
            feed = feedparser.parse(resp.content)
        except Exception:
            continue  # one dead feed must never break the app
        for entry in feed.entries[:max_per_feed]:
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            tok = _tokens(title)
            if any(_near_duplicate(tok, s) for s in seen):
                continue
            seen.append(tok)
            headlines.append(
                {"headline": title, "source": feed.feed.get("title", url), "link": entry.get("link", "")}
            )
    return headlines


def classify(headline: str) -> tuple:
    """Return (event_type_or_None, set_of_channels) for one headline."""
    event = None
    for ev, patterns in COMPILED_EVENTS.items():
        if any(p.search(headline) for p in patterns):
            event = ev
            break
    channels = {ch for ch, pats in COMPILED_CHANNELS.items()
                 if any(p.search(headline) for p in pats)}
    return event, channels


def compute_geo_context(nlp, max_per_feed: int = 25) -> dict:
    """Full auto-detected geopolitical context, ready for the composite score."""
    headlines = fetch_geo_headlines(max_per_feed)
    if not headlines:
        # Total failure: neutral, never crash the decision layer
        return {"index": 0.0, "events": [], "channels": {},
                "headlines": [], "status": "unavailable — neutral climate applied"}

    # FinBERT sentiment on every geopolitical headline
    for h in headlines:
        res = nlp(h["headline"][:512])[0]
        h["label"] = res["label"].lower()
        h["score"] = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}[h["label"]]

    n = len(headlines)
    neg = [h for h in headlines if h["score"] < 0]
    tension_hits = [h for h in headlines if TENSION_RE.search(h["headline"])]

    # Climate index: negative-sentiment pressure weighted by tension share.
    # Range [-1, +1]; scaled so a calm day sits near 0.
    neg_share = len(neg) / n
    tension_share = len(tension_hits) / n
    avg_sent = sum(h["score"] for h in headlines) / n
    index = round(max(-1.0, min(1.0, -(neg_share * 0.5 + tension_share * 0.5) + avg_sent * 0.5)), 3)

    # Event detection: count + average sentiment per event type
    ev_count, ev_sent, ev_examples = defaultdict(int), defaultdict(list), defaultdict(list)
    ch_intensity = defaultdict(list)
    for h in headlines:
        event, channels = classify(h["headline"])
        if event:
            ev_count[event] += 1
            ev_sent[event].append(h["score"])
            if len(ev_examples[event]) < 3:
                ev_examples[event].append(h)
        for ch in channels:
            ch_intensity[ch].append(h["score"])

    events = sorted(
        (
            {
                "event": ev,
                "articles": c,
                "avg_sentiment": round(sum(ev_sent[ev]) / len(ev_sent[ev]), 3),
                "examples": ev_examples[ev],
            }
            for ev, c in ev_count.items()
        ),
        key=lambda e: -e["articles"],
    )

    # Channel intensity: how negative the coverage touching each channel is,
    # scaled 0-1. These replace the manual sliders in the composite score.
    channels = {
        ch: round(min(1.0, max(0.0, 0.5 - (sum(scores) / len(scores)) * 0.5)), 3)
        for ch, scores in ch_intensity.items()
    }

    return {
        "index": index,                        # feed the 10% macro weight
        "events": events,                       # show as expanders in the UI
        "channels": channels,                   # per-sector transmission impact
        "headlines": headlines,                 # audit trail
        "n_articles": n,
        "status": "live — %d headlines analyzed" % n,
    }


# ---------------------------------------------------------------------------
# Streamlit integration (drop into app.py)
# ---------------------------------------------------------------------------
# from geo_engine import compute_geo_context
# import streamlit as st
#
# @st.cache_data(ttl=3600, show_spinner=False)
# def get_geo_context():
#     return compute_geo_context(load_model())
#
# geo = get_geo_context()
#
# # 1. Replace the manual event dropdown with auto-detection:
# st.metric("Geopolitical climate (auto)", f"{geo['index']:+.2f}")
# for ev in geo["events"]:
#     with st.expander(f"{ev['event'].replace('_', ' ').title()} — {ev['articles']} articles"):
#         st.markdown(f"Average sentiment: **{ev['avg_sentiment']:+.2f}**")
#         for ex in ev["examples"]:
#             st.markdown(f"> {ex['headline']} — *{ex['source']}*")
#
# # 2. Feed the composite score:
# macro_weight = 0.10 * geo["index"]          # instead of manual VIX+slider
# # and use geo["channels"]["energy_freight"] etc. to modulate sector impact
# ---------------------------------------------------------------------------
