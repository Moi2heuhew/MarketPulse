#!/usr/bin/env python3
"""
geo_engine — Contexte geopolitique auto-detete depuis la presse mondiale.
=========================================================================
Ce module lit de VRAIS flux d'actualite internationale (RSS, aucune cle API,
aucune dependance hors bibliotheque standard), detecte automatiquement les
evenements geopolitiques dominants et leurs canaux de transmission, et
construit un indice de climat que le score composite consomme directement.

L'utilisateur ne choisit plus l'evenement : l'application le lit dans le monde
reel. Si aucun flux ne repond, le contexte est "indisponible" et l'app retombe
sur ses ancres internes (VIX + vocabulaire des titres financiers) sans planter.
"""

import re
import urllib.request
import xml.etree.ElementTree as ET

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# --- Sources : presse internationale, regions et biais mixtes ---------------
FEEDS = [
    ("https://feeds.bbci.co.uk/news/world/rss.xml", "BBC World"),
    ("https://www.aljazeera.com/xml/rss/all.xml", "Al Jazeera"),
    ("https://www.euronews.com/rss?level=theme&name=news", "Euronews"),
    ("https://news.un.org/feed/subscribe/en/news/topic/peace-and-security/feed/rss.xml", "UN News"),
]

# --- Taxonomie des evenements (frontieres de mots uniquement) ---------------
EVENT_PATTERNS = {
    "energy_shock": [
        r"\boil price", r"\bbrent\b", r"\bopec\w*", r"\bgas suppl",
        r"\bembargo", r"\benergy crisis", r"\bpipeline", r"\bstrait of",
    ],
    "maritime_conflict": [
        r"\bblockade", r"\bstrait\b", r"\bshipping lane", r"\btanker",
        r"\bnaval\b", r"\bmaritime",
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
        r"\bcoup\b", r"\bstate of emergency", r"\bgovernment collapse",
        r"\bcivil unrest", r"\bcurfew",
    ],
}

# Correspondance geo_engine -> EVENEMENTS_GEO de app.py (cles FR)
EVENTS_FR = {
    "energy_shock": "Sanctions et embargo énergétique",
    "maritime_conflict": "Conflit militaire avec risque de blocus maritime",
    "trade_war": "Guerre commerciale et hausse des tarifs",
    "military_escalation": "Escalade nucléaire ou attaque directe",
    "political_instability": "Changement de régime politique",
}

# Canaux de transmission (memes cles que CANAUX dans app.py)
CHANNEL_PATTERNS = {
    "energie": [r"\boil\b", r"\bgas\b", r"\bbrent\b", r"\bfreight", r"\benergy",
                r"\bpipeline", r"\btanker", r"\bstrait"],
    "taux": [r"\binflation", r"\binterest rate", r"\bcentral bank", r"\byields?\b",
             r"\bbond market", r"\bcurrenc(y|ies)"],
    "chine": [r"\bchina\b", r"\bchinese\b", r"\btariff", r"\bexport", r"\bimport",
              r"\btrade", r"\bsemiconductor", r"\bchips?\b"],
    "defense": [r"\bmilitary", r"\bdefense (budget|spending|minister)", r"\bnato\b",
                r"\barms?\b", r"\bweapons"],
    "confiance": [r"\bwar\b", r"\bconflict", r"\bcrisis", r"\battack", r"\bsanction",
                  r"\bescalat", r"\buncertainty", r"\brecession"],
}

TENSION_RE = re.compile(
    r"\b(war|conflict|crisis|attack|strike|sanction|embargo|tariff|escalat"
    r"|invasion|missile|coup|blockade|nuclear|recession|emergency)\w*\b",
    re.I,
)

COMPILED_EVENTS = {k: [re.compile(p, re.I) for p in v] for k, v in EVENT_PATTERNS.items()}
COMPILED_CHANNELS = {k: [re.compile(p, re.I) for p in v] for k, v in CHANNEL_PATTERNS.items()}

MOTS_VIDES = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "as", "at", "by",
    "with", "from", "is", "are", "its", "it", "this", "that", "after", "over",
    "amid", "says", "said", "will", "has", "have", "be", "or", "new",
}


def _signature(titre: str) -> frozenset:
    """Empreinte d'un titre : mots significatifs les plus longs (dedup)."""
    mots = [m for m in re.findall(r"[a-z0-9]{4,}", str(titre).lower()) if m not in MOTS_VIDES]
    return frozenset(sorted(mots, key=len, reverse=True)[:6])


def _http_get(url: str, timeout: int = 12) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def fetch_geo_headlines(max_par_flux: int = 25) -> list:
    """Recupere les titres de la presse mondiale, deduplique, RSS via stdlib.

    Un flux mort ne casse jamais l'application : il est simplement ignore.
    """
    titres, signatures = [], []
    for url, nom in FEEDS:
        try:
            racine = ET.fromstring(_http_get(url))
        except Exception:
            continue
        for item in list(racine.iter("item"))[:max_par_flux]:
            titre = (item.findtext("title") or "").strip()
            if not titre:
                continue
            sig = _signature(titre)
            if sig and any(len(sig & s) / len(sig | s) >= 0.6 for s in signatures if s):
                continue
            if sig:
                signatures.append(sig)
            titres.append({"headline": titre, "source": nom})
    return titres


def _classifier(headline: str):
    """Retourne (cle_evenement ou None, ensemble_de_canaux) pour un titre."""
    evenement = None
    for cle, motifs in COMPILED_EVENTS.items():
        if any(m.search(headline) for m in motifs):
            evenement = cle
            break
    canaux = {c for c, motifs in COMPILED_CHANNELS.items()
              if any(m.search(headline) for m in motifs)}
    return evenement, canaux


def construire_contexte(titres: list, scores: list) -> dict:
    """Assemble le contexte geopolitique a partir des titres et de FinBERT.

    scores : liste de scores de sentiment [-1, +1], un par titre (FinBERT).
    L'intensite d'un evenement est une heuristique explicite : negativite
    moyenne de sa couverture + volume d'articles, plafonnee a 1.
    """
    if not titres:
        return {"index": None, "events": [], "channels": {},
                "exemples_negatifs": [], "n_articles": 0,
                "status": "flux indisponibles — retombée sur VIX et vocabulaire local"}

    for t, s in zip(titres, scores):
        t["score"] = float(s)

    n = len(titres)
    negatifs = [t for t in titres if t["score"] < 0]
    tension = [t for t in titres if TENSION_RE.search(t["headline"])]
    part_neg = len(negatifs) / n
    part_tension = len(tension) / n
    sent_moyen = sum(t["score"] for t in titres) / n
    index = round(max(-1.0, min(1.0, -(part_neg * 0.5 + part_tension * 0.5) + sent_moyen * 0.5)), 3)

    # Evenements dominants, comptes et classes
    compte, sent_ev, exemples = {}, {}, {}
    canaux_scores = {}
    for t in titres:
        ev, canaux = _classifier(t["headline"])
        if ev:
            compte[ev] = compte.get(ev, 0) + 1
            sent_ev.setdefault(ev, []).append(t["score"])
            exemples.setdefault(ev, [])
            if len(exemples[ev]) < 3:
                exemples[ev].append((t["headline"], t["source"]))
        for c in canaux:
            canaux_scores.setdefault(c, []).append(t["score"])

    events = []
    for ev, nb in compte.items():
        sm = sum(sent_ev[ev]) / len(sent_ev[ev])
        # Heuristique d'intensite, affichee comme telle : negativite + volume
        intensite = round(min(1.0, max(0.0, -sm + nb / 15.0)), 2)
        events.append({
            "evenement": ev,
            "evenement_fr": EVENTS_FR.get(ev, ev),
            "articles": nb,
            "sentiment_moyen": round(sm, 3),
            "intensite": intensite,
            "exemples": exemples[ev],
        })
    events.sort(key=lambda e: -e["articles"])

    # Intensite par canal : negativite de la couverture touchant chaque canal
    channels = {
        c: round(min(1.0, max(0.0, 0.5 - (sum(s) / len(s)) * 0.5)), 3)
        for c, s in canaux_scores.items()
    }

    return {
        "index": index,
        "events": events,
        "channels": channels,
        "exemples_negatifs": [t["headline"] for t in negatifs[:3]],
        "n_articles": n,
        "status": f"en direct — {n} titres de la presse mondiale analysés",
    }
