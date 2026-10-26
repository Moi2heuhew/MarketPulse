#!/usr/bin/env python3
"""
MarketPulse Web — radar de sentiment de marche en direct
=========================================================
Entrez une liste de tickers : l'application recupere les VRAIS titres de presse
financiere (Yahoo Finance, avec Yahoo RSS puis Google News en secours), les
analyse avec FinBERT et affiche un classement haussier / baissier.

Lancement :
    streamlit run app.py

Dependances :
    pip install streamlit yfinance transformers torch pandas

Citations :
    - Modele : ProsusAI/finbert (Hugging Face, Apache-2.0)
    - yfinance, transformers, streamlit (Apache-2.0)
"""

from __future__ import annotations

import datetime as dt
import email.utils
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd
import streamlit as st

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
MAX_HEADLINES = 25
SEUIL_FORT = 0.35
SEUIL_LEGER = 0.12

st.set_page_config(page_title="MarketPulse", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# 1. Normalisation de la saisie
# ---------------------------------------------------------------------------
def parse_tickers(saisie: str) -> list[str]:
    """Transforme "aapl, msft  nvda" en ["AAPL", "MSFT", "NVDA"]."""
    brut = re.split(r"[,\s;|]+", (saisie or "").strip())
    vus: list[str] = []
    for jeton in brut:
        t = jeton.strip().upper()
        if not t:
            continue
        t = t.replace("$", "")
        if not re.fullmatch(r"[\^A-Z0-9][A-Z0-9.\-=]{0,11}", t):
            continue
        if t not in vus:
            vus.append(t)
    return vus[:15]


# ---------------------------------------------------------------------------
# 2. Recuperation des titres de presse (3 sources en cascade)
# ---------------------------------------------------------------------------
def _http_get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def _to_date(valeur) -> dt.date | None:
    """Accepte un timestamp epoch, une chaine ISO ou une date RFC 822."""
    if valeur is None:
        return None
    try:
        if isinstance(valeur, (int, float)):
            return dt.datetime.fromtimestamp(float(valeur), tz=dt.timezone.utc).date()
        if isinstance(valeur, dt.datetime):
            return valeur.date()
        if isinstance(valeur, dt.date):
            return valeur
        texte = str(valeur).strip()
        if not texte:
            return None
        if re.fullmatch(r"\d{9,11}", texte):
            return dt.datetime.fromtimestamp(int(texte), tz=dt.timezone.utc).date()
        try:
            return dt.datetime.fromisoformat(texte.replace("Z", "+00:00")).date()
        except ValueError:
            pass
        parsed = email.utils.parsedate_to_datetime(texte)
        return parsed.date() if parsed else None
    except Exception:
        return None


def _parse_news_item(item) -> dict | None:
    """Gere l'ANCIEN format Yahoo (title, providerPublishTime) ET le NOUVEAU
    format (item["content"]["title"], item["content"]["pubDate"]...)."""
    if not isinstance(item, dict):
        return None
    contenu = item.get("content") if isinstance(item.get("content"), dict) else {}
    titre = contenu.get("title") or item.get("title")
    if not titre:
        return None

    editeur = ""
    prov = contenu.get("provider")
    if isinstance(prov, dict):
        editeur = prov.get("displayName") or ""
    if not editeur:
        editeur = item.get("publisher") or ""

    lien = ""
    canon = contenu.get("canonicalUrl")
    if isinstance(canon, dict):
        lien = canon.get("url") or ""
    if not lien:
        lien = item.get("link") or ""

    date = _to_date(
        contenu.get("pubDate")
        or contenu.get("displayTime")
        or item.get("providerPublishTime")
        or item.get("pubDate")
    )
    return {"date": date, "headline": str(titre).strip(), "editeur": editeur, "lien": lien}


def _news_via_yfinance(ticker: str, erreurs: list[str]) -> list[dict]:
    items: list[dict] = []
    try:
        import yfinance as yf
    except Exception as e:  # pragma: no cover
        erreurs.append(f"yfinance non installe ({e.__class__.__name__})")
        return items

    brut = []
    for methode in ("ticker", "search"):
        try:
            if methode == "ticker":
                brut = list(yf.Ticker(ticker).news or [])
            else:
                brut = list(getattr(yf.Search(ticker), "news", []) or [])
        except Exception as e:
            erreurs.append(f"yfinance/{methode} : {e.__class__.__name__}: {e}")
            brut = []
        parsed = [p for p in (_parse_news_item(x) for x in brut) if p]
        if parsed:
            return parsed
    if not brut:
        erreurs.append("yfinance : liste de news vide (limite de debit ou ticker inconnu)")
    else:
        erreurs.append(f"yfinance : {len(brut)} element(s) recu(s) mais aucun titre exploitable")
    return items


def _news_via_rss(url: str, ticker: str, erreurs: list[str], nom: str) -> list[dict]:
    try:
        racine = ET.fromstring(_http_get(url))
    except Exception as e:
        erreurs.append(f"{nom} : {e.__class__.__name__}: {e}")
        return []
    out: list[dict] = []
    for item in racine.iter("item"):
        titre = (item.findtext("title") or "").strip()
        if not titre:
            continue
        out.append(
            {
                "date": _to_date(item.findtext("pubDate")),
                "headline": titre,
                "editeur": (item.findtext("source") or nom).strip(),
                "lien": (item.findtext("link") or "").strip(),
            }
        )
    if not out:
        erreurs.append(f"{nom} : flux vide pour {ticker}")
    return out


@st.cache_data(ttl=900, show_spinner=False)
def fetch_headlines(ticker: str) -> dict:
    """Renvoie {'ticker', 'titres', 'source', 'erreurs'} pour un ticker."""
    erreurs: list[str] = []

    titres = _news_via_yfinance(ticker, erreurs)
    source = "Yahoo Finance (yfinance)"

    if not titres:
        url_yahoo = (
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s="
            + urllib.parse.quote(ticker)
            + "&region=US&lang=en-US"
        )
        titres = _news_via_rss(url_yahoo, ticker, erreurs, "Yahoo RSS")
        source = "Yahoo Finance (RSS)"

    if not titres:
        requete = urllib.parse.quote(f'"{ticker}" stock OR shares OR results')
        url_google = (
            f"https://news.google.com/rss/search?q={requete}&hl=en-US&gl=US&ceid=US:en"
        )
        titres = _news_via_rss(url_google, ticker, erreurs, "Google News RSS")
        source = "Google News (RSS)"

    vus, uniques = set(), []
    for t in titres:
        cle = t["headline"].lower()
        if cle in vus:
            continue
        vus.add(cle)
        uniques.append(t)

    uniques.sort(key=lambda t: (t["date"] or dt.date(1970, 1, 1)), reverse=True)
    return {
        "ticker": ticker,
        "titres": uniques[:MAX_HEADLINES],
        "source": source if uniques else "aucune",
        "erreurs": erreurs,
    }


# ---------------------------------------------------------------------------
# 3. Analyse FinBERT
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def charger_modele():
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        top_k=None,
    )


def scorer(titres: list[str]) -> list[dict]:
    clf = charger_modele()
    sorties = clf(titres, truncation=True, batch_size=16)
    resultats = []
    for sortie in sorties:
        probs = {d["label"].lower(): float(d["score"]) for d in sortie}
        score = probs.get("positive", 0.0) - probs.get("negative", 0.0)
        if score >= SEUIL_FORT:
            etiquette = "🟢 haussier"
        elif score >= SEUIL_LEGER:
            etiquette = "🟢 legerement haussier"
        elif score <= -SEUIL_FORT:
            etiquette = "🔴 baissier"
        elif score <= -SEUIL_LEGER:
            etiquette = "🔴 legerement baissier"
        else:
            etiquette = "🟡 neutre"
        resultats.append(
            {
                "score": round(score, 3),
                "confiance": round(max(probs.values()) if probs else 0.0, 3),
                "etiquette": etiquette,
            }
        )
    return resultats


def verdict(score: float) -> str:
    if score >= SEUIL_FORT:
        return "🟢 ACHAT FORT"
    if score >= SEUIL_LEGER:
        return "🟢 ACHAT"
    if score <= -SEUIL_FORT:
        return "🔴 VENTE FORTE"
    if score <= -SEUIL_LEGER:
        return "🔴 PRUDENCE"
    return "🟡 NEUTRE"


def momentum(dates, scores) -> float | None:
    paires = [(d, s) for d, s in zip(dates, scores) if d is not None]
    if len(paires) < 2:
        return None
    paires.sort(key=lambda p: p[0])
    moitie = len(paires) // 2
    ancien = sum(s for _, s in paires[:moitie]) / max(moitie, 1)
    recent = sum(s for _, s in paires[moitie:]) / max(len(paires) - moitie, 1)
    return round(recent - ancien, 3)


# ---------------------------------------------------------------------------
# 4. Interface
# ---------------------------------------------------------------------------
st.title("📈 MarketPulse")
st.caption(
    "Radar de sentiment de marche — titres de presse reels, analyses par FinBERT "
    "(ProsusAI/finbert, Hugging Face)."
)

with st.sidebar:
    st.header("Parametres")
    saisie = st.text_input(
        "Tickers (separes par une virgule)",
        value="AAPL, NVDA, TSLA, MC.PA",
        help="Actions US (AAPL), indices (^GSPC), Europe (MC.PA, SAP.DE), ETF (SPY)...",
    )
    nb_max = st.slider("Titres analyses par ticker", 5, MAX_HEADLINES, 15)
    lancer = st.button("Analyser", type="primary", use_container_width=True)
    st.divider()
    fichier = st.file_uploader(
        "Ou chargez votre propre CSV (colonnes : date, ticker, headline)",
        type=["csv"],
        help="Mode hors-ligne : les resultats restent demontrables sans acces reseau.",
    )

tickers = parse_tickers(saisie)

if fichier is not None:
    try:
        df_csv = pd.read_csv(fichier)
        df_csv.columns = [c.strip().lower() for c in df_csv.columns]
        if not {"headline", "ticker"}.issubset(df_csv.columns):
            st.error("Le CSV doit contenir au minimum les colonnes ticker et headline.")
            st.stop()
        if "date" not in df_csv.columns:
            df_csv["date"] = pd.NaT
        df_csv["date"] = pd.to_datetime(df_csv["date"], errors="coerce").dt.date
        df_csv["ticker"] = df_csv["ticker"].astype(str).str.upper().str.strip()
        lignes = df_csv.to_dict("records")
        with st.spinner(f"Analyse de {len(lignes)} titres avec FinBERT..."):
            scores = scorer([str(l["headline"]) for l in lignes])
        details = pd.DataFrame(
            [
                {
                    "date": l.get("date"),
                    "ticker": l["ticker"],
                    "headline": l["headline"],
                    "sentiment": s["etiquette"],
                    "score": s["score"],
                    "confiance": s["confiance"],
                }
                for l, s in zip(lignes, scores)
            ]
        )
        rapport_sources = []
        source_label = "CSV importe"
    except Exception as e:
        st.error(f"Lecture du CSV impossible : {e}")
        st.stop()
else:
    if lancer and tickers:
        details_liste, rapport_sources = [], []
        barre = st.progress(0.0, text="Recuperation des titres de presse...")
        for i, ticker in enumerate(tickers):
            barre.progress((i + 0.5) / len(tickers), text=f"Recuperation : {ticker}")
            paquet = fetch_headlines(ticker)
            rapport_sources.append(
                {
                    "Ticker": ticker,
                    "Source": paquet["source"],
                    "Titres": len(paquet["titres"]),
                    "Detail": " | ".join(paquet["erreurs"]) or "ok",
                }
            )
            details_liste.extend(
                {"ticker": ticker, **t} for t in paquet["titres"][:nb_max]
            )
            barre.progress((i + 1) / len(tickers), text=f"Recuperation : {ticker}")
        barre.empty()

        if not details_liste:
            st.error(
                "Aucun titre de presse recupere. Voir le diagnostic ci-dessous "
                "(reseau de l'ecole, proxy, ou limite de debit Yahoo)."
            )
            st.dataframe(pd.DataFrame(rapport_sources), use_container_width=True)
            st.stop()

        with st.spinner(f"Analyse de {len(details_liste)} titres avec FinBERT..."):
            scores = scorer([str(d["headline"]) for d in details_liste])
        details = pd.DataFrame(
            [
                {
                    "date": d.get("date"),
                    "ticker": d["ticker"],
                    "headline": d["headline"],
                    "editeur": d.get("editeur", ""),
                    "lien": d.get("lien", ""),
                    "sentiment": s["etiquette"],
                    "score": s["score"],
                    "confiance": s["confiance"],
                }
                for d, s in zip(details_liste, scores)
            ]
        )
        source_label = "sources en direct"
    else:
        st.info(
            "Renseignez des tickers dans la barre laterale puis cliquez sur **Analyser**. "
            "Aucun marche n'est ouvert ? Le mode CSV fonctionne hors-ligne."
        )
        st.stop()

# --- Agregation par ticker -------------------------------------------------
resume = []
for ticker, groupe in details.groupby("ticker"):
    resume.append(
        {
            "ticker": ticker,
            "score": round(groupe["score"].mean(), 3),
            "verdict": verdict(groupe["score"].mean()),
            "momentum": momentum(list(groupe["date"]), list(groupe["score"])),
            "nb_titres": len(groupe),
            "positifs": int((groupe["score"] > SEUIL_LEGER).sum()),
            "negatifs": int((groupe["score"] < -SEUIL_LEGER).sum()),
        }
    )
resume = sorted(resume, key=lambda r: r["score"], reverse=True)

st.subheader("Classement par sentiment")
colonnes = st.columns(min(len(resume), 4))
for i, ligne in enumerate(resume[:4]):
    with colonnes[i % len(colonnes)]:
        st.metric(
            label=ligne["ticker"],
            value=ligne["verdict"],
            delta=(f"momentum {ligne['momentum']:+.2f}" if ligne["momentum"] is not None else None),
        )

st.dataframe(
    pd.DataFrame(resume).rename(
        columns={
            "ticker": "Ticker",
            "score": "Score",
            "verdict": "Verdict",
            "momentum": "Momentum",
            "nb_titres": "Titres",
            "positifs": "Positifs",
            "negatifs": "Negatifs",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.bar_chart(
    pd.DataFrame(resume).set_index("ticker")["score"],
    y_label="Score de sentiment (-1 a +1)",
)

st.subheader("Detail des titres analyses")
choix = st.multiselect("Filtrer par sentiment", sorted(details["sentiment"].unique()))
vue = details if not choix else details[details["sentiment"].isin(choix)]
st.dataframe(
    vue.rename(
        columns={
            "date": "Date",
            "ticker": "Ticker",
            "headline": "Titre de presse",
            "editeur": "Editeur",
            "lien": "Lien",
            "sentiment": "Sentiment",
            "score": "Score",
            "confiance": "Confiance",
        }
    ),
    use_container_width=True,
    hide_index=True,
    column_config={"Lien": st.column_config.LinkColumn("Lien")},
)

st.download_button(
    "Telecharger les signaux (CSV)",
    data=details.to_csv(index=False).encode("utf-8"),
    file_name="signaux_marketpulse.csv",
    mime="text/csv",
)

with st.expander("Diagnostic des sources de donnees"):
    st.caption(f"Origine des donnees : {source_label}")
    st.dataframe(pd.DataFrame(rapport_sources), use_container_width=True, hide_index=True)
    st.markdown(
        "Si la colonne **Titres** vaut 0 partout : Yahoo Finance limite le debit "
        "(erreur 429) ou le reseau filtre la requete. Reessayez dans une minute, "
        "testez un autre reseau, ou utilisez le mode CSV."
    )
