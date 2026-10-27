#!/usr/bin/env python3
"""
MarketPulse Web — radar de sentiment de marche et aide a la decision
=====================================================================
Entrez des tickers ou choisissez un panier sectoriel : l'application recupere
des titres de presse financiere reels, les analyse avec FinBERT, puis combine
ce sentiment avec des facteurs de marche (tendance 5 jours et 1 mois, position
vs moyenne mobile, volumes, volatilite, repli) et un indicateur de climat
externe (stress de marche VIX + tension geopolitique detectee dans la presse).

Sortie : un score composite, un niveau de confiance, une recommandation
(achat / accumuler / attente / prudence / vente) et la liste des facteurs qui
la motivent, avec le poids et la contribution de chacun.

Principe de conception : toute regle utilisee est mesuree sur l'historique, et
le resultat de cette mesure est affiche tel quel. Quand le signal est proche du
hasard, l'application le dit et retombe sur une recommandation d'attente.

Lancement local :
    streamlit run app.py

Dependances : voir requirements.txt

Citations :
    - Modele de sentiment : ProsusAI/finbert (Hugging Face, Apache-2.0), Araci (2019)
    - yfinance, transformers, streamlit, numpy, pandas (licences libres)
"""

from __future__ import annotations

import datetime as dt
import email.utils
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
MAX_TITRES = 25
SEUIL_FORT = 0.35
SEUIL_LEGER = 0.12
PANIER_DEFAUT = "AAPL, NVDA, TSLA"

# --- Paniers sectoriels : un clic, zero saisie (usage mobile) --------------
PANNIERS = {
    "Big Tech US": "AAPL, MSFT, NVDA, GOOGL, META",
    "Luxe & CAC 40": "MC.PA, RMS.PA, KER.PA, OR.PA",
    "Automobile": "TSLA, RNO.PA, MBG.DE, STLA.PA",
    "Énergie": "XOM, CVX, TTE.PA, SHEL.L",
    "Finance": "JPM, GS, BNP.PA, HSBC",
}

# --- Themes : lecture thematique des titres --------------------------------
# Chaque motif exige une frontiere de mot, sinon « ai » est trouve dans « said »,
# « Thailand » ou « chairman », et « eu » dans « Reuters ».
LEXIQUE_THEMES = {
    "résultats et marges": r"\b(earnings|profits?|revenues?|margins?|quarterly|results?|bénéfices?|résultats?)\b",
    "intelligence artificielle et semi-conducteurs": r"\b(ai|a\.i\.|artificial intelligence|chips?|semiconductors?|gpus?|data cent(?:er|re)s?|cloud)\b",
    "risque réglementaire et juridique": r"\b(regulat\w*|antitrust|lawsuit|lawsuits|probes?|fines?|courts?|investigations?|european commission)\b",
    "demande et commandes": r"\b(demand|orders?|contracts?|deals?|partnerships?|backlog|sales)\b",
    "pression sur les prix": r"\b(price cuts?|discount\w*|pricing|rebates?|cheaper|inflation)\b",
    "emploi et restructuration": r"\b(layoffs?|job cuts?|restructur\w*|strikes?|unions?)\b",
    "guidance et perspectives": r"\b(guidance|outlook|forecasts?|warnings?|estimates?)\b",
    "direction et gouvernance": r"\b(ceo|cfo|boards?|resign\w*|appoint\w*|shareholders?|buybacks?)\b",
}
THEMES_COMPILES = {nom: re.compile(motif, re.I) for nom, motif in LEXIQUE_THEMES.items()}


# ---------------------------------------------------------------------------
# 1. Saisie des tickers
# ---------------------------------------------------------------------------
def parse_tickers(saisie: str) -> list[str]:
    """Transforme "aapl, msft nvda" en ["AAPL", "MSFT", "NVDA"]."""
    jetons = re.split(r"[,\s;|]+", (saisie or "").strip())
    vus: list[str] = []
    for jeton in jetons:
        t = jeton.strip().upper().replace("$", "")
        if not t or not re.fullmatch(r"[\^A-Z0-9][A-Z0-9.\-=]{0,11}", t):
            continue
        if t not in vus:
            vus.append(t)
    return vus[:15]


# ---------------------------------------------------------------------------
# 2. Recuperation des titres de presse (3 sources en cascade)
# ---------------------------------------------------------------------------
def _http_get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"},
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
    """Gere l'ancien format Yahoo (title, providerPublishTime) ET le nouveau
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
    try:
        import yfinance as yf
    except Exception as e:  # pragma: no cover
        erreurs.append(f"yfinance indisponible ({e.__class__.__name__})")
        return []

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
        erreurs.append("yfinance : liste vide (limite de débit ou ticker inconnu)")
    else:
        erreurs.append(f"yfinance : {len(brut)} élément(s) sans titre exploitable")
    return []


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


MOTS_VIDES = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "as", "at", "by",
    "with", "from", "is", "are", "its", "it", "this", "that", "after", "over",
    "amid", "says", "said", "will", "has", "have", "be", "or", "new",
}


def _signature(titre: str) -> frozenset:
    """Empreinte d'un titre : ses mots significatifs les plus longs."""
    mots = [
        m for m in re.findall(r"[a-z0-9]{4,}", str(titre).lower()) if m not in MOTS_VIDES
    ]
    return frozenset(sorted(mots, key=len, reverse=True)[:6])


def dedupliquer(titres: list[dict], seuil: float = 0.6) -> list[dict]:
    """Retire les reprises d'une meme depeche par plusieurs medias.

    Deux titres sont juges redondants si leurs empreintes se recouvrent a 60 %.
    Sans cela, une depeche syndiquee six fois pese six fois dans la moyenne.
    """
    gardes: list[dict] = []
    signatures: list[frozenset] = []
    for t in titres:
        sig = _signature(t["headline"])
        if sig:
            redondant = False
            for s in signatures:
                union = len(sig | s)
                if union and len(sig & s) / union >= seuil:
                    redondant = True
                    break
            if redondant:
                continue
        gardes.append(t)
        signatures.append(sig)
    return gardes


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

    avant = len(titres)
    titres = dedupliquer(titres)
    if avant and len(titres) < avant:
        erreurs.append(f"{avant - len(titres)} reprise(s) de dépêche écartée(s)")

    titres.sort(key=lambda t: (t["date"] or dt.date(1970, 1, 1)), reverse=True)
    return {
        "ticker": ticker,
        "titres": titres[:MAX_TITRES],
        "source": source if titres else "aucune",
        "erreurs": erreurs,
    }


# ---------------------------------------------------------------------------
# 3. Analyse FinBERT
# ---------------------------------------------------------------------------
def charger_modele():
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        top_k=None,
    )


def scorer(titres: list[str], clf=None) -> list[dict]:
    clf = clf or charger_modele()
    sorties = clf(titres, truncation=True, batch_size=16)
    resultats = []
    for sortie in sorties:
        probs = {d["label"].lower(): float(d["score"]) for d in sortie}
        score = probs.get("positive", 0.0) - probs.get("negative", 0.0)
        if score >= SEUIL_FORT:
            etiquette = "haussier"
        elif score >= SEUIL_LEGER:
            etiquette = "légèrement haussier"
        elif score <= -SEUIL_FORT:
            etiquette = "baissier"
        elif score <= -SEUIL_LEGER:
            etiquette = "légèrement baissier"
        else:
            etiquette = "neutre"
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
        return "🟢 favorable"
    if score >= SEUIL_LEGER:
        return "🟢 plutôt favorable"
    if score <= -SEUIL_FORT:
        return "🔴 défavorable"
    if score <= -SEUIL_LEGER:
        return "🔴 plutôt défavorable"
    return "🟡 neutre"


def momentum(dates, scores) -> float | None:
    """Evolution du sentiment entre deux periodes distinctes.

    Les articles sont regroupes par date reelle, puis la premiere moitie des
    jours est comparee a la seconde. Le calcul exige au moins deux dates
    distinctes : sans cela, l'ordre d'arrivee des articles deciderait du signe,
    ce qui produisait des valeurs hors de l'intervalle [-1, +1].
    """
    par_jour: dict[dt.date, list[float]] = {}
    for d, s in zip(dates, scores):
        if d is None or not np.isfinite(s):
            continue
        par_jour.setdefault(d, []).append(float(s))
    if len(par_jour) < 2:
        return None
    jours = sorted(par_jour)
    moyennes = [float(np.mean(par_jour[j])) for j in jours]
    moitie = len(jours) // 2
    ancien = float(np.mean(moyennes[:moitie]))
    recent = float(np.mean(moyennes[moitie:]))
    return round(recent - ancien, 3)
# ---------------------------------------------------------------------------
# 5bis. Decision assistee : facteurs de marche, climat macro, banc de validation
# ---------------------------------------------------------------------------
# Poids du score composite (somme = 1). Chaque facteur est ramene sur [-1, +1]
# avant ponderation : les sentiments sont bornes naturellement, les facteurs de
# marche sont exprimes en percentile centre de leur propre historique.
POIDS_FACTEURS = {
    "sentiment": 0.30,
    "tendance5": 0.15,
    "tendance21": 0.15,
    "position_sma20": 0.10,
    "volume": 0.05,
    "volatilite": 0.10,
    "repli": 0.05,
    "macro": 0.10,
}

NOMS_FACTEURS = {
    "sentiment": "Sentiment de presse (FinBERT)",
    "tendance5": "Tendance courte (5 jours ouvrés)",
    "tendance21": "Tendance 1 mois (21 jours ouvrés)",
    "position_sma20": "Position vs moyenne mobile 20 j",
    "volume": "Momentum des volumes",
    "volatilite": "Volatilité réalisée (pénalité de risque)",
    "repli": "Profondeur du repli sur 1 mois",
    "macro": "Climat macro et géopolitique",
}

GEO_MOTS = re.compile(
    r"\b(guerre|war|invasion|sanctions?|sanctioned|ceasefire|missile|airstrike|"
    r"strike on|embargo|tariffs?|trade war|geopolitical|geopolitique|opec|"
    r"strait|coup|military|militaire)\b"
)


def historique_marche(ticker: str):
    """Recupere trois ans de cours quotidiens et calcule les indicateurs techniques.

    Renvoie (df_indicateurs ou None, message d'erreur ou None).
    """
    try:
        import yfinance as yf
    except Exception as e:  # pragma: no cover
        return None, f"yfinance indisponible ({e.__class__.__name__})"
    try:
        # 3 ans : la decomposition de tendance, le banc de validation et la
        # distribution conditionnelle puisent dans la meme serie, ce qui evite
        # un second appel reseau par ticker.
        brut = yf.Ticker(ticker).history(period="3y", interval="1d", auto_adjust=True)
    except Exception as e:
        return None, f"historique inaccessible : {e.__class__.__name__}"
    if brut is None or len(brut) < 60:
        return None, "historique insuffisant (moins de 60 séances)"
    df = brut.rename(columns=str.lower)[["close", "volume"]].dropna()
    if len(df) < 60:
        return None, "historique insuffisant après nettoyage"

    c = df["close"]
    v = df["volume"]
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    perte = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / perte.replace(0, np.nan))
    df["ret5"] = c.pct_change(5)
    df["ret21"] = c.pct_change(21)
    df["ecart_sma20"] = c / c.rolling(20).mean() - 1
    df["vol_realisee"] = c.pct_change().rolling(21).std() * np.sqrt(252)
    df["repli30"] = c / c.rolling(21).max() - 1
    df["vol_relatif"] = v / v.rolling(21).mean()
    return df, None


def percentile_centre(serie, valeur, fenetre: int = 252, min_obs: int = 60):
    """Position de la valeur dans son propre historique, recentree sur [-1, +1].

    Le percentile est monotone : contrairement au z-score sur fenetre courte,
    il ne s'inverse pas sur une serie tendancielle.
    """
    h = serie.dropna().tail(fenetre)
    if len(h) < min_obs or valeur is None or not np.isfinite(valeur):
        return None
    return 2.0 * float((h < valeur).mean()) - 1.0


def vix_percentile():
    """Percentile du VIX dans son historique d'un an : bas = climat calme."""
    try:
        import yfinance as yf
        brut = yf.Ticker("^VIX").history(period="1y", interval="1d")["Close"].dropna()
        if len(brut) < 60:
            return None, "historique VIX insuffisant"
        return float((brut < brut.iloc[-1]).mean()), None
    except Exception as e:
        return None, f"VIX inaccessible : {e.__class__.__name__}"


def tension_geopolitique(titres) -> tuple[float, list[str]]:
    """Part des titres evoquant un enjeu geopolitique, avec exemples."""
    liste = [str(t) for t in titres if str(t).strip()]
    if not liste:
        return 0.0, []
    touches = [t for t in liste if GEO_MOTS.search(t.lower())]
    return len(touches) / len(liste), touches[:3]


def climat_macro(vix_pct, part_geo: float) -> tuple[float, dict]:
    """Combine stress de marche (VIX) et tension geopolitique detectee dans les titres.

    Absence de titre geopolitique = contribution neutre (0), pas « climat calme » :
    on ne recompense pas le silence des sources.
    """
    vix_part = -(2.0 * vix_pct - 1.0) if vix_pct is not None else 0.0
    geo_part = -(2.0 * min(1.0, part_geo * 4.0) - 1.0) if part_geo > 0 else 0.0
    macro = 0.6 * vix_part + 0.4 * geo_part
    return round(macro, 3), {"vix": round(vix_part, 3), "geo": round(geo_part, 3)}


def facteurs_decision(ticker: str, sentiment: float, macro: float, hist=None, utiliser_marche: bool = True):
    """Construit la liste des facteurs normalises et le score composite du ticker."""
    df = hist
    err = None
    if df is None and utiliser_marche:
        df, err = historique_marche(ticker)

    lignes = [
        {
            "cle": "sentiment",
            "valeur": round(float(np.clip(sentiment, -1, 1)), 3),
            "poids": POIDS_FACTEURS["sentiment"],
            "brut": f"score de presse {sentiment:+.2f}",
        },
        {
            "cle": "macro",
            "valeur": macro,
            "poids": POIDS_FACTEURS["macro"],
            "brut": "climat macro et géopolitique centré",
        },
    ]

    marche_dispo = df is not None
    if marche_dispo:
        l = df.iloc[-1]
        specs = [
            ("tendance5", percentile_centre(df["ret5"], l["ret5"]), f"rendement 5 j : {l['ret5']:+.1%}", 1),
            ("tendance21", percentile_centre(df["ret21"], l["ret21"]), f"rendement 1 mois : {l['ret21']:+.1%}", 1),
            ("position_sma20", percentile_centre(df["ecart_sma20"], l["ecart_sma20"]), f"écart à la MM20 : {l['ecart_sma20']:+.1%}", 1),
            ("volume", percentile_centre(df["vol_relatif"], l["vol_relatif"]), f"volume relatif : {l['vol_relatif']:.2f}×", 1),
            ("volatilite", percentile_centre(df["vol_realisee"], l["vol_realisee"]), f"volatilité annualisée : {l['vol_realisee']:.0%}", -1),
            ("repli", percentile_centre(df["repli30"], l["repli30"]), f"repli vs plus haut 1 mois : {l['repli30']:+.1%}", 1),
        ]
        for cle, pc, brut, signe in specs:
            lignes.append(
                {
                    "cle": cle,
                    "valeur": None if pc is None else round(signe * pc, 3),
                    "poids": POIDS_FACTEURS[cle],
                    "brut": brut,
                }
            )

    dispo = [x for x in lignes if x["valeur"] is not None]
    somme_poids = sum(x["poids"] for x in dispo) or 1.0
    composite_brut = sum(np.clip(x["valeur"], -2, 2) * x["poids"] for x in dispo) / somme_poids
    return {
        "facteurs": lignes,
        "composite": round(float(np.tanh(composite_brut)), 3),
        "marche_dispo": marche_dispo,
        "erreur_marche": err,
        "rsi": (round(float(df.iloc[-1]["rsi14"]), 0) if marche_dispo and np.isfinite(df.iloc[-1]["rsi14"]) else None),
    }


def banc_validation(df):
    """Mesure le taux de reussite passe de la regle de tendance, sur 12 mois.

    Signal : percentile du rendement 1 mois >= 0,60 (hausse attendue) ou
    <= 0,40 (baisse attendue) ; sinon abstention. Cible : signe du rendement
    des 5 seances suivantes. Aucune donnee future n'est utilisee.
    """
    if df is None or len(df) < 140:
        return None
    c = df["close"]
    ret = c.pct_change()
    ret21 = c.pct_change(21)
    futur = ret.rolling(5).sum().shift(-5)
    pct = ret21.rolling(252, min_periods=100).rank(pct=True)
    bons = tot = 0
    for i in range(len(df)):
        p, f = pct.iloc[i], futur.iloc[i]
        if not np.isfinite(p) or not np.isfinite(f):
            continue
        if p >= 0.60:
            pred = 1
        elif p <= 0.40:
            pred = -1
        else:
            continue
        tot += 1
        bons += int(np.sign(f) == pred)
    if tot < 30:
        return None
    return {"taux": bons / tot, "n": tot}


def confiance_decision(nb_titres: int, ecart_type: float, marche_dispo: bool) -> float:
    """Confiance [0,1] : taille d'echantillon, accord entre titres, donnees de marche."""
    taille = min(1.0, nb_titres / 10.0)
    accord = max(0.0, 1.0 - min(1.0, ecart_type / 0.8))
    donnees = 1.0 if marche_dispo else 0.7
    return round(taille * accord * donnees, 2)


def recommandation(composite: float, confiance: float, nb_titres: int, bench) -> tuple[str, list[str]]:
    """Transforme le score composite en recommandation, avec zone d'abstention."""
    notes = []
    if nb_titres < 5:
        return "Attente — échantillon insuffisant", [
            f"{nb_titres} titre(s) analysé(s) : le score de sentiment n'est pas stable.",
        ]
    if confiance < 0.35:
        return "Attente — signal peu fiable", [
            f"Confiance {confiance:.0%} : titres trop peu nombreux ou trop dispersés.",
        ]
    if composite >= 0.35:
        label = "Achat — conviction modérée"
    elif composite >= 0.15:
        label = "Accumuler — biais haussier"
    elif composite > -0.15:
        label = "Attente — signal non tranché"
    elif composite > -0.35:
        label = "Prudence — biais baissier"
    else:
        label = "Vente / éviter — conviction modérée"

    if bench:
        if bench["taux"] < 0.52:
            notes.append(
                f"Règle de tendance mesurée à {bench['taux']:.0%} sur {bench['n']} décisions "
                "(12 mois) : proche du hasard ; la recommandation repose surtout sur le "
                "sentiment de presse et le climat macro."
            )
        else:
            notes.append(
                f"Règle de tendance mesurée à {bench['taux']:.0%} sur {bench['n']} décisions "
                "(12 mois) : signal faible mais non nul."
            )
    notes.append(
        "Aide à la décision fondée sur des données publiques ; ne constitue pas un "
        "conseil en investissement."
    )
    return label, notes
# ---------------------------------------------------------------------------
# 5quater. Transparence de la tendance, canaux geopolitiques, scenarios
# ---------------------------------------------------------------------------
# Cette couche ne produit ni nouvelle date ni nouveau cours : elle decompose ce
# que l'application mesure deja, et rend explicites les canaux par lesquels un
# evenement exterieur atteint un titre.

# --- 1. Ce que mesure chaque facteur de tendance, en clair ------------------
HORIZONS_TENDANCE = [
    ("1_semaine", 5, "5 séances"),
    ("1_mois", 21, "21 séances"),
    ("3_mois", 63, "63 séances"),
    ("6_mois", 126, "126 séances"),
    ("12_mois", 252, "252 séances"),
]

EXPLICATION_HORIZON = {
    "1_semaine": "Bruit dominant : sert au momentum court, jamais à une décision seule.",
    "1_mois": "Horizon du facteur de tendance principal du score composite.",
    "3_mois": "Sépare un rebond technique d'une inflexion de tendance réelle.",
    "6_mois": "Fenêtre de référence des gérants actions ; lisse les effets saisonniers.",
    "12_mois": "Base de comparaison annuelle : situe le titre dans son cycle.",
}

SEUIL_TENDANCE = 0.02  # ±2 % : en-dessous, la variation est lue comme non significative


def _rendement(close: pd.Series, n: int):
    if len(close) <= n:
        return None
    v = close.iloc[-1] / close.iloc[-1 - n] - 1.0
    return float(v) if np.isfinite(v) else None


def decomposer_tendance(df) -> dict | None:
    """Rend explicite la tendance utilisee : par horizon, par moyenne, par regime.

    Chaque ligne porte sa lecture en clair, pour que l'utilisateur verifie ce que
    l'application voit au lieu de faire confiance a un score unique.
    """
    if df is None or len(df) < 30:
        return None
    close = df["close"].dropna()
    horizons = []
    for cle, n, libelle in HORIZONS_TENDANCE:
        r = _rendement(close, n)
        if r is None:
            continue
        if r >= SEUIL_TENDANCE:
            lecture = "hausse établie"
        elif r <= -SEUIL_TENDANCE:
            lecture = "baisse établie"
        else:
            lecture = "quasi-stable"
        horizons.append(
            {
                "cle": cle,
                "libelle": libelle,
                "rendement": r,
                "lecture": lecture,
                "commentaire": EXPLICATION_HORIZON[cle],
            }
        )

    moyennes = {}
    for periode in (20, 50, 200):
        if len(close) >= periode:
            m = float(close.rolling(periode).mean().iloc[-1])
            moyennes[f"MM{periode}"] = {
                "valeur": m,
                "ecart": float(close.iloc[-1] / m - 1.0),
            }

    c = float(close.iloc[-1])
    plus_haut = float(close.tail(252).max())
    plus_bas = float(close.tail(252).min())
    vol = df["vol_realisee"].dropna()
    vol_pct = percentile_centre(df["vol_realisee"], df["vol_realisee"].iloc[-1], fenetre=252)
    repli_max = float((close / close.cummax() - 1.0).min())

    # Regime : regles explicites et auditables, pas un classement appris
    mm50 = moyennes.get("MM50", {}).get("valeur")
    mm200 = moyennes.get("MM200", {}).get("valeur")
    r3 = _rendement(close, 63)
    if mm50 is not None and mm200 is not None and c > mm50 > mm200:
        regime = "tendance haussière confirmée"
        regle = "cours > MM50 > MM200"
    elif mm50 is not None and mm200 is not None and c < mm50 < mm200:
        regime = "tendance baissière confirmée"
        regle = "cours < MM50 < MM200"
    elif mm50 is not None and r3 is not None and r3 > SEUIL_TENDANCE and c > mm50:
        regime = "reprise, sans confirmation longue"
        regle = "rendement 3 mois > +2 % et cours au-dessus de la MM50, moyennes non alignées"
    elif mm50 is not None and r3 is not None and r3 < -SEUIL_TENDANCE and c < mm50:
        regime = "repli dans une structure incertaine"
        regle = "rendement 3 mois < -2 % et cours sous la MM50, sans alignement baissier"
    else:
        regime = "sans direction (range)"
        regle = "aucune condition d'alignement des moyennes n'est remplie"

    return {
        "horizons": horizons,
        "moyennes": moyennes,
        "regime": regime,
        "regle_regime": regle,
        "distance_plus_haut": float(c / plus_haut - 1.0),
        "distance_plus_bas": float(c / plus_bas - 1.0),
        "volatilite": float(vol.iloc[-1]) if len(vol) else None,
        "volatilite_percentile": vol_pct,
        "repli_max": repli_max,
        "cours": c,
    }


# --- 2. Consequences documentees des evenements geopolitiques --------------
# Chaque entree separe l'ordre 1 (ce qui bouge tout de suite) de l'ordre 2 (la
# cascade), et distingue le transitoire du structurel. Les ordres de grandeur
# decrivent des moyennes historiques, pas une prevision.
CANAUX = {
    "energie": "Prix de l'énergie (pétrole, gaz, fret, assurance maritime)",
    "taux": "Taux et prime de risque (inflation anticipée, duration)",
    "chine": "Chaînes d'approvisionnement et demande chinoise",
    "defense": "Dépenses de défense et carnets de commandes militaires",
    "confiance": "Confiance des ménages et des entreprises, report des décisions",
}

EVENEMENTS_GEO = {
    "Sanctions et embargo énergétique": {
        "canal_principal": "energie",
        # Poids de propagation : le canal principal domine, les autres suivent.
        "propagation": {"energie": 1.5, "taux": 1.0, "confiance": 0.6, "chine": 0.3},
        "ordre1": "Le brut monte ; le fret et l'assurance maritime renchérissent.",
        "ordre2": "Inflation anticipée en hausse → taux longs tendus → valorisations à "
                  "duration longue sous pression ; marges des secteurs énergivores comprimées.",
        "duree": "structurel si les infrastructures sont touchées, transitoire si l'offre "
                 "est simplement réacheminée",
        "exemples": "Embargo pétrolier de 1973 ; sanctions sur le brut russe (2022).",
        "reference": "Le canal énergétique est le principal vecteur macro des chocs "
                     "géopolitiques vers les marchés actions.",
    },
    "Conflit militaire avec risque de blocus maritime": {
        "canal_principal": "energie",
        "propagation": {"energie": 1.3, "confiance": 1.0, "defense": 0.8, "taux": 0.5},
        "ordre1": "Prime de risque immédiate sur le pétrole, l'or et le VIX ; aviation et "
                  "transport maritime sous pression.",
        "ordre2": "Rotation vers les secteurs défensifs (énergie, santé, services aux "
                  "collectivités) ; les paris momentum se dénouent.",
        "duree": "choc bref puis récupération : les marchés baissent avant que l'issue soit "
                 "connue et se redressent dès que le périmètre se clarifie",
        "exemples": "Guerre du Golfe 1990-91 ; invasion de l'Ukraine en 2022.",
        "reference": "Sur les conflits majeurs depuis 1945, le S&P 500 recule d'environ 7 % "
                     "en moyenne et efface ses pertes en une cinquantaine de séances.",
    },
    "Guerre commerciale et hausse des tarifs": {
        "canal_principal": "chine",
        "propagation": {"chine": 1.5, "confiance": 0.8, "taux": 0.6, "energie": 0.2},
        "ordre1": "Révisions de bénéfices pour les exportateurs exposés ; volatilité sur "
                  "les devises et les chaînes d'approvisionnement.",
        "ordre2": "Reconfiguration des chaînes de production, hausse des coûts d'intrant, "
                  "rotation sectorielle durable plutôt que rebond rapide.",
        "duree": "structurel : les tarifs survivent souvent à la crise qui les a motivés",
        "exemples": "Tarifs américains de 2018 ; tensions commerciales récurrentes.",
        "reference": "Le canal commercial agit par les bénéfices attendus, pas par la seule "
                     "aversion au risque, ce qui le rend plus persistant qu'un choc militaire.",
    },
    "Escalade nucléaire ou attaque directe": {
        "canal_principal": "confiance",
        "propagation": {"confiance": 1.5, "taux": 1.0, "energie": 0.6, "defense": 0.4},
        "ordre1": "Fuite vers la qualité : obligations souveraines, or, dollar ; ventes "
                  "mécaniques sur les actifs risqués.",
        "ordre2": "Report des investissements : la recherche mesure une baisse de "
                  "l'investissement des entreprises d'environ 1,5 % un an après un choc de "
                  "risque géopolitique de grande ampleur.",
        "duree": "les chocs d'actes sont plus persistants que les chocs de menaces",
        "exemples": "Crise des missiles de Cuba (1962) ; attentats du 11 septembre 2001.",
        "reference": "Caldara & Iacoviello (AER 2022) : un risque géopolitique élevé annonce "
                     "un investissement, des cours et un emploi plus faibles.",
    },
    "Changement de régime politique": {
        "canal_principal": "confiance",
        "propagation": {"confiance": 1.5, "taux": 0.7, "chine": 0.4},
        "ordre1": "Écartement des spreads souverains ; volatilité des devises émergentes.",
        "ordre2": "Réévaluation de la prime de risque pays, refonte réglementaire possible.",
        "duree": "variable, souvent structurel pour les actifs du pays concerné",
        "exemples": "Printemps arabes (2011) ; coups d'État récents en Afrique de l'Ouest.",
        "reference": "L'effet est concentré sur les actifs du pays et ses voisins "
                     "commerciaux, rarement sur les indices mondiaux.",
    },
}


# --- 3. Sensibilite sectorielle aux canaux ---------------------------------
# Valeurs qualitatives, non estimees : elles donnent le SIGNE et l'ordre de
# grandeur relatif de l'exposition, pas un coefficient de regression. Elles sont
# affichees telles quelles dans l'interface, pour etre contestables.
PROFILS_SECTEURS = {
    "technologie": {
        "libelle": "Technologie et semi-conducteurs",
        "energie": -0.3, "taux": -0.8, "chine": -0.7, "defense": 0.1, "confiance": -0.5,
    },
    "luxe": {
        "libelle": "Luxe et consommation discrétionnaire",
        "energie": -0.2, "taux": -0.5, "chine": -0.9, "defense": 0.0, "confiance": -0.7,
    },
    "automobile": {
        "libelle": "Automobile",
        "energie": -0.9, "taux": -0.6, "chine": -0.6, "defense": 0.0, "confiance": -0.5,
    },
    "energie_secteur": {
        "libelle": "Énergie et matières premières",
        "energie": 0.9, "taux": -0.3, "chine": 0.1, "defense": 0.0, "confiance": 0.2,
    },
    "finance": {
        "libelle": "Finance et banques",
        "energie": -0.2, "taux": 0.4, "chine": -0.3, "defense": 0.0, "confiance": -0.6,
    },
    "defense_secteur": {
        "libelle": "Défense et aéronautique",
        "energie": 0.0, "taux": -0.2, "chine": -0.2, "defense": 1.0, "confiance": 0.3,
    },
    "sante": {
        "libelle": "Santé",
        "energie": -0.2, "taux": -0.3, "chine": -0.3, "defense": 0.0, "confiance": -0.2,
    },
    "generaliste": {
        "libelle": "Généraliste (exposition moyenne du marché)",
        "energie": -0.3, "taux": -0.4, "chine": -0.4, "defense": 0.0, "confiance": -0.4,
    },
}

SECTEUR_PAR_TICKER = {
    "AAPL": "technologie", "MSFT": "technologie", "NVDA": "technologie",
    "GOOGL": "technologie", "META": "technologie", "AMZN": "technologie",
    "TSM": "technologie", "ASML": "technologie", "SAP.DE": "technologie",
    "MC.PA": "luxe", "RMS.PA": "luxe", "KER.PA": "luxe", "OR.PA": "luxe",
    "TSLA": "automobile", "RNO.PA": "automobile", "MBG.DE": "automobile",
    "STLA.PA": "automobile", "BMW.DE": "automobile", "VOW3.DE": "automobile",
    "XOM": "energie_secteur", "CVX": "energie_secteur", "TTE.PA": "energie_secteur",
    "SHEL.L": "energie_secteur", "COP": "energie_secteur",
    "JPM": "finance", "GS": "finance", "BNP.PA": "finance", "HSBC": "finance",
    "ACA.PA": "finance", "DBK.DE": "finance",
    "LMT": "defense_secteur", "RTX": "defense_secteur", "NOC": "defense_secteur",
    "GD": "defense_secteur", "AIR.PA": "defense_secteur", "SAF.PA": "defense_secteur",
    "SAN.PA": "sante", "NOVN.SW": "sante", "PFE": "sante", "JNJ": "sante",
}

NOMS_SECTEURS = {k: v["libelle"] for k, v in PROFILS_SECTEURS.items()}


def secteur_de(ticker: str) -> str:
    return SECTEUR_PAR_TICKER.get(str(ticker).upper(), "generaliste")


def impact_evenement(ticker: str, evenement: str, intensite: float = 1.0) -> dict:
    """Traduit un evenement en exposition attendue d'un titre, canal par canal.

    Le resultat est une MOYENNE PONDEREE des expositions par les poids de
    propagation : il reste donc dans [-1, +1] sans saturer, et l'ordre entre
    secteurs est preserve (c'est ce qui manquait a une simple somme bornee).
    L'intensite (0 a 1) est choisie par l'utilisateur, jamais estimee.
    """
    ev = EVENEMENTS_GEO.get(evenement)
    if ev is None:
        return {}
    sec = secteur_de(ticker)
    profil = PROFILS_SECTEURS[sec]
    propagation = ev["propagation"]
    total_poids = sum(propagation.values()) or 1.0

    details = []
    for canal, poids in propagation.items():
        exposition = float(profil.get(canal, 0.0))
        contribution = exposition * poids * intensite
        if abs(contribution) < 0.02:
            continue
        details.append(
            {
                "canal": canal,
                "libelle_canal": CANAUX[canal],
                "poids_propagation": poids,
                "exposition": exposition,
                "contribution": round(contribution, 3),
            }
        )

    total = float(np.clip(sum(d["contribution"] for d in details) / total_poids, -1, 1))
    return {
        "ticker": ticker,
        "secteur": NOMS_SECTEURS[sec],
        "evenement": evenement,
        "canal_principal": ev["canal_principal"],
        "impact": round(total, 3),
        "lecture": (
            "exposition favorable" if total > 0.15
            else "exposition défavorable" if total < -0.15
            else "exposition peu marquée"
        ),
        "details": sorted(details, key=lambda d: d["contribution"]),
    }


# --- 4. Comment la situation pourrait evoluer -----------------------------
# Les variations ci-dessous sont des variations de STRESS, pas des variations
# favorables : +1 signifie que la pression monte sur ce canal, -1 qu'elle
# retombe. Les expositions sectorielles sont ecrites dans le meme sens (une
# valeur negative = le secteur souffre quand le stress monte), ce qui evite les
# doubles negations dans le calcul.
SCENARIOS = {
    "Apaisement": {
        "description": "Le périmètre se clarifie, la pression sur l'énergie retombe et la "
                       "prime de risque se referme.",
        "canaux": {"energie": -0.6, "confiance": -0.5, "taux": -0.2},
        "banc": "La majorité des chocs géopolitiques majeurs s'efface en quelques "
                "semaines à quelques mois.",
    },
    "Statu quo": {
        "description": "L'épisode dure sans s'étendre : le marché s'adapte, la prime de "
                       "risque reste mais cesse de croître.",
        "canaux": {"energie": 0.0, "confiance": 0.0, "taux": 0.0},
        "banc": "Le marché cesse de réagir aux nouvelles de même nature : cas le plus "
                "fréquent des conflits longs.",
    },
    "Escalade": {
        "description": "Extension géographique ou nouvelle rupture d'approvisionnement : "
                       "la pression sur l'énergie monte, l'aversion au risque domine.",
        "canaux": {"energie": +0.8, "confiance": +0.7, "taux": +0.4},
        "banc": "Les chocs d'actes, et non de menaces, sont les plus persistants pour "
                "l'investissement et les valorisations.",
    },
}


def impact_scenario(ticker: str, scenario: str) -> dict:
    """Decalage du score composite d'un titre sous un scenario externe.

    Ne previsionne aucun cours : repond a « si ces canaux bougeaient ainsi, de
    combien mon score composite bougerait, toutes choses egales par ailleurs ».
    """
    sc = SCENARIOS.get(scenario)
    if sc is None:
        return {}
    sec = secteur_de(ticker)
    profil = PROFILS_SECTEURS[sec]
    contributions = []
    for canal, variation in sc["canaux"].items():
        exposition = float(profil.get(canal, 0.0))
        contributions.append(
            {
                "canal": canal,
                "libelle_canal": CANAUX[canal],
                "variation_canal": variation,
                "exposition": exposition,
                "effet": round(variation * exposition, 3),
            }
        )
    effet = float(np.clip(sum(c["effet"] for c in contributions) / 2.0, -1, 1))
    return {
        "scenario": scenario,
        "effet": round(effet, 3),
        "contributions": sorted(contributions, key=lambda c: c["effet"]),
        "description": sc["description"],
        "banc": sc["banc"],
    }


# --- 5. Distribution conditionnelle : comment cela a evolue par le passe ---
def distribution_conditionnelle(df_long, regime_cible: str) -> dict | None:
    """Distribution des rendements a 1 mois dans le meme regime, par le passe.

    Le signal n'utilise que le passe disponible ; la cible est le rendement des
    21 seances suivantes. Les fenetres se recouvrant, l'echantillon effectif
    vaut environ n / 21 : c'est indique a l'ecran, et un echantillon trop faible
    interdit de conclure plutot que d'afficher une moyenne instable.
    """
    if df_long is None or len(df_long) < 300:
        return None
    close = df_long["close"].dropna()
    ret21 = close.pct_change(21)
    futur = close.shift(-21) / close - 1.0

    mm50 = close.rolling(50).mean()
    mm200 = close.rolling(200).mean()
    en_hausse = ((close > mm50) & (mm50 > mm200)).fillna(False)
    en_baisse = ((close < mm50) & (mm50 < mm200)).fillna(False)

    if regime_cible.startswith("tendance haussière"):
        masque = en_hausse
    elif regime_cible.startswith("tendance baissière"):
        masque = en_baisse
    elif regime_cible.startswith("reprise"):
        masque = ((ret21 > SEUIL_TENDANCE) & (close > mm50) & ~en_hausse).fillna(False)
    elif regime_cible.startswith("repli"):
        masque = ((ret21 < -SEUIL_TENDANCE) & (close < mm50) & ~en_baisse).fillna(False)
    else:
        masque = (~en_hausse & ~en_baisse)

    masque = masque.astype(bool)
    echantillon = futur[masque].dropna()
    reference = futur.dropna()

    n = int(len(echantillon))
    if n < 40:
        return {"regime": regime_cible, "n": n, "insuffisant": True}

    q = echantillon.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "regime": regime_cible,
        "n": n,
        "n_effectif": max(1, n // 21),
        "insuffisant": False,
        "mediane": float(q.loc[0.5]),
        "q10": float(q.loc[0.1]),
        "q25": float(q.loc[0.25]),
        "q75": float(q.loc[0.75]),
        "q90": float(q.loc[0.9]),
        "taux_positif": float((echantillon > 0).mean()),
        "mediane_reference": float(reference.median()),
        "taux_positif_reference": float((reference > 0).mean()),
        "ecart_reference": float(q.loc[0.5] - reference.median()),
        # Une fenetre de 21 seances qui se recouvre ne fournit que n/21
        # observations independantes : en dessous d'une vingtaine, l'ecart a la
        # reference ne peut pas etre distingue du bruit.
        "interpretable": bool(max(1, n // 21) >= 20),
    }


# --- 6. Ce qui ferait changer la recommandation ----------------------------
PALIERS_RECO = [
    (0.35, "Achat — conviction modérée"),
    (0.15, "Accumuler — biais haussier"),
    (-0.15, "Attente — signal non tranché"),
    (-0.35, "Prudence — biais baissier"),
]


def _inverse_tanh(c: float) -> float:
    c = float(np.clip(c, -0.999, 0.999))
    return float(np.arctanh(c))


def changements_de_recommandation(dec: dict, nb_titres: int) -> dict | None:
    """Calcule le palier a franchir et les valeurs de facteurs qui y menerainet.

    Rend la decision falsifiable : l'utilisateur sait quelle valeur de quel
    facteur ferait changer le verdict, au lieu de subir une etiquette.
    """
    if not dec or not dec.get("facteurs"):
        return None
    facteurs = [f for f in dec["facteurs"] if f["valeur"] is not None and f["poids"] > 0]
    if not facteurs:
        return None
    poids_total = sum(f["poids"] for f in facteurs) or 1.0
    composite = dec["composite"]

    # Palier a franchir selon la position actuelle
    if composite >= 0.35:
        cible, direction = 0.35, "baisse"
    elif composite >= 0.15:
        cible, direction = 0.15, "baisse"
    elif composite > -0.15:
        cible = 0.15 if (0.15 - composite) <= (composite + 0.15) else -0.15
        direction = "hausse" if cible > 0 else "baisse"
    elif composite > -0.35:
        cible, direction = -0.35, "baisse"
    else:
        cible, direction = -0.35, "hausse"

    marge = -0.001 if direction == "baisse" else 0.001
    s_actuel = _inverse_tanh(composite) * poids_total
    s_cible = _inverse_tanh(cible + marge) * poids_total
    delta_s = s_cible - s_actuel

    candidats = []
    for f in sorted(facteurs, key=lambda x: abs(x["valeur"] * x["poids"]), reverse=True):
        delta_facteur = delta_s / f["poids"]
        nouveau = f["valeur"] + delta_facteur
        if not (-1.0 <= nouveau <= 1.0):
            continue
        entree = {
            "facteur": NOMS_FACTEURS[f["cle"]],
            "actuel": f["valeur"],
            "requis": round(nouveau, 3),
            "delta": round(delta_facteur, 3),
            "brut": f["brut"],
        }
        if f["cle"] == "sentiment" and nb_titres > 0:
            basculement = abs(delta_facteur) * nb_titres / 2.0
            entree["equivalent"] = (
                f"soit {min(nb_titres, max(1, round(basculement)))} article(s) sur "
                f"{nb_titres} changeant d'orientation"
            )
        candidats.append(entree)

    if not candidats:
        return None
    return {
        "palier_vise": cible,
        "direction": direction,
        "candidats": candidats[:3],
    }


# ---------------------------------------------------------------------------
# 4. Synthese redigee — narrateur local (aucune cle requise)
# ---------------------------------------------------------------------------
def qualificatif(score: float) -> str:
    if score >= SEUIL_FORT:
        return "franchement favorable"
    if score >= SEUIL_LEGER:
        return "plutôt favorable"
    if score <= -SEUIL_FORT:
        return "franchement défavorable"
    if score <= -SEUIL_LEGER:
        return "plutôt défavorable"
    return "mitigé"


def themes_du_groupe(titres) -> list[str]:
    """Detecte les themes dominants, avec frontieres de mots.

    La recherche par sous-chaine produisait des faux positifs (« ai » dans
    « said », « eu » dans « Reuters ») : les motifs sont donc des expressions
    regulieres ancrees sur des limites de mot.
    """
    texte = " ".join(str(t) for t in titres)
    detectes = []
    for theme, motif in THEMES_COMPILES.items():
        occurrences = len(motif.findall(texte))
        if occurrences:
            detectes.append((occurrences, theme))
    detectes.sort(reverse=True)
    return [theme for _, theme in detectes[:2]]


def _tronque(texte: str, limite: int = 110) -> str:
    texte = re.sub(r"\s+", " ", str(texte)).strip()
    return texte if len(texte) <= limite else texte[: limite - 1].rstrip() + "..."


def _facteurs_dominants(decision: dict, n: int = 2) -> list[dict]:
    """Classe les facteurs par contribution absolue au score composite."""
    if not decision:
        return []
    dispo = [f for f in decision["facteurs"] if f["valeur"] is not None]
    dispo.sort(key=lambda f: abs(f["valeur"] * f["poids"]), reverse=True)
    return dispo[:n]


def synthese_locale(ticker: str, groupe: pd.DataFrame, ecart=None, decision=None, geo=None) -> str:
    """Redige une synthese a partir des scores, des themes et des facteurs."""
    score = float(groupe["score"].mean())
    n = len(groupe)
    positifs = groupe[groupe["score"] > SEUIL_LEGER]
    negatifs = groupe[groupe["score"] < -SEUIL_LEGER]
    neutres = n - len(positifs) - len(negatifs)

    phrases = []
    phrases.append(
        f"Sur les {n} titres collectés pour {ticker}, le ton de la presse est "
        f"{qualificatif(score)} (score moyen {score:+.2f}, entre -1 et +1)."
    )

    repartition = (
        f"{len(positifs)} titre(s) orienté(s) favorablement, "
        f"{len(negatifs)} défavorablement"
    )
    if neutres:
        repartition += f", {neutres} sans orientation nette"
    phrases.append(f"Répartition : {repartition}.")

    themes = themes_du_groupe(groupe["headline"])
    if len(themes) == 1:
        phrases.append(f"Thème dominant dans la couverture : {themes[0]}.")
    elif len(themes) > 1:
        phrases.append(
            f"Thèmes dominants dans la couverture : {themes[0]} et {themes[1]}."
        )

    if len(positifs):
        meilleur = positifs.sort_values("score", ascending=False).iloc[0]
        phrases.append(
            f"Le titre le plus favorable évoqué : « {_tronque(meilleur['headline'])} »."
        )
    if len(negatifs):
        pire = negatifs.sort_values("score").iloc[0]
        phrases.append(
            f"Le principal point de friction vient de : « {_tronque(pire['headline'])} »."
        )

    mom = momentum(list(groupe["date"]), list(groupe["score"]))
    if mom is None:
        phrases.append(
            "L'évolution du sentiment n'est pas calculable : les articles ne couvrent "
            "pas au moins deux dates distinctes."
        )
    elif mom <= -SEUIL_LEGER:
        phrases.append("Le sentiment se dégrade sur la période couverte.")
    elif mom >= SEUIL_LEGER:
        phrases.append("Le sentiment s'améliore sur la période couverte.")
    else:
        phrases.append("Le sentiment reste stable sur la période couverte.")

    # --- Apport de la couche decisionnelle
    if decision:
        phrases.append(
            f"Recommandation : {decision['recommandation'].lower()} "
            f"(score composite {decision['composite']:+.2f}, confiance "
            f"{decision['confiance']:.0%})."
        )
        dominants = _facteurs_dominants(decision)
        if dominants:
            details = ", ".join(
                f"{NOMS_FACTEURS[f['cle']].lower()} ({'favorable' if f['valeur'] > 0 else 'défavorable'}, "
                f"{f['valeur']:+.2f})"
                for f in dominants
            )
            phrases.append(f"Facteurs les plus déterminants : {details}.")
        if not decision["marche_dispo"]:
            phrases.append(
                "Les données de marché n'ont pas pu être récupérées : la recommandation "
                "repose sur le seul sentiment de presse et le climat macroéconomique."
            )

    if geo and abs(geo.get("impact", 0.0)) >= 0.15:
        sens = "favorable" if geo["impact"] > 0 else "défavorable"
        phrases.append(
            f"Exposition {sens} à l'événement retenu ({geo['evenement'].lower()}) : "
            f"{geo['impact']:+.2f} pour le secteur {geo['secteur'].lower()}, via le canal "
            f"{CANAUX[geo['canal_principal']].lower()}."
        )

    if ecart is not None:
        if ecart >= SEUIL_LEGER:
            phrases.append("Le titre se situe au-dessus de la moyenne de son panier.")
        elif ecart <= -SEUIL_LEGER:
            phrases.append("Le titre se situe sous la moyenne de son panier.")
        else:
            phrases.append("Le titre se situe dans la moyenne de son panier.")

    phrases.append(
        "À lire comme une synthèse d'indicateurs publics, pas comme une prévision de cours."
    )
    return " ".join(phrases)


# ---------------------------------------------------------------------------
# 5. Synthese redigee — modele de langage (optionnel, cle API)
# ---------------------------------------------------------------------------
def construire_prompt(ticker, groupe, panier, decision=None, geo=None) -> str:
    score = float(groupe["score"].mean())
    lignes = []
    for _, r in groupe.sort_values("score", ascending=False).head(8).iterrows():
        lignes.append(f"- ({r['score']:+.2f}) {_tronque(r['headline'], 150)}")
    contexte = f" Panier comparé : {panier}." if panier else ""

    bloc_decision = ""
    if decision:
        facteurs = "\n".join(
            f"- {NOMS_FACTEURS[f['cle']]} : {f['valeur']:+.2f} "
            f"(poids {f['poids']:.2f}, {f['brut']})"
            for f in decision["facteurs"]
            if f["valeur"] is not None
        )
        bloc_decision = (
            f"\nScore composite calculé : {decision['composite']:+.2f}. "
            f"Recommandation du modèle : {decision['recommandation']}. "
            f"Confiance : {decision['confiance']:.0%}.\nFacteurs :\n{facteurs}\n"
        )

    bloc_geo = ""
    if geo:
        bloc_geo = (
            f"\nExposition de son secteur à l'événement « {geo['evenement']} » : "
            f"{geo['impact']:+.2f} (canal dominant : "
            f"{CANAUX[geo['canal_principal']]}).\n"
        )

    return (
        "Tu es analyste de veille médiatique. Rédige en français un briefing de "
        "quatre phrases maximum sur la situation de l'action "
        f"{ticker}.{contexte} Score de sentiment moyen de la presse : {score:+.2f} "
        "sur une échelle de -1 (très négatif) à +1 (très positif).\n"
        "Titres récents :\n" + "\n".join(lignes) + "\n" + bloc_decision + bloc_geo + "\n"
        "Consignes : phrases courtes et factuelles, appuyées uniquement sur ces "
        "éléments. Explique ce qui pèse le plus dans la recommandation. Ne promets "
        "aucun rendement, ne donne aucun objectif de cours. Termine par une réserve "
        "d'usage d'une demi-phrase rappelant qu'il ne s'agit pas d'un conseil en "
        "investissement."
    )


def _post_json(url: str, charge: dict, entetes: dict | None = None, timeout: int = 45) -> dict:
    corps = json.dumps(charge).encode("utf-8")
    entetes = dict(entetes or {})
    entetes.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=corps, headers=entetes, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def synthese_llm(prompt: str, fournisseur: str, cle: str, modele: str) -> str:
    """Appelle un modele de langage distant et renvoie le texte genere."""
    if not cle:
        raise ValueError("clé API absente")

    if fournisseur == "Google Gemini":
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            + urllib.parse.quote(modele)
            + ":generateContent?key="
            + urllib.parse.quote(cle)
        )
        charge = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 300},
        }
        data = _post_json(url, charge)
        candidats = data.get("candidates") or []
        if not candidats:
            raise ValueError("réponse vide du modèle")
        return candidats[0]["content"]["parts"][0]["text"].strip()

    if fournisseur == "Hugging Face":
        url = "https://api-inference.huggingface.co/models/" + urllib.parse.quote(modele)
        charge = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.4,
                "return_full_text": False,
            },
        }
        data = _post_json(url, charge, {"Authorization": f"Bearer {cle}"})
        if isinstance(data, list) and data:
            return str(data[0].get("generated_text", "")).strip()
        raise ValueError("réponse inattendue du modèle")

    raise ValueError(f"fournisseur inconnu : {fournisseur}")


# ---------------------------------------------------------------------------
# 6. Interface — saisie dans la page principale (lisible sur telephone)
# ---------------------------------------------------------------------------
import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="MarketPulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

fetch_headlines_cached = st.cache_data(ttl=900, show_spinner=False)(fetch_headlines)
historique_cached = st.cache_data(ttl=3600, show_spinner=False)(historique_marche)
vix_cached = st.cache_data(ttl=3600, show_spinner=False)(vix_percentile)

from geo_engine import fetch_geo_headlines, construire_contexte

fetch_geo_cached = st.cache_data(ttl=3600, show_spinner=False)(fetch_geo_headlines)
modele_cache = st.cache_resource(show_spinner=False)(charger_modele)


def scorer_avec_modele(titres: list[str]) -> list[dict]:
    return scorer(titres, clf=modele_cache())


st.title("📈 MarketPulse")
st.caption(
    "Sentiment de presse analysé par FinBERT, combiné aux tendances de marché et aux "
    "canaux de transmission géopolitiques : un score composite, un niveau de confiance, "
    "une recommandation expliquée et sa sensibilité à chaque facteur."
)

if "tickers_input" not in st.session_state:
    st.session_state.tickers_input = PANIER_DEFAUT


def appliquer_panier() -> None:
    choix = st.session_state.get("panier_choisi", "Saisie libre")
    if choix in PANNIERS:
        st.session_state.tickers_input = PANNIERS[choix]


st.selectbox(
    "Panier sectoriel — un clic, aucune saisie",
    ["Saisie libre"] + list(PANNIERS.keys()),
    key="panier_choisi",
    on_change=appliquer_panier,
    help="Choisir un panier remplit automatiquement la liste de tickers ci-dessous.",
)

st.text_input(
    "Tickers (séparés par une virgule)",
    key="tickers_input",
    help="Actions US : AAPL, MSFT · Europe : MC.PA, SAP.DE · Défense : LMT, AIR.PA · Indice : ^GSPC",
)

col_a, col_b = st.columns([3, 1])
with col_a:
    nb_max = st.slider("Titres analysés par ticker", 5, MAX_TITRES, 15)
with col_b:
    lancer = st.button("Analyser", type="primary", use_container_width=True)

with st.expander("Réglages du moteur de décision"):
    st.caption(
        "Le score composite est la moyenne pondérée de facteurs ramenés sur [-1, +1]. "
        "Chaque facteur de marché est exprimé en percentile de son propre historique, "
        "puis recentré : ce choix évite les inversions du z-score sur les séries "
        "tendancielles."
    )
    if st.checkbox("Ajuster les poids des facteurs", value=False):
        poids = {}
        for cle in POIDS_FACTEURS:
            poids[cle] = st.slider(
                NOMS_FACTEURS[cle], 0.0, 0.60, POIDS_FACTEURS[cle], 0.05, key=f"poids_{cle}"
            )
        total = sum(poids.values()) or 1.0
        POIDS_FACTEURS.update({k: v / total for k, v in poids.items()})
        st.caption("Les poids sont renormalisés pour totaliser 1.")

    mode_synthese = st.radio(
        "Moteur de rédaction",
        ["Narrateur local (aucune clé requise)", "Modèle de langage (clé API)"],
        help=(
            "Le narrateur local rédige à partir des scores, des thèmes détectés, des "
            "facteurs dominants et de l'exposition géopolitique. Le modèle de langage "
            "produit une formulation plus naturelle, via une clé API personnelle."
        ),
    )
    fournisseur = modele_llm = cle_api = None
    if mode_synthese.startswith("Modèle"):
        fournisseur = st.selectbox("Fournisseur", ["Google Gemini", "Hugging Face"])
        defaut_modele = (
            "gemini-2.0-flash"
            if fournisseur == "Google Gemini"
            else "mistralai/Mistral-7B-Instruct-v0.2"
        )
        modele_llm = st.text_input("Nom du modèle", value=defaut_modele)
        cle_api = st.text_input(
            "Clé API",
            type="password",
            help="La clé reste dans votre session et n'est jamais écrite dans le code.",
        )
        st.caption(
            "Sans clé valide, l'application revient automatiquement au narrateur local : "
            "la démonstration reste possible."
        )

with st.expander("Mode hors-ligne : analyser votre propre CSV"):
    st.caption("Colonnes attendues : date, ticker, headline (date facultative).")
    fichier = st.file_uploader("Fichier CSV", type=["csv"], label_visibility="collapsed")


# ---------------------------------------------------------------------------
# 7. Calculs
# ---------------------------------------------------------------------------
def construire_resume(details: pd.DataFrame) -> list[dict]:
    lignes = []
    for ticker, groupe in details.groupby("ticker"):
        lignes.append(
            {
                "ticker": ticker,
                "score": round(groupe["score"].mean(), 3),
                "verdict": verdict(groupe["score"].mean()),
                "momentum": momentum(list(groupe["date"]), list(groupe["score"])),
                "nb_titres": len(groupe),
                "positifs": int((groupe["score"] > SEUIL_LEGER).sum()),
                "negatifs": int((groupe["score"] < -SEUIL_LEGER).sum()),
                "dispersion": round(float(groupe["score"].std(ddof=0) or 0.0), 3),
            }
        )
    return lignes


def calculer_decisions(details: pd.DataFrame, resume: list[dict], avec_marche: bool) -> dict:
    """Couche decisionnelle : macro, facteurs, tendance, confiance, banc, seuils."""
    tous_titres = list(details["headline"])
    vix_pct, vix_err = vix_cached() if avec_marche else (None, "données de marché désactivées")
    part_geo, exemples_geo = None, []
    if avec_marche and contexte_geo and contexte_geo.get("index") is not None:
        part_geo = min(1.0, max(0.0, -contexte_geo["index"]))
        exemples_geo = contexte_geo["exemples_negatifs"]
    else:
        part_geo, exemples_geo = tension_geopolitique(tous_titres)
    macro, detail_macro = climat_macro(vix_pct, part_geo)

    decisions, erreurs = {}, []
    for ligne in resume:
        ticker = ligne["ticker"]
        hist = None
        if avec_marche:
            hist, err = historique_cached(ticker)
            if err:
                erreurs.append(f"{ticker} : {err}")
        dec = facteurs_decision(
            ticker, ligne["score"], macro, hist=hist, utiliser_marche=avec_marche
        )
        conf = confiance_decision(ligne["nb_titres"], ligne["dispersion"], dec["marche_dispo"])
        bench = banc_validation(hist) if dec["marche_dispo"] else None
        label, notes = recommandation(dec["composite"], conf, ligne["nb_titres"], bench)

        tendance = decomposer_tendance(hist) if dec["marche_dispo"] else None
        distribution = None
        if tendance:
            distribution = distribution_conditionnelle(hist, tendance["regime"])

        dec.update(
            {
                "ticker": ticker,
                "confiance": conf,
                "recommandation": label,
                "notes": notes,
                "banc": bench,
                "tendance": tendance,
                "distribution": distribution,
                "changements": changements_de_recommandation(dec, ligne["nb_titres"]),
            }
        )
        decisions[ticker] = dec
    return {
        "par_ticker": decisions,
        "macro": macro,
        "detail_macro": detail_macro,
        "vix_pct": vix_pct,
        "vix_err": vix_err,
        "part_geo": part_geo,
        "exemples_geo": exemples_geo,
        "erreurs": erreurs,
    }


def rediger_syntheses(details, resume, panier_nom, decisions, evenement, intensite):
    syntheses, avertissements = {}, []
    moyenne_panier = (
        sum(r["score"] for r in resume) / len(resume) if len(resume) > 1 else None
    )
    for r in resume:
        ticker = r["ticker"]
        groupe = details[details["ticker"] == ticker]
        ecart = None if moyenne_panier is None else r["score"] - moyenne_panier
        dec = decisions["par_ticker"].get(ticker)
        geo = impact_evenement(ticker, evenement, intensite) if evenement else None
        texte = synthese_locale(ticker, groupe, ecart, dec, geo)

        if mode_synthese.startswith("Modèle") and cle_api:
            prompt = construire_prompt(ticker, groupe, panier_nom, dec, geo)
            try:
                texte_llm = synthese_llm(prompt, fournisseur, cle_api, modele_llm)
                if texte_llm and len(texte_llm) > 40:
                    texte = texte_llm
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as e:
                avertissements.append(
                    f"{ticker} : modèle indisponible ({e}). Narrateur local utilisé."
                )
            except Exception as e:  # pragma: no cover
                avertissements.append(
                    f"{ticker} : erreur inattendue ({e.__class__.__name__}). "
                    "Narrateur local utilisé."
                )
        syntheses[ticker] = texte
    return syntheses, avertissements, moyenne_panier


def executer(details, sources, source_label, panier_nom, avec_marche, evenement, intensite):
    resume = construire_resume(details)
    with st.spinner("Calcul des tendances, des facteurs et du climat externe..."):
        decisions = calculer_decisions(details, resume, avec_marche)
    with st.spinner("Rédaction des synthèses..."):
        syntheses, avertissements, moyenne_panier = rediger_syntheses(
            details, resume, panier_nom, decisions, evenement, intensite
        )
    st.session_state.resultat = {
        "details": details,
        "resume": resume,
        "syntheses": syntheses,
        "avertissements": avertissements + decisions["erreurs"],
        "moyenne_panier": moyenne_panier,
        "decisions": decisions,
        "sources": sources,
        "source_label": source_label,
        "panier": panier_nom,
        "horodatage": dt.datetime.now().strftime("%d/%m/%Y %H:%M"),
    }


panier_nom = st.session_state.get("panier_choisi", "Saisie libre")
tickers = parse_tickers(st.session_state.tickers_input)
avec_marche = st.checkbox(
    "Intégrer les facteurs de marché et le climat externe (cours, volatilité, VIX, géopolitique)",
    value=True,
    help="Décochez pour une analyse fondée sur la seule presse, plus rapide.",
)

# Contexte géopolitique AUTO-DÉTECTÉ depuis la presse mondiale (geo_engine)
contexte_geo = None
evenement_geo, intensite_geo = None, 0.0
if avec_marche:
    try:
        with st.spinner("Lecture de l'actualité géopolitique mondiale…"):
            titres_geo = fetch_geo_cached()
        if titres_geo:
            scores_geo = [r["score"] for r in scorer_avec_modele(
                [t["headline"] for t in titres_geo]
            )]
            contexte_geo = construire_contexte(titres_geo, scores_geo)
            if contexte_geo["events"]:
                ev0 = contexte_geo["events"][0]
                evenement_geo = ev0["evenement_fr"]
                intensite_geo = ev0["intensite"]
    except Exception:
        contexte_geo = None
if evenement_geo:
    st.info(
        f"Événement géopolitique détecté automatiquement : **{evenement_geo}** "
        f"(intensité estimée {intensite_geo:.2f}, "
        f"{contexte_geo['events'][0]['articles']} articles de presse mondiale)."
    )

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
        lignes_csv = df_csv.to_dict("records")
        with st.spinner(f"Analyse de {len(lignes_csv)} titres avec FinBERT..."):
            scores_csv = scorer_avec_modele([str(l["headline"]) for l in lignes_csv])
        details = pd.DataFrame(
            [
                {
                    "date": l.get("date"),
                    "ticker": l["ticker"],
                    "headline": l["headline"],
                    "editeur": "",
                    "lien": "",
                    "sentiment": s["etiquette"],
                    "score": s["score"],
                    "confiance": s["confiance"],
                }
                for l, s in zip(lignes_csv, scores_csv)
            ]
        )
        executer(details, [], "CSV importé", None, avec_marche, evenement_geo, intensite_geo)
    except Exception as e:
        st.error(f"Lecture du CSV impossible : {e}")
        st.stop()

elif lancer and tickers:
    details_liste, rapport_sources = [], []
    barre = st.progress(0.0, text="Récupération des titres de presse...")
    for i, ticker in enumerate(tickers):
        barre.progress((i + 0.5) / len(tickers), text=f"Récupération : {ticker}")
        paquet = fetch_headlines_cached(ticker)
        rapport_sources.append(
            {
                "Ticker": ticker,
                "Source": paquet["source"],
                "Titres": len(paquet["titres"]),
                "Détail": " | ".join(paquet["erreurs"]) or "ok",
            }
        )
        details_liste.extend({"ticker": ticker, **t} for t in paquet["titres"][:nb_max])
    barre.empty()

    if not details_liste:
        st.error(
            "Aucun titre de presse récupéré. Voir le diagnostic ci-dessous : réseau "
            "filtré, proxy, ou limite de débit de la source."
        )
        st.dataframe(pd.DataFrame(rapport_sources), use_container_width=True)
        st.stop()

    with st.spinner(f"Analyse de {len(details_liste)} titres avec FinBERT..."):
        scores = scorer_avec_modele([str(d["headline"]) for d in details_liste])
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
    executer(
        details,
        rapport_sources,
        "sources en direct",
        None if panier_nom == "Saisie libre" else panier_nom,
        avec_marche,
        evenement_geo,
        intensite_geo,
    )

if "resultat" not in st.session_state:
    st.info(
        "Choisissez un panier sectoriel ou saisissez des tickers, puis appuyez sur "
        "**Analyser**. Aucun accès réseau ? Le mode CSV fonctionne hors-ligne."
    )
    st.stop()


# ---------------------------------------------------------------------------
# 8. Affichage
# ---------------------------------------------------------------------------
res = st.session_state.resultat
details = res["details"]
resume = res["resume"]
syntheses = res["syntheses"]
decisions = res["decisions"]
moyenne_panier = res["moyenne_panier"]
par_ticker = decisions["par_ticker"]

entete = f"Analyse du {res['horodatage']}"
if res["panier"]:
    entete += f" · panier {res['panier']}"
if evenement_geo:
    entete += f" · événement de référence : {evenement_geo} (intensité {intensite_geo:.1f})"
st.caption(entete)

if res["avertissements"]:
    with st.expander("Avertissements de calcul", expanded=False):
        for a in res["avertissements"]:
            st.write(f"· {a}")

# --- 8.1 Recommendations par action ---------------------------------------
st.subheader("Recommandations par action")
for ligne in resume:
    ticker = ligne["ticker"]
    dec = par_ticker[ticker]
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 3])
        with c1:
            st.markdown(f"### {ticker}")
            st.caption(f"presse : {ligne['verdict']} ({ligne['score']:+.2f})")
        with c2:
            st.metric("Recommandation", dec["recommandation"].split(" — ")[0])
            st.caption(
                f"score composite {dec['composite']:+.2f} · confiance {dec['confiance']:.0%}"
            )
        with c3:
            st.caption(
                f"{ligne['nb_titres']} article(s) · dispersion {ligne['dispersion']:.2f}"
                + (
                    f" · momentum {ligne['momentum']:+.2f}"
                    if ligne["momentum"] is not None
                    else " · momentum non calculable"
                )
                + (f" · RSI {dec['rsi']:.0f}" if dec["rsi"] else "")
            )
        for note in dec["notes"]:
            st.caption(f"↳ {note}")

        onglets = st.tabs(
            ["Facteurs", "Tendance", "Géopolitique", "Scénarios", "Seuils de bascule"]
        )

        # --- Facteurs
        with onglets[0]:
            lignes_facteurs = []
            for f in dec["facteurs"]:
                lignes_facteurs.append(
                    {
                        "Facteur": NOMS_FACTEURS[f["cle"]],
                        "Valeur normalisée": (
                            f"{f['valeur']:+.2f}" if f["valeur"] is not None else "indisponible"
                        ),
                        "Poids": f"{f['poids']:.2f}",
                        "Contribution": (
                            f"{f['valeur'] * f['poids']:+.3f}"
                            if f["valeur"] is not None
                            else "—"
                        ),
                        "Mesure": f["brut"],
                        "Lecture": (
                            "favorable"
                            if (f["valeur"] or 0) > 0.05
                            else "défavorable"
                            if (f["valeur"] or 0) < -0.05
                            else "neutre"
                        ),
                    }
                )
            st.dataframe(
                pd.DataFrame(lignes_facteurs), use_container_width=True, hide_index=True
            )
            if dec["banc"]:
                b = dec["banc"]
                st.caption(
                    f"Banc de validation — règle de tendance rejouée sur l'historique : "
                    f"{b['taux']:.0%} de décisions correctes sur {b['n']} décisions. "
                    "50 % correspond au hasard."
                )
            elif dec["marche_dispo"]:
                st.caption(
                    "Banc de validation indisponible : historique trop court pour mesurer "
                    "la règle de tendance."
                )

        # --- Tendance
        with onglets[1]:
            tend = dec["tendance"]
            if not tend:
                st.caption("Données de marché indisponibles : décomposition impossible.")
            else:
                st.markdown(f"**Régime détecté : {tend['regime']}**")
                st.caption(f"Règle appliquée : {tend['regle_regime']}")
                df_h = pd.DataFrame(
                    [
                        {
                            "Horizon": h["libelle"],
                            "Rendement": h["rendement"],
                            "Lecture": h["lecture"],
                            "À quoi cela sert": h["commentaire"],
                        }
                        for h in tend["horizons"]
                    ]
                )
                st.dataframe(
                    df_h,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Rendement": st.column_config.NumberColumn(format="%.1f%%")
                    },
                )
                st.bar_chart(
                    pd.DataFrame(
                        {"rendement": {h["libelle"]: h["rendement"] * 100 for h in tend["horizons"]}}
                    ),
                    y_label="Rendement cumulé (%)",
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric(
                        "Distance au plus haut 12 mois",
                        f"{tend['distance_plus_haut']:+.1%}",
                        help="Un titre loin de son plus haut est en repli, pas forcément en "
                        "tendance baissière.",
                    )
                with c2:
                    st.metric(
                        "Volatilité annualisée",
                        f"{tend['volatilite']:.0%}" if tend["volatilite"] is not None else "n/d",
                        help="Écart-type des rendements quotidiens, annualisé.",
                    )
                with c3:
                    st.metric(
                        "Repli maximal depuis un sommet",
                        f"{tend['repli_max']:.1%}",
                        help="Pire baisse depuis un plus haut, sur tout l'historique disponible.",
                    )
                if tend["moyennes"]:
                    st.caption(
                        "Moyennes mobiles : "
                        + " · ".join(
                            f"{k} à {v['valeur']:.2f} (cours {v['ecart']:+.1%})"
                            for k, v in tend["moyennes"].items()
                        )
                    )
                dist = dec["distribution"]
                if dist and not dist.get("insuffisant"):
                    st.markdown("**Rendements à 1 mois observés dans ce même régime**")
                    st.caption(
                        f"{dist['n']} jours de bourse concernés, soit environ "
                        f"{dist['n_effectif']} observations indépendantes (fenêtres de "
                        "21 séances qui se recouvrent)."
                    )
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Médiane", f"{dist['mediane']:+.1%}")
                    with c2:
                        st.metric("10 % les plus défavorables", f"{dist['q10']:+.1%}")
                    with c3:
                        st.metric("10 % les plus favorables", f"{dist['q90']:+.1%}")
                    st.caption(
                        f"Part de périodes positives : {dist['taux_positif']:.0%} "
                        f"(référence toutes périodes confondues : "
                        f"{dist['taux_positif_reference']:.0%}, médiane "
                        f"{dist['mediane_reference']:+.1%})."
                    )
                    if not dist["interpretable"]:
                        st.warning(
                            "Échantillon indépendant trop faible : l'écart à la référence "
                            "ne peut pas être distingué du hasard. À lire comme une "
                            "illustration, pas comme une probabilité."
                        )
                elif dist:
                    st.caption(
                        f"Rendements conditionnels non calculables : seulement {dist['n']} "
                        "observations dans ce régime."
                    )

        # --- Géopolitique
        with onglets[2]:
            if not evenement_geo:
                st.caption(
                    "Choisissez un événement de référence au-dessus de la zone d'analyse "
                    "pour chiffrer l'exposition de ce titre."
                )
            else:
                imp = impact_evenement(ticker, evenement_geo, intensite_geo)
                ev = EVENEMENTS_GEO[evenement_geo]
                st.markdown(
                    f"**{evenement_geo}** — exposition estimée : `{imp['impact']:+.2f}` "
                    f"({imp['lecture']})"
                )
                st.caption(
                    f"Secteur retenu : {imp['secteur']} · canal dominant : "
                    f"{CANAUX[imp['canal_principal']]}"
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Canal": d["libelle_canal"],
                                "Poids de propagation": d["poids_propagation"],
                                "Exposition du secteur": d["exposition"],
                                "Contribution": d["contribution"],
                            }
                            for d in imp["details"]
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Effet immédiat (ordre 1)**")
                    st.write(ev["ordre1"])
                    st.markdown("**Cascade (ordre 2)**")
                    st.write(ev["ordre2"])
                with c2:
                    st.markdown("**Durée attendue**")
                    st.write(ev["duree"])
                    st.markdown("**Précédents comparables**")
                    st.write(ev["exemples"])
                    st.caption(ev["reference"])
                st.caption(
                    "Les expositions sectorielles sont des appréciations qualitatives "
                    "affichées telles quelles, pas des coefficients estimés sur données. "
                    "L'intensité est choisie par vous, jamais déduite par le modèle."
                )

        # --- Scénarios
        with onglets[3]:
            lignes_sc = []
            for nom_sc in SCENARIOS:
                r = impact_scenario(ticker, nom_sc)
                lignes_sc.append(
                    {
                        "Scénario": nom_sc,
                        "Effet sur le score composite": r["effet"],
                        "Score composite simulé": round(
                            float(np.clip(dec["composite"] + r["effet"], -1, 1)), 3
                        ),
                        "Description": r["description"],
                    }
                )
            st.dataframe(
                pd.DataFrame(lignes_sc),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Effet sur le score composite": st.column_config.NumberColumn(format="%+.2f")
                },
            )
            st.caption(
                "Ces valeurs répondent à « si la pression sur ces canaux bougeait ainsi, de "
                "combien mon score bougerait, toutes choses égales ». Aucun cours n'est "
                "prévu."
            )
            for nom_sc in ["Escalade", "Apaisement"]:
                st.caption(f"· {nom_sc} : {SCENARIOS[nom_sc]['banc']}")

        # --- Seuils de bascule
        with onglets[4]:
            ch = dec["changements"]
            if not ch:
                st.caption("Aucun seuil calculable : les facteurs disponibles saturent déjà.")
            else:
                sens = "abaisserait" if ch["direction"] == "baisse" else "relèverait"
                st.markdown(
                    f"Pour {sens} la recommandation au palier "
                    f"`{ch['palier_vise']:+.2f}`, il faudrait :"
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Facteur": c["facteur"],
                                "Valeur actuelle": c["actuel"],
                                "Valeur nécessaire": c["requis"],
                                "Écart": c["delta"],
                                "Traduction": c.get("equivalent", "—"),
                                "Mesure actuelle": c["brut"],
                            }
                            for c in ch["candidats"]
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "Rend la recommandation falsifiable : on sait exactement quelle mesure "
                    "devrait changer, et de combien, pour que le verdict bascule."
                )

if moyenne_panier is not None:
    meilleur = max(resume, key=lambda r: par_ticker[r["ticker"]]["composite"])
    st.markdown(
        f"**Comparaison sectorielle** — moyenne de sentiment du panier : "
        f"`{moyenne_panier:+.2f}`. Meilleur score composite : "
        f"**{meilleur['ticker']}** ({par_ticker[meilleur['ticker']]['composite']:+.2f})."
    )

# --- 8.2 Climat externe ----------------------------------------------------
with st.expander("Climat externe pris en compte", expanded=bool(decisions["exemples_geo"])):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Stress de marché (VIX)",
            f"{decisions['vix_pct']:.0%}" if decisions["vix_pct"] is not None else "n/d",
            help="Percentile du VIX sur un an. Un percentile élevé pèse négativement.",
        )
    with c2:
        st.metric(
            "Titres à tonalité géopolitique",
            f"{decisions['part_geo']:.0%}",
            help="Part des articles mentionnant guerre, sanctions, tarifs, OPEC, etc.",
        )
    with c3:
        st.metric("Contribution du climat", f"{decisions['macro']:+.2f}")
    if decisions["exemples_geo"]:
        st.caption("Exemples détectés :")
        for ex in decisions["exemples_geo"]:
            st.caption(f"· {_tronque(ex, 140)}")
    if contexte_geo and contexte_geo.get("events"):
        st.caption(f"Contexte détecté automatiquement ({contexte_geo['status']}) :")
        for ev in contexte_geo["events"]:
            with st.expander(
                f"{ev['evenement_fr']} — {ev['articles']} articles "
                f"(sentiment {ev['sentiment_moyen']:+.2f}, intensité {ev['intensite']:.2f})"
            ):
                for titre_geo, source_geo in ev["exemples"]:
                    st.caption(f"· {titre_geo} — *{source_geo}*")
        st.caption(
            "Intensité = heuristique (négativité de la couverture + volume d'articles), "
            "pas une estimation officielle."
        )
    if decisions["vix_err"]:
        st.caption(f"VIX indisponible : {decisions['vix_err']}")
    st.caption(
        "Le climat n'est pas un score de risque géopolitique officiel : c'est un "
        "indicateur dérivé de la volatilité attendue et du vocabulaire détecté dans les "
        "titres analysés."
    )

# --- 8.3 Comprendre les canaux ---------------------------------------------
with st.expander("Comment une décision géopolitique atteint la bourse : les cinq canaux"):
    st.markdown(
        "Un événement extérieur ne fait jamais bouger une action directement. Il passe par "
        "cinq canaux, et chaque secteur y est exposé différemment. C'est cette chaîne que "
        "l'application chiffre."
    )
    st.dataframe(
        pd.DataFrame(
            [{"Canal": CANAUX[k], "Clé": k} for k in CANAUX]
        ),
        use_container_width=True,
        hide_index=True,
    )
    matrice = pd.DataFrame(
        {
            "Secteur": [v["libelle"] for v in PROFILS_SECTEURS.values()],
            **{
                CANAUX[canal]: [v.get(canal, 0.0) for v in PROFILS_SECTEURS.values()]
                for canal in CANAUX
            },
        }
    )
    st.dataframe(matrice, use_container_width=True, hide_index=True)
    st.caption(
        "Lecture : une valeur négative signifie que le secteur souffre quand la pression "
        "monte sur ce canal. Ces appréciations sont qualitatives et contestables : elles "
        "sont affichées pour être discutées, pas pour être crues sur parole."
    )
    for nom_ev, ev in EVENEMENTS_GEO.items():
        st.markdown(f"**{nom_ev}** — canal dominant : {CANAUX[ev['canal_principal']]}")
        st.write(f"Ordre 1 : {ev['ordre1']}")
        st.write(f"Ordre 2 : {ev['ordre2']}")
        st.write(f"Durée : {ev['duree']}")
        st.caption(f"Précédents : {ev['exemples']} — {ev['reference']}")

# --- 8.4 Tableau comparatif ------------------------------------------------
tableau = pd.DataFrame(
    [
        {
            "Ticker": r["ticker"],
            "Sentiment presse": r["score"],
            "Score composite": par_ticker[r["ticker"]]["composite"],
            "Confiance": par_ticker[r["ticker"]]["confiance"],
            "Recommandation": par_ticker[r["ticker"]]["recommandation"],
            "Régime de tendance": (
                par_ticker[r["ticker"]]["tendance"]["regime"]
                if par_ticker[r["ticker"]]["tendance"]
                else "n/d"
            ),
            "Exposition géo": (
                impact_evenement(r["ticker"], evenement_geo, intensite_geo)["impact"]
                if evenement_geo
                else None
            ),
            "Titres": r["nb_titres"],
            "Momentum": r["momentum"],
        }
        for r in resume
    ]
)
st.dataframe(tableau, use_container_width=True, hide_index=True)

st.bar_chart(
    pd.DataFrame(
        {
            "sentiment": {r["ticker"]: r["score"] for r in resume},
            "composite": {r["ticker"]: par_ticker[r["ticker"]]["composite"] for r in resume},
        }
    ),
    y_label="Score (-1 à +1)",
)

# --- 8.5 Syntheses ---------------------------------------------------------
st.subheader("Lecture rédigée par action")
for ligne in resume:
    ticker = ligne["ticker"]
    dec = par_ticker[ticker]
    titre_bloc = f"{ticker} — {dec['recommandation']} (composite {dec['composite']:+.2f})"
    with st.expander(titre_bloc, expanded=len(resume) <= 3):
        st.write(syntheses.get(ticker, "Synthèse indisponible."))
        extraits = details[details["ticker"] == ticker].sort_values(
            "score", ascending=False
        )
        for _, r in extraits.head(3).iterrows():
            date_txt = f"{r['date']} — " if r["date"] else ""
            st.caption(f"{date_txt}{r['headline']} ({r['score']:+.2f})")

# --- 8.6 Detail des titres -------------------------------------------------
st.subheader("Détail des titres analysés")
choix = st.multiselect("Filtrer par sentiment", sorted(details["sentiment"].unique()))
vue = details if not choix else details[details["sentiment"].isin(choix)]
st.dataframe(
    vue.rename(
        columns={
            "date": "Date",
            "ticker": "Ticker",
            "headline": "Titre de presse",
            "editeur": "Éditeur",
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

# --- 8.7 Export ------------------------------------------------------------
export = details.copy()
export["synthese"] = export["ticker"].map(syntheses)
export["recommandation"] = export["ticker"].map(
    {t: d["recommandation"] for t, d in par_ticker.items()}
)
export["score_composite"] = export["ticker"].map(
    {t: d["composite"] for t, d in par_ticker.items()}
)
export["confiance_decision"] = export["ticker"].map(
    {t: d["confiance"] for t, d in par_ticker.items()}
)
export["regime_tendance"] = export["ticker"].map(
    {t: (d["tendance"]["regime"] if d["tendance"] else "") for t, d in par_ticker.items()}
)
if evenement_geo:
    export["exposition_geo"] = export["ticker"].map(
        {
            t: impact_evenement(t, evenement_geo, intensite_geo)["impact"]
            for t in par_ticker
        }
    )
st.download_button(
    "Télécharger les signaux et recommandations (CSV)",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name="signaux_marketpulse.csv",
    mime="text/csv",
)

# --- 8.8 Diagnostic -------------------------------------------------------
if res["sources"]:
    with st.expander("Diagnostic des sources de données"):
        st.caption(f"Origine des données : {res['source_label']}")
        st.dataframe(
            pd.DataFrame(res["sources"]), use_container_width=True, hide_index=True
        )
        st.markdown(
            "Colonne **Titres** à 0 partout : la source limite le débit (erreur 429) ou le "
            "réseau filtre la requête. Réessayez dans une minute ou utilisez le mode CSV."
        )

st.divider()
st.caption(
    "Outil d'aide à la décision fondé sur des données publiques : ton de la presse "
    "analysé par FinBERT, tendances de cours sur cinq horizons, volatilité, et "
    "indicateur de climat externe. Les scores sont des indicateurs de contexte, pas des "
    "prévisions de rendement. Aucune recommandation ne constitue un conseil en "
    "investissement."
)
