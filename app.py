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
    """Recupere un an de cours quotidiens et calcule les indicateurs techniques.

    Renvoie (df_indicateurs ou None, message d'erreur ou None).
    """
    try:
        import yfinance as yf
    except Exception as e:  # pragma: no cover
        return None, f"yfinance indisponible ({e.__class__.__name__})"
    try:
        brut = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
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


def synthese_locale(ticker: str, groupe: pd.DataFrame, ecart=None, decision=None) -> str:
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
def construire_prompt(ticker, groupe, panier, decision=None) -> str:
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

    return (
        "Tu es analyste de veille médiatique. Rédige en français un briefing de "
        "quatre phrases maximum sur la situation de l'action "
        f"{ticker}.{contexte} Score de sentiment moyen de la presse : {score:+.2f} "
        "sur une échelle de -1 (très négatif) à +1 (très positif).\n"
        "Titres récents :\n" + "\n".join(lignes) + "\n" + bloc_decision + "\n"
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
modele_cache = st.cache_resource(show_spinner=False)(charger_modele)


def scorer_avec_modele(titres: list[str]) -> list[dict]:
    return scorer(titres, clf=modele_cache())


st.title("📈 MarketPulse")
st.caption(
    "Sentiment de presse analysé par FinBERT, combiné aux facteurs de marché et au "
    "climat externe : un score composite, un niveau de confiance et une "
    "recommandation expliquée."
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
    help="Actions US : AAPL, MSFT · Europe : MC.PA, SAP.DE · Indice : ^GSPC · ETF : SPY",
)

col_a, col_b = st.columns([3, 1])
with col_a:
    nb_max = st.slider("Titres analysés par ticker", 5, MAX_TITRES, 15)
with col_b:
    lancer = st.button("Analyser", type="primary", use_container_width=True)

with st.expander("Pondération des facteurs et moteur de synthèse"):
    st.caption(
        "Le score composite est la moyenne pondérée de facteurs ramenés sur "
        "[-1, +1]. Chaque facteur de marché est exprimé en percentile de son "
        "propre historique sur un an, puis recentré : ce choix évite les "
        "inversions du z-score sur les séries tendancielles."
    )
    if st.checkbox("Ajuster les poids", value=False):
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
            "Le narrateur local rédige à partir des scores, des thèmes détectés et "
            "des facteurs dominants. Le modèle de langage produit une formulation "
            "plus naturelle, via une clé API personnelle."
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
            "Sans clé valide, l'application revient automatiquement au narrateur "
            "local : la démonstration reste possible."
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
    """Construit la couche decisionnelle : macro, facteurs, confiance, banc."""
    tous_titres = list(details["headline"])
    vix_pct, vix_err = vix_cached() if avec_marche else (None, "données de marché désactivées")
    part_geo, exemples_geo = tension_geopolitique(tous_titres)
    macro, detail_macro = climat_macro(vix_pct, part_geo)

    decisions, erreurs = {}, []
    for ligne in resume:
        ticker = ligne["ticker"]
        groupe = details[details["ticker"] == ticker]
        hist = None
        err = None
        if avec_marche:
            hist, err = historique_cached(ticker)
        dec = facteurs_decision(ticker, ligne["score"], macro, hist=hist, utiliser_marche=avec_marche)
        if dec["erreur_marche"]:
            erreurs.append(f"{ticker} : {dec['erreur_marche']}")
        conf = confiance_decision(ligne["nb_titres"], ligne["dispersion"], dec["marche_dispo"])
        bench = banc_validation(hist) if dec["marche_dispo"] else None
        label, notes = recommandation(dec["composite"], conf, ligne["nb_titres"], bench)
        dec.update(
            {
                "ticker": ticker,
                "confiance": conf,
                "recommandation": label,
                "notes": notes,
                "banc": bench,
                "rsi": dec.get("rsi"),
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


def rediger_syntheses(details, resume, panier_nom, decisions):
    syntheses, avertissements = {}, []
    moyenne_panier = (
        sum(r["score"] for r in resume) / len(resume) if len(resume) > 1 else None
    )
    for r in resume:
        ticker = r["ticker"]
        groupe = details[details["ticker"] == ticker]
        ecart = None if moyenne_panier is None else r["score"] - moyenne_panier
        dec = decisions["par_ticker"].get(ticker)
        texte = synthese_locale(ticker, groupe, ecart, dec)

        if mode_synthese.startswith("Modèle") and cle_api:
            prompt = construire_prompt(ticker, groupe, panier_nom, dec)
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


def executer(details, sources, source_label, panier_nom, avec_marche):
    resume = construire_resume(details)
    with st.spinner("Calcul des facteurs de marché et du climat externe..."):
        decisions = calculer_decisions(details, resume, avec_marche)
    with st.spinner("Rédaction des synthèses..."):
        syntheses, avertissements, moyenne_panier = rediger_syntheses(
            details, resume, panier_nom, decisions
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
        executer(details, [], "CSV importé", None, avec_marche)
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
            "Aucun titre de presse récupéré. Voir le diagnostic ci-dessous : "
            "réseau filtré, proxy, ou limite de débit de la source."
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
st.caption(entete)

if res["avertissements"]:
    with st.expander("Avertissements de calcul", expanded=False):
        for a in res["avertissements"]:
            st.write(f"· {a}")

# --- Classement enrichi avec la recommandation ------------------------------
st.subheader("Recommandations par action")
for ligne in resume:
    dec = par_ticker[ligne["ticker"]]
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 3])
        with c1:
            st.markdown(f"### {ligne['ticker']}")
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
        with st.expander("Détail des facteurs"):
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
                    f"Banc de validation — règle de tendance appliquée sur 12 mois : "
                    f"{b['taux']:.0%} de décisions correctes sur {b['n']} décisions. "
                    "50 % correspond au hasard."
                )
            elif dec["marche_dispo"]:
                st.caption(
                    "Banc de validation indisponible : historique trop court pour "
                    "mesurer la règle de tendance."
                )

if moyenne_panier is not None:
    meilleur = max(resume, key=lambda r: par_ticker[r["ticker"]]["composite"])
    st.markdown(
        f"**Comparaison sectorielle** — moyenne de sentiment du panier : "
        f"`{moyenne_panier:+.2f}`. Meilleur score composite : "
        f"**{meilleur['ticker']}** ({par_ticker[meilleur['ticker']]['composite']:+.2f})."
    )

# --- Climat externe --------------------------------------------------------
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
    if decisions["vix_err"]:
        st.caption(f"VIX indisponible : {decisions['vix_err']}")
    st.caption(
        "Le climat n'est pas un score de risque géopolitique officiel : c'est un "
        "indicateur dérivé de la volatilité attendue et du vocabulaire détecté "
        "dans les titres analysés."
    )

# --- Tableau comparatif ----------------------------------------------------
tableau = pd.DataFrame(
    [
        {
            "Ticker": r["ticker"],
            "Sentiment presse": r["score"],
            "Score composite": par_ticker[r["ticker"]]["composite"],
            "Confiance": par_ticker[r["ticker"]]["confiance"],
            "Recommandation": par_ticker[r["ticker"]]["recommandation"],
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

# --- Syntheses -------------------------------------------------------------
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

# --- Detail des titres -----------------------------------------------------
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

# --- Export ----------------------------------------------------------------
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
st.download_button(
    "Télécharger les signaux et recommandations (CSV)",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name="signaux_marketpulse.csv",
    mime="text/csv",
)

# --- Diagnostic ------------------------------------------------------------
if res["sources"]:
    with st.expander("Diagnostic des sources de données"):
        st.caption(f"Origine des données : {res['source_label']}")
        st.dataframe(
            pd.DataFrame(res["sources"]), use_container_width=True, hide_index=True
        )
        st.markdown(
            "Colonne **Titres** à 0 partout : la source limite le débit (erreur 429) "
            "ou le réseau filtre la requête. Réessayez dans une minute ou utilisez "
            "le mode CSV."
        )

st.divider()
st.caption(
    "Outil d'aide à la décision fondé sur des données publiques : ton de la presse "
    "analysé par FinBERT, tendance des cours, volatilité et indicateur de climat "
    "externe. Les scores sont des indicateurs de contexte, pas des prévisions de "
    "rendement. Aucune recommandation ne constitue un conseil en investissement."
)
