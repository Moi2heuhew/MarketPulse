#!/usr/bin/env python3
"""
MarketPulse Web — radar de sentiment de marche
==============================================
Interface web (Streamlit) : saisissez des tickers ou choisissez un panier
sectoriel, l'application recupere des titres de presse financière réels, les
analyse avec FinBERT et affiche un classement haussier / baissier accompagné
d'une synthèse rédigée.

Deux modes de synthèse :
  - narrateur local (par défaut, aucune clé requise, instantané) ;
  - modèle de langage (optionnel) via une clé API Google Gemini ou Hugging Face.

Lancement local :
    streamlit run app.py

Dépendances : voir requirements.txt

Citations :
    - Modèle de sentiment : ProsusAI/finbert (Hugging Face, Apache-2.0), Araci (2019)
    - yfinance, transformers, streamlit (Apache-2.0)
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

# --- Themes : lecture thematique des titres, sans dependance externe -------
LEXIQUE_THEMES = {
    "résultats et marges": (
        "earnings", "profit", "revenue", "margin", "quarter", "results",
        "bénéfice", "chiffre d'affaires", "résultat",
    ),
    "intelligence artificielle et semi-conducteurs": (
        "ai", "artificial intelligence", "chip", "chips", "semiconductor",
        "gpu", "data center", "data centre", "cloud",
    ),
    "risque réglementaire et juridique": (
        "regulator", "regulation", "antitrust", "lawsuit", "probe", "fine",
        "court", "investigation", "eu", "european commission",
    ),
    "demande et commandes": (
        "demand", "order", "orders", "contract", "deal", "partnership",
        "wins", "backlog", "sales",
    ),
    "pression sur les prix": (
        "price cut", "price cuts", "discount", "pricing", "rebate",
        "cheaper", "inflation",
    ),
    "emploi et restructuration": (
        "layoff", "layoffs", "job cuts", "restructuring", "strike", "union",
    ),
    "guidance et perspectives": (
        "guidance", "outlook", "forecast", "warning", "cuts forecast",
        "raises forecast", "estimates",
    ),
    "direction et gouvernance": (
        "ceo", "cfo", "board", "resign", "appoint", "shareholder", "buyback",
    ),
}


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
        "titres": uniques[:MAX_TITRES],
        "source": source if uniques else "aucune",
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
    paires = [(d, s) for d, s in zip(dates, scores) if d is not None]
    if len(paires) < 2:
        return None
    paires.sort(key=lambda p: p[0])
    moitie = len(paires) // 2
    ancien = sum(s for _, s in paires[:moitie]) / max(moitie, 1)
    recent = sum(s for _, s in paires[moitie:]) / max(len(paires) - moitie, 1)
    return round(recent - ancien, 3)


# ---------------------------------------------------------------------------
# 4. Synthese redigee — narrateur local (aucune cle requise)
# ---------------------------------------------------------------------------
def qualificatif(score: float) -> str:
    if score >= SEUIL_FORT:
        return "franchement favorable"
    if score >= SEUIL_LEGER:
        return "plutôt favorable"
    if score <= -SEUIL_FORT:
        return "franchement defavorable"
    if score <= -SEUIL_LEGER:
        return "plutôt défavorable"
    return "mitigé"


def themes_du_groupe(titres) -> list[str]:
    """Detecte les themes dominants dans les titres d'un ticker."""
    texte = " ".join(str(t).lower() for t in titres)
    scores = []
    for theme, mots in LEXIQUE_THEMES.items():
        hits = sum(texte.count(m) for m in mots)
        if hits:
            scores.append((hits, theme))
    scores.sort(reverse=True)
    return [theme for _, theme in scores[:2]]


def _tronque(texte: str, limite: int = 110) -> str:
    texte = re.sub(r"\s+", " ", str(texte)).strip()
    return texte if len(texte) <= limite else texte[: limite - 1].rstrip() + "..."


def synthese_locale(ticker: str, groupe: pd.DataFrame, ecart: float | None = None) -> str:
    """Rédige une synthèse de trois à cinq phrases à partir des chiffres et des titres."""
    score = float(groupe["score"].mean())
    n = len(groupe)
    positifs = groupe[groupe["score"] > SEUIL_LEGER]
    negatifs = groupe[groupe["score"] < -SEUIL_LEGER]
    neutres = n - len(positifs) - len(negatifs)

    phrases = []
    phrases.append(
        f"Sur les {n} titres collectés pour {ticker}, le ton d'ensemble est "
        f"{qualificatif(score)} (score moyen {score:+.2f}, entre -1 et +1)."
    )

    repartition = f"{len(positifs)} titre(s) orienté(s) favorablement, {len(negatifs)} défavorablement"
    if neutres:
        repartition += f", {neutres} sans orientation nette"
    phrases.append(f"Répartition : {repartition}.")

    themes = themes_du_groupe(groupe["headline"])
    if themes:
        if len(themes) == 1:
            phrases.append(f"Thème dominant dans la couverture : {themes[0]}.")
        else:
            phrases.append(f"Thèmes dominants dans la couverture : {themes[0]} et {themes[1]}.")

    if len(positifs):
        meilleur = positifs.sort_values("score", ascending=False).iloc[0]
        phrases.append(f"Le titre le plus favorable évoqué : « {_tronque(meilleur['headline'])} ».")
    if len(negatifs):
        pire = negatifs.sort_values("score").iloc[0]
        phrases.append(f"Le principal point de friction vient de : « {_tronque(pire['headline'])} ».")

    mom = momentum(list(groupe["date"]), list(groupe["score"]))
    if mom is not None:
        if mom <= -SEUIL_LEGER:
            phrases.append("La tendance se dégrade sur la période couverte.")
        elif mom >= SEUIL_LEGER:
            phrases.append("La tendance s'améliore sur la période couverte.")
        else:
            phrases.append("La tendance reste stable sur la période couverte.")

    if ecart is not None:
        if ecart >= SEUIL_LEGER:
            phrases.append("Le titre se situe au-dessus de la moyenne de son panier.")
        elif ecart <= -SEUIL_LEGER:
            phrases.append("Le titre se situe sous la moyenne de son panier.")
        else:
            phrases.append("Le titre se situe dans la moyenne de son panier.")

    phrases.append(
        "À lire comme un indicateur de tonalité médiatique, pas comme une prévision de cours."
    )
    return " ".join(phrases)


# ---------------------------------------------------------------------------
# 5. Synthese redigee — modele de langage (optionnel, cle API)
# ---------------------------------------------------------------------------
def construire_prompt(ticker: str, groupe: pd.DataFrame, panier: str | None) -> str:
    score = float(groupe["score"].mean())
    lignes = []
    for _, r in groupe.sort_values("score", ascending=False).head(8).iterrows():
        lignes.append(f"- ({r['score']:+.2f}) {_tronque(r['headline'], 150)}")
    contexte = f" Panier compare : {panier}." if panier else ""
    return (
        "Tu es analyste de veille médiatique. Rédige en français un briefing de "
        "trois phrases maximum sur le sentiment de presse de l'action "
        f"{ticker}.{contexte} Score de sentiment moyen : {score:+.2f} sur une "
        "échelle de -1 (très négatif) à +1 (très positif).\n"
        "Titres recents :\n" + "\n".join(lignes) + "\n\n"
        "Consignes : phrases courtes, factuelles, appuyées uniquement sur ces "
        "titres. Ne donne aucune recommandation d'investissement, aucun objectif "
        "de cours. Termine par une réserve d'usage d'une demi-phrase."
    )


def _post_json(url: str, charge: dict, entetes: dict | None = None, timeout: int = 45) -> dict:
    corps = json.dumps(charge).encode("utf-8")
    entetes = dict(entetes or {})
    entetes.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=corps, headers=entetes, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def synthese_llm(prompt: str, fournisseur: str, cle: str, modele: str) -> str:
    """Appelle un modele de langage distant et renvoie le texte genere.

    Leve une exception explicite si l'appel echoue : l'interface retombe alors
    sur le narrateur local.
    """
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
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 260},
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
                "max_new_tokens": 260,
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
import streamlit as st  # noqa: E402  (importe apres les fonctions pour la lisibilite)

st.set_page_config(
    page_title="MarketPulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

fetch_headlines_cached = st.cache_data(ttl=900, show_spinner=False)(fetch_headlines)
modele_cache = st.cache_resource(show_spinner=False)(charger_modele)


def scorer_avec_modele(titres: list[str]) -> list[dict]:
    return scorer(titres, clf=modele_cache())


st.title("📈 MarketPulse")
st.caption(
    "Radar de sentiment de marché : titres de presse financière réels analysés "
    "par FinBERT (ProsusAI/finbert, Hugging Face), puis résumés en clair."
)

# --- Choix du panier sectoriel (Piste 3) ------------------------------------
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

# --- Reglages de la synthese (Piste 1) -------------------------------------
with st.expander("Réglages de la synthèse rédigée"):
    mode_synthese = st.radio(
        "Moteur de synthèse",
        ["Narrateur local (aucune clé requise)", "Modèle de langage (clé API)"],
        horizontal=False,
        help=(
            "Le narrateur local rédige un résumé à partir des scores et des thèmes "
            "détectés dans les titres. Le modèle de langage produit une formulation "
            "plus naturelle, via une clé API personnelle."
        ),
    )
    fournisseur = modele_llm = cle_api = None
    if mode_synthese.startswith("Modele"):
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
            }
        )
    return sorted(lignes, key=lambda r: r["score"], reverse=True)


def rediger_syntheses(details, resume, panier_nom):
    """Redige une synthese par ticker : narrateur local, puis LLM si demande."""
    moyenne_panier = (
        sum(r["score"] for r in resume) / len(resume) if len(resume) > 1 else None
    )
    syntheses, avertissements = {}, []
    for r in resume:
        ticker = r["ticker"]
        groupe = details[details["ticker"] == ticker]
        ecart = None if moyenne_panier is None else r["score"] - moyenne_panier
        texte = synthese_locale(ticker, groupe, ecart)

        if mode_synthese.startswith("Modele") and cle_api:
            prompt = construire_prompt(ticker, groupe, panier_nom)
            try:
                texte_llm = synthese_llm(prompt, fournisseur, cle_api, modele_llm)
                if texte_llm and len(texte_llm) > 40:
                    texte = texte_llm
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as e:
                avertissements.append(f"{ticker} : modèle indisponible ({e}). Narrateur local utilisé.")
            except Exception as e:  # pragma: no cover
                avertissements.append(f"{ticker} : erreur inattendue ({e.__class__.__name__}). Narrateur local utilisé.")
        syntheses[ticker] = texte
    return syntheses, avertissements, moyenne_panier


panier_nom = st.session_state.get("panier_choisi", "Saisie libre")
tickers = parse_tickers(st.session_state.tickers_input)

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
        resume = construire_resume(details)
        with st.spinner("Rédaction des synthèses..."):
            syntheses, avertissements, moyenne_panier = rediger_syntheses(
                details, resume, None
            )
        st.session_state.resultat = {
            "details": details,
            "resume": resume,
            "syntheses": syntheses,
            "avertissements": avertissements,
            "moyenne_panier": moyenne_panier,
            "sources": [],
            "source_label": "CSV importe",
            "panier": None,
            "horodatage": dt.datetime.now().strftime("%d/%m/%Y %H:%M"),
        }
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
    resume = construire_resume(details)
    with st.spinner("Rédaction des synthèses..."):
        syntheses, avertissements, moyenne_panier = rediger_syntheses(
            details, resume, None if panier_nom == "Saisie libre" else panier_nom
        )
    st.session_state.resultat = {
        "details": details,
        "resume": resume,
        "syntheses": syntheses,
        "avertissements": avertissements,
        "moyenne_panier": moyenne_panier,
        "sources": rapport_sources,
        "source_label": "sources en direct",
        "panier": None if panier_nom == "Saisie libre" else panier_nom,
        "horodatage": dt.datetime.now().strftime("%d/%m/%Y %H:%M"),
    }

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
moyenne_panier = res["moyenne_panier"]

entete = f"Analyse du {res['horodatage']}"
if res["panier"]:
    entete += f" · panier {res['panier']}"
st.caption(entete)

if res["avertissements"]:
    st.warning(" | ".join(res["avertissements"]))

st.subheader("Classement par sentiment")
colonnes = st.columns(min(len(resume), 4))
for i, ligne in enumerate(resume[:4]):
    with colonnes[i % len(colonnes)]:
        st.metric(
            label=ligne["ticker"],
            value=ligne["verdict"],
            delta=(
                f"momentum {ligne['momentum']:+.2f}"
                if ligne["momentum"] is not None
                else None
            ),
        )

# --- Comparaison sectorielle (Piste 3) -------------------------------------
if moyenne_panier is not None:
    meilleur = resume[0]
    ecart = meilleur["score"] - moyenne_panier
    st.markdown(
        f"**Comparaison sectorielle** — moyenne du panier : `{moyenne_panier:+.2f}`. "
        f"{meilleur['ticker']} se situe {ecart:+.2f} point(s) au-dessus de cette moyenne."
    )

tableau = pd.DataFrame(resume).rename(
    columns={
        "ticker": "Ticker",
        "score": "Score",
        "verdict": "Tonalité",
        "momentum": "Momentum",
        "nb_titres": "Titres",
        "positifs": "Favorables",
        "negatifs": "Défavorables",
    }
)
if moyenne_panier is not None:
    tableau["Écart panier"] = tableau["Score"] - moyenne_panier
st.dataframe(tableau, use_container_width=True, hide_index=True)

st.bar_chart(
    pd.DataFrame(resume).set_index("ticker")["score"],
    y_label="Score de sentiment (-1 à +1)",
)

# --- Syntheses redigees (Piste 1) ------------------------------------------
st.subheader("Lecture rédigée par action")
for ligne in resume:
    ticker = ligne["ticker"]
    titre_bloc = f"{ticker} — {ligne['verdict']} (score {ligne['score']:+.2f})"
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
choix = st.multiselect(
    "Filtrer par sentiment", sorted(details["sentiment"].unique())
)
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
st.download_button(
    "Télécharger les signaux et synthèses (CSV)",
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
    "Prototype pédagogique. Les scores reflètent le ton d'articles de presse, "
    "pas la valeur d'un titre ni l'évolution de son cours. Les synthèses sont "
    "générées automatiquement et peuvent comporter des erreurs. "
    "Ne constitue pas un conseil en investissement."
)
