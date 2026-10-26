# MarketPulse

Radar de sentiment de marche : des titres de presse financiere aux signaux
haussiers / baissiers par action, avec FinBERT.

Application de demonstration realisee dans le cadre du cours
*Advanced AI Algorithms* (AAA Group Assignment).

## Ce que fait l'application

- Saisie de plusieurs tickers (AAPL, MSFT, MC.PA, ^GSPC...).
- Recuperation de titres de presse recents, en cascade : yfinance / Yahoo Finance,
  puis flux RSS Yahoo Finance, puis Google News.
- Analyse de chaque titre par `ProsusAI/finbert` (Hugging Face).
- Score agrege par ticker entre -1 et +1, verdict, momentum, graphique et export CSV.
- Mode hors-ligne : import d'un CSV `date,ticker,headline`.

## Lancer l'application

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Le navigateur s'ouvre sur <http://localhost:8501>. La premiere execution
telecharge le modele FinBERT (environ 440 Mo), mis en cache ensuite.

## Format du CSV

```
date,ticker,headline
2024-05-02,AAPL,Apple beats quarterly revenue expectations
2024-05-02,KO,Coca-Cola cuts full-year guidance
```

Le fichier `exemple_titres.csv` fourni permet de tester l'application sans
connexion.

## Limites

- Les sources d'actualites sont des services tiers : disponibilite, volume et
  couverture ne sont pas garantis. Yahoo Finance limite le debit des requetes.
- Le sentiment d'un article n'indique pas la reaction du cours. L'outil ne prend
  en compte ni les resultats financiers detailles, ni les anticipations de marche.
- Prototype pedagogique : ne constitue pas un conseil en investissement.

## Citations

- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained
  Language Models.* arXiv:1908.10063.
- Modele : <https://huggingface.co/ProsusAI/finbert> (Apache-2.0).
- Bibliotheques : [Transformers](https://github.com/huggingface/transformers),
  [yfinance](https://github.com/ranaroussi/yfinance),
  [Streamlit](https://streamlit.io) (Apache-2.0).

## Structure du depot

```
app.py               interface Streamlit
requirements.txt     dependances
exemple_titres.csv   jeu de donnees de test hors-ligne
README.md            ce document
```
