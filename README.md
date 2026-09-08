# Online Shopper Purchase Intent Predictor

Neuro Five Solutions Internship — Task 11 (Capstone Project)

**🔗 Live app: _[add your Streamlit Community Cloud link here after deploying — see "Deploying" below]_**

## Problem statement

Can we predict, from a visitor's on-site browsing behavior during a
session, whether that session is going to end in a purchase? This is a
real conversion-rate-optimization (CRO) problem faced by every e-commerce
business, Shopify stores included: most visitors never buy, and knowing
*during* a session that a visitor looks low-intent is what makes an
exit-intent discount, a live-chat prompt, or a retargeting ad worth
triggering — instead of interrupting a visitor who was going to buy
anyway, or ignoring one who was about to leave.

I picked this problem specifically because it sits directly on top of my
Shopify + AI positioning — a live version of this is a genuinely sellable
feature for an e-commerce client, not just a portfolio exercise.

## Dataset

The **UCI Online Shoppers Purchasing Intention Dataset** (Sakar et al.) —
12,330 real e-commerce sessions collected over one year, each session
belonging to a different visitor (to avoid bias toward any one campaign,
special day, or user). 18 columns: page-view counts and durations across 3
page categories, bounce/exit rates, a Google Analytics "PageValues" score,
calendar/traffic context, and the target `Revenue` (did the session end in
a purchase).

## Workflow

### 1. EDA

- **Class balance**: 15.5% of sessions end in a purchase (84.5% don't) —
  realistic for e-commerce, and imbalanced enough that accuracy alone
  would be misleading (a model that always predicts "no purchase" would
  score 84.5% while catching zero real buyers).
- **PageValues is the standout signal**: average PageValues is **1.98**
  for non-purchasers vs. **27.26** for purchasers — a 13x difference, and
  by far the strongest correlation with the target (0.493, more than 3x
  the next-strongest feature).
- **Visitor type matters**: New visitors convert at 24.9%, vs. only 13.9%
  for returning visitors — counter to the naive assumption that loyal
  repeat visitors buy more often; here, new visitors arriving with
  purchase intent (e.g. from an ad) convert better than browsing regulars.
- **Seasonality**: November has the highest purchase rate (25.4%,
  presumably Black Friday effects), February the lowest (1.6%).
- **Bounce/exit rates correlate negatively** with purchase, as expected —
  visitors who bounce quickly rarely buy.

See `eda_shopper_patterns.png` for the charts.

### 2. Cleaning

The dataset has **zero missing values** — no imputation needed. `Revenue`
and `Weekend` converted from boolean to the types scikit-learn expects.

### 3. Feature engineering

3 new engineered features:
1. **`TotalPages`** = Administrative + Informational + ProductRelated page
   counts — total pages viewed across all categories.
2. **`TotalDuration`** = sum of all 3 duration columns — total time spent
   on-site.
3. **`AvgTimePerPage`** = TotalDuration / TotalPages — engagement depth
   per page, distinguishing a visitor lingering deeply on a few pages
   from one quickly skimming many.

### 4. Modeling

3 models, each a complete scikit-learn `Pipeline`
(`StandardScaler` on numeric features, `OneHotEncoder` on categorical —
Month, VisitorType, OperatingSystems, Browser, Region, TrafficType,
Weekend) trained with class weighting (`class_weight="balanced"` for
Logistic Regression/Random Forest, `scale_pos_weight` for XGBoost) to
handle the 85/15 imbalance, on an 80/20 stratified split.

## Results

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| Accuracy | 0.8410 | 0.8682 | **0.8783** |
| Precision (Purchase) | 0.4913 | 0.5527 | **0.5840** |
| Recall (Purchase) | **0.7435** | 0.7827 | 0.7461 |
| F1 (Purchase) | 0.5917 | 0.6479 | **0.6552** |
| ROC-AUC | 0.8930 | 0.9073 | **0.9200** |

(Full table in `model_comparison.csv`.)

**XGBoost was selected as the best model** — highest F1 on the Purchase
class and highest ROC-AUC (0.92, meaning it ranks a random purchasing
session above a random non-purchasing one 92% of the time). It correctly
identifies 74.6% of actual purchasing sessions while keeping false alarms
lower than Random Forest's.

**Top drivers** (XGBoost `.feature_importances_`): `PageValues` dominates
(0.172, more than double the next feature), followed by `Month_Nov`,
`Month_May`, `Month_Mar`, and `Month_Sep` — season/month carries real
signal beyond raw browsing behavior, and `TotalPages` (an engineered
feature) also cracks the top 10. See `feature_importance.png`.

## How to run

```bash
pip install -r requirements.txt
python shopper_model.py     # trains all 3 models, saves shopper_pipeline.joblib
streamlit run app.py        # launches the live demo app at localhost:8501
```

## Deploying (Streamlit Community Cloud, free)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. **New app** → pick this repo → branch `main` → main file `app.py` → **Deploy**.
4. First build takes 1-2 minutes; you'll get a public `.streamlit.app` URL.
5. Paste it into the placeholder at the top of this README, commit, push.

## Files

- `shopper_model.py` — full pipeline: EDA, cleaning, feature engineering, 3-model training/comparison, saves the best pipeline
- `app.py` — the Streamlit deployment app
- `data/online_shoppers_intention.csv` — raw dataset
- `shopper_pipeline.joblib` — the saved, fitted, best (XGBoost) pipeline
- `model_comparison.csv` — comparison table (model, metric, score)
- `eda_shopper_patterns.png`, `feature_importance.png` — charts
- `CASE_STUDY.md` — half-page business case study
- `requirements.txt` — exact package versions used

## Next steps

- Tune the classification threshold via a precision-recall curve — the
  current default 0.5 threshold is a reasonable start, but a real
  intervention (e.g. discount popup) has its own cost, so the optimal
  threshold depends on that cost, not just F1.
- Add session-sequence features (e.g. page-view order) if raw clickstream
  data becomes available — the current dataset only has aggregated
  per-session stats.
- A/B test a live version: trigger an intervention only on sessions the
  model flags as low-probability, and measure lift in actual conversions.
