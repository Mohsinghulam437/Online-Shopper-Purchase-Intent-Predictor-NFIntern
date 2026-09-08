# Case Study: Predicting Purchase Intent from On-Site Behavior

**Problem.** Most e-commerce visitors leave without buying — in this
dataset, 84.5% of sessions end with no purchase. Every one of those
sessions is a missed opportunity to intervene: a well-timed discount
popup, a live-chat prompt, or a retargeting ad, shown at the right moment,
can turn a browsing session into a sale. But showing that intervention to
*every* visitor is expensive (discounts cut margin, popups annoy people
who were already buying) and showing it to *no one* leaves conversions on
the table. The business question is: can we tell, from how someone is
browsing right now, whether they're likely to buy — so an intervention
only fires on the sessions that actually need one?

**Approach.** I trained and compared three models (Logistic Regression,
Random Forest, XGBoost) on 12,330 real e-commerce sessions, using signals
already available to any store from standard analytics: pages viewed,
time spent, bounce/exit rates, a Google Analytics "PageValues" score, and
calendar context (month, weekend, proximity to a special day). No new
data collection required — this runs on what a store's analytics tooling
already tracks. XGBoost came out on top: it correctly flags about 75% of
sessions that go on to purchase, and ranks a random buying session above a
random non-buying one 92% of the time (ROC-AUC 0.92).

**Real-world value.** For a Shopify store, this translates into a
concrete conversion-rate-optimization tool: instead of showing an
exit-intent discount to every visitor (cutting margin on people who were
buying anyway) or to no one (missing recoverable sales), the store can
trigger it selectively on the ~25% of sessions the model flags as
lowest-probability but still active — the segment most likely to respond
to a nudge. The single strongest signal, by a wide margin, is
`PageValues` — a metric already sitting in most stores' Google Analytics
account, meaning the hardest part of building this (getting the right
data) is often already done. The model is intentionally built to run on
live session data mid-visit, not historical batch data, which is what
makes real-time intervention possible rather than just after-the-fact
reporting.

**Limitations, honestly stated.** Precision on the "will purchase" class
is 58% — meaning roughly 2 in 5 sessions flagged as high-intent won't
actually convert. For a cheap intervention (a subtle on-page nudge) that's
an acceptable trade-off; for an expensive one (a large discount code),
the classification threshold would need tuning against the actual cost of
a false positive versus a missed sale, not just optimized for F1. This is
a foundation to build and test on, not a finished, tuned production
system — the natural next step is an A/B test measuring actual lift in
conversions when the model gates a real intervention, not just an offline
metric.
