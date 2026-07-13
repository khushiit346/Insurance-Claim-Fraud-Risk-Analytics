# Insurance Claim Fraud Risk Analytics

An end-to-end **unsupervised fraud detection pipeline** for insurance claims that flags high-risk claims for investigation, quantifies dollar exposure, and produces an agent-level risk scorecard for internal audit teams.

## Problem

Insurance carriers process thousands of claims with no reliable way to know which ones deserve a closer look — and audit teams can't review everything. This project ranks claims by risk and quantifies the dollar exposure in each risk tier, rather than producing a simple flagged/not-flagged label.

## Approach

- **Dual independent models** — a deep autoencoder and an Isolation Forest run separately. With no ground-truth fraud label in the data, *agreement between the two models* serves as the confidence signal in place of labeled validation.
- **Hard collusion rules** — claims where an agent and customer share a bank account, address, or surname are force-flagged regardless of model scores.
- **Business translation** — flags roll up into dollar exposure by risk tier and an agent-level scorecard, so the output is "which agents to investigate first," not just a list of suspicious rows.

Claims are bucketed into four tiers: **CRITICAL** (collusion rule fired), **HIGH** (both models agree), **MEDIUM** (one model flags), **LOW** (neither).

## Key Results

Run against a 10,000-claim / $165.6M portfolio:

- **738 claims (7.4%)** flagged, representing **$11.2M (6.8%)** of total exposure
- **12 CRITICAL collusion cases** surfaced independent of model scoring
- High-risk-segment customers flagged at **9.8%** vs. 7.4% portfolio average


**Honest negative finding:** the model is *not* just a large-claims detector — reviewing the top 5% of claims by risk score captures only ~5% of dollar value, no better than random. It catches structurally unusual claims rather than large ones.


## Tech Stack

Python · pandas · scikit-learn (Isolation Forest) · TensorFlow/Keras (autoencoder) · matplotlib

## Running 

```bash
pip install -r requirements.txt
python run_analysis.py                 # defaults to 95th percentile threshold
python run_analysis.py --percentile 90 # widen the net
```

Outputs: flagged claims CSV, agent risk scorecard, executive summary, and charts — all written to `outputs/`.



Want me to save this as a `README.md` file in the project, or is pasting it into GitHub directly enough? I can also add a project-structure tree if you want it slightly more complete.
