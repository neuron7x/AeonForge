# Phase 3 Randomized Controlled Trial (RCT) Protocol

## 1. Overview
The Phase 3 trial validates the end-to-end CBC-Ω² intervention in a production
setting. The study randomizes eligible participants to receive the CBC-Ω² driven
coaching program or a control program. Continuous telemetry flows into the
observability stack to support rapid detection of safety issues and efficacy
signal drift.

## 2. Objectives
- **Primary:** Demonstrate a statistically significant improvement in composite
  recovery outcomes (sleep quality, resting HR, subjective readiness) measured at
  12 weeks compared with control.
- **Secondary:** Quantify adherence, engagement, and adverse event rates.
- **Exploratory:** Evaluate cohort-specific response curves to support dynamic
  personalization in subsequent releases.

## 3. Eligibility and Enrollment
- Adults 21–60, with wearable-compatible hardware and 60 days of historical
  biometrics.
- Exclusion: acute cardiovascular events in last 6 months, uncontrolled chronic
  conditions, or inability to consent.
- Participants consent electronically; baseline assessments captured in REDCap
  and mirrored into the analytics warehouse.

## 4. Study Arms
| Arm | Description | Sample Size |
| --- | ----------- | ----------- |
| Intervention | CBC-Ω² engine with adaptive coaching + alerting | 250 |
| Control | Standard care content without adaptive personalization | 250 |

## 5. Randomization and Blinding
- Stratified block randomization (block size 4) by site, sex, and baseline
  readiness score.
- Allocation managed by the Phase 2 weight-optimization service to ensure
  consistent cohort tagging.
- Outcomes assessed by blinded analysts; participants and coaches aware of arm
  assignment.

## 6. Intervention Delivery
- Intervention cohort receives daily CBC-Ω² recommendations delivered through
  the mobile app. Alerts and escalations integrate with the clinician console.
- Control cohort receives weekly generic wellness nudges.
- All content delivery events emit Prometheus counters for dose quantification.

## 7. Outcome Measures
- **Primary endpoint:** Change in composite recovery index (z-score aggregated
  from HRV, resting HR, and sleep efficiency) at week 12.
- **Secondary endpoints:**
  - Adherence rate ≥4 interactions/week.
  - Adverse events logged by study clinicians.
  - Subjective readiness (Likert scale) weekly.
- **Safety monitoring:** Automatic detection of >10% decline in recovery index
  triggers clinician review within 24 hours.

## 8. Data Capture & Instrumentation
1. Wearable ingestions flow through the Phase 1 ingestion stack. The
   `phase1_pipeline` script persists fold-level ROC metrics to
   `artifacts/phase1/` for traceability.
2. Phase 2 grid-search updates weight schedules weekly; results are exported to
   `artifacts/phase2/weights.json` and versioned for reproducibility.
3. Trial events stream to Kafka topics (`trial.events`, `trial.alerts`). The
   `instrumentation.py` exporter (see below) consumes summaries and exposes
   Prometheus metrics.
4. Prometheus scrapes the exporter every 15 seconds. Dashboards in Grafana
   provide:
   - Enrollment funnel (gauge + rate panels).
   - Adherence heatmaps per cohort.
   - Safety watchlist (alert thresholds tied to recovery index declines).

## 9. Statistical Analysis Plan
- Intention-to-treat, mixed-effects linear model adjusting for baseline covariates.
- Interim analysis at 50% enrollment using O'Brien–Fleming boundaries.
- Missing data addressed with multiple imputation (m=5) and sensitivity analysis
  using last-observation-carried-forward.

## 10. Monitoring and Governance
- Data Safety Monitoring Board (DSMB) reviews weekly dashboards and monthly
  comprehensive reports.
- Automated PagerDuty alerts fire on:
  - Exporter downtime >5 minutes.
  - Prometheus alert `cbc_omega2_adverse_event_rate` crossing safety threshold.
- All deviations logged in Jira with links to Grafana panel snapshots.

## 11. Deliverables
- Prometheus scrape configs and Grafana dashboard JSON committed under
  `deploy/monitoring/` (tracked separately).
- Weekly Phase 2 optimisation summary appended to the shared Confluence space.
- Final RCT report combining clinical endpoints and telemetry health metrics.

---
The complementary instrumentation implementation is in
[`instrumentation.py`](./instrumentation.py).
