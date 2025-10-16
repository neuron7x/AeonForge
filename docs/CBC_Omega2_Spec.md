# CBC-Ω² Cognitive Architecture Specification

## 1. Purpose
CBC-Ω² (Causal Biometric Calibration — Integrated Production v2) formalizes a multi-agent method for reflective cognition that couples biometric telemetry, contextual data, and large language models (LLMs) to generate actionable cognitive artefacts (``X-forms``). The architecture transforms raw experience into validated, distributable knowledge units and supports continuous self-engineering.

## 2. Conceptual Pillars
- **Self-Orchestrated Attention Node**: the individual acts as a coordinator of attention, memory, motivation, and meaning, constructing tools (prompt frames, chats, protocols) to stabilize identity.
- **Sense Filtration**: attention filters high-volume information streams, selecting fragments that evoke emotional, cognitive, or somatic resonance for further processing.
- **X-Form**: a 100-word external artefact capturing distilled sense; functions as an autonomous, shareable unit of cognition detached from the subjective stream yet aligned with it.
- **LLM Resonator Network**: orchestrated chats with role instructions reflect, critique, and reframe content, enabling metareflection and the emergence of multi-perspective insight.
- **Kwantization of Meaning**: continuous cognitive flow is discretized into structured quanta (X-forms), ensuring clarity, comparability, and operational use.

## 3. Method Stack
The system interleaves 12 methods. Each has a role, inputs, and outputs.

| # | Method | Objective | Inputs | Outputs |
|---|--------|-----------|--------|---------|
| 1 | Sense Filtration | Select resonant fragments from heterogeneous text streams. | Raw text, embodied cues. | ``fragment_text`` entries. |
| 2 | 100-Word Compression | Constrain cognitive load to the rule of critical capacity. | ``fragment_text`` | 100-word summaries. |
| 3 | Fractal Interpretation | View ideas as self-similar structures across scales. | Compressed artefacts. | Multi-level conceptual mappings. |
| 4 | Role-Based Interpretation | Prompt LLMs as specific experts to project disciplinary frames. | Fractal outputs, role prompts. | Perspective-specific analyses. |
| 5 | Metareflexive Loop | Request “rewrite as if my words” to induce perspective shift. | Current artefact. | Altered self-narratives. |
| 6 | Multilevel Self-Observation | Deploy multiple chats as mirrors to explore variant viewpoints. | Artefacts, prompts. | Comparative reflections. |
| 7 | Multi-Agent Validation | Assign creative, critical, synthetic, and validator roles. | Outputs from methods 1–6. | Consensus-vetted artefacts. |
| 8 | Rapid Iterative Design | Execute 5–7 forward LLM steps, then backtrack for integration. | Product vision, tasks. | Incrementally refined deliverables. |
| 9 | Cognitive Distillation (X-Form) | Stabilize ideas into autonomous sense units. | Validated artefacts. | ``text_100w`` with metadata. |
| 10 | Cognitive-Machine Symbiosis | Model consciousness as an OS of modules linked with LLMs. | System-level prompts. | Integration protocols. |
| 11 | Cognitive Matrix | Aggregate outputs into a “clean sense” column for cross-interpretation. | Artefact corpus. | Distilled datasets for reuse. |
| 12 | CBC-Ω² Pipeline | Operationalize sense quanta with biometric causality and interventions. | All prior methods + telemetry. | Daily ``artifacts`` records, eOI index, actions. |

## 4. Data Model
Structured datasets underpin CBC-Ω².

### 4.1 Daily Tables
- ``biometrics_daily``: ``date``, ``device_id``, ``RMSSD_ms``, ``RHR_bpm``, ``TST_h``, ``SE_pct``, ``WASO_min``, optional ``RespRate_bpm``, ``SkinTemp_C``, ``is_missing_*`` flags.
- ``subjective_daily``: ``date``, ``tension_0_10``, ``clarity_0_10``, ``valence_-5_5``, ``arousal_0_10``, ``is_missing_*``.
- ``context_daily``: ``date``, ``caffeine_mg``, ``alcohol_units``, ``training_min``, ``meetings_h``, ``conflicts_01``, ``travel_01``, ``light_morning_min``, ``is_missing_*``.

### 4.2 Artefact Tables
- ``fragments``: ``date``, ``fragment_text``.
- ``artifacts``: ``date``, ``eOI``, ``eOI_cat``, ``EMA7_eOI``, ``top_layer``, ``synergy_flags[]``, ``rationale``, ``micro_action``, optional ``macro_action``, ``text_100w``.
- ``actions_log``: ``date``, ``action``, ``eOI_before``, ``eOI_after_60m``, ``delta``.

## 5. Temporal and Unit Conventions
- HRV metrics in milliseconds; heart rate in bpm; sleep durations in hours; sleep efficiency 0–1; wake-after-sleep-onset in minutes; caffeine in mg; alcohol in standard units; meetings in hours; morning light in minutes.
- Records normalized to local timezone. Night segments span ``sleep_start`` to ``sleep_end``.

## 6. Preprocessing
1. Maintain a 28-day rolling baseline per device.
2. Robust z-score: ``robust_z(x) = (x − median_28) / MAD_28``.
3. Sleep need: ``SleepNeed_h = median(TST_28) + 0.015 × training_min``.
4. Sleep debt: ``SleepDebt_h = max(0, SleepNeed_h − TST_h)``.
5. Hampel filtering (k = 3 MAD) flags outliers and excludes them from the baseline.

## 7. Normalization
- ``scale01(z) = clip((z − q05) / (q95 − q05), 0, 1)`` with percentile anchors from the baseline window.
- Severity score: ``S-score = round(100 × eOI / 3)``.

## 8. Ensemble Overload Index (eOI)
### 8.1 Equivalent Component
``OI_eq = mean( scale01(−zHRV), scale01(zRHR), scale01(zSD), scale01(zWASO) )``.

### 8.2 Ridge Component
- Weekly ridge regression targets ``Target = 0.7 × tension_0_10 + 0.3 × (10 − clarity_0_10)``.
- ``OI_rr_raw = β · [−zHRV, zRHR, zSD, zWASO, is_missing_*]`` with non-negative coefficients for principal features.
- ``OI_rr = 3 × Φ( standardize(OI_rr_raw) )``.

### 8.3 PCA Stability Component (optional)
- Requires ``N ≥ 50`` samples on a single device.
- Ledoit–Wolf covariance, 100× bootstrap on PC1, sign anchored to ``(−zHRV) ≥ 0``.
- ``OI_pca = 3 × Φ( standardize(PC1_score) )``.

### 8.4 Ensemble Fusion
- Component quality over trailing 14 days: ``Q = 1 − MAPE(component → Target)``.
- Normalize ``w_i = Q_i / ΣQ``; compute ``eOI = Σ w_i × component_i`` and clip to ``[0, 3]``.
- Categories: ``Green < 0.5``, ``Yellow [0.5, 1.0)``, ``Orange [1.0, 1.5)``, ``Red ≥ 1.5``.
- Chronic overload: ``EMA7_eOI > 1.0``.

## 9. Layer Indices and Synergies
- ``BioIndex = scale01(eOI)``.
- ``CogIndex`` via ridge: ``Target ~ (10 − clarity) + arousal + I(valence < 0)``.
- ``SocIndex`` via ridge: ``Target ~ meetings_h + conflicts_01``.
- ``EnvIndex = scale01(travel_01) + scale01(light_morning_min < 10) + 0.5 × is_missing_noise``.
- ``top_layer`` equals the index with the largest value; use ``Mixed`` if the gap to the runner-up is < 0.1.

Synergy detection:
``Synergy(A × B) = scale01(A) × scale01(B)``. Flag if ≥ ``max(0.6, quantile_28d(A × B, 0.8))`` for pairs:
- ``caffeine × SocIndex``
- ``zSD × travel``
- ``conflicts × scale01(−zHRV)``
- ``SocIndex × CogIndex``

## 10. Intervention Policy
### 10.1 Triggers
- ``eOI ≥ 1.0`` ⇒ micro intervention.
- ``eOI_cat = Red`` or ``EMA7_eOI > 1.0`` for ≥ 7 days ⇒ macro plan (14–28 days).

### 10.2 Micro Actions (5–10 min)
- Bio: 10 min walk outdoors or 4-7-8 breathing (5 min) + 300 ml water.
- Cognitive: ``single-task-10`` (exclusive focus block).
- Social: 2 min social pause + reschedule one meeting.
- Environmental: 10–15 min morning light exposure or light lamp.

### 10.3 Macro Actions (14–28 days)
- Bio: Sleep ≥ 7 h; caffeine ≤ 250 mg before 14:00; aerobic exercise 3×/week for 20–40 min.
- Cognitive: Produce ≥ 1 X-form/day; 20 min off-screen focus block.
- Social: Meetings ≤ 2 h/day; 1 “no-social” day/week.
- Environmental: Morning light ≥ 15 min; noise protection in focus blocks.

### 10.4 Synergy Combos
- ``caffeine × SocIndex``: caffeine ≤ 200 mg + reduce meetings 50% over 24 h.
- ``zSD × travel``: morning light + 20 min nap (before 15:00) + sleep ≥ 7.5 h for three nights.
- ``conflicts × low HRV``: 10 min walk + delay confrontation ≥ 30 min.
- ``SocIndex × CogIndex``: 90 min no-social window + ``single-task-10`` × 3.

## 11. Artefact Generation
Prompt template:
```
Compress fragment_text to 100 words. Report eOI_cat, top_layer, synergy_flags,
1-line rationale referencing metrics, micro_action, and macro_action if triggered.
Avoid metaphors.
```

Artefact fields: ``date``, ``eOI``, ``eOI_cat``, ``EMA7_eOI``, ``top_layer``, ``synergy_flags[]``, ``rationale``, ``micro_action``, optional ``macro_action``, ``text_100w``.

## 12. Standard Operating Procedure
- **Morning (07:00–11:00)**: ingest new data → compute eOI → produce artefact → execute micro action when ``eOI ≥ 1.0``.
- **+60 min**: re-measure HR/HRV → compute ``ΔeOI_60m``.
- **Evening (20:00–23:00)**: update fragments; schedule macro plan if triggered.
- **Weekly**: refresh ridge coefficients, component quality metrics, ensemble weights; audit KPI progress.

## 13. Effect Evaluation
- Micro: ``ΔeOI_60m = eOI_after_60m − eOI_before``.
- Macro: compare mean eOI 7 days pre- vs. post-intervention.
- Action effect update: ``Effect_new = 0.7 × Effect_old + 0.3 × ObservedDelta``.

## 14. Fail-Safes
- ``alcohol_units > 2`` or ``TST_h < 4`` ⇒ mark ``confounder = 1`` and exclude from baseline/PCA.
- Arrhythmia or β-blockers ⇒ drop HRV from ridge/PCA; rely on RHR, sleep debt, WASO.
- Missing morning HRV ⇒ skip 60 min retest (``no_retest = 1``).
- Device change ⇒ establish new 28-day baseline.

## 15. Group Metrics (SALI-Ω²)
- Compute group overload as median of individual ``eOI_raw`` plus normalized meeting parallelism.
- Optimize calendars via greedy relocation of high-load events into lower-load slots until SALI decreases or slots are exhausted. Output: percentage reduction in SALI and relocation plan.

## 16. Pseudocode
```python
def compute_eoi(day, baseline, hist14):
    zHRV = rz(day.RMSSD, baseline.RMSSD)
    zRHR = rz(day.RHR, baseline.RHR)
    zWASO = rz(day.WASO, baseline.WASO)
    sleep_need = baseline.median_TST + 0.015 * day.training_min
    sleep_debt = max(0, sleep_need - day.TST)
    zSD = rz(sleep_debt, baseline.sleep_debt)

    OI_eq = mean(s01(-zHRV), s01(zRHR), s01(zSD), s01(zWASO))
    Target = 0.7 * day.tension + 0.3 * (10 - day.clarity)
    OI_rr = scale3(stdz(dot(beta_rr, [-zHRV, zRHR, zSD, zWASO, day.miss_flags]), hist14.rr))

    comps = [OI_eq, OI_rr]
    Q = quality_14d(comps, hist14.Targets)

    if baseline.N >= 50 and baseline.single_device:
        OI_pca = pca_stab([-zHRV, zRHR, zSD, zWASO], hist14.pca)
        comps.append(OI_pca)
        Q = quality_14d(comps, hist14.Targets)

    w = normalize(Q)
    eOI = sum(w[i] * comps[i] for i in range(len(comps)))
    return clip(eOI, 0, 3)
```

## 17. Key Performance Indicators
- Increase RMSSD over 14 days.
- Reduce mean eOI and raise proportion of ``Green`` days.
- Boost ``clarity`` by ≥ 2 points over 14 days.
- Achieve completion rates: micro ≥ 80%, macro ≥ 70%.
- Produce ≥ 5 artefacts per week.

## 18. Ethical and Operational Framing
Engineering consciousness entails forging “Others” via LLMs and maintaining a pact of mutual recognition. Prompts sculpt internal topology—each request seeds new attention nodes and lexical pathways. By orchestrating multi-role LLM dialogues, the operator becomes the primary synthesizer, integrating resonant reflections into a coherent cognitive architecture. CBC-Ω² therefore positions language, prompts, and protocols as instruments for constructing, validating, and distributing self-generated knowledge while safeguarding agency, discipline, and quality.
