# Recommended Text Sources for RAG
## Cardiac Rehab, HRV, VO2, and Clinical Agent Grounding

This list focuses on **text sources only** for retrieval-augmented generation. The goal is to give the multi-agent system a strong medical and physiologic grounding layer for **in-clinic cardiac rehabilitation**, while also supporting patient-friendly coaching, clinician Q&A, and structured report generation.

---

# 1. Highest-priority core corpus

These are the first sources I would ingest because they define the medical logic, program structure, and physiologic interpretation backbone of the system.

## 1. Core Components of Cardiac Rehabilitation Programs: 2024 Update
- **Type:** AHA/AACVPR scientific statement
- **Best use:** Duty Doctor Agent, Clinical Assistant Agent, overall rehab policy and workflow grounding
- **Why include it:** This is the single most important cardiac rehab text source because it updates the scientific basis of modern cardiac rehab and explicitly covers patient assessment, nutritional counseling, weight management, cardiovascular risk-factor management, psychosocial management, aerobic training, strength training, physical activity counseling, and program quality. It also reflects newer delivery models and quality-improvement thinking. [web:6][web:7][web:9]
- **Summary for RAG:** Use this as the main source for “what a cardiac rehab program should include,” how sessions should be framed, and how the agents should talk about core components, program goals, and quality standards.

## 2. HRV Standards of Measurement, Physiological Interpretation, and Clinical Use
- **Type:** Foundational HRV standards statement
- **Best use:** Foundation physiologic grounding, Duty Doctor Agent, Clinical Assistant Agent
- **Why include it:** This is still the canonical text for defining HRV terminology, time-domain and frequency-domain measures, physiologic interpretation, and clinical framing. It is essential if your system is going to discuss RMSSD, SDNN, autonomic balance, recovery behavior, or risk interpretation in a medically coherent way. [web:11][web:14]
- **Summary for RAG:** Use this as the reference text for what HRV metrics mean, how they should be interpreted, and how to explain the relationship between autonomic function, sinus rhythm variability, and clinical context.

## 3. Validation of a Neural Network for Estimating VO2max from Wearables
- **Type:** Wearable VO2 / cardiorespiratory fitness estimation study
- **Best use:** VO2 estimation grounding, Nurse Agent explanation, Duty Doctor summary
- **Why include it:** Your system already uses wearable-derived exercise and recovery context, so this source is directly relevant to explaining why wearable-based VO2max estimation is valid and where its limitations are. It is especially useful for grounding the system’s language around free-living fitness estimation rather than treadmill-only lab testing. [web:15]
- **Summary for RAG:** Use this as the evidence base for explaining that wearable data can estimate VO2max with meaningful accuracy and can identify patients with lower functional fitness who may benefit from intervention.

## 4. FRIEND cardiorespiratory fitness reference standards
- **Type:** Registry-based reference standards for cardiorespiratory fitness
- **Best use:** VO2 interpretation, percentile reporting, clinical benchmarking
- **Why include it:** The FRIEND registry gives you the normative and clinical reference language needed to convert a raw VO2 estimate or CPX-derived measure into something more interpretable, such as percentile rank or category-relative fitness. It is valuable when you want the agents to say whether a patient’s functional capacity is low, average, or improved relative to relevant populations. [web:21][web:22][web:24][web:25]
- **Summary for RAG:** Use this for cardiorespiratory fitness percentiles, disease-category benchmarking, and interpretation of functional capacity relative to age, sex, and cardiovascular disease cohorts.

## 5. Borg Rating of Perceived Exertion scales
- **Type:** Exercise intensity / symptom self-report scale reference
- **Best use:** Nurse Agent, patient coaching, exercise-intensity language
- **Why include it:** Cardiac rehab is not only about objective numbers. RPE is central to matching physiologic state with patient-reported effort, fatigue, and breathlessness. This is especially important in beta-blocked patients where heart rate alone may underrepresent exertion. [web:26][web:28][web:33]
- **Summary for RAG:** Use this to map patient-friendly statements about effort to standard exertion scales and to align session coaching with accepted exercise-monitoring language.

---

# 2. Cardiac rehab clinical workflow corpus

These sources help the clinician-facing agents produce documentation and recommendations that look and feel like real cardiac rehab workflow outputs.

## 6. AACVPR cardiac program certification materials
- **Type:** Program standards / certification materials
- **Best use:** Duty Doctor Agent, Clinical Assistant Agent, program-structure grounding
- **Why include it:** These materials help the agents understand how real cardiac rehab programs structure assessment, intervention, education, reassessment, and outcomes. This is particularly useful if you want outputs that sound aligned with actual rehab operations rather than generic cardiology summaries. [web:13][web:29]
- **Summary for RAG:** Use this to structure language around admission screening, rehab goals, reassessment, symptom improvement, and program-quality expectations.

## 7. AACVPR cardiac ITP checklist
- **Type:** Individualized treatment plan structure
- **Best use:** SOAP-note generation, treatment-plan formatting, clinician documentation
- **Why include it:** If your MedGemma agents are expected to generate rehab notes that resemble chart-ready documentation, the ITP checklist is extremely useful. It includes the kinds of sections clinicians expect, such as psychosocial assessment, plan, goals, interventions, education, reassessment, and follow-up. [web:34]
- **Summary for RAG:** Use this to shape generated care plans and documentation into AACVPR-style structures rather than freeform prose.

## 8. AHA/ACC chronic coronary disease and related long-term management guidelines
- **Type:** Disease-management guideline family
- **Best use:** Clinical Assistant Agent, medication-aware reasoning, patient-context interpretation
- **Why include it:** Cardiac rehab patients are often on beta-blockers, ACE inhibitors, anticoagulants, statins, or antianginal therapy. The clinician-facing agents need source text that explains how long-term disease management interacts with exercise response and symptom interpretation.
- **Summary for RAG:** Use this to ground medication-aware reasoning, chronic disease context, and the interpretation of exercise findings in medically managed patients.

---

# 3. HRV and autonomic recovery corpus

These sources specifically strengthen the system’s language around recovery, autonomic function, and prognostic framing.

## 9. Heart rate variability in the prediction of mortality
- **Type:** Review / prognostic synthesis
- **Best use:** Duty Doctor Agent, Clinical Assistant Agent
- **Why include it:** This review supports the idea that lower HRV values are associated with higher mortality across populations and helps distinguish which classes of HRV measures matter for different outcomes. [web:37]
- **Summary for RAG:** Use this when the system needs to explain why reduced variability may be clinically meaningful, especially in longitudinal trend summaries.

## 10. Heart Rate Variability as a Predictor of Mortality in Heart Failure
- **Type:** Meta-analysis
- **Best use:** Clinical Assistant Agent, risk-context explanation
- **Why include it:** This is useful for clinician-facing context because it specifically links impaired HRV, especially lower SDNN, to mortality risk in heart failure populations and shows HRV’s value beyond conventional markers. [web:35][web:36]
- **Summary for RAG:** Use this for high-level prognostic language in appropriate heart-failure contexts, especially when comparing HRV trends over time.

## 11. Heart rate variability as predictive factor for sudden cardiac death
- **Type:** Review
- **Best use:** Clinical Assistant Agent, cautionary risk explanation
- **Why include it:** This source helps anchor any discussion of HRV as a risk-stratification tool, especially for explaining why chronically depressed HRV can matter clinically without overstating what a single rehab session proves. [web:39]
- **Summary for RAG:** Use this for careful language around HRV and cardiac risk, especially to avoid overclaiming while still preserving medical relevance.

## 12. The role of heart rate variability in prognosis for different modes of death
- **Type:** Review
- **Best use:** Foundation-model grounding, clinician explanations
- **Why include it:** This source helps differentiate which HRV domains are most informative in different prognostic contexts and supports more nuanced reasoning. [web:45]
- **Summary for RAG:** Use this when you need agent responses to distinguish global HRV suppression, sudden-death risk context, and broader mortality-risk interpretation.

---

# 4. VO2, functional capacity, and exercise-tolerance corpus

These sources help your agents discuss fitness, exercise capacity, and clinical performance during rehab.

## 13. FRIEND reference standards for cardiorespiratory fitness measured with CPX
- **Type:** Registry norms
- **Best use:** VO2 percentile framing, clinician benchmarking, report generation
- **Why include it:** This is one of the most practical texts for transforming VO2-related outputs into understandable percentile and category language. [web:22][web:25]
- **Summary for RAG:** Use this to contextualize fitness measures against normative datasets and to support percentile-based output in reports.

## 14. Reference standards for cardiorespiratory fitness by cardiovascular disease category
- **Type:** CVD-specific reference standards
- **Best use:** Clinician-facing benchmarking
- **Why include it:** This is stronger than generic fitness norms when you are working in a cardiac rehab setting, because it provides disease-category-specific standards and comparisons to apparently healthy individuals. [web:21]
- **Summary for RAG:** Use this for clinically relevant benchmarking of CRF in CABG, MI, PCI, and HF populations.

## 15. 6-minute walking test: a useful tool in the management of heart failure
- **Type:** Review
- **Best use:** Functional-capacity explanation, clinic workflow support
- **Why include it:** The 6MWT is common, practical, and highly relevant in rehab settings where not every patient undergoes full CPX. It gives the system a way to discuss submaximal functional capacity in clinically familiar language. [web:40]
- **Summary for RAG:** Use this to relate wearable-derived session behavior to broader ideas of daily-life functional capacity and submaximal exercise tolerance.

## 16. ATS Guidelines for the Six-Minute Walk Test
- **Type:** Practical test guideline
- **Best use:** Clinical Assistant Agent, protocol grounding
- **Why include it:** This is useful if your system ever references walking performance, rehabilitation function, or standardized submaximal exercise testing methods. [web:46]
- **Summary for RAG:** Use this for protocol-aware language, standardization, and interpretation boundaries around 6MWT discussions.

## 17. Six-Minute Walk Test and Cardiopulmonary Exercise Testing in Heart Failure
- **Type:** Comparative study
- **Best use:** nuanced clinician responses
- **Why include it:** This source helps the system explain the difference between a simple functional test and deeper prognostic variables from CPX. It is especially useful to stop the model from overclaiming what walking distance alone proves. [web:43][web:48]
- **Summary for RAG:** Use this to teach the model that 6MWT is practical and useful, but CPX-derived variables remain stronger prognostic markers.

---

# 5. Exercise prescription and safety corpus

These sources make the system safer and more practical for in-clinic rehab.

## 18. Borg scale / modified Borg exertion documents
- **Type:** Exercise-intensity monitoring aids
- **Best use:** Nurse Agent, exercise prescription language
- **Why include it:** In cardiac rehab, patient effort matters as much as measured intensity. Borg documents help the system align subjective effort with monitored physiologic response. [web:26][web:28][web:33]
- **Summary for RAG:** Use this for phrases like “moderate effort,” “somewhat hard,” or “strong exertion,” especially when heart-rate response is medication-modified.

## 19. Safety and Outcomes of Cardiac Rehabilitation for Patients with SCAD
- **Type:** Safety/outcomes study
- **Best use:** Clinician-facing safety narratives
- **Why include it:** Even though it is disease-specific, it gives concrete text showing how rehab protocols can be individualized and monitored safely, including aerobic plus resistance training progression and blood-pressure monitoring. [web:41]
- **Summary for RAG:** Use this as an example source for how structured rehab programs can safely progress exercise and monitor outcomes.

## 20. Safety of Exercise Training for Cardiac Patients
- **Type:** Historical safety framing
- **Best use:** general safety context
- **Why include it:** This gives a broader background for how supervised exercise safety is framed in cardiac populations. [web:44]
- **Summary for RAG:** Use this to support the safety rationale for monitored rehab rather than unsupervised exertion.

---

# 6. Foundation-model and retrieval-support corpus

These sources are especially helpful for grounding outputs from your ECG foundation-model and similarity-retrieval layer.

## 21. PhysioNet dataset metadata and label documentation
- **Type:** Dataset documentation
- **Best use:** retrieval explanation, Clinical Assistant Agent, auditability
- **Why include it:** If your system retrieves similar rhythms or morphology patterns from datasets like MIT-BIH, the agents need text explaining what those datasets are, what labels they include, and what types of rhythms they contain.
- **Summary for RAG:** Use this so the model can explain retrieval provenance clearly and avoid hallucinating what a dataset contains.

## 22. CLEF foundation-model paper text
- **Type:** model description / representation-learning reference
- **Best use:** system self-explanation, clinician trust-building
- **Why include it:** Since your pipeline uses ECG embeddings for similarity search, the model should be able to explain in simple language what the embedding captures and why nearest-neighbor retrieval is clinically useful.
- **Summary for RAG:** Use this to describe single-lead ECG embedding space, similarity-based reasoning, and the idea of comparing a patient to learned physiologic reference patterns.

## 23. ECG-FM model paper text
- **Type:** model description / diagnostic support reference
- **Best use:** technical justification, clinician-facing interpretability
- **Why include it:** This source helps explain what classifier-like outputs are available beyond raw embeddings and can support careful language around arrhythmia-class support and morphology interpretation.
- **Summary for RAG:** Use this when the system needs to explain how a secondary ECG model supports label-oriented interpretation.

---

# 7. How to split these sources across agents

## Duty Doctor Agent
Use the most formal, clinical, and guideline-heavy corpus:
- Core Components of Cardiac Rehabilitation Programs: 2024 Update
- AACVPR program/certification materials
- AACVPR ITP checklist
- HRV standards statement
- HRV prognosis reviews and meta-analyses
- FRIEND standards
- 6MWT guidance
- disease-management guidelines

**Why:** This agent writes structured summaries, reviews trends, and generates SOAP-like notes. It needs formal medical language and workflow structure.

## Clinical Assistant Agent
Use all Duty Doctor sources plus retrieval-specific and model-explanation sources:
- Core Components 2024
- HRV standards and prognostic papers
- FRIEND standards
- 6MWT and CPX comparison papers
- PhysioNet metadata
- CLEF and ECG-FM text
- medication / chronic coronary disease guidance

**Why:** This agent must answer nuanced clinician questions and explain why the system reached a conclusion.

## Nurse Agent
Use a smaller, safer, simplified corpus:
- Core Components 2024
- Borg scales
- selected rehab education materials
- simplified exercise-safety text
- limited VO2 / recovery interpretation text
- carefully filtered HRV explanations

**Why:** This agent should coach, encourage, and educate without sounding like a physician or overinterpreting risk.

---

# 8. Practical ingestion strategy

## Tier 1: ingest immediately
1. Core Components of Cardiac Rehabilitation Programs: 2024 Update
2. HRV standards statement
3. VO2max from wearables validation paper
4. FRIEND standards
5. Borg scale documents
6. AACVPR ITP checklist

## Tier 2: ingest next
7. HRV mortality/prognosis reviews
8. 6MWT guideline and review papers
9. AACVPR certification/program materials
10. chronic coronary disease / medication-management guidelines

## Tier 3: specialized grounding
11. PhysioNet metadata
12. CLEF paper text
13. ECG-FM paper text
14. disease-specific rehab safety papers

---

# 9. Best final shortlist

If you want the smallest high-value corpus to start with, I would begin with these 10:

1. Core Components of Cardiac Rehabilitation Programs: 2024 Update
2. HRV Standards of Measurement, Physiological Interpretation, and Clinical Use
3. Validation of a Neural Network for Estimating VO2max from Wearables
4. FRIEND reference standards for cardiorespiratory fitness
5. Reference standards for cardiorespiratory fitness by cardiovascular disease category
6. Borg Rating of Perceived Exertion scales
7. AACVPR cardiac ITP checklist
8. Heart rate variability in the prediction of mortality
9. 6-minute walking test review / ATS 6MWT guideline
10. PhysioNet dataset metadata for the ECG retrieval datasets

That gives you a compact but very strong corpus for:
- rehab program logic,
- HRV interpretation,
- VO2 and fitness context,
- patient coaching,
- report generation,
- and retrieval transparency.