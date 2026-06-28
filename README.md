# Redrob Candidate Ranker

A two-phase CPU-only pipeline that ranks 100,000 candidates against a Senior AI Engineer job description using BM25 text matching, structured feature scoring, behavioral signal analysis, and honeypot detection — producing a top-100 submission CSV in under 5 seconds at inference time.

---

## Approach

Two-phase pipeline separates heavy computation from timed ranking.

**Phase 1 — `precompute.py` (no time limit):** Streams all 100K candidates, extracts four feature scores per candidate (JD match via BM25 + structured skills, experience & company background, behavioral availability, location), builds a BM25 index over concatenated career text, runs 7-rule honeypot detection, and saves compact artifacts to disk (~230MB total).

**Phase 2 — `rank.py` (≤5 min, CPU only, no network):** Loads artifacts, computes final scores as vectorized numpy operations in under 1 second, generates template-based reasoning using only real profile facts (no LLM), writes submission CSV.

**Scoring formula:**
```
jd_match = 0.65 × bm25_score + 0.35 × skills_score
base     = 0.50 × jd_match + 0.35 × experience_score + 0.15 × location_score
final    = base × (0.40 + 0.60 × behavioral_score) × honeypot_penalty
```

Honeypot detection uses 7 rules covering impossible timelines, title-description mismatches, and implausible skill durations. Reasoning is assembled from real data fields only — no hallucination possible.

---

## Setup

### Prerequisites
- Python 3.10+
- 16 GB RAM recommended for full 100K dataset
- CPU only for the ranking step (`rank.py`)
- Pre-computation (`precompute.py`) can use any hardware

### Install dependencies
```bash
pip install -r requirements.txt
```

### Directory structure
```
redrob-ranker/
├── README.md
├── requirements.txt
├── submission_metadata.yaml
├── precompute.py
├── rank.py
├── app.py
├── validate_submission.py        ← provided by competition organisers
├── team_xxx.csv                  ← final submission (The Results)
│
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── honeypot.py
│   ├── bm25_index.py
│   ├── signals.py
│   └── reasoning.py
│
├── data/
│   ├── candidates.jsonl          ← NOT in git (465MB)
│   ├── sample_candidates.json    ← in git (used by sandbox)
│   ├── sample_submission.csv     ← in git (format reference)
│   └── candidate_schema.json     ← in git (reference)
│
├── artifacts/                    ← NOT in git (~230MB, regenerate with precompute.py)
│   ├── candidate_ids.json
│   ├── feature_matrix.npy
│   ├── bm25_index.pkl
│   ├── signal_summaries.json
│   └── candidate_meta.json
│
└── docs/                         ← NOT in git (competition docs)
    ├── job_description.docx
    ├── redrob_signals_doc.docx
    ├── submission_spec.docx
    └── README.docx
```

---

## Reproducing the Submission

### Step 1 — Pre-computation (run once, no time limit)

Place `candidates.jsonl` in the `data/` folder first, then:

```bash
python precompute.py --candidates ./data/candidates.jsonl --artifacts ./artifacts/
```

Expected output:
```
Loading candidates from ./data/candidates.jsonl...
100000 candidates found.
Feature extraction: 100000 candidates in ~180s
BM25 index built and saved.
All artifacts saved to ./artifacts/
Honeypot knockouts: ~75 / 100000
precompute.py completed in ~8.5 minutes.
```

Artifacts produced (`./artifacts/`):

| File | Size | Contents |
|---|---|---|
| `candidate_ids.json` | ~4 MB | Ordered list of 100K candidate IDs |
| `feature_matrix.npy` | ~2.4 MB | Float32 array (100000 × 6): skills, exp, behavioral, location, honeypot, bm25 |
| `bm25_index.pkl` | ~150 MB | Pickled BM25Okapi index over career text |
| `signal_summaries.json` | ~45 MB | Human-readable signal labels per candidate |
| `candidate_meta.json` | ~30 MB | Lightweight profile dict per candidate |

### Step 2 — Ranking (the timed step, ≤5 minutes)

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./team_xxx.csv
```

Expected output:
```
Loading artifacts...        done (2.1s)
Scoring 100K candidates...  done (0.3s)
Generating reasoning...     done (0.8s)
Writing CSV...              done (0.1s)
Validation passed ✓
Done. Total time: 3.3s
```

> **Note:** `--candidates` is required by the submission spec but is not read during ranking — all scoring uses precomputed artifacts. The file must exist on disk.

### Step 3 — Validate before submitting

```bash
python validate_submission.py ./team_xxx.csv
```

This is the same validator provided by competition organisers. Run it before every upload.

---

## Running the Sandbox (Streamlit)

The sandbox is hosted at: `{FILL_IN — your Streamlit Cloud / HuggingFace URL}`

To run locally:
```bash
streamlit run app.py
# Navigate to http://localhost:8501
```

Upload a candidates `.json` or `.jsonl` file (≤100 candidates), or click **"Load Sample Candidates"** to use the bundled `data/sample_candidates.json`. The sandbox runs the full pipeline inline on the small sample and produces a downloadable ranked CSV.

---

## Scoring Formula

```
# JD text match (BM25 over career text + structured skills)
jd_match_score = 0.65 × bm25_score + 0.35 × skills_score

# Base score (three components)
base_score = 0.50 × jd_match_score
           + 0.35 × experience_score
           + 0.15 × location_score

# Availability multiplier from 23 behavioral signals (range: 0.40 → 1.00)
availability_multiplier = 0.40 + 0.60 × behavioral_score

# Final score — honeypot_penalty is 0.0 for knockout candidates, 1.0 for clean
final_score = base_score × availability_multiplier × honeypot_penalty
```

All component scores are in [0, 1]. A honeypot candidate gets `final_score = 0.0` and cannot appear in the top 100.

---

## Honeypot Detection

Seven rules are applied per candidate. Any knockout rule sets `honeypot_penalty = 0.0`.

| Rule | Type | Description |
|---|---|---|
| `SKILL_DURATION_EXCEEDS_CAREER` | Knockout | 2+ skills with `duration_months` > 120% of total career months |
| `TITLE_DESC_MISMATCH` | Knockout | Career role descriptions describe a completely different job than the title |
| `TIMELINE_REVERSED` | Knockout | A role's `end_date` is earlier than its `start_date` |
| `TIMELINE_OVERLAP` | Soft | Two non-current roles overlap by more than 3 months |
| `EXPERT_ZERO_DURATION` | Soft | 5+ skills with `proficiency=expert` and `duration_months=0` |
| `ASSESSMENT_PROFICIENCY_MISMATCH` | Soft | Claims "expert" but platform assessment score < 40 on 3+ skills |
| `IMPLAUSIBLE_TOTAL_SKILL_DURATION` | Soft | Sum of all skill durations > 8× total career months |

Soft rules apply a multiplicative penalty (0.4–0.85) rather than a full knockout.

---

## Feature Components

### Skills Score
Structured scoring over `skills[]` list: each skill scored by proficiency weight (0.25–1.0), duration (capped at 36 months), endorsements (capped at 30), and platform assessment score if available. Priority skills from the JD (FAISS, Pinecone, embeddings, NLP, etc.) weighted 2×. Combined with BM25 score over full career text.

### Experience Score
Four sub-scores: years of experience (peak 5–9 years), company type (BIG_TECH → 1.0, PRODUCT → 0.9, CONSULTING → 0.15), ML role ratio (fraction of career months in ML/AI/search titles), average tenure per role (anti-title-chaser).

### Behavioral Score
Six signals from `redrob_signals`: recency of last activity, open-to-work flag, recruiter response rate + response time + application activity (composite), notice period, interview completion rate, GitHub activity score.

### Location Score
India + major tech cities (Bangalore, Hyderabad, Pune, etc.) → 0.95. India + willing to relocate → 0.80. Outside India + willing to relocate → 0.40. Outside India, not relocating → 0.15.

---

## AI Tools Declaration

Claude (claude.ai) was used for architecture design discussion, scoring formula refinement, and generating initial drafts of each module. GitHub Copilot was used for autocomplete during development. All module code was reviewed, adapted, and tested by the team. No candidate data was passed to any external LLM. `rank.py` makes zero network calls — all scoring uses precomputed artifacts and pure Python/numpy logic.

See `submission_metadata.yaml` for full declaration.

---

## File Sizes (Approximate)

| File | Size |
|---|---|
| `data/candidates.jsonl` | ~465 MB |
| `artifacts/bm25_index.pkl` | ~150 MB |
| `artifacts/signal_summaries.json` | ~45 MB |
| `artifacts/candidate_meta.json` | ~30 MB |
| `artifacts/feature_matrix.npy` | ~2.4 MB |
| `artifacts/candidate_ids.json` | ~4 MB |
| **Total artifacts** | **~231 MB** (well within 5 GB limit) |
