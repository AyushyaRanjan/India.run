"""
src/honeypot.py — Honeypot & impossible-profile detection.

The competition dataset contains ~80 fabricated "honeypot" candidates.
Submissions with honeypot rate > 10% in top 100 → DISQUALIFIED at Stage 3.

KEY INSIGHT from data analysis:
  skill.duration_months includes PRE-PROFESSIONAL time (college, bootcamps,
  side projects), but profile.years_of_experience is PROFESSIONAL ONLY.
  A candidate with 1yr professional experience can legitimately have Python
  for 36 months (used in college). We add PRE_CAREER_BUFFER_MONTHS = 60
  before any skill-duration comparison to avoid false positives.

TARGET: ~50-150 knockouts out of 100K (competition says ~80 honeypots exist).
  Better to miss a genuine honeypot than to knockout a real candidate.
"""

import re
from datetime import datetime

# =============================================================================
# THRESHOLDS — all named constants so they're easy to tune
# =============================================================================

# Skill duration: add this to pro career months before comparing.
# Accounts for college + bootcamp + side-project skill usage.
PRE_CAREER_BUFFER_MONTHS = 60           # 5 years pre-professional buffer

# Rule 1 — Skill duration vs effective career
SKILL_HARD_MULTIPLIER   = 3.0           # per-skill: > 3× effective max = very suspicious
SKILL_SOFT_MULTIPLIER   = 2.0           # per-skill: > 2× effective max = mild flag
SKILL_HARD_MIN_COUNT    = 4             # need 4+ hard-suspicious skills → knockout

# Rule 6 — Total skill duration (sum of ALL skills) vs effective career
# With 15 skills used in parallel, total = 15× career — perfectly normal.
TOTAL_SKILL_MULTIPLIER  = 35.0          # only flag truly absurd totals

# Rule 4 — Expert skills with zero duration months (spec explicitly mentions this)
# Competition spec says: "expert proficiency in 10 skills with 0 years used" is a honeypot.
EXPERT_ZERO_KNOCKOUT    = 10            # 10+ expert skills ALL with 0 months → knockout
EXPERT_ZERO_SOFT        = 6             # 6-9  expert skills with 0 months → soft penalty

# Rule 5 — Assessment score vs self-reported proficiency
ASSESSMENT_FAIL_SCORE   = 35            # score below this on an "expert" claim = mismatch
ASSESSMENT_MISMATCH_MIN = 8             # need 8+ mismatches for soft penalty

# Rule 2 — Title-description domain mismatch
# How many keywords from the WRONG domain must appear in a role description.
MISMATCH_KW_THRESHOLD   = 3             # 3+ wrong-domain keywords in a role = mismatch
MISMATCH_ROLE_KNOCKOUT  = 3             # 3+ mismatched roles → knockout
MISMATCH_ROLE_SOFT      = 2             # 2   mismatched roles → 0.5 penalty
MISMATCH_ROLE_SINGLE    = 1             # 1   mismatched role  → 0.8 penalty (mild)

# Rule 3c — Timeline overlaps (soft signal — some people freelance in parallel)
OVERLAP_MIN_MONTHS      = 4             # only count overlaps > 4 months
OVERLAP_SOFT_COUNT      = 5             # 5+ overlapping role pairs → soft penalty

# Rule 7 — Unexplained career gap
GAP_MONTHS_THRESHOLD    = 48            # gaps > 4 years trigger a mild soft flag

# =============================================================================
# KEYWORD SETS FOR TITLE-DESCRIPTION MISMATCH
# =============================================================================

# Job titles that belong to non-technical domains.
# A description for these roles should NOT have heavy ML / hardware / writing content.
NON_TECH_TITLES = {
    "marketing manager", "operations manager", "hr manager",
    "human resources manager", "sales manager", "account manager",
    "project manager", "business analyst", "content writer",
    "seo specialist", "accountant", "finance manager",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "graphic designer", "customer success manager",
    "business development manager", "supply chain manager",
    "procurement manager", "logistics manager"
}

# Strongly ML/AI-specific phrases unlikely to appear in non-tech job descriptions
ML_KEYWORDS = {
    "machine learning", "neural network", "deep learning",
    "tensorflow", "pytorch", "embedding pipeline",
    "vector database", "recommendation system", "ranking model",
    "faiss index", "pinecone", "transformer model",
    "bert", "large language model", "generative ai"
}

# Hardware engineering phrases that shouldn't appear in a marketing/hr/sales role
HARDWARE_KEYWORDS = {
    "solidworks", "creo", "ansys", "dfm", "dfma",
    "production tooling", "fea simulation", "cnc machining",
    "mechanical design", "hardware prototype", "injection moulding"
}

# Content/writing phrases that shouldn't appear in an operations/engineering role
WRITING_KEYWORDS = {
    "seo strategy", "content brief", "editorial calendar",
    "copywriting", "longform", "editorial standards",
    "press release", "journalist", "blog articles"
}


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def get_honeypot_penalty(candidate: dict) -> tuple[float, list[str]]:
    """
    Evaluate a candidate for honeypot / impossible-profile signals.

    Returns:
        (penalty_multiplier, flags)
        penalty_multiplier : 0.0 = hard knockout (final_score → 0)
                             0.4–0.9 = soft penalty
                             1.0 = clean, no issues found
        flags : list of short uppercase tags for logging / debugging
    """
    penalty = 1.0
    flags   = []

    profile        = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills         = candidate.get("skills", [])
    signals        = candidate.get("redrob_signals", {})

    pro_months       = profile.get("years_of_experience", 0.0) * 12
    effective_max    = pro_months + PRE_CAREER_BUFFER_MONTHS  # includes pre-career

    # ------------------------------------------------------------------
    # RULE 3a: TIMELINE REVERSED  — hard knockout, provably impossible
    # ------------------------------------------------------------------
    for role in career_history:
        sd = role.get("start_date")
        ed = role.get("end_date")
        if not sd or not ed:
            continue
        try:
            s = datetime.strptime(sd, "%Y-%m-%d")
            e = datetime.strptime(ed, "%Y-%m-%d")
            if e < s:
                flags.append("TIMELINE_REVERSED")
                return 0.0, flags
        except (ValueError, TypeError):
            continue

    # ------------------------------------------------------------------
    # RULE 1: SKILL DURATION EXCEEDS EFFECTIVE CAREER
    # ------------------------------------------------------------------
    hard_dur_count = 0
    soft_dur_count = 0
    sum_skill_months = 0

    for skill in skills:
        dur = skill.get("duration_months", 0)
        sum_skill_months += dur
        if effective_max > 0:
            ratio = dur / effective_max
            if ratio > SKILL_HARD_MULTIPLIER:
                hard_dur_count += 1
            elif ratio > SKILL_SOFT_MULTIPLIER:
                soft_dur_count += 1

    if hard_dur_count >= SKILL_HARD_MIN_COUNT:
        flags.append("SKILL_DURATION_EXCEEDS_CAREER")
        return 0.0, flags
    elif hard_dur_count >= 2 or soft_dur_count >= 8:
        flags.append("SKILL_DURATION_SUSPICIOUS")
        penalty = min(penalty, 0.75)

    # ------------------------------------------------------------------
    # RULE 6: IMPLAUSIBLE TOTAL SKILL DURATION
    # ------------------------------------------------------------------
    if effective_max > 0 and sum_skill_months / effective_max > TOTAL_SKILL_MULTIPLIER:
        flags.append("IMPLAUSIBLE_TOTAL_SKILL_DURATION")
        penalty = min(penalty, 0.75)

    # ------------------------------------------------------------------
    # RULE 4: EXPERT SKILLS WITH ZERO DURATION
    # Competition spec explicitly calls out: "expert in 10 skills, 0 years used"
    # We skip skills that have a platform assessment score — those are verified.
    # ------------------------------------------------------------------
    assessments  = signals.get("skill_assessment_scores", {})
    assess_lower = {k.lower(): v for k, v in assessments.items()}

    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency", "").lower() == "expert"
        and s.get("duration_months", 0) == 0
        and s.get("name", "").lower() not in assess_lower
    )

    if expert_zero >= EXPERT_ZERO_KNOCKOUT:
        flags.append("EXPERT_ZERO_DURATION")
        return 0.0, flags
    elif expert_zero >= EXPERT_ZERO_SOFT:
        flags.append("EXPERT_ZERO_DURATION_SOFT")
        penalty = min(penalty, 0.65)

    # ------------------------------------------------------------------
    # RULE 5: ASSESSMENT SCORE vs CLAIMED PROFICIENCY
    # ------------------------------------------------------------------
    if assess_lower:
        mismatches = sum(
            1 for s in skills
            if s.get("proficiency", "").lower() == "expert"
            and s.get("name", "").lower() in assess_lower
            and assess_lower[s.get("name", "").lower()] < ASSESSMENT_FAIL_SCORE
        )
        if mismatches >= ASSESSMENT_MISMATCH_MIN:
            flags.append("ASSESSMENT_PROFICIENCY_MISMATCH")
            penalty = min(penalty, 0.75)

    # ------------------------------------------------------------------
    # RULE 2: TITLE-TO-DESCRIPTION DOMAIN MISMATCH
    # Classic honeypot: Marketing Manager whose description is about FAISS
    # embeddings, or mechanical engineering, or content writing strategy.
    # ------------------------------------------------------------------
    mismatch_count = 0

    for role in career_history:
        title = role.get("title", "").lower()
        desc  = role.get("description", "").lower()

        is_non_tech = any(nt in title for nt in NON_TECH_TITLES)
        if not is_non_tech:
            continue

        ml_hits = sum(1 for kw in ML_KEYWORDS if kw in desc)
        hw_hits = sum(1 for kw in HARDWARE_KEYWORDS if kw in desc)
        wr_hits = sum(1 for kw in WRITING_KEYWORDS if kw in desc)

        # Is this title normally associated with hardware or writing?
        # If so, those keyword hits are NOT mismatches.
        is_hw_title = any(t in title for t in ["mechanical", "civil", "electrical"])
        is_wr_title = any(t in title for t in ["content", "seo", "marketing", "sales", "copywriting"])

        mismatch = False
        if ml_hits >= MISMATCH_KW_THRESHOLD:
            mismatch = True   # Non-tech role heavy on ML jargon
        elif hw_hits >= MISMATCH_KW_THRESHOLD and not is_hw_title:
            mismatch = True   # Non-hardware role with hardware engineering detail
        elif wr_hits >= MISMATCH_KW_THRESHOLD and not is_wr_title:
            mismatch = True   # Non-writing role with content/SEO detail

        if mismatch:
            mismatch_count += 1

    if mismatch_count >= MISMATCH_ROLE_KNOCKOUT:
        flags.append("TITLE_DESC_MISMATCH_MULTIPLE")
        return 0.0, flags
    elif mismatch_count >= MISMATCH_ROLE_SOFT:
        flags.append("TITLE_DESC_MISMATCH")
        penalty = min(penalty, 0.50)
    elif mismatch_count >= MISMATCH_ROLE_SINGLE:
        flags.append("TITLE_DESC_MISMATCH_SINGLE")
        penalty = min(penalty, 0.80)

    # ------------------------------------------------------------------
    # RULE 3b: TIMELINE OVERLAPS  — soft signal only
    # Some candidates genuinely freelance/consult alongside a full-time role.
    # Only flag truly excessive overlap counts.
    # ------------------------------------------------------------------
    parsed = []
    for role in career_history:
        sd = role.get("start_date")
        ed = role.get("end_date")
        if not sd:
            continue
        try:
            s = datetime.strptime(sd, "%Y-%m-%d")
            e = datetime.strptime(ed, "%Y-%m-%d") if ed else None
            parsed.append({"s": s, "e": e, "cur": role.get("is_current", False)})
        except (ValueError, TypeError):
            continue

    parsed.sort(key=lambda x: x["s"])
    overlap_count = 0

    for i in range(len(parsed) - 1):
        a, b = parsed[i], parsed[i + 1]
        # Only check two completed (non-current) roles
        if a["e"] and b["e"] and not a["cur"] and not b["cur"]:
            overlap_days = (min(a["e"], b["e"]) - max(a["s"], b["s"])).days
            if overlap_days > OVERLAP_MIN_MONTHS * 30:
                overlap_count += 1

    if overlap_count >= OVERLAP_SOFT_COUNT:
        flags.append("TIMELINE_OVERLAP")
        penalty = min(penalty, 0.75)

    # ------------------------------------------------------------------
    # RULE 7: LARGE UNEXPLAINED CAREER GAP  — very soft, one flag only
    # ------------------------------------------------------------------
    for i in range(len(parsed) - 1):
        a, b = parsed[i], parsed[i + 1]
        if a["e"]:
            gap_days = (b["s"] - a["e"]).days
            if gap_days > GAP_MONTHS_THRESHOLD * 30:
                flags.append("LARGE_UNEXPLAINED_GAP")
                penalty = min(penalty, 0.90)
                break

    return penalty, flags


# =============================================================================
# DIAGNOSTICS HELPER
# =============================================================================

def run_diagnostics(candidates: list[dict]) -> dict:
    """
    Run honeypot detection over a list and return a calibration summary.
    Call this after precompute.py to check your knockout count is reasonable.
    """
    knockout_count = 0
    flag_freq: dict[str, int] = {}

    for c in candidates:
        pen, flags = get_honeypot_penalty(c)
        if pen == 0.0:
            knockout_count += 1
        for f in flags:
            flag_freq[f] = flag_freq.get(f, 0) + 1

    return {
        "total":          len(candidates),
        "knockouts":      knockout_count,
        "knockout_pct":   round(knockout_count / max(len(candidates), 1) * 100, 3),
        "flag_frequency": dict(sorted(flag_freq.items(), key=lambda x: -x[1]))
    }


# =============================================================================
# SELF TEST
# =============================================================================

if __name__ == "__main__":

    tests = [
        # 1. Clean candidate — no flags expected
        ("Clean ML engineer", {
            "candidate_id": "T01",
            "profile": {"years_of_experience": 6.0},
            "career_history": [{
                "title": "ML Engineer", "is_current": True,
                "description": "Built ranking systems using PyTorch and FAISS.",
                "start_date": "2018-01-01", "end_date": None, "duration_months": 72
            }],
            "skills": [
                {"name": "Python", "proficiency": "expert", "duration_months": 72},
                {"name": "PyTorch", "proficiency": "advanced", "duration_months": 48},
            ],
            "redrob_signals": {"skill_assessment_scores": {"Python": 88}}
        }),

        # 2. Expert-zero-duration honeypot (spec's exact example: 10 expert, 0 months)
        ("Spec honeypot: 10 expert skills, 0 months each", {
            "candidate_id": "T02",
            "profile": {"years_of_experience": 5.0},
            "career_history": [{
                "title": "Senior AI Engineer", "is_current": True,
                "description": "General AI work.",
                "start_date": "2019-01-01", "end_date": None, "duration_months": 60
            }],
            "skills": [
                {"name": f"Skill{i}", "proficiency": "expert", "duration_months": 0}
                for i in range(10)
            ],
            "redrob_signals": {"skill_assessment_scores": {}}
        }),

        # 3. Title-description mismatch × 3 roles → hard knockout
        ("Honeypot: 3 domain-mismatch roles", {
            "candidate_id": "T03",
            "profile": {"years_of_experience": 4.0},
            "career_history": [
                {
                    "title": "Marketing Manager", "is_current": False,
                    "description": "Trained ranking model using FAISS embedding pipeline and transformer model for recommendation system.",
                    "start_date": "2020-01-01", "end_date": "2021-06-01", "duration_months": 18
                },
                {
                    "title": "Operations Manager", "is_current": False,
                    "description": "Used SolidWorks and ANSYS for dfm and production tooling design on hardware prototype.",
                    "start_date": "2021-07-01", "end_date": "2023-01-01", "duration_months": 18
                },
                {
                    "title": "Business Analyst", "is_current": True,
                    "description": "Wrote longform blog articles, managed editorial calendar, ran seo strategy campaigns.",
                    "start_date": "2023-02-01", "end_date": None, "duration_months": 16
                }
            ],
            "skills": [],
            "redrob_signals": {"skill_assessment_scores": {}}
        }),

        # 4. Reversed timeline → hard knockout
        ("Honeypot: reversed end/start date", {
            "candidate_id": "T04",
            "profile": {"years_of_experience": 3.0},
            "career_history": [{
                "title": "Data Scientist", "is_current": False,
                "description": "ML work.",
                "start_date": "2023-01-01", "end_date": "2021-01-01",
                "duration_months": 24
            }],
            "skills": [],
            "redrob_signals": {"skill_assessment_scores": {}}
        }),

        # 5. Borderline — junior dev with skills used during college
        ("Borderline: 1yr pro exp, skills from college (should PASS)", {
            "candidate_id": "T05",
            "profile": {"years_of_experience": 1.5},
            "career_history": [{
                "title": "Junior ML Engineer", "is_current": True,
                "description": "Deployed ML models to production.",
                "start_date": "2023-01-01", "end_date": None, "duration_months": 18
            }],
            "skills": [
                {"name": "Python", "proficiency": "advanced", "duration_months": 48},
                {"name": "TensorFlow", "proficiency": "intermediate", "duration_months": 36},
            ],
            "redrob_signals": {"skill_assessment_scores": {}}
        }),

        # 6. Single title-description mismatch → mild penalty only
        ("Single mismatch: mild penalty only", {
            "candidate_id": "T06",
            "profile": {"years_of_experience": 5.0},
            "career_history": [
                {
                    "title": "Marketing Manager", "is_current": False,
                    "description": "Ran FAISS embedding pipeline and transformer model with ranking model and recommendation system deep learning.",
                    "start_date": "2019-01-01", "end_date": "2022-01-01", "duration_months": 36
                },
                {
                    "title": "ML Engineer", "is_current": True,
                    "description": "Built search systems using embeddings and vector databases.",
                    "start_date": "2022-02-01", "end_date": None, "duration_months": 28
                }
            ],
            "skills": [],
            "redrob_signals": {"skill_assessment_scores": {}}
        }),
    ]

    print("=" * 65)
    print(f"{'Test':<45} {'Result':<20} {'Flags'}")
    print("=" * 65)

    for label, cand in tests:
        pen, flags = get_honeypot_penalty(cand)
        status = "KNOCKOUT ❌" if pen == 0.0 else f"penalty={pen:.2f} ✓"
        flag_str = ", ".join(flags) if flags else "none"
        print(f"{label[:44]:<45} {status:<20} {flag_str}")

    # Diagnostics on sample data if available
    import os, json
    sample_path = "./data/sample_candidates.json"
    if not os.path.exists(sample_path):
        sample_path = "/mnt/user-data/uploads/sample_candidates.json"

    if os.path.exists(sample_path):
        with open(sample_path) as f:
            sample = json.load(f)
        diag = run_diagnostics(sample)
        print()
        print("=" * 65)
        print(f"DIAGNOSTICS — {sample_path}")
        print(f"  Total        : {diag['total']}")
        print(f"  Knockouts    : {diag['knockouts']} ({diag['knockout_pct']}%)")
        print(f"  Flag counts  : {diag['flag_frequency']}")
        print()
        print("  Soft-penalised candidates:")
        for c in sample:
            pen, flags = get_honeypot_penalty(c)
            if 0.0 < pen < 1.0:
                p = c['profile']
                print(f"    {c['candidate_id']} | {p['current_title'][:35]} | pen={pen:.2f} | {flags}")