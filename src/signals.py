# src/signals.py
from datetime import datetime, date

def compute_behavioral_score(signals: dict) -> float:
    """
    Computes a comprehensive behavioral score (0.0 to 1.0) based on platform
    engagement and availability signals.
    """
    today = date.today()
    
    # 1. Recency Score
    last_active_str = signals.get("last_active_date")
    recency_score = 0.20  # Default if missing or unparseable
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_since = (today - last_active).days
            
            if days_since <= 14: recency_score = 1.00
            elif days_since <= 30: recency_score = 0.90
            elif days_since <= 60: recency_score = 0.75
            elif days_since <= 90: recency_score = 0.55
            elif days_since <= 180: recency_score = 0.30
            else: recency_score = 0.10
        except (ValueError, TypeError):
            pass

    # 2. Open to Work Score
    open_to_work = signals.get("open_to_work_flag")
    open_to_work_score = 1.00 if open_to_work is True else 0.35

    # 3. Response Score
    rr_raw = signals.get("recruiter_response_rate")
    rr = float(rr_raw) if rr_raw is not None else 0.50
    
    rt_raw = signals.get("avg_response_time_hours")
    if rt_raw is None:
        rt_score = 0.50
    else:
        rt = float(rt_raw)
        if rt <= 2: rt_score = 1.00
        elif rt <= 6: rt_score = 0.85
        elif rt <= 24: rt_score = 0.65
        elif rt <= 72: rt_score = 0.40
        else: rt_score = 0.15
        
    apps_raw = signals.get("applications_submitted_30d")
    apps = int(apps_raw) if apps_raw is not None else 0
    engagement_bonus = min(apps / 5.0, 1.0)
    
    response_score = (0.50 * rr) + (0.35 * rt_score) + (0.15 * engagement_bonus)

    # 4. Notice Score
    notice_raw = signals.get("notice_period_days")
    if notice_raw is None:
        notice_score = 0.35  # Assume 90 days if unknown as safe default
    else:
        notice = int(notice_raw)
        if notice == 0: notice_score = 1.00
        elif notice <= 15: notice_score = 0.95
        elif notice <= 30: notice_score = 0.85
        elif notice <= 45: notice_score = 0.70
        elif notice <= 60: notice_score = 0.55
        elif notice <= 90: notice_score = 0.35
        else: notice_score = 0.15

    # 5. Interview Score
    int_raw = signals.get("interview_completion_rate")
    interview_score = float(int_raw) if int_raw is not None else 0.50

    # 6. GitHub Score
    gh_raw = signals.get("github_activity_score")
    if gh_raw is None or float(gh_raw) == -1:
        github_score = 0.30
    else:
        github_score = float(gh_raw) / 100.0

    # Final Weighted Average
    behavioral_score = (
        0.30 * recency_score +
        0.15 * open_to_work_score +
        0.20 * response_score +
        0.15 * notice_score +
        0.10 * interview_score +
        0.10 * github_score
    )
    
    return min(max(behavioral_score, 0.0), 1.0)


def extract_signal_summary(signals: dict) -> dict:
    """
    Extracts a structured, human-readable summary of candidate behavioral
    signals for downstream reasoning text generation.
    """
    today = date.today()
    summary = {}

    # --- Days Since Active ---
    last_active_str = signals.get("last_active_date")
    days_since = -1
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_since = (today - last_active).days
        except (ValueError, TypeError):
            pass
    summary["days_since_active"] = days_since

    # --- Open to Work ---
    summary["open_to_work"] = bool(signals.get("open_to_work_flag", False))

    # --- Notice Period ---
    notice_days = signals.get("notice_period_days")
    summary["notice_period_days"] = int(notice_days) if notice_days is not None else -1
    
    if notice_days is None:
        summary["notice_label"] = "notice period unknown"
    else:
        nd = int(notice_days)
        if nd == 0:
            summary["notice_label"] = "immediately available"
        elif 1 <= nd <= 30:
            summary["notice_label"] = "30-day notice or less"
        elif 31 <= nd <= 60:
            summary["notice_label"] = f"{nd}-day notice"
        elif 61 <= nd <= 90:
            summary["notice_label"] = f"long notice ({nd} days)"
        else:
            summary["notice_label"] = f"very long notice ({nd} days)"

    # --- Response Rates ---
    rr_raw = signals.get("recruiter_response_rate")
    summary["recruiter_response_rate_pct"] = int(float(rr_raw) * 100) if rr_raw is not None else 0

    rt_raw = signals.get("avg_response_time_hours")
    if rt_raw is None:
        summary["avg_response_time_label"] = "response time unknown"
    else:
        rt = float(rt_raw)
        if rt <= 2: summary["avg_response_time_label"] = "responds within 2h"
        elif rt <= 6: summary["avg_response_time_label"] = "responds within 6h"
        elif rt <= 24: summary["avg_response_time_label"] = "responds within 24h"
        elif rt <= 72: summary["avg_response_time_label"] = f"slower responder (~{rt:.0f}h)"
        else: summary["avg_response_time_label"] = f"slow responder ({rt:.0f}h)"

    # --- GitHub Status ---
    gh_raw = signals.get("github_activity_score")
    if gh_raw is None or float(gh_raw) == -1:
        summary["github_status"] = "no github linked"
    else:
        gh = float(gh_raw)
        if gh < 30: summary["github_status"] = f"low github activity (score: {gh:.0f})"
        elif gh < 70: summary["github_status"] = f"moderate github activity (score: {gh:.0f})"
        else: summary["github_status"] = f"strong github activity (score: {gh:.0f})"

    # --- Interview Completion ---
    int_comp = signals.get("interview_completion_rate")
    summary["interview_completion_pct"] = int(float(int_comp) * 100) if int_comp is not None else 0

    # --- Offer Acceptance ---
    oa_raw = signals.get("offer_acceptance_rate")
    if oa_raw is None or float(oa_raw) == -1:
        summary["offer_acceptance_label"] = "no offer history"
    else:
        oa = float(oa_raw)
        if oa < 0.4: summary["offer_acceptance_label"] = f"low historical acceptance ({oa:.0%})"
        elif oa < 0.7: summary["offer_acceptance_label"] = f"moderate offer acceptance ({oa:.0%})"
        else: summary["offer_acceptance_label"] = f"strong offer acceptance ({oa:.0%})"

    # --- Salary Expectations ---
    sal = signals.get("expected_salary_range_inr_lpa", {})
    if isinstance(sal, dict) and "min" in sal and "max" in sal:
        summary["salary_expectation_label"] = f"expects {sal['min']}-{sal['max']} inr/yr"
    else:
        summary["salary_expectation_label"] = "not specified"

    # --- Misc Profile Info ---
    summary["work_mode_preference"] = str(signals.get("preferred_work_mode", "unknown")).lower()
    summary["willing_to_relocate"] = bool(signals.get("willing_to_relocate", False))
    
    comp_pct = signals.get("profile_completeness_score")
    summary["profile_completeness_pct"] = int(float(comp_pct)) if comp_pct is not None else 0

    # --- Verification Label ---
    email_ver = signals.get("verified_email", False)
    phone_ver = signals.get("verified_phone", False)
    if email_ver and phone_ver:
        summary["verification_label"] = "email+phone verified"
    elif email_ver:
        summary["verification_label"] = "email verified"
    else:
        summary["verification_label"] = "unverified"

    # --- Top Assessed Skills ---
    assessments = signals.get("skill_assessment_scores", {})
    if isinstance(assessments, dict) and assessments:
        # Sort by score descending
        sorted_skills = sorted(assessments.items(), key=lambda item: float(item[1]), reverse=True)
        top_3 = sorted_skills[:3]
        summary["top_assessed_skills"] = [f"{k} ({v})" for k, v in top_3]
    else:
        summary["top_assessed_skills"] = []

    # --- Attach Behavioral Score ---
    summary["behavioral_score"] = round(compute_behavioral_score(signals), 4)

    return summary


if __name__ == "__main__":
    from datetime import timedelta
    import json
    
    today = date.today()
    yesterday_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    old_date_str = (today - timedelta(days=200)).strftime("%Y-%m-%d")

    highly_engaged = {
        "last_active_date": yesterday_str,
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.95,
        "avg_response_time_hours": 1.5,
        "applications_submitted_30d": 10,
        "notice_period_days": 15,
        "interview_completion_rate": 0.90,
        "github_activity_score": 85.0,
        "offer_acceptance_rate": 0.80,
        "expected_salary_range_inr_lpa": {"min": 25, "max": 35},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "profile_completeness_score": 95,
        "verified_email": True,
        "verified_phone": True,
        "skill_assessment_scores": {"Python": 92, "Machine Learning": 88, "SQL": 75}
    }

    passive = {
        "last_active_date": old_date_str,
        "open_to_work_flag": False,
        "recruiter_response_rate": 0.10,
        "avg_response_time_hours": 96.0,
        "applications_submitted_30d": 0,
        "notice_period_days": 90,
        "interview_completion_rate": 0.40,
        "github_activity_score": -1,
        "offer_acceptance_rate": -1,
        "expected_salary_range_inr_lpa": {},
        "preferred_work_mode": "remote",
        "willing_to_relocate": False,
        "profile_completeness_score": 55,
        "verified_email": True,
        "verified_phone": False,
        "skill_assessment_scores": {}
    }

    print("--- Highly Engaged Candidate ---")
    score_active = compute_behavioral_score(highly_engaged)
    summary_active = extract_signal_summary(highly_engaged)
    print(f"Behavioral Score: {score_active:.4f}")
    print(json.dumps(summary_active, indent=2))
    
    print("\n--- Passive Candidate ---")
    score_passive = compute_behavioral_score(passive)
    summary_passive = extract_signal_summary(passive)
    print(f"Behavioral Score: {score_passive:.4f}")
    print(json.dumps(summary_passive, indent=2))