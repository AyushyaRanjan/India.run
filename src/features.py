# src/features.py
import json
from datetime import datetime

# ==============================================================================
# SCORING CONSTANTS & WEIGHTS (TUNABLE)
# ==============================================================================

# --- SKILLS CONSTANTS ---
HIGH_PRIORITY_SKILLS = {
    "machine learning", "python", "information retrieval", "recommendation systems", 
    "vector search", "embeddings", "faiss", "pinecone", "qdrant", "weaviate", 
    "elasticsearch", "retrieval", "ranking", "nlp", "natural language processing", 
    "transformers", "sentence transformers", "pytorch", "tensorflow", "xgboost", 
    "lightgbm", "search", "rag", "langchain"
}
IGNORE_SKILLS = {
    "microsoft office", "excel", "powerpoint", "communication", "leadership", 
    "teamwork", "time management", "ms word"
}
PROFICIENCY_WEIGHTS = {
    "beginner": 0.25,
    "intermediate": 0.50,
    "advanced": 0.75,
    "expert": 1.00
}
DEFAULT_ASSESSMENT_BONUS = 0.4

# --- EXPERIENCE CONSTANTS ---
EXP_WEIGHTS = {
    "yoe": 0.30,
    "company_type": 0.35,
    "ml_ratio": 0.20,
    "tenure": 0.15
}
CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "tech mahindra", 
    "capgemini", "hcl", "hexaware", "mphasis", "ltimindtree", "l&t infotech"
}
PRODUCT_INDUSTRIES = {
    "software", "ai/ml", "fintech", "food delivery", "transportation", 
    "e-commerce", "healthcare tech", "edtech", "gaming", "saas"
}
BIG_TECH_COMPANIES = {
    "google", "microsoft", "amazon", "meta", "apple", "netflix", "uber", 
    "airbnb", "linkedin", "twitter", "openai", "anthropic", "deepmind", 
    "salesforce", "adobe"
}
RESEARCH_INDUSTRIES = {"research", "academia", "university"}
RESEARCH_KEYWORDS = {"iit", "iim", "iisc", "mit", "stanford", "cmu"}
ML_TITLE_KEYWORDS = {
    "ml", "machine learning", "ai", "data scientist", "nlp", "search", "ranking", 
    "recommendation", "computer vision", "deep learning", "research scientist", 
    "applied scientist"
}

# --- BEHAVIORAL CONSTANTS ---
BEHAVIORAL_WEIGHTS = {
    "recency": 0.30,
    "open_to_work": 0.15,
    "response": 0.20,
    "notice": 0.15,
    "interview": 0.10,
    "github": 0.10
}

# --- LOCATION CONSTANTS ---
TARGET_CITIES = {
    "bangalore", "bengaluru", "hyderabad", "pune", "mumbai", "delhi", 
    "noida", "gurugram", "gurgaon", "chennai"
}


# ==============================================================================
# FEATURE EXTRACTION LOGIC
# ==============================================================================

def _safe_str(val) -> str:
    """Helper to convert to string and handle None."""
    return str(val) if val is not None else ""

def _calculate_skills_score(skills: list, assessment_scores: dict) -> float:
    """
    Computes a weighted quality score for technical skills based on proficiency,
    duration, endorsements, and explicit skill assessments.
    """
    if not skills:
        return 0.0

    # Lowercase assessment keys for reliable matching
    assess_lower = {k.lower(): v for k, v in (assessment_scores or {}).items()}
    
    weighted_sum = 0.0
    max_possible_sum = 0.0

    for skill in skills:
        name_lower = skill.get("name", "").lower()
        if name_lower in IGNORE_SKILLS:
            continue
            
        prof_str = skill.get("proficiency", "beginner").lower()
        prof_score = PROFICIENCY_WEIGHTS.get(prof_str, 0.25)
        
        duration = skill.get("duration_months", 0)
        duration_score = min(duration / 36.0, 1.0)
        
        endorsements = skill.get("endorsements", 0)
        endorse_score = min(endorsements / 30.0, 1.0)
        
        # Get assessment bonus
        if name_lower in assess_lower:
            assess_bonus = assess_lower[name_lower] / 100.0
        else:
            assess_bonus = DEFAULT_ASSESSMENT_BONUS
            
        quality = (prof_score * 0.30) + (duration_score * 0.30) + (endorse_score * 0.20) + (assess_bonus * 0.20)
        
        weight = 2.0 if name_lower in HIGH_PRIORITY_SKILLS else 1.0
        
        weighted_sum += (quality * weight)
        max_possible_sum += (1.0 * weight)
        
    if max_possible_sum == 0:
        return 0.0
        
    return min(weighted_sum / max_possible_sum, 1.0)

def _calculate_experience_score(profile: dict, career_history: list) -> float:
    """
    Computes an experience score based on YOE bounds, company pedigree, 
    ML role concentration, and average tenure length.
    """
    # 1. YOE Score
    yoe = float(profile.get("years_of_experience", 0))
    if yoe < 2.0:
        yoe_score = 0.1
    elif yoe <= 4.0:
        yoe_score = 0.4
    elif yoe <= 5.0:
        yoe_score = 0.7
    elif yoe <= 9.0:
        yoe_score = 1.0
    elif yoe <= 12.0:
        yoe_score = 0.8
    else:
        yoe_score = 0.6

    # 2. Company Type Score & 3. ML Ratio Score & 4. Tenure Score
    total_months = 0
    ml_months = 0
    company_type_weighted_sum = 0.0
    valid_roles_for_tenure = []
    
    for role in career_history:
        duration = role.get("duration_months", 0)
        company = role.get("company", "").lower()
        industry = role.get("industry", "").lower()
        title = role.get("title", "").lower()
        size = role.get("company_size", "")
        
        # Track ML duration
        if any(kw in title for kw in ML_TITLE_KEYWORDS):
            ml_months += duration
            
        total_months += duration
        
        # Track for tenure (ignore current role if < 12 months)
        is_current = role.get("is_current", False)
        if not (is_current and duration < 12):
            valid_roles_for_tenure.append(duration)

        # Classify Company Type
        if any(kw in company for kw in BIG_TECH_COMPANIES):
            type_weight = 1.0
        elif industry in PRODUCT_INDUSTRIES and size != "10001+":
            type_weight = 0.9
        elif industry in RESEARCH_INDUSTRIES or any(kw in company for kw in RESEARCH_KEYWORDS):
            type_weight = 0.5
        elif any(kw in company for kw in CONSULTING_COMPANIES):
            type_weight = 0.15
        else:
            type_weight = 0.4
            
        company_type_weighted_sum += (type_weight * duration)
        
    company_type_score = (company_type_weighted_sum / total_months) if total_months > 0 else 0.4
    ml_role_score = (ml_months / total_months) if total_months > 0 else 0.0
    
    # Calculate Tenure Score
    if valid_roles_for_tenure:
        avg_tenure = sum(valid_roles_for_tenure) / len(valid_roles_for_tenure)
    else:
        avg_tenure = 0
        
    if avg_tenure < 12:
        tenure_score = 0.2
    elif avg_tenure <= 18:
        tenure_score = 0.5
    elif avg_tenure <= 24:
        tenure_score = 0.7
    elif avg_tenure <= 36:
        tenure_score = 0.9
    else:
        tenure_score = 1.0
        
    exp_score = (
        (EXP_WEIGHTS["yoe"] * yoe_score) +
        (EXP_WEIGHTS["company_type"] * company_type_score) +
        (EXP_WEIGHTS["ml_ratio"] * ml_role_score) +
        (EXP_WEIGHTS["tenure"] * tenure_score)
    )
    return min(exp_score, 1.0)

def _calculate_behavioral_score(signals: dict) -> float:
    """
    Computes a behavioral score combining platform recency, explicit openness
    to work, recruiter response rates, and interview completion reliability.
    """
    # 1. Recency Score
    last_active = signals.get("last_active_date")
    recency_score = 0.15 # Default if missing
    if last_active:
        try:
            last_active_date = datetime.strptime(last_active, "%Y-%m-%d").date()
            days_since = (datetime.today().date() - last_active_date).days
            if days_since <= 30: recency_score = 1.0
            elif days_since <= 60: recency_score = 0.85
            elif days_since <= 90: recency_score = 0.65
            elif days_since <= 180: recency_score = 0.4
            else: recency_score = 0.15
        except ValueError:
            pass
            
    # 2. Open to Work
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.35
    
    # 3. Response Score
    response_score = float(signals.get("recruiter_response_rate", 0.0))
    
    # 4. Notice Period
    notice_days = signals.get("notice_period_days", 90)
    if notice_days == 0: notice_score = 1.0
    elif notice_days <= 30: notice_score = 0.95
    elif notice_days <= 60: notice_score = 0.75
    elif notice_days <= 90: notice_score = 0.50
    else: notice_score = 0.20
    
    # 5. Interview Completion
    interview_score = float(signals.get("interview_completion_rate", 0.0))
    
    # 6. GitHub Score
    gh_raw = float(signals.get("github_activity_score", -1.0))
    github_score = 0.3 if gh_raw == -1 else (gh_raw / 100.0)
    
    score = (
        (BEHAVIORAL_WEIGHTS["recency"] * recency_score) +
        (BEHAVIORAL_WEIGHTS["open_to_work"] * open_to_work) +
        (BEHAVIORAL_WEIGHTS["response"] * response_score) +
        (BEHAVIORAL_WEIGHTS["notice"] * notice_score) +
        (BEHAVIORAL_WEIGHTS["interview"] * interview_score) +
        (BEHAVIORAL_WEIGHTS["github"] * github_score)
    )
    return min(score, 1.0)

def _calculate_location_score(profile: dict, signals: dict) -> float:
    """
    Computes a location match score favoring candidates already in 
    target Indian cities or strongly willing to relocate.
    """
    country = _safe_str(profile.get("country")).strip().lower()
    location = _safe_str(profile.get("location")).strip().lower()
    willing = signals.get("willing_to_relocate", False)
    
    if country == "india":
        if any(city in location for city in TARGET_CITIES):
            return 0.95
        return 0.80 if willing else 0.65
    else:
        return 0.40 if willing else 0.15

def _extract_career_text(profile: dict, career_history: list, skills: list) -> str:
    """
    Aggregates profile text data into a single string corpus for BM25 indexing.
    """
    parts = []
    
    # Core Profile
    parts.append(_safe_str(profile.get("headline")))
    parts.append(_safe_str(profile.get("summary")))
    
    # History
    for role in career_history:
        parts.append(_safe_str(role.get("title")))
        parts.append(_safe_str(role.get("description")))
        
    # Skills
    for skill in skills:
        parts.append(_safe_str(skill.get("name")))
        
    # Filter out empty strings and join
    return " ".join(p.strip() for p in parts if p.strip())


# ==============================================================================
# MAIN EXPORT
# ==============================================================================

def extract_features(candidate: dict) -> dict:
    """
    Extracts core scoring features from a candidate profile dictionary.
    Returns a dictionary of normalized 0.0-1.0 scores and the raw career text.
    """
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    skills_score = _calculate_skills_score(skills, signals.get("skill_assessment_scores", {}))
    experience_score = _calculate_experience_score(profile, career_history)
    behavioral_score = _calculate_behavioral_score(signals)
    location_score = _calculate_location_score(profile, signals)
    career_text = _extract_career_text(profile, career_history, skills)
    
    return {
        "skills_score": round(skills_score, 4),
        "experience_score": round(experience_score, 4),
        "behavioral_score": round(behavioral_score, 4),
        "location_score": round(location_score, 4),
        "career_text": career_text
    }


# ==============================================================================
# SANITY CHECK / TESTING
# ==============================================================================
if __name__ == "__main__":
    import os
    
    test_file = "sample_candidates.json"
    
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            try:
                candidates = json.load(f)
                
                print(f"{'ID':<15} | {'Skills':<8} | {'Exp':<8} | {'Behav':<8} | {'Loc':<8}")
                print("-" * 55)
                
                for cand in candidates[:5]:
                    c_id = cand.get("candidate_id", "UNKNOWN")
                    feats = extract_features(cand)
                    print(f"{c_id:<15} | {feats['skills_score']:<8.4f} | {feats['experience_score']:<8.4f} | "
                          f"{feats['behavioral_score']:<8.4f} | {feats['location_score']:<8.4f}")
                          
            except json.JSONDecodeError:
                print("Error parsing sample_candidates.json")
    else:
        print(f"Test file {test_file} not found. Skipping local test block.")