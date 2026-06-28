# src/reasoning.py
import re
import hashlib
from typing import List, Dict

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

ML_KEYWORDS = {
    "ml", "machine learning", "ai", "nlp", "search", "ranking", 
    "recommendation", "data scientist", "applied scientist", "research"
}

JD_CORE_SKILLS = {
    "retrieval", "ranking", "recommendation", "embeddings", 
    "vector search", "faiss", "pinecone", "nlp", "information retrieval", 
    "transformers", "search", "rag"
}

PRODUCT_INDUSTRIES = {
    "AI/ML", "Food Delivery", "Fintech", "Transportation", 
    "E-commerce", "SaaS", "Healthcare Tech", "Edtech", "Gaming"
}

CONSULTING_KEYWORDS = ["IT Services", "Consulting", "Outsourcing"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _vary(options: List[str], seed: str) -> str:
    """
    Deterministically picks an option from a list based on a hash of the seed.
    Ensures the same candidate always gets the same phrasing, but different 
    candidates get varied phrasings.
    """
    if not options:
        return ""
    # Use MD5 to get a stable hash across different Python runs/sessions
    h = int(hashlib.md5(seed.encode('utf-8')).hexdigest(), 16)
    return options[h % len(options)]


def clean_reasoning(text: str) -> str:
    """
    Post-processes the assembled string for grammar, formatting, and limits.
    """
    if not text:
        return ""
        
    # Strip and handle spacing
    text = text.strip()
    text = " ".join(text.split())  # collapse multiple spaces
    
    # Fix punctuation spacing
    text = text.replace(" ;", ";").replace(" ,", ",")
    
    # Capitalize first character securely
    if text:
        text = text[0].upper() + text[1:]
        
    # Ensure it ends with exactly one period
    text = text.replace("..", ".")
    if not text.endswith("."):
        if text.endswith(";"):
            text = text[:-1] + "."
        else:
            text += "."
            
    # Max length truncation
    if len(text) > 300:
        truncated = text[:300]
        last_space = truncated.rfind(" ")
        if last_space != -1:
            text = truncated[:last_space] + "."
        else:
            text = truncated + "."
            
    return text


# =============================================================================
# CORE REASONING GENERATION
# =============================================================================

def generate_reasoning(
    rank: int,
    candidate_id: str,
    meta: Dict,
    signals: Dict,
    scores: Dict
) -> str:
    """
    Generates a 1-2 sentence reasoning string based on factual data arrays.
    Constructed in 5 logic-based clauses to prevent hallucination while
    maintaining human-like variety.
    """
    
    # Safe extracts
    yoe = meta.get("yoe", 0.0)
    title = meta.get("title", "Candidate")
    company = meta.get("company", "Unknown")
    jd_match = scores.get("jd_match_score", 0.0)
    top_skills = meta.get("top_skills", [])
    
    # --- CLAUSE A: OPENING ---
    is_ml_title = any(kw in title.lower() for kw in ML_KEYWORDS)
    has_retrieval_ranking = any(kw in s.lower() for s in top_skills for kw in ["retrieval", "ranking"])
    
    if rank <= 5:
        if jd_match > 0.75:
            if has_retrieval_ranking:
                clause_a = _vary([
                    f"{yoe:.0f}-year applied ML engineer",
                    f"Strong retrieval/ranking background with {yoe:.0f} years"
                ], candidate_id + "_A1")
            else:
                clause_a = f"{yoe:.0f}-year applied ML engineer"
        else:
            clause_a = f"{title} at {company} with {yoe:.0f} years of applied ML experience"
            
    elif rank <= 20:
        if is_ml_title:
            clause_a = _vary([
                f"{yoe:.0f} years as {title}", 
                f"ML engineer with {yoe:.0f} years' experience"
            ], candidate_id + "_A2")
        else:
            clause_a = f"{yoe:.0f} years as {title}"
            
    elif rank <= 50:
        clause_a = f"{title} with {yoe:.0f} years of experience"
    else:
        clause_a = f"Candidate is a {title} with {yoe:.0f} years of experience"

    # --- CLAUSE B: SKILLS/JD FIT ---
    matching = [s for s in top_skills if any(kw in s.lower() for kw in JD_CORE_SKILLS)]
    non_matching = [s for s in top_skills if s not in matching]
    
    if len(matching) >= 2:
        clause_b = f"; core skills in {', '.join(matching[:3])} directly match JD requirements"
    elif len(matching) == 1:
        support = non_matching[0] if non_matching else "broader ML background"
        clause_b = f"; relevant skill in {matching[0]}, supported by {support}"
    else:
        skills_str = ", ".join(top_skills[:3]) if top_skills else "general engineering"
        clause_b = f"; skills in {skills_str} (adjacent, not core IR/search)"

    top_assessed = signals.get("top_assessed_skills", [])
    if top_assessed:
        clause_b += f" (platform-assessed: {', '.join(top_assessed[:2])})"

    # --- CLAUSE C: BACKGROUND QUALITY ---
    hist = meta.get("industry_history", [])
    prod_count = sum(1 for i in hist if i in PRODUCT_INDUSTRIES)
    cons_count = sum(1 for i in hist if any(kw in i for kw in CONSULTING_KEYWORDS))
    
    if prod_count >= 2:
        clause_c = "; multiple product-company roles"
    elif prod_count == 1 and cons_count == 0:
        clause_c = "; product-company background"
    elif cons_count > 0 and prod_count > 0:
        clause_c = "; mixed consulting/product background"
    elif cons_count >= 1 and prod_count == 0:
        clause_c = "; primarily consulting background — less direct product exposure"
    else:
        clause_c = ""
        
    if meta.get("edu_tier") == "tier_1":
        clause_c += "; tier-1 education"
        
    clause_c += "."

    # --- CLAUSE D: AVAILABILITY ---
    days_active = signals.get("days_since_active", 999)
    otw = signals.get("open_to_work", False)
    
    if otw and days_active <= 14:
        clause_d = "Actively looking and engaged on platform"
    elif otw and days_active <= 60:
        clause_d = "Open to opportunities, recently active"
    elif not otw and days_active <= 30:
        clause_d = "Active on platform but not explicitly open to work"
    elif days_active > 180:
        clause_d = f"Last active {days_active} days ago — outreach responsiveness uncertain"
    else:
        clause_d = "Platform activity moderate"
        
    notice_label = signals.get("notice_label", "notice unknown")
    clause_d += f"; {notice_label}"
    
    gh_status = signals.get("github_status", "")
    if gh_status.startswith("strong"):
        clause_d += f"; {gh_status}"

    # --- CLAUSE E: CONCERNS ---
    concerns = []
    flags = []
    
    notice_period = signals.get("notice_period_days", 0)
    if notice_period > 90:
        concerns.append(f"notice period is {notice_period} days")
        
    offer_acc = signals.get("offer_acceptance_label", "")
    if offer_acc.startswith("low"):
        concerns.append(offer_acc)
        
    interview_pct = signals.get("interview_completion_pct", 100)
    if interview_pct < 50:
        concerns.append(f"low interview completion ({interview_pct}%)")
        
    if scores.get("experience_score", 1.0) < 0.35:
        concerns.append("experience profile below JD target range")
        
    if scores.get("jd_match_score", 1.0) < 0.25 and rank > 50:
        concerns.append("limited direct IR/search signal in profile")
        
    if days_active > 180:
        flags.append("low recent activity")
        
    if concerns:
        clause_e = ". Concern(s): " + "; ".join(concerns) + "."
    elif flags:
        clause_e = " — " + "; ".join(flags) + "."
    else:
        clause_e = "."

    # --- ASSEMBLE AND CLEAN ---
    raw_reasoning = f"{clause_a}{clause_b}{clause_c} {clause_d}{clause_e}"
    
    return clean_reasoning(raw_reasoning)


def batch_generate_reasoning(ranked_candidates: List[Dict]) -> Dict[str, str]:
    """
    Generate reasoning for all top-100 candidates at once.
    Expects a list of dictionaries, each containing:
    { 'rank', 'candidate_id', 'meta', 'signals', 'scores' }
    """
    results = {}
    for cand in ranked_candidates:
        cid = cand.get("candidate_id")
        if not cid:
            continue
            
        reasoning = generate_reasoning(
            rank=cand.get("rank", 999),
            candidate_id=cid,
            meta=cand.get("meta", {}),
            signals=cand.get("signals", {}),
            scores=cand.get("scores", {})
        )
        results[cid] = reasoning
        
    return results


# =============================================================================
# SANITY CHECK / TESTING
# =============================================================================
if __name__ == "__main__":
    
    test_cases = [
        {
            "rank": 1,
            "candidate_id": "CAND_0000001",
            "meta": {
                "title": "Senior AI Engineer",
                "company": "Swiggy",
                "yoe": 6.5,
                "top_skills": ["Retrieval Augmented Generation", "Pinecone", "Python", "Ranking"],
                "industry_history": ["Food Delivery", "AI/ML"],
                "edu_tier": "tier_1"
            },
            "signals": {
                "days_since_active": 2,
                "open_to_work": True,
                "notice_period_days": 30,
                "notice_label": "30-day notice or less",
                "github_status": "strong github activity (score: 85)",
                "offer_acceptance_label": "strong offer acceptance (85%)",
                "interview_completion_pct": 95,
                "top_assessed_skills": ["Python (98)"]
            },
            "scores": {
                "jd_match_score": 0.88,
                "experience_score": 0.90
            }
        },
        {
            "rank": 25,
            "candidate_id": "CAND_0000025",
            "meta": {
                "title": "Machine Learning Engineer",
                "company": "Infosys",
                "yoe": 4.0,
                "top_skills": ["NLP", "Transformers", "SQL"],
                "industry_history": ["IT Services"],
                "edu_tier": "tier_2"
            },
            "signals": {
                "days_since_active": 45,
                "open_to_work": True,
                "notice_period_days": 90,
                "notice_label": "long notice (90 days)",
                "github_status": "moderate github activity (score: 50)",
                "offer_acceptance_label": "moderate offer acceptance (60%)",
                "interview_completion_pct": 80,
                "top_assessed_skills": []
            },
            "scores": {
                "jd_match_score": 0.60,
                "experience_score": 0.40
            }
        },
        {
            "rank": 95,
            "candidate_id": "CAND_0000095",
            "meta": {
                "title": "Data Analyst",
                "company": "Unknown",
                "yoe": 2.0,
                "top_skills": ["Excel", "Tableau", "SQL"],
                "industry_history": ["Consulting"],
                "edu_tier": "tier_3"
            },
            "signals": {
                "days_since_active": 200,
                "open_to_work": False,
                "notice_period_days": 0,
                "notice_label": "immediately available",
                "github_status": "no github linked",
                "offer_acceptance_label": "low historical acceptance (20%)",
                "interview_completion_pct": 40,
                "top_assessed_skills": []
            },
            "scores": {
                "jd_match_score": 0.15,
                "experience_score": 0.20
            }
        }
    ]
    
    print("-" * 80)
    for tc in test_cases:
        reasoning = generate_reasoning(
            rank=tc["rank"],
            candidate_id=tc["candidate_id"],
            meta=tc["meta"],
            signals=tc["signals"],
            scores=tc["scores"]
        )
        print(f"Rank {tc['rank']} ({tc['candidate_id']}):\n{reasoning}\n")
    print("-" * 80)