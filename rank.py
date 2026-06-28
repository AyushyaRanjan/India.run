"""
rank.py — Timed ranking step for Redrob Hackathon submission.

Constraints: ≤5 min wall-clock, ≤16GB RAM, CPU only, no network.
Reads precomputed artifacts from ./artifacts/ (produced by precompute.py).

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import os
import sys
import json
import csv
import time
import argparse
import numpy as np


def generate_reasoning(rank: int, candidate_id: str, meta: dict, signals: dict) -> str:
    """
    Template-based reasoning generator strictly relying on extracted facts.
    Produces a 1-2 sentence string justifying the candidate's rank.
    """
    yoe = meta.get("yoe", 0.0)
    title = meta.get("title", "Candidate")
    company = meta.get("company", "Unknown")

    # PART 1: Opening clause
    if rank <= 10:
        p1 = f"{yoe:.0f}-year applied ML background" if yoe >= 5 else f"Strong ML profile with {yoe:.0f} years of applied experience"
    elif rank <= 30:
        p1 = f"{title} at {company} with {yoe:.0f} years experience"
    else:
        p1 = f"Candidate has {yoe:.0f} years experience as {title}"

    # PART 2: Skills clause
    top_skills = meta.get("top_skills", [])[:3]
    p2 = "; skills in " + ", ".join(top_skills) if top_skills else ""

    # PART 3: Company/background clause
    ind_hist = meta.get("industry_history", [])
    prod_industries = {"AI/ML", "Food Delivery", "Fintech", "Transportation", "E-commerce", "SaaS"}
    if any(ind in prod_industries for ind in ind_hist):
        p3 = "; product-company background"
    elif "Research" in ind_hist or "Academia" in ind_hist:
        p3 = "; primarily research background (less production focus)"
    else:
        p3 = ""

    # PART 4: Availability clause
    days_active = signals.get("days_since_active", 999)
    open_to_work = signals.get("open_to_work", False)
    notice_label = signals.get("notice_label", "notice period unknown")

    if days_active <= 30 and open_to_work:
        p4 = f"; actively looking, {notice_label}"
    elif days_active <= 60:
        p4 = f"; recently active, {notice_label}"
    elif days_active > 180:
        p4 = f"; last active {days_active} days ago — may not be responsive"
    else:
        p4 = f"; {notice_label}"

    # PART 5: Concern clause
    concerns = []
    notice_period = signals.get("notice_period_days", 0)
    offer_acc = signals.get("offer_acceptance_label", "")

    if notice_period > 90:
        concerns.append(f"long notice period ({notice_period}d)")
    if offer_acc.startswith("low"):
        concerns.append(offer_acc)
    if days_active > 180:
        concerns.append("low recent activity")
    if yoe < 4:
        concerns.append("below recommended experience range")

    p5 = ". Concerns: " + "; ".join(concerns) + "." if concerns else "."

    # Concatenate and clean up formatting
    full = p1 + p2 + p3 + p4 + p5
    
    # Capitalize first letter safely
    if full:
        full = full[0].upper() + full[1:]
        
    # Clean up any potential double punctuation or spacing
    full = " ".join(full.split())
    full = full.replace(" .", ".").replace("..", ".").replace(";;", ";")
    
    return full


def main():
    parser = argparse.ArgumentParser(description="Generate rank submission for Redrob hackathon.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv", help="Output path for the submission CSV")
    parser.add_argument("--artifacts", default="./artifacts/", help="Directory containing precomputed artifacts")
    parser.add_argument("--team-id", default="team_xxx", help="Team ID (used for default filename if needed)")
    args = parser.parse_args()

    t_start = time.time()

    if not os.path.exists(args.candidates):
        print(f"WARNING: Candidates file {args.candidates} not found. Proceeding using precomputed artifacts...")

    # --- 1. Load Artifacts ---
    print("Loading artifacts...")
    try:
        with open(os.path.join(args.artifacts, "candidate_ids.json"), "r", encoding="utf-8") as f:
            candidate_ids = json.load(f)
            
        feature_matrix = np.load(os.path.join(args.artifacts, "feature_matrix.npy"))
        
        with open(os.path.join(args.artifacts, "signal_summaries.json"), "r", encoding="utf-8") as f:
            signal_summaries = json.load(f)
            
        with open(os.path.join(args.artifacts, "candidate_meta.json"), "r", encoding="utf-8") as f:
            candidate_meta = json.load(f)
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Could not load artifact - {e}")
        sys.exit(1)

    t_load = time.time()
    print(f"Loaded artifacts in {t_load - t_start:.2f}s")

    # --- 2. Scoring 100K Candidates ---
    print("Scoring 100K candidates...")
    skills_score = feature_matrix[:, 0]
    experience_score = feature_matrix[:, 1]
    behavioral_score = feature_matrix[:, 2]
    location_score = feature_matrix[:, 3]
    honeypot_penalty = feature_matrix[:, 4]
    bm25_score = feature_matrix[:, 5]

    jd_match_score = 0.65 * bm25_score + 0.35 * skills_score
    base_score = 0.50 * jd_match_score + 0.35 * experience_score + 0.15 * location_score
    availability_multiplier = 0.40 + 0.60 * behavioral_score
    final_scores = base_score * availability_multiplier * honeypot_penalty

    # Python's built-in sort is highly optimized for tie-breaking on mixed types
    # Primary sort: negative score (descending), Secondary sort: candidate_id (ascending)
    scores_ids = [(float(score), cid) for score, cid in zip(final_scores, candidate_ids)]
    scores_ids.sort(key=lambda x: (-x[0], x[1]))
    
    top_100_raw = scores_ids[:100]

    t_score = time.time()
    print(f"Scored candidates in {t_score - t_load:.2f}s")

    # --- 3. Generating Reasoning ---
    print("Generating reasoning...")
    submission_data = []
    seen_ids = set()
    candidate_id_set = set(candidate_ids)

    for i, (score, cid) in enumerate(top_100_raw):
        rank = i + 1
        meta = candidate_meta.get(cid, {})
        signals = signal_summaries.get(cid, {})
        
        reasoning = generate_reasoning(rank, cid, meta, signals)
        
        submission_data.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 6),
            "reasoning": reasoning
        })
        seen_ids.add(cid)

    t_reason = time.time()
    print(f"Generated reasoning in {t_reason - t_score:.2f}s")

    # --- 4. Validation Checks ---
    try:
        if len(submission_data) != 100:
            raise ValueError(f"Submission has {len(submission_data)} rows instead of 100.")
            
        ranks = [row["rank"] for row in submission_data]
        if ranks != list(range(1, 101)):
            raise ValueError("Ranks are not exactly 1 through 100.")
            
        if len(seen_ids) != 100:
            raise ValueError("Candidate IDs in the top 100 are not unique.")
            
        invalid_ids = [cid for cid in seen_ids if cid not in candidate_id_set]
        if invalid_ids:
            raise ValueError(f"Found invalid candidate IDs not in original pool: {invalid_ids}")
            
        for i in range(len(submission_data) - 1):
            if submission_data[i]["score"] < submission_data[i+1]["score"]:
                raise ValueError("Scores are not monotonically non-increasing.")
                
        print("Validation passed ✓")
    except ValueError as e:
        print(f"Validation Failed: {e}")
        sys.exit(1)

    # --- 5. Writing CSV ---
    print("Writing CSV...")
    
    # If out path is a directory, append team_id.csv
    out_path = args.out
    if os.path.isdir(out_path):
        out_path = os.path.join(out_path, f"{args.team_id}.csv")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in submission_data:
            writer.writerow([
                row["candidate_id"],
                row["rank"],
                f"{row['score']:.6f}",
                row["reasoning"]
            ])

    t_write = time.time()
    print(f"Wrote CSV in {t_write - t_reason:.2f}s")
    print(f"Done. Total time: {t_write - t_start:.1f}s")


if __name__ == "__main__":
    main()