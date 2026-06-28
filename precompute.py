# precompute.py
import os
import json
import time
import shutil
import argparse
from typing import Generator
import numpy as np
from tqdm import tqdm

# Import internal modules (assuming they are in PYTHONPATH or run from root)
from src.features import extract_features
from src.honeypot import get_honeypot_penalty
from src.bm25_index import build_bm25_index, score_all_candidates, save_bm25
from src.signals import compute_behavioral_score, extract_signal_summary


def count_lines(path: str) -> int:
    """Quickly counts the number of lines in a file."""
    print(f"Counting candidates in {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        count = sum(1 for _ in f)
    print(f"Total candidates to process: {count}")
    return count


def load_candidates(path: str) -> Generator[dict, None, None]:
    """Streams candidates line-by-line to prevent massive RAM usage."""
    print(f"Loading candidates from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON on line {line_num}. Skipping.")


def build_candidate_meta(candidate: dict) -> dict:
    """Extracts lightweight metadata for Stage 4 manual review generation."""
    profile = candidate.get("profile", {})
    
    # Estimate top skills by duration since we don't have individual qualities here
    skills = candidate.get("skills", [])
    sorted_skills = sorted(skills, key=lambda x: x.get("duration_months", 0), reverse=True)
    top_skills = [s.get("name") for s in sorted_skills[:5] if s.get("name")]

    # Unique industries
    industries = list(set(
        role.get("industry") for role in candidate.get("career_history", []) 
        if role.get("industry")
    ))

    # Best education tier (1 is best)
    tier_map = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": 5}
    best_tier_val = 5
    best_tier_str = "unknown"
    for edu in candidate.get("education", []):
        t = edu.get("tier", "unknown").lower()
        if tier_map.get(t, 5) < best_tier_val:
            best_tier_val = tier_map.get(t, 5)
            best_tier_str = t

    return {
        "name": profile.get("anonymized_name", ""),
        "title": profile.get("current_title", ""),
        "company": profile.get("current_company", ""),
        "yoe": float(profile.get("years_of_experience", 0.0)),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "top_skills": top_skills,
        "industry_history": industries,
        "edu_tier": best_tier_str,
        "summary_snippet": profile.get("summary", "")[:200]
    }


def main(candidates_path: str, artifacts_dir: str) -> None:
    start_time = time.time()
    success = False
    
    # Ensure clean directory
    if os.path.exists(artifacts_dir):
        print(f"Clearing existing artifacts directory: {artifacts_dir}")
        shutil.rmtree(artifacts_dir)
    os.makedirs(artifacts_dir, exist_ok=True)

    try:
        # --- Step 1: Initialization ---
        total_lines = count_lines(candidates_path)
        
        candidate_ids = []
        feature_rows = []
        signal_summaries = {}
        candidate_meta = {}
        career_texts = []
        
        extraction_start = time.time()

        # --- Step 2: Feature Extraction Pass ---
        for candidate in tqdm(load_candidates(candidates_path), total=total_lines, desc="Extracting features"):
            cid = candidate.get("candidate_id")
            if not cid:
                continue
                
            try:
                # Core features
                features = extract_features(candidate)
                
                # Penalties and Signals
                penalty, _ = get_honeypot_penalty(candidate)
                signals = candidate.get("redrob_signals", {})
                behavioral_score = compute_behavioral_score(signals)
                sig_summary = extract_signal_summary(signals)
                
                # Meta dict
                meta = build_candidate_meta(candidate)
                
                # Append to RAM stores
                candidate_ids.append(cid)
                
                # feature_matrix cols: [skills, experience, behavioral, location, honeypot, bm25 (placeholder)]
                feature_rows.append([
                    features["skills_score"],
                    features["experience_score"],
                    behavioral_score,
                    features["location_score"],
                    penalty,
                    0.0  # Placeholder for BM25
                ])
                
                signal_summaries[cid] = sig_summary
                candidate_meta[cid] = meta
                career_texts.append(features["career_text"])
                
            except Exception as e:
                print(f"\nError processing candidate {cid}: {e}. Using zero-filled row.")
                # Graceful degradation on individual candidate failure
                candidate_ids.append(cid)
                feature_rows.append([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                signal_summaries[cid] = {}
                candidate_meta[cid] = {}
                career_texts.append("")
                
        extraction_time = time.time() - extraction_start
        print(f"\nFeature extraction: {len(candidate_ids)} candidates in {extraction_time:.1f}s")

        # --- Step 3: BM25 Indexing ---
        print("\nStarting BM25 Indexing...")
        bm25_index = build_bm25_index(career_texts)
        bm25_scores = score_all_candidates(bm25_index)
        
        # Inject BM25 scores into the feature matrix (Column 5)
        feature_matrix = np.array(feature_rows, dtype=np.float32)
        feature_matrix[:, 5] = bm25_scores
        
        bm25_path = os.path.join(artifacts_dir, "bm25_index.pkl")
        save_bm25(bm25_index, bm25_path)
        print("BM25 index built and saved.")

        # --- Step 4: Save all Artifacts ---
        print("\nSaving JSON and NumPy artifacts...")
        
        ids_path = os.path.join(artifacts_dir, "candidate_ids.json")
        with open(ids_path, 'w', encoding='utf-8') as f:
            json.dump(candidate_ids, f)
            
        matrix_path = os.path.join(artifacts_dir, "feature_matrix.npy")
        np.save(matrix_path, feature_matrix)
        
        signals_path = os.path.join(artifacts_dir, "signal_summaries.json")
        with open(signals_path, 'w', encoding='utf-8') as f:
            json.dump(signal_summaries, f)
            
        meta_path = os.path.join(artifacts_dir, "candidate_meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(candidate_meta, f)
            
        print(f"candidate_ids.json:    {os.path.getsize(ids_path) / (1024*1024):.2f} MB")
        print(f"feature_matrix.npy:    {os.path.getsize(matrix_path) / (1024*1024):.2f} MB")
        print(f"signal_summaries.json: {os.path.getsize(signals_path) / (1024*1024):.2f} MB")
        print(f"candidate_meta.json:   {os.path.getsize(meta_path) / (1024*1024):.2f} MB")
        print(f"All artifacts saved to {artifacts_dir}")

        # --- Step 5: Sanity Check ---
        print("\nRunning Sanity Checks on saved artifacts...")
        
        with open(ids_path, 'r', encoding='utf-8') as f:
            loaded_ids = json.load(f)
        loaded_matrix = np.load(matrix_path)
        
        num_candidates = len(loaded_ids)
        assert num_candidates > 0, "No candidates were processed."
        assert loaded_matrix.shape == (num_candidates, 6), f"Matrix shape mismatch: {loaded_matrix.shape}"
        
        # Bounds check
        out_of_bounds = (loaded_matrix < 0.0) | (loaded_matrix > 1.0)
        if out_of_bounds.any():
            print("WARNING: Some feature values are outside the [0, 1] range!")
            
        # Knockouts count
        knockouts = np.sum(loaded_matrix[:, 4] == 0.0)
        print(f"Honeypot knockouts: {knockouts} / {num_candidates}")
        
        # Top 5 preview calculation
        skills_col = loaded_matrix[:, 0]
        exp_col = loaded_matrix[:, 1]
        behav_col = loaded_matrix[:, 2]
        loc_col = loaded_matrix[:, 3]
        penalty_col = loaded_matrix[:, 4]
        bm25_col = loaded_matrix[:, 5]
        
        raw_composite = (
            0.50 * (0.65 * bm25_col + 0.35 * skills_col) +
            0.35 * exp_col +
            0.15 * loc_col
        ) * (0.40 + 0.60 * behav_col) * penalty_col
        
        top_indices = np.argsort(raw_composite)[::-1][:5]
        print("\nTop 5 Candidates (Preview Ranking):")
        for rank, idx in enumerate(top_indices, 1):
            c_id = loaded_ids[idx]
            score = raw_composite[idx]
            print(f"  Rank {rank}: {c_id} (Score: {score:.4f})")
            
        success = True

    except Exception as e:
        print(f"\nCRITICAL FAILURE during precompute: {e}")
        raise
        
    finally:
        if not success:
            print(f"\nPrecompute failed. Cleaning up partial artifacts in {artifacts_dir}...")
            shutil.rmtree(artifacts_dir, ignore_errors=True)
        else:
            total_time = (time.time() - start_time) / 60
            print(f"\nprecompute.py completed successfully in {total_time:.1f} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute features and artifacts for candidate ranking.")
    parser.add_argument("--candidates", type=str, default="./candidates.jsonl", 
                        help="Path to the uncompressed candidates.jsonl file.")
    parser.add_argument("--artifacts", type=str, default="./artifacts/", 
                        help="Directory to save the computed artifacts.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.candidates):
        print(f"Error: Candidate file not found at {args.candidates}")
        print("Please ensure the dataset is unzipped and located at the specified path.")
    else:
        main(args.candidates, args.artifacts)