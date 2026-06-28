# Redrob Hackathon — Demo Sandbox
# Runs on small candidate samples (≤100) for submission verification.
# Deploy to Streamlit Cloud: streamlit run app.py
# Full 100K ranking: python rank.py --candidates candidates.jsonl --out submission.csv

import os
import json
import csv
import io
import streamlit as st

# Handle graceful imports for optional dependencies
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

# Internal ML modules
from src.features import extract_features
from src.honeypot import get_honeypot_penalty
from src.bm25_index import build_bm25_index, score_all_candidates
from src.signals import compute_behavioral_score, extract_signal_summary
from src.reasoning import generate_reasoning

st.set_page_config(page_title="Redrob Ranker", page_icon="🎯", layout="wide")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_cached_bm25_index(career_texts):
    """Caches the BM25 index creation to speed up re-runs with different weights."""
    return build_bm25_index(career_texts)

def build_candidate_meta(candidate: dict) -> dict:
    """Builds the lightweight metadata dict needed by generate_reasoning."""
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    
    # Sort by duration to approximate "top" skills
    sorted_skills = sorted(skills, key=lambda x: x.get("duration_months", 0), reverse=True)
    top_skills = [s.get("name") for s in sorted_skills[:5] if s.get("name")]
    
    industries = list(set(
        r.get("industry") for r in candidate.get("career_history", []) if r.get("industry")
    ))
    
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

def parse_upload(file_obj) -> list:
    """Parses JSON or JSONL file into a list of dicts."""
    content = file_obj.read().decode("utf-8")
    if file_obj.name.endswith(".jsonl"):
        candidates = []
        for line in content.splitlines():
            if line.strip():
                candidates.append(json.loads(line))
        return candidates
    else:
        return json.loads(content)

# Initialize session state
if "candidates" not in st.session_state:
    st.session_state["candidates"] = []
if "ranked_results" not in st.session_state:
    st.session_state["ranked_results"] = None
if "all_results" not in st.session_state:
    st.session_state["all_results"] = None


# =============================================================================
# HEADER
# =============================================================================

st.title("Redrob Candidate Ranker — Hackathon Demo")
st.markdown("Upload a candidate JSON file (≤100 candidates) to see the top-ranked results.")

# =============================================================================
# SECTION 1 — INPUT
# =============================================================================

tab1, tab2 = st.tabs(["Upload File", "Use Sample Data"])

with tab1:
    uploaded_file = st.file_uploader("Upload candidates JSON or JSONL", type=["json", "jsonl"])
    if uploaded_file is not None:
        try:
            cands = parse_upload(uploaded_file)
            if not cands:
                st.error("Uploaded file contains 0 candidates.")
            else:
                if len(cands) > 100:
                    st.warning(f"File contains {len(cands)} candidates. Capping at 100 for sandbox.")
                    cands = cands[:100]
                st.session_state["candidates"] = cands
                st.success(f"Loaded {len(cands)} candidates successfully.")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

with tab2:
    st.info("Using the bundled sample_candidates.json (first 50 candidates)")
    if st.button("Load Sample Candidates"):
        if os.path.exists("./sample_candidates.json"):
            try:
                with open("./sample_candidates.json", "r", encoding="utf-8") as f:
                    cands = json.load(f)
                    st.session_state["candidates"] = cands[:100]
                    st.success(f"Loaded {len(st.session_state['candidates'])} sample candidates.")
            except Exception as e:
                st.error(f"Error loading sample file: {e}")
        else:
            st.error("sample_candidates.json not found. Please upload candidates manually.")


# =============================================================================
# SECTION 2 — CONFIGURATION (Sidebar)
# =============================================================================

st.sidebar.header("Scoring Weights")
raw_jd_weight = st.sidebar.slider("JD Match (BM25 + Skills)", 0.0, 1.0, 0.50, 0.05)
raw_exp_weight = st.sidebar.slider("Experience & Background", 0.0, 1.0, 0.35, 0.05)
raw_loc_weight = st.sidebar.slider("Location", 0.0, 1.0, 0.15, 0.05)

auto_normalize = st.sidebar.checkbox("Auto-normalize weights to sum to 1.0", value=True)

if auto_normalize:
    total = raw_jd_weight + raw_exp_weight + raw_loc_weight
    if total > 0:
        jd_match_weight = raw_jd_weight / total
        experience_weight = raw_exp_weight / total
        location_weight = raw_loc_weight / total
    else:
        jd_match_weight, experience_weight, location_weight = 0.333, 0.333, 0.333
    
    st.sidebar.caption(f"Normalized: JD: {jd_match_weight:.2f} | Exp: {experience_weight:.2f} | Loc: {location_weight:.2f}")
else:
    jd_match_weight = raw_jd_weight
    experience_weight = raw_exp_weight
    location_weight = raw_loc_weight

st.sidebar.header("Output")
top_n = st.sidebar.slider("Show top N candidates", 5, 100, 20)
team_id = st.sidebar.text_input("Team ID (for CSV filename)", value="team_xxx")


# =============================================================================
# SECTION 3 — RUN RANKING
# =============================================================================

st.markdown("---")
has_candidates = len(st.session_state.get("candidates", [])) > 0

if st.button("▶ Run Ranking", disabled=not has_candidates, type="primary"):
    candidates = st.session_state["candidates"]
    n_cands = len(candidates)
    
    with st.spinner(f"Ranking {n_cands} candidates..."):
        try:
            # Step 1: Extract features
            results = []
            for cand in candidates:
                feat = extract_features(cand)
                penalty, flags = get_honeypot_penalty(cand)
                signals = cand.get("redrob_signals", {})
                beh = compute_behavioral_score(signals)
                sig_summary = extract_signal_summary(signals)
                meta = build_candidate_meta(cand)
                
                results.append({
                    "candidate_id": cand.get("candidate_id", "UNKNOWN"),
                    "features": feat,
                    "penalty": penalty,
                    "flags": flags,
                    "behavioral_score": beh,
                    "signal_summary": sig_summary,
                    "meta": meta
                })

            # Step 2: Build BM25 index on small batch
            career_texts = [r["features"]["career_text"] for r in results]
            bm25 = get_cached_bm25_index(tuple(career_texts))  # Tuple for hashing
            bm25_scores = score_all_candidates(bm25)

            # Step 3: Score all candidates
            for i in range(n_cands):
                skills = results[i]["features"]["skills_score"]
                exp = results[i]["features"]["experience_score"]
                loc = results[i]["features"]["location_score"]
                beh = results[i]["behavioral_score"]
                penalty = results[i]["penalty"]
                b_score = float(bm25_scores[i])

                jd_match = 0.65 * b_score + 0.35 * skills
                base = (jd_match_weight * jd_match + 
                        experience_weight * exp + 
                        location_weight * loc)
                
                availability = 0.40 + 0.60 * beh
                final = base * availability * penalty
                
                results[i]["final_score"] = final
                results[i]["score_components"] = {
                    "final_score": final,
                    "jd_match_score": jd_match,
                    "skills_score": skills,
                    "experience_score": exp,
                    "behavioral_score": beh,
                    "location_score": loc,
                    "bm25_score": b_score,
                    "honeypot_penalty": penalty,
                    "base_score": base
                }

            # Step 4: Sort and rank
            results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
            
            # Ensure we don't request more than we have
            n_to_show = min(top_n, n_cands)
            ranked_results = []
            
            # Step 5: Generate reasoning for top candidates
            for i in range(n_to_show):
                r = results[i]
                rank = i + 1
                reasoning = generate_reasoning(
                    rank=rank,
                    candidate_id=r["candidate_id"],
                    meta=r["meta"],
                    signals=r["signal_summary"],
                    scores=r["score_components"]
                )
                r["rank"] = rank
                r["reasoning"] = reasoning
                ranked_results.append(r)

            st.session_state["all_results"] = results
            st.session_state["ranked_results"] = ranked_results
            st.success("Ranking complete!")

        except Exception as e:
            st.error(f"Error during ranking pipeline: {e}")
            st.info("Please check your candidates file format.")


# =============================================================================
# SECTION 4 — RESULTS DISPLAY
# =============================================================================

if st.session_state.get("ranked_results"):
    all_res = st.session_state["all_results"]
    ranked_res = st.session_state["ranked_results"]
    
    st.header("Results Analysis")
    
    # 4a — Summary metrics bar
    col1, col2, col3, col4 = st.columns(4)
    knockouts = sum(1 for r in all_res if r["penalty"] == 0.0)
    top_score = ranked_res[0]["final_score"] if ranked_res else 0.0
    avg_beh = sum(r["behavioral_score"] for r in all_res) / len(all_res) if all_res else 0.0
    
    col1.metric("Candidates Ranked", len(all_res))
    col2.metric("Honeypot Knockouts", knockouts)
    col3.metric("Top Score", f"{top_score:.3f}")
    col4.metric("Avg Behavioral Score", f"{avg_beh:.3f}")

    # 4b — Score distribution chart
    st.subheader("Final Score Distribution (all candidates)")
    scores_only = [r["final_score"] for r in all_res]
    st.bar_chart(scores_only)

    # 4c — Top N results table
    st.subheader(f"Top {len(ranked_res)} Candidates")
    if pd:
        df_data = []
        for r in ranked_res:
            df_data.append({
                "Rank": r["rank"],
                "Candidate ID": r["candidate_id"],
                "Title": r["meta"]["title"],
                "Company": r["meta"]["company"],
                "YoE": r["meta"]["yoe"],
                "Top Skills": ", ".join(r["meta"]["top_skills"][:3]),
                "Final Score": round(r["final_score"], 4),
                "Reasoning": r["reasoning"]
            })
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("Pandas not installed. Tabular view disabled.")

    # 4d — Expandable detail cards
    st.subheader("Top 5 Deep Dive")
    for r in ranked_res[:5]:
        title = r["meta"]["title"] or "Unknown Title"
        company = r["meta"]["company"] or "Unknown Company"
        exp_title = f"#{r['rank']} — {title} @ {company} (Score: {r['final_score']:.3f})"
        
        with st.expander(exp_title):
            # Breakdown Chart
            sc = r["score_components"]
            st.markdown("**Score Breakdown**")
            
            if go:
                fig = go.Figure(go.Bar(
                    x=[sc["jd_match_score"], sc["experience_score"], sc["behavioral_score"], sc["location_score"]],
                    y=['JD Match', 'Experience', 'Behavioral', 'Location'],
                    orientation='h'
                ))
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.progress(sc["jd_match_score"], text=f"JD Match: {sc['jd_match_score']:.2f}")
                st.progress(sc["experience_score"], text=f"Experience: {sc['experience_score']:.2f}")
                st.progress(sc["behavioral_score"], text=f"Behavioral: {sc['behavioral_score']:.2f}")
                st.progress(sc["location_score"], text=f"Location: {sc['location_score']:.2f}")

            st.markdown("**Signal Summary**")
            sig = r["signal_summary"]
            scol1, scol2 = st.columns(2)
            scol1.write(f"- Notice: {sig.get('notice_label')}")
            scol1.write(f"- Active: {sig.get('days_since_active')} days ago")
            scol1.write(f"- Open to Work: {sig.get('open_to_work')}")
            scol2.write(f"- Response Rate: {sig.get('recruiter_response_rate_pct')}%")
            scol2.write(f"- Interview Completion: {sig.get('interview_completion_pct')}%")
            scol2.write(f"- Verified: {sig.get('verification_label')}")

            st.markdown("**Reasoning**")
            st.info(r["reasoning"])
            
            if r["flags"] or r["penalty"] < 1.0:
                st.warning(f"Flags Triggered: {', '.join(r['flags'])} (Penalty Modifier: {r['penalty']})")


# =============================================================================
# SECTION 5 — DOWNLOAD
# =============================================================================

if st.session_state.get("ranked_results"):
    st.markdown("---")
    st.subheader("Downloads")
    
    col_dl1, col_dl2 = st.columns(2)
    
    # Generate CSV content
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in st.session_state["ranked_results"]:
        writer.writerow([
            r["candidate_id"],
            r["rank"],
            f"{r['final_score']:.6f}",
            r["reasoning"]
        ])
    
    col_dl1.download_button(
        label="⬇ Download Submission CSV",
        data=csv_buffer.getvalue().encode('utf-8'),
        file_name=f"{team_id}.csv",
        mime="text/csv",
        type="primary"
    )
    
    # Generate JSON debug content
    json_data = json.dumps(st.session_state["all_results"], indent=2)
    col_dl2.download_button(
        label="⬇ Download Full Scores JSON",
        data=json_data.encode('utf-8'),
        file_name="debug_scores.json",
        mime="application/json"
    )