# src/bm25_index.py
import re
import pickle
import numpy as np
from rank_bm25 import BM25Okapi

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "i", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "their", "this", "that", "these", "those", "what", "which", "who",
    "how", "when", "where", "than", "also", "such", "into", "through",
    "during", "including", "until", "across", "between", "each", "both",
    "about", "against", "over", "then", "once", "other", "own", "same",
    "so", "just", "more", "most", "very", "its"
}

STEM_MAP = {
    "retrieval": "retriev", "retrieve": "retriev",
    "ranking": "rank", "ranked": "rank", "ranks": "rank",
    "recommendation": "recommend", "recommendations": "recommend",
    "embedding": "embed", "embeddings": "embed",
    "searching": "search", "searches": "search",
    "training": "train", "trained": "train", "trains": "train",
    "building": "build", "built": "build", "builds": "build",
    "shipping": "ship", "shipped": "ship", "ships": "ship"
}

JD_QUERY = """
retrieval ranking recommendation embeddings vector search
retrieval ranking recommendation embeddings vector search
faiss pinecone qdrant weaviate elasticsearch vector database
python pytorch tensorflow machine learning applied ml
production ml systems end to end pipeline deployment
two tower model hnsw ann approximate nearest neighbor
nlp natural language processing transformers sentence transformers bert
xgboost lightgbm gradient boosting feature engineering
ndcg mrr map evaluation offline online ab testing
search engine information retrieval semantic search dense retrieval
ship build product company real users scale latency
senior engineer founding team applied scientist ml engineer
"""

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def tokenize(text: str) -> list[str]:
    """
    Tokenize a single text string. Used both for indexing and querying.
    - Lowercase
    - Replace punctuation with spaces (except hyphens between words)
    - Split on whitespace
    - Remove tokens in STOPWORDS
    - Apply STEM_MAP for key term normalization
    - Remove tokens shorter than 2 characters
    """
    if not text:
        return []
        
    # Lowercase and replace non-alphanumeric/non-hyphen characters with spaces
    text = text.lower()
    text = re.sub(r'[^a-z0-9\-]', ' ', text)
    
    tokens = text.split()
    processed_tokens = []
    
    for token in tokens:
        # Remove standalone hyphens or trailing/leading hyphens
        token = token.strip('-')
        
        if len(token) < 2 or token in STOPWORDS:
            continue
            
        # Apply basic stemming map
        token = STEM_MAP.get(token, token)
        processed_tokens.append(token)
        
    return processed_tokens

def build_bm25_index(career_texts: list[str]) -> BM25Okapi:
    """
    Build a BM25 index from a list of career_text strings.
    Tokenizes the entire corpus and instantiates a rank_bm25.BM25Okapi object.
    """
    print(f"Building BM25 index over {len(career_texts)} documents...")
    
    tokenized_corpus = [tokenize(text) for text in career_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print("BM25 index built successfully.")
    return bm25

def score_all_candidates(bm25: BM25Okapi, jd_query: str = None) -> np.ndarray:
    """
    Score all candidates in the index against the JD query.
    Returns a numpy array of shape (n_candidates,) with min-max normalized BM25 scores.
    """
    if jd_query is None:
        jd_query = JD_QUERY
        
    tokenized_query = tokenize(jd_query)
    scores = np.array(bm25.get_scores(tokenized_query))
    
    scores_min = scores.min()
    scores_max = scores.max()
    
    if scores_max > scores_min:
        normalized_scores = (scores - scores_min) / (scores_max - scores_min)
        return normalized_scores
    else:
        return np.zeros_like(scores)

def save_bm25(bm25: BM25Okapi, path: str) -> None:
    """Save BM25 index to disk using pickle."""
    with open(path, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved to {path}")

def load_bm25(path: str) -> BM25Okapi:
    """Load BM25 index from disk."""
    with open(path, 'rb') as f:
        bm25 = pickle.load(f)
    print(f"BM25 index loaded from {path}")
    return bm25


# ==============================================================================
# MAIN TEST BLOCK
# ==============================================================================
if __name__ == "__main__":
    # Fake candidate career_text strings
    # 2 highly relevant, 3 completely irrelevant
    dummy_career_texts = [
        # Candidate 0: Relevant
        "Senior AI Engineer built retrieval ranking recommendation systems using python pytorch embeddings faiss pinecone. Shipped end to end ml systems measuring ndcg mrr.",
        # Candidate 1: Irrelevant
        "Graphic Designer photoshop illustrator indesign branding logos typography creative direction. visual arts graphic design UI UX.",
        # Candidate 2: Irrelevant
        "Marketing Manager social media SEO content strategy facebook ads google analytics. led campaigns increasing engagement.",
        # Candidate 3: Relevant
        "Applied Scientist semantic search natural language processing transformers. ranked two tower models vector database qdrant tensorflow elasticsearch scale latency.",
        # Candidate 4: Irrelevant
        "Financial Analyst excel financial modeling accounting forecasting budgets. CPA investment banking private equity."
    ]

    print("--- Tokenizer Test ---")
    print(f"Original: {dummy_career_texts[0]}")
    print(f"Tokenized: {tokenize(dummy_career_texts[0])}\n")

    # Build and score
    bm25_index = build_bm25_index(dummy_career_texts)
    normalized_scores = score_all_candidates(bm25_index)

    print("\n--- BM25 Scores ---")
    for i, score in enumerate(normalized_scores):
        print(f"Candidate {i}: Score = {score:.4f} -> {dummy_career_texts[i][:50]}...")
        
    assert normalized_scores[0] > normalized_scores[1], "Relevant candidate 0 should score higher than irrelevant candidate 1"
    assert normalized_scores[3] > normalized_scores[2], "Relevant candidate 3 should score higher than irrelevant candidate 2"