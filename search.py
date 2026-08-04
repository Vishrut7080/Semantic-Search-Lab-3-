from pathlib import Path
import json, os, numpy as np
from embeddings import get_embedding, EMBEDDING_MODE

path=Path('documents.json')
CACHE_PATH=Path('embeddings_cache.json')

def load_documents(path):
    """Read the corpus JSON file and return it as a list of dicts
    (each with 'id', 'topic', 'text')."""
    with open(path,'r') as file:
        # if it was a normal file
        # content=file.read()

        # for json
        content=json.load(file)     
    return content

def _load_cache():
    """Load the on-disk embeddings cache, if it exists.
    Returns an empty dict on first run, on missing file, or on a
    corrupted/unreadable cache file (so a bad cache never crashes the pipeline —
    it just gets treated as 'nothing cached yet' and gets rebuilt)."""
    if not CACHE_PATH.exists():
        return {}
    try:
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

def _save_cache(cache):
    """Persist the full cache dict (all modes) back to disk as JSON."""
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f)

def build_embedding_matrix(documents):
    # cache is split by embedding mode so switching offline <-> api
    # doesn't reuse the wrong kind of vector

    cache=_load_cache()
    mode_cache=cache.setdefault(EMBEDDING_MODE, {})

    embeddings=[]
    cache_dirty=False

    for doc in documents:
        doc_id=str(doc['id'])
        text=doc['text']

        cache_entry=mode_cache.get(doc_id)

        if cache_entry is not None and cache_entry.get('text')==text:
            vec=cache_entry['embedding']
        else:
            vec=get_embedding(text, input_type='passage')
            mode_cache[doc_id]={
                'text':text,
                'embedding': vec
                }
            cache_dirty=True

        embeddings.append(vec)

    if cache_dirty:
        _save_cache(cache)

    return np.array(embeddings, dtype=np.float64)

def cosine_similarity(vector, matrix):
    """Cosine similarity between a single query vector and every row of matrix.
    Returns a 1D array of scores, one per document."""
    vector_norm=np.linalg.norm(vector)
    matrix_norms=np.linalg.norm(matrix, axis=1)
    denom=matrix_norms*vector_norm
    denom[denom==0]=1e-10
    return (matrix@vector)/denom

def search(query, embedding_matrix, documents, top_k):
    """Embed the query, score it against every document, and return the
    top_k most similar documents with their scores."""
    query_vec=np.array(get_embedding(query, input_type='query'), dtype=np.float64)
    scores=cosine_similarity(query_vec, embedding_matrix)

    top_indices=np.argsort(-scores)[:top_k]

    results=[]

    for idx in top_indices:
        doc=documents[idx]

        results.append({
            'id': doc['id'],
            'topic':doc['topic'],
            'text': doc['text'],
            'score': float(scores[idx])
        })

    return results

if __name__=='__main__':
    # simple manual smoke test when running `python search.py` directly
    documents=load_documents(path)
    embedding_matrix=build_embedding_matrix(documents)
    print(f'Loaded {len(documents)} documents.')
    print(f'Embedding matrix shape: {embedding_matrix.shape}')
    sample_text=input('Enter a sample text passage: \n')
    response=search(sample_text, embedding_matrix, documents, 3)[0]['topic']
    print(f'Output: {response}')