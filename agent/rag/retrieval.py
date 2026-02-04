import os
from typing import List, Dict
from rank_bm25 import BM25Okapi
import glob
import re

class Retriever:
    def __init__(self, docs_dir: str = "docs/"):
        self.docs_dir = docs_dir
        self.chunks: List[Dict] = []
        self.bm25 = None
        self._load_and_chunk_docs()

    def _load_and_chunk_docs(self):
        """
        Loads markdown files from docs_dir and splits them into logical chunks.
        Uses section-based chunking (## headers) to keep related content together.
        """
        md_files = glob.glob(os.path.join(self.docs_dir, "*.md"))
        
        for file_path in md_files:
            filename = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Split by ## headers to keep sections together
            # This ensures "## Summer Beverages 2017" stays with its dates
            sections = re.split(r'\n(?=## )', content)
            
            for i, section in enumerate(sections):
                section = section.strip()
                if section:
                    self.chunks.append({
                        "id": f"{filename}::section{i}",
                        "content": section,
                        "source": filename
                    })
        
        # Initialize BM25
        tokenized_corpus = [chunk["content"].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieves top_k relevant chunks for the query.
        Augments query with synonyms for better recall.
        """
        if not self.chunks:
            return []
        
        # Query augmentation for better recall
        augmented_query = query.lower()
        if "policy" in augmented_query:
            augmented_query += " return returns"
        if "return" in augmented_query:
            augmented_query += " policy"
        if "beverages" in augmented_query or "beverage" in augmented_query:
            augmented_query += " beverages unopened"
            
        tokenized_query = augmented_query.split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_n_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = scores[idx]
            results.append(chunk)
            
        return results

if __name__ == "__main__":
    # Test the retriever
    retriever = Retriever()
    print(f"Loaded {len(retriever.chunks)} chunks.")
    
    query = "return policy for beverages"
    results = retriever.retrieve(query)
    print(f"\nQuery: {query}")
    for res in results:
        print(f"- [{res['score']:.4f}] {res['id']}: {res['content'][:50]}...")
