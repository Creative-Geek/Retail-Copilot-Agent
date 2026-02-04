import json
import uuid
import argparse
from agent.graph_hybrid import app, trace_logger
from typing import Dict, Any

def process_question(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single question through the LangGraph agent.
    """
    # Add a unique ID to bust Ollama's response cache
    cache_buster = f"[Query ID: {uuid.uuid4().hex[:8]}] "
    
    initial_state = {
        "question": cache_buster + item["question"],
        "format_hint": item["format_hint"],
        "mode": "",
        "repair_count": 0,
        "errors": [],
        "retrieved_docs": [],
        "constraints": "",
        "sql_query": "",
        "sql_results": {},
        "sql_error": "",
        "final_answer": None,
        "explanation": "",
        "citations": []
    }
    
    try:
        result = app.invoke(initial_state)
        
        # Check for confidence override (e.g., validation failures)
        if result.get("_confidence_override") is not None:
            confidence = result["_confidence_override"]
        else:
            # Calculate confidence based on multiple factors
            confidence = 0.9
            
            # Penalty for errors
            if result.get("errors"):
                confidence -= 0.2 * len(result["errors"])
            
            # Penalty for repair attempts
            if result.get("repair_count", 0) > 0:
                confidence -= 0.1 * result["repair_count"]
            
            # Penalty for null/None answer
            if result.get("final_answer") is None:
                confidence -= 0.4
            
            # Penalty for empty or zero SQL results
            sql_results = result.get("sql_results", {})
            if sql_results:
                rows = sql_results.get("rows", [])
                if len(rows) == 0:
                    # No rows returned
                    confidence -= 0.3
                elif len(rows) == 1 and len(rows[0]) == 1:
                    # Single value result - check if it's zero/null
                    val = rows[0][0]
                    if val is None or val == 0 or val == 0.0:
                        confidence -= 0.2
            
            # Penalty for poor retrieval coverage (hybrid/rag modes)
            mode = result.get("mode", "")
            if mode in ("rag", "hybrid"):
                docs = result.get("retrieved_docs", [])
                if len(docs) == 0:
                    confidence -= 0.3
                elif len(docs) < 2:
                    confidence -= 0.1
        
        confidence = max(0.1, min(1.0, confidence))
        
        return {
            "id": item["id"],
            "final_answer": result.get("final_answer"),
            "sql": result.get("sql_query", ""),
            "confidence": round(confidence, 2),
            "explanation": result.get("explanation", "")[:200],
            "citations": result.get("citations", [])
        }
    except Exception as e:
        print(f"Error processing {item['id']}: {str(e)}")
        return {
            "id": item["id"],
            "final_answer": None,
            "sql": "",
            "confidence": 0.0,
            "explanation": f"Error: {str(e)}",
            "citations": []
        }

def main():
    parser = argparse.ArgumentParser(description="Run Retail Analytics Copilot")
    parser.add_argument("--batch", required=True, help="Input JSONL file")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    args = parser.parse_args()

    results = []
    
    # Read input
    with open(args.batch, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    # Process items
    print(f"Processing {len(items)} questions...")
    for item in items:
        print(f"\n{'='*50}")
        print(f"Running: {item['id']}")
        result = process_question(item)
        results.append(result)
        print(f"Answer: {result['final_answer']}")

    # Write output
    with open(args.out, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    
    # Trace has been written incrementally to agent_trace.jsonl
    print(f"Done. Results written to {args.out}")
    print(f"Trace log written to agent_trace.jsonl (session: {trace_logger.session_id})")

if __name__ == "__main__":
    main()
