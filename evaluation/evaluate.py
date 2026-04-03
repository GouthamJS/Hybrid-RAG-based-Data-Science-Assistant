"""
Evaluates the true hybrid RAG pipeline on standard DS/ML questions.
Checks JSON structure, sources presence, confidence extraction, and avoidance
of hallucination triggers (like directly mentioning "fallback").
"""
import os
import sys
import json
import uuid
import pandas as pd

# Add project root to sys.path to allow importing from the 'rag' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.chain import HybridRAGChain

def run_evaluations():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    questions_path = os.path.join(base_dir, "evaluation", "test_questions.json")
    
    with open(questions_path, "r") as f:
        questions = json.load(f)
        
    chain = HybridRAGChain()
    results = []
    
    print("-" * 80)
    print(" EVALUATION SUITE STARTING ".center(80, "="))
    print("-" * 80)
    
    session_id = str(uuid.uuid4())
    
    for q in questions:
        q_text = q["question"]
        print(f"\nProcessing Q{q['id']}: {q_text}")
        
        passed = True
        reason = []
        
        try:
            resp = chain.process_query(q_text, session_id)
            
            # Check 1: Answer exists and is not raw error string
            if "answer" not in resp or not isinstance(resp["answer"], str):
                passed = False
                reason.append("Missing/Invalid 'answer'")
                
            # Check 2: Sources logic
            if resp.get("intent") != "csv_query":
                # For non-CSV queries, if doing well, sources exist
                # Unless it triggered fallback limit
                conf_str = str(resp.get("confidence", "0%")).replace('%', '')
                conf = float(conf_str)
                if conf >= 40.0 and not resp.get("sources"):
                    passed = False
                    reason.append("High confidence but no sources listed")
                    
            # Check 3: Confidence format
            if "confidence" not in resp or not isinstance(resp["confidence"], str) or "%" not in resp["confidence"]:
                passed = False
                reason.append("Invalid 'confidence' format")
                
            # Check 4: Hallucination trigger
            # The system prompt instructions must be followed, so we check if LLM hallucinated
            if "hallucin" in resp.get("answer", "").lower():
                passed = False
                reason.append("Contains hallucination trigger text")
                
        except Exception as e:
            passed = False
            reason.append(f"Exception: {str(e)}")
            
        print(f"Result: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(f"Reason: {', '.join(reason)}")
            
        results.append({
            "Q_id": q["id"],
            "Question": q_text,
            "Passed": passed,
            "Reason": ", ".join(reason) if reason else "Valid output structure"
        })
        
    print("\n" + "=" * 80)
    print(" EVALUATION SUMMARY ".center(80, "="))
    print("=" * 80)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_evaluations()
