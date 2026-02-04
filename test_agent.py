"""
Comprehensive Test Script for Retail Analytics Copilot
======================================================

This script tests the agent against known questions with pre-computed correct answers.
Run with: uv run python test_agent.py

Tests include:
- Original 6 evaluation questions (from assignment)
- Extended test questions
- Edge case queries (least selling, most consistent, etc.)
"""

import json
import uuid
from typing import Any, Dict, List, Tuple
from agent.graph_hybrid import app

# ============================================================================
# TEST CASES WITH EXPECTED ANSWERS
# ============================================================================

# Pre-computed correct answers from direct SQL queries
TEST_CASES = [
    # --- ORIGINAL 6 EVALUATION QUESTIONS (from sample_questions_hybrid_eval.jsonl) ---
    {
        "id": "rag_policy_beverages_return_days",
        "question": "According to the product policy, what is the return window (days) for unopened Beverages? Return an integer.",
        "format_hint": "int",
        "expected": 14,
        "tolerance": 0,  # Exact match for int
        "category": "eval"
    },
    {
        "id": "hybrid_top_category_qty_summer_1997",
        "question": "During 'Summer Beverages 2017' as defined in the marketing calendar, which product category had the highest total quantity sold? Return {category:str, quantity:int}.",
        "format_hint": "{category:str, quantity:int}",
        "expected": {"category": "Confections", "quantity": 17791},
        "tolerance": 100,  # Allow some variance in quantity
        "category": "eval"
    },
    {
        "id": "hybrid_aov_winter_1997",
        "question": "Using the AOV definition from the KPI docs, what was the Average Order Value during 'Winter Classics 2017'? Return a float rounded to 2 decimals.",
        "format_hint": "float",
        "expected": 21032.34,
        "tolerance": 100,  # AOV can have slight variance
        "category": "eval"
    },
    {
        "id": "sql_top3_products_by_revenue_alltime",
        "question": "Top 3 products by total revenue all-time. Revenue uses Order Details: SUM(UnitPrice*Quantity*(1-Discount)). Return list[{product:str, revenue:float}].",
        "format_hint": "list[{product:str, revenue:float}]",
        "expected": [
            {"product": "Côte de Blaye", "revenue": 53265895.23},
            {"product": "Thüringer Rostbratwurst", "revenue": 24623469.23},
            {"product": "Mishi Kobe Niku", "revenue": 19423037.5}
        ],
        "tolerance": 1000,  # Allow revenue variance
        "category": "eval"
    },
    {
        "id": "hybrid_revenue_beverages_summer_1997",
        "question": "Total revenue from the 'Beverages' category during 'Summer Beverages 2017' dates. Return a float rounded to 2 decimals.",
        "format_hint": "float",
        "expected": 591887.18,
        "tolerance": 5000,  # Allow some variance
        "category": "eval"
    },
    {
        "id": "hybrid_best_customer_margin_1997",
        "question": "Per the KPI definition of gross margin, who was the top customer by gross margin in 2017? Assume CostOfGoods is approximated by 70% of UnitPrice if not available. Return {customer:str, margin:float}.",
        "format_hint": "{customer:str, margin:float}",
        "expected": {"customer": "Wilman Kala", "margin": 251847.49},
        "tolerance": 1000,
        "category": "eval"
    },
    
    # --- EXTENDED TEST QUESTIONS ---
    {
        "id": "rag_policy_perishables_return_days",
        "question": "According to the product policy, what is the return window (days) for unopened Perishables? Return an integer.",
        "format_hint": "int",
        "expected": [3, 7],  # "3-7 days" - accept either min or max
        "tolerance": 0,
        "category": "extended",
        "multi_valid": True  # Flag for multiple valid answers
    },
    {
        "id": "sql_total_orders_count",
        "question": "How many total orders are in the database?",
        "format_hint": "int",
        "expected": 16282,
        "tolerance": 0,
        "category": "extended"
    },
    {
        "id": "hybrid_orders_count_summer_2017",
        "question": "How many orders were placed during 'Summer Beverages 2017' campaign (June 1-30, 2017)?",
        "format_hint": "int",
        "expected": 131,
        "tolerance": 5,
        "category": "extended"
    },
    {
        "id": "hybrid_top_product_qty_summer_2017",
        "question": "During 'Summer Beverages 2017' as defined in the marketing calendar, which product had the highest total quantity sold? Return as {product:str, quantity:int}.",
        "format_hint": "{product:str, quantity:int}",
        "expected": {"product": "Sir Rodney's Scones", "quantity": 1701},
        "tolerance": 100,
        "category": "extended"
    },
    {
        "id": "hybrid_revenue_confections_winter_2017",
        "question": "Total revenue from the 'Confections' category during 'Winter Classics 2017' campaign (Dec 1-31, 2017).",
        "format_hint": "float",
        "expected": 497871.04,
        "tolerance": 10000,
        "category": "extended"
    },
    
    # --- EDGE CASE QUERIES ---
    {
        "id": "edge_least_selling_product",
        "question": "What is the least selling product by revenue all-time?",
        "format_hint": "{product:str, revenue:float}",
        "expected": {"product": "Geitost", "revenue": 507120.63},
        "tolerance": 1000,
        "category": "edge"
    },
    {
        "id": "edge_most_consistent_product",
        "question": "What is the most consistent selling product (appears in most orders)?",
        "format_hint": "{product:str, quantity:int}",
        "expected": {"product": "Louisiana Hot Spiced Okra", "quantity": 8040},
        "tolerance": 100,
        "category": "edge"
    },
    {
        "id": "edge_invalid_product",
        "question": "What is the revenue for thingy in 2017?",
        "format_hint": "float",
        "expected": None,  # Should return None for non-existent product
        "tolerance": 0,
        "category": "edge",
        "expect_failure": True  # We expect validation to fail
    },
]


def run_agent(test_case: Dict) -> Dict:
    """Run the agent on a single test case."""
    # Add a unique ID to bust Ollama's response cache
    cache_buster = f"[Query ID: {uuid.uuid4().hex[:8]}] "
    
    initial_state = {
        "question": cache_buster + test_case["question"],
        "format_hint": test_case["format_hint"],
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
        "citations": [],
        "_validation_failed": False,
        "_validation_msg": "",
        "_confidence_override": None
    }
    
    try:
        result = app.invoke(initial_state)
        return {
            "id": test_case["id"],
            "answer": result.get("final_answer"),
            "sql": result.get("sql_query", ""),
            "explanation": result.get("explanation", ""),
            "validation_failed": result.get("_validation_failed", False),
            "error": None
        }
    except Exception as e:
        return {
            "id": test_case["id"],
            "answer": None,
            "sql": "",
            "explanation": "",
            "validation_failed": False,
            "error": str(e)
        }


def check_answer(result: Dict, test_case: Dict) -> Tuple[bool, str]:
    """Check if the answer matches expected value within tolerance."""
    answer = result["answer"]
    expected = test_case["expected"]
    tolerance = test_case.get("tolerance", 0)
    expect_failure = test_case.get("expect_failure", False)
    
    # Handle expected failures (like invalid product queries)
    if expect_failure:
        if result["validation_failed"] or answer is None:
            return True, "Correctly rejected invalid query"
        return False, f"Should have failed but got: {answer}"
    
    if answer is None and expected is not None:
        return False, f"Got None, expected {expected}"
    
    if answer is None and expected is None:
        return True, "Both None"
    
    # Handle multiple valid answers
    multi_valid = test_case.get("multi_valid", False)
    if multi_valid and isinstance(expected, list):
        for valid_answer in expected:
            try:
                if isinstance(valid_answer, int):
                    ans_val = int(answer) if not isinstance(answer, int) else answer
                    if ans_val == valid_answer:
                        return True, f"{ans_val} is a valid answer (accepts: {expected})"
                elif isinstance(valid_answer, float):
                    ans_val = float(answer) if not isinstance(answer, float) else answer
                    if abs(ans_val - valid_answer) <= tolerance:
                        return True, f"{ans_val} ~= {valid_answer}"
                elif str(answer).lower() == str(valid_answer).lower():
                    return True, f"'{answer}' matches"
            except:
                continue
        return False, f"{answer} not in valid answers: {expected}"
    
    # Check based on type
    if isinstance(expected, int):
        try:
            ans_val = int(answer) if not isinstance(answer, int) else answer
            if abs(ans_val - expected) <= tolerance:
                if ans_val == expected:
                    return True, f"{ans_val} == {expected}"
                return True, f"{ans_val} ~= {expected} (within tolerance: {tolerance})"
            return False, f"{ans_val} != {expected} (tolerance: {tolerance})"
        except:
            return False, f"Could not convert {answer} to int"
    
    elif isinstance(expected, float):
        try:
            ans_val = float(answer) if not isinstance(answer, float) else answer
            if abs(ans_val - expected) <= tolerance:
                return True, f"{ans_val} ~= {expected}"
            return False, f"{ans_val} != {expected} (tolerance: {tolerance})"
        except:
            return False, f"Could not convert {answer} to float"
    
    elif isinstance(expected, dict):
        if not isinstance(answer, dict):
            return False, f"Expected dict, got {type(answer)}"
        
        # Check each key
        for key, exp_val in expected.items():
            if key not in answer:
                return False, f"Missing key: {key}"
            
            ans_val = answer[key]
            if isinstance(exp_val, (int, float)):
                try:
                    if abs(float(ans_val) - float(exp_val)) > tolerance:
                        return False, f"{key}: {ans_val} != {exp_val}"
                except:
                    return False, f"Could not compare {key}"
            elif isinstance(exp_val, str):
                # String comparison - check if it's contained (fuzzy match)
                if exp_val.lower() not in str(ans_val).lower() and str(ans_val).lower() not in exp_val.lower():
                    return False, f"{key}: '{ans_val}' != '{exp_val}'"
        
        return True, "Dict matches"
    
    elif isinstance(expected, list):
        if not isinstance(answer, list):
            return False, f"Expected list, got {type(answer)}"
        
        if len(answer) != len(expected):
            return False, f"List length {len(answer)} != {len(expected)}"
        
        # Check each item
        for i, (ans_item, exp_item) in enumerate(zip(answer, expected)):
            if isinstance(exp_item, dict):
                for key, exp_val in exp_item.items():
                    if key not in ans_item:
                        return False, f"Item {i} missing key: {key}"
                    ans_val = ans_item[key]
                    if isinstance(exp_val, (int, float)):
                        if abs(float(ans_val) - float(exp_val)) > tolerance:
                            return False, f"Item {i}.{key}: {ans_val} != {exp_val}"
        
        return True, "List matches"
    
    else:
        # String comparison
        if str(answer).lower() == str(expected).lower():
            return True, "String match"
        return False, f"'{answer}' != '{expected}'"


def run_tests(categories: List[str] = None, verbose: bool = True):
    """Run all tests or specific categories."""
    if categories is None:
        categories = ["eval", "extended", "edge"]
    
    results = {
        "passed": 0,
        "failed": 0,
        "total": 0,
        "details": []
    }
    
    # Filter test cases by category
    tests_to_run = [t for t in TEST_CASES if t["category"] in categories]
    
    print("=" * 70)
    print("RETAIL ANALYTICS COPILOT - TEST SUITE")
    print("=" * 70)
    print(f"Running {len(tests_to_run)} tests in categories: {categories}")
    print()
    
    for test_case in tests_to_run:
        results["total"] += 1
        
        if verbose:
            print(f"\n{'─' * 70}")
            print(f"TEST: {test_case['id']}")
            print(f"Q: {test_case['question'][:80]}...")
        
        # Run agent
        result = run_agent(test_case)
        
        # Check answer
        passed, reason = check_answer(result, test_case)
        
        if passed:
            results["passed"] += 1
            status = "✓ PASS"
        else:
            results["failed"] += 1
            status = "✗ FAIL"
        
        results["details"].append({
            "id": test_case["id"],
            "category": test_case["category"],
            "passed": passed,
            "reason": reason,
            "answer": result["answer"],
            "expected": test_case["expected"],
            "sql": result.get("sql", "")[:100]
        })
        
        if verbose:
            print(f"Expected: {test_case['expected']}")
            print(f"Got:      {result['answer']}")
            print(f"Status:   {status} - {reason}")
            if result.get("error"):
                print(f"Error:    {result['error']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total:  {results['total']}")
    print(f"Passed: {results['passed']} ({100*results['passed']/results['total']:.1f}%)")
    print(f"Failed: {results['failed']} ({100*results['failed']/results['total']:.1f}%)")
    
    # Category breakdown
    print("\nBy Category:")
    for cat in categories:
        cat_tests = [d for d in results["details"] if d["category"] == cat]
        cat_passed = sum(1 for d in cat_tests if d["passed"])
        print(f"  {cat}: {cat_passed}/{len(cat_tests)}")
    
    # Failed tests detail
    if results["failed"] > 0:
        print("\nFailed Tests:")
        for detail in results["details"]:
            if not detail["passed"]:
                print(f"  - {detail['id']}: {detail['reason']}")
    
    return results


def run_quick_test():
    """Run only the 6 original evaluation questions."""
    return run_tests(categories=["eval"], verbose=True)


def run_full_test():
    """Run all tests including extended and edge cases."""
    return run_tests(categories=["eval", "extended", "edge"], verbose=True)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            run_quick_test()
        elif sys.argv[1] == "--eval":
            run_tests(categories=["eval"])
        elif sys.argv[1] == "--edge":
            run_tests(categories=["edge"])
        elif sys.argv[1] == "--extended":
            run_tests(categories=["extended"])
        else:
            print("Usage: python test_agent.py [--quick|--eval|--edge|--extended]")
            print("  --quick     Run only 6 eval questions")
            print("  --eval      Run evaluation questions")
            print("  --edge      Run edge case tests")
            print("  --extended  Run extended tests")
            print("  (no args)   Run all tests")
    else:
        run_full_test()
