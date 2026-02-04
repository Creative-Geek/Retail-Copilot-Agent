import dspy
import uuid
from dspy.teleprompt import BootstrapFewShot
from agent.dspy_signatures import GenerateSQL
from agent.tools.sqlite_tool import SQLiteTool

# --- Configuration ---
lm = dspy.LM('ollama_chat/phi3.5:3.8b-mini-instruct-q4_K_M', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

def add_cache_buster(question: str) -> str:
    """Add a unique ID to bust Ollama's response cache."""
    return f"[Query ID: {uuid.uuid4().hex[:8]}] {question}"

def evaluate_baseline(module, test_data, tool):
    """Evaluate model before optimization."""
    valid_count = 0
    for example in test_data:
        try:
            pred = module(
                question=add_cache_buster(example.question),
                db_schema=example.db_schema,
                constraints=example.constraints
            )
            sql = pred.sql_query
            if sql:
                # Clean up markdown if present
                if sql.startswith("```"):
                    import re
                    sql = re.sub(r'^```\w*\n?', '', sql)
                    sql = re.sub(r'\n?```$', '', sql)
                result = tool.execute_sql(sql)
                if result["error"] is None:
                    valid_count += 1
        except Exception as e:
            pass
    return valid_count / len(test_data) if test_data else 0

def optimize_sql_module():
    # 1. Define Training Data (Question, Schema -> SQL)
    tool = SQLiteTool()
    schema = tool.get_schema()
    
    # High-quality examples with correct SQLite syntax for Northwind
    # Note: Cache busters are added at runtime to prevent Ollama caching
    train_data_raw = [
        dspy.Example(
            question="What is the total revenue from 'Beverages'?",
            db_schema=schema,
            constraints="",
            sql_query='SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as Revenue FROM "Order Details" od JOIN Products p ON od.ProductID = p.ProductID JOIN Categories c ON p.CategoryID = c.CategoryID WHERE c.CategoryName = \'Beverages\''
        ).with_inputs("question", "db_schema", "constraints"),
        
        dspy.Example(
            question="List top 3 products by unit price.",
            db_schema=schema,
            constraints="",
            sql_query="SELECT ProductName, UnitPrice FROM Products ORDER BY UnitPrice DESC LIMIT 3"
        ).with_inputs("question", "db_schema", "constraints"),
        
        dspy.Example(
            question="How many orders were placed in 2017?",
            db_schema=schema,
            constraints="",
            sql_query="SELECT COUNT(*) FROM Orders WHERE strftime('%Y', OrderDate) = '2017'"
        ).with_inputs("question", "db_schema", "constraints"),
        
        dspy.Example(
            question="What is the average order value?",
            db_schema=schema,
            constraints="AOV = SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID)",
            sql_query='SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID), 2) as AOV FROM Orders o JOIN "Order Details" od ON o.OrderID = od.OrderID'
        ).with_inputs("question", "db_schema", "constraints"),
        
        dspy.Example(
            question="Which customer has the most orders?",
            db_schema=schema,
            constraints="",
            sql_query="SELECT c.CompanyName, COUNT(o.OrderID) as OrderCount FROM Customers c JOIN Orders o ON c.CustomerID = o.CustomerID GROUP BY c.CustomerID ORDER BY OrderCount DESC LIMIT 1"
        ).with_inputs("question", "db_schema", "constraints"),
        
        dspy.Example(
            question="Top 3 products by total revenue all-time",
            db_schema=schema,
            constraints="",
            sql_query='SELECT p.ProductName, ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) as Revenue FROM "Order Details" od JOIN Products p ON od.ProductID = p.ProductID GROUP BY od.ProductID ORDER BY Revenue DESC LIMIT 3'
        ).with_inputs("question", "db_schema", "constraints"),
        
        dspy.Example(
            question="Total revenue from Beverages in June 2017",
            db_schema=schema,
            constraints="Date range: 2017-06-01 to 2017-06-30",
            sql_query='SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) as Revenue FROM Orders o JOIN "Order Details" od ON o.OrderID = od.OrderID JOIN Products p ON od.ProductID = p.ProductID JOIN Categories c ON p.CategoryID = c.CategoryID WHERE c.CategoryName = \'Beverages\' AND o.OrderDate BETWEEN \'2017-06-01\' AND \'2017-06-30\''
        ).with_inputs("question", "db_schema", "constraints"),
        
        dspy.Example(
            question="Which category had highest quantity sold in June 2017?",
            db_schema=schema,
            constraints="Date range: 2017-06-01 to 2017-06-30",
            sql_query='SELECT c.CategoryName, SUM(od.Quantity) as TotalQty FROM Orders o JOIN "Order Details" od ON o.OrderID = od.OrderID JOIN Products p ON od.ProductID = p.ProductID JOIN Categories c ON p.CategoryID = c.CategoryID WHERE o.OrderDate BETWEEN \'2017-06-01\' AND \'2017-06-30\' GROUP BY c.CategoryID ORDER BY TotalQty DESC LIMIT 1'
        ).with_inputs("question", "db_schema", "constraints"),
        
        dspy.Example(
            question="Top customer by gross margin in 2017",
            db_schema=schema,
            constraints="GM = SUM((UnitPrice - 0.7*UnitPrice) * Quantity * (1 - Discount))",
            sql_query='SELECT cu.CompanyName, ROUND(SUM((od.UnitPrice - 0.7 * od.UnitPrice) * od.Quantity * (1 - od.Discount)), 2) as GrossMargin FROM Orders o JOIN "Order Details" od ON o.OrderID = od.OrderID JOIN Customers cu ON o.CustomerID = cu.CustomerID WHERE strftime(\'%Y\', o.OrderDate) = \'2017\' GROUP BY o.CustomerID ORDER BY GrossMargin DESC LIMIT 1'
        ).with_inputs("question", "db_schema", "constraints"),
        
        dspy.Example(
            question="Average Order Value in December 2017",
            db_schema=schema,
            constraints="Date range: 2017-12-01 to 2017-12-31. AOV = SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID)",
            sql_query='SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID), 2) as AOV FROM Orders o JOIN "Order Details" od ON o.OrderID = od.OrderID WHERE o.OrderDate BETWEEN \'2017-12-01\' AND \'2017-12-31\''
        ).with_inputs("question", "db_schema", "constraints"),
    ]
    
    # Add cache busters to prevent Ollama from returning cached responses
    train_data = [
        dspy.Example(
            question=add_cache_buster(ex.question),
            db_schema=ex.db_schema,
            constraints=ex.constraints,
            sql_query=ex.sql_query
        ).with_inputs("question", "db_schema", "constraints")
        for ex in train_data_raw
    ]

    # 2.5 Test data for evaluation (subset of train data)
    test_data = train_data[:5]  # Use first 5 examples for quick evaluation
    
    # 2.6 Evaluate baseline (before optimization)
    print("\n=== BASELINE EVALUATION (Before Optimization) ===")
    baseline_module = dspy.ChainOfThought(GenerateSQL)
    baseline_accuracy = evaluate_baseline(baseline_module, test_data, tool)
    print(f"Baseline Valid SQL Rate: {baseline_accuracy:.1%}")

    # 3. Define Metric
    def validate_sql(example, pred, trace=None):
        # Simple validation: Check if it executes without error
        sql = pred.sql_query
        if not sql:
            return False
        # Clean up markdown if present
        if sql.startswith("```"):
            import re
            sql = re.sub(r'^```\w*\n?', '', sql)
            sql = re.sub(r'\n?```$', '', sql)
        result = tool.execute_sql(sql)
        return result["error"] is None

    # 4. Compile
    print("\n=== DSPy OPTIMIZATION ===")
    print("Compiling GenerateSQL with BootstrapFewShot...")
    teleprompter = BootstrapFewShot(metric=validate_sql, max_bootstrapped_demos=4, max_labeled_demos=4)
    
    module = dspy.ChainOfThought(GenerateSQL)
    
    compiled_module = teleprompter.compile(module, trainset=train_data)
    
    # 5. Evaluate after optimization
    print("\n=== POST-OPTIMIZATION EVALUATION ===")
    optimized_accuracy = evaluate_baseline(compiled_module, test_data, tool)
    print(f"Optimized Valid SQL Rate: {optimized_accuracy:.1%}")
    
    # 6. Print comparison
    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Before: {baseline_accuracy:.1%}")
    print(f"After:  {optimized_accuracy:.1%}")
    print(f"Improvement: {(optimized_accuracy - baseline_accuracy):.1%}")
    
    # 7. Save
    print("\nSaving compiled module...")
    compiled_module.save("agent/compiled_sql_gen.json")
    print("Done. Module saved to agent/compiled_sql_gen.json")

if __name__ == "__main__":
    optimize_sql_module()
