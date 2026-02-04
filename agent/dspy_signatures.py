import dspy
from typing import List, Optional

class GenerateSQL(dspy.Signature):
    """Generate a valid SQLite query for the Northwind retail database.
    
    CRITICAL SQLite rules:
    - Use double quotes for table names with spaces: "Order Details"
    - Use strftime('%Y', OrderDate) for year extraction, NOT YEAR()
    - Use BETWEEN for date ranges: OrderDate BETWEEN '2017-06-01' AND '2017-06-30'
    - Join tables: Orders -> "Order Details" -> Products -> Categories
    - Revenue formula: SUM(od.UnitPrice * od.Quantity * (1 - od.Discount))
    - Gross Margin: SUM((od.UnitPrice * 0.3) * od.Quantity * (1 - od.Discount)) (30% margin = UnitPrice - 0.7*UnitPrice)
    - When asked for product names, SELECT p.ProductName NOT p.ProductID
    - When asked for customer names, SELECT c.CompanyName NOT c.CustomerID
    - When asked for category names, SELECT c.CategoryName NOT c.CategoryID
    """
    question = dspy.InputField(desc="The user's question about retail analytics.")
    db_schema = dspy.InputField(desc="The database schema (tables and columns).")
    constraints = dspy.InputField(desc="Extracted constraints like date ranges, KPI formulas from docs.", default="")
    sql_query = dspy.OutputField(desc="A valid SQLite query. Return names not IDs. No markdown code blocks.")

class SynthesizeAnswer(dspy.Signature):
    """Synthesize a final answer based on SQL results or RAG context."""
    question = dspy.InputField()
    context = dspy.InputField(desc="Retrieved document chunks or SQL results.")
    format_hint = dspy.InputField(desc="The expected format of the answer (e.g., int, float, list).")
    final_answer = dspy.OutputField(desc="The final answer matching the format hint.")
    explanation = dspy.OutputField(desc="A brief explanation (<= 2 sentences).")
    citations = dspy.OutputField(desc="List of DB tables and doc chunks used.")

class Router(dspy.Signature):
    """Decide whether to use RAG, SQL, or Hybrid approach."""
    question = dspy.InputField()
    choice = dspy.OutputField(desc="One of: 'rag', 'sql', 'hybrid'")

class Planner(dspy.Signature):
    """Extract constraints and entities from the question."""
    question = dspy.InputField()
    context = dspy.InputField(desc="Retrieved document chunks.")
    constraints = dspy.OutputField(desc="Extracted constraints (dates, KPIs, categories).")
