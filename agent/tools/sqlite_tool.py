import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional

class SQLiteTool:
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = db_path

    def execute_sql(self, sql: str) -> Dict[str, Any]:
        """
        Executes a SQL query and returns the results (columns, rows) or error.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            return {
                "columns": columns,
                "rows": rows,
                "error": None
            }
        except Exception as e:
            return {
                "columns": [],
                "rows": [],
                "error": str(e)
            }

    def get_schema(self) -> str:
        """
        Returns a string representation of the database schema (tables and columns).
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_str = ""
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info('{table_name}');")
                columns = cursor.fetchall()
                
                column_names = [col[1] for col in columns]
                schema_str += f"Table: {table_name}\nColumns: {', '.join(column_names)}\n\n"
            
            conn.close()
            return schema_str
        except Exception as e:
            return f"Error retrieving schema: {str(e)}"

if __name__ == "__main__":
    # Test the tool
    tool = SQLiteTool()
    print("Schema:")
    print(tool.get_schema())
    print("\nTest Query (SELECT * FROM Products LIMIT 2):")
    print(tool.execute_sql("SELECT * FROM Products LIMIT 2"))
