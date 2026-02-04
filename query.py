#!/usr/bin/env python
"""
Retail Analytics Copilot - GUI Query Interface

A simple one-turn query interface. Type your question, get an answer.
Note: Each query is independent - there is no conversation history.
"""

import wx
import threading
from rewx import Component, wsx, render, create_element
from rewx import components as c
from agent.graph_hybrid import app as agent_app


def calculate_confidence(result: dict) -> float:
    """Calculate confidence based on multiple factors."""
    confidence = 0.9
    
    if result.get("errors"):
        confidence -= 0.2 * len(result["errors"])
    
    if result.get("repair_count", 0) > 0:
        confidence -= 0.1 * result["repair_count"]
    
    if result.get("final_answer") is None:
        confidence -= 0.4
    
    sql_results = result.get("sql_results", {})
    if sql_results:
        rows = sql_results.get("rows", [])
        if len(rows) == 0:
            confidence -= 0.3
        elif len(rows) == 1 and len(rows[0]) == 1:
            val = rows[0][0]
            if val is None or val == 0 or val == 0.0:
                confidence -= 0.2
    
    mode = result.get("mode", "")
    if mode in ("rag", "hybrid"):
        docs = result.get("retrieved_docs", [])
        if len(docs) == 0:
            confidence -= 0.3
        elif len(docs) < 2:
            confidence -= 0.1
    
    return max(0.1, min(1.0, confidence))


def run_query(question: str, format_hint: str = "str") -> dict:
    """Run a query through the agent and return formatted results."""
    initial_state = {
        "question": question,
        "format_hint": format_hint,
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
    
    result = agent_app.invoke(initial_state)
    confidence = calculate_confidence(result)
    
    return {
        "answer": result.get("final_answer"),
        "confidence": confidence,
        "sql": result.get("sql_query", ""),
        "explanation": result.get("explanation", ""),
        "citations": result.get("citations", []),
        "mode": result.get("mode", "")
    }


class QueryApp(Component):
    """Main GUI component for the query interface."""
    
    def __init__(self, props):
        super().__init__(props)
        self.state = {
            "question": "",
            "format_hint": "str",
            "output": "Results will appear here after you submit a query.\n\n⚠️ Note: Each query is independent. There is no conversation history.",
            "is_loading": False
        }
    
    def on_question_change(self, event):
        """Clear output when user starts typing a new question."""
        if not self.state["is_loading"]:
            self.set_state({
                **self.state,
                "question": event.String,
                "output": ""  # Clear output on new input
            })
    
    def on_format_change(self, event):
        """Update format hint selection."""
        if not self.state["is_loading"]:
            self.set_state({
                **self.state,
                "format_hint": event.String
            })
    
    def _run_query_thread(self, question, format_hint):
        """Run query in background thread."""
        try:
            result = run_query(question, format_hint)
            
            # Format output
            confidence_emoji = "🟢" if result["confidence"] >= 0.8 else "🟡" if result["confidence"] >= 0.5 else "🔴"
            
            output_lines = [
                "═" * 50,
                "📊 RESULT",
                "═" * 50,
                "",
                f"💡 Answer: {result['answer']}",
                "",
                f"{confidence_emoji} Confidence: {result['confidence']:.0%}",
                f"🔀 Mode: {result['mode'].upper()}",
                "",
            ]
            
            if result["sql"]:
                output_lines.extend([
                    "📝 SQL Query:",
                    result["sql"],
                    "",
                ])
            
            if result["explanation"]:
                output_lines.extend([
                    f"📖 Explanation: {result['explanation']}",
                    "",
                ])
            
            if result["citations"]:
                output_lines.extend([
                    f"📚 Sources: {', '.join(result['citations'][:5])}",
                ])
            
            output_lines.extend([
                "",
                "─" * 50,
                "⚠️ Each query is independent (no history)",
            ])
            
            output_text = "\n".join(output_lines)
            
        except Exception as e:
            output_text = f"❌ Error: {str(e)}"
        
        # Update UI from main thread
        wx.CallAfter(self._on_query_complete, output_text)
    
    def _on_query_complete(self, output_text):
        """Called when query completes - updates UI from main thread."""
        self.set_state({
            **self.state,
            "output": output_text,
            "is_loading": False
        })
    
    def on_submit(self, event):
        """Run the query in a background thread."""
        question = self.state["question"].strip()
        if not question:
            self.set_state({
                **self.state,
                "output": "⚠️ Please enter a question."
            })
            return
        
        if self.state["is_loading"]:
            return  # Already processing
        
        # Show loading state and disable inputs
        self.set_state({
            **self.state,
            "output": "🔄 Processing your query...\n\nPlease wait, this may take a few seconds.",
            "is_loading": True
        })
        
        # Run query in background thread
        thread = threading.Thread(
            target=self._run_query_thread,
            args=(question, self.state["format_hint"]),
            daemon=True
        )
        thread.start()
    
    def render(self):
        is_loading = self.state["is_loading"]
        
        return wsx(
            [c.Frame, {"title": "Retail Analytics Copilot", "show": True, 
                       "size": (700, 600)},
             [c.Block, {"orient": wx.VERTICAL, "flag": wx.EXPAND | wx.ALL, "border": 15},
              
              # Header
              [c.StaticText, {"label": "🛒 Retail Analytics Copilot",
                             "style": wx.ALIGN_CENTER,
                             "flag": wx.EXPAND | wx.BOTTOM, "border": 10}],
              
              # Question input section
              [c.StaticText, {"label": "Enter your question:" + (" (processing...)" if is_loading else ""),
                             "flag": wx.BOTTOM, "border": 5}],
              [c.TextCtrl, {"value": self.state["question"],
                           "on_change": self.on_question_change,
                           "style": wx.TE_MULTILINE,
                           "size": (-1, 80),
                           "enabled": not is_loading,
                           "flag": wx.EXPAND | wx.BOTTOM, "border": 10}],
              
              # Format hint row
              [c.Block, {"orient": wx.HORIZONTAL, "flag": wx.EXPAND | wx.BOTTOM, "border": 10},
               [c.StaticText, {"label": "Expected format: ",
                              "flag": wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, "border": 5}],
               [c.ComboBox, {"value": self.state["format_hint"],
                            "choices": ["str", "int", "float", 
                                       "{category:str, quantity:int}",
                                       "{customer:str, margin:float}",
                                       "list[{product:str, revenue:float}]"],
                            "on_change": self.on_format_change,
                            "enabled": not is_loading,
                            "flag": wx.EXPAND, "proportion": 1}],
               [c.Button, {"label": "🔄 Processing..." if is_loading else "🔍 Ask", 
                          "on_click": self.on_submit,
                          "enabled": not is_loading,
                          "flag": wx.LEFT, "border": 10}]
              ],
              
              # Output section
              [c.StaticText, {"label": "Results:",
                             "flag": wx.BOTTOM, "border": 5}],
              [c.TextCtrl, {"value": self.state["output"],
                           "style": wx.TE_MULTILINE | wx.TE_READONLY,
                           "flag": wx.EXPAND, "proportion": 1}],
              
              # Footer hint
              [c.StaticText, {"label": "💡 Examples: \"Top 3 products by revenue\" • \"Return policy for Beverages\" • \"AOV during Summer 2017\"",
                             "flag": wx.TOP | wx.ALIGN_CENTER, "border": 10}]
             ]
            ]
        )


def main():
    """Launch the GUI application."""
    print("Loading Retail Analytics Copilot GUI...")
    print("(Loading the AI model, please wait...)")
    
    app = wx.App()
    frame = render(create_element(QueryApp, {}), None)
    app.MainLoop()


if __name__ == "__main__":
    main()

