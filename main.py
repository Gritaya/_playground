import os
import time
from typing import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. State Definition
class AgentState(TypedDict):
    query: str
    retrieved_snippets: str
    final_report: str

# 2. Custom RAG Tool (Keyword Search)
def retrieve_information(query: str) -> str:
    """Reads knowledge_base.txt and performs basic keyword search."""
    try:
        with open("knowledge_base.txt", "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return "Error: knowledge_base.txt not found."
    
    # Filter out question words for better keyword matching
    stop_words = {"what", "when", "where", "which", "who", "why", "how", "can", "could", "would", "should"}
    clean_query = query.replace('?', '').replace('.', '').replace(',', '')
    keywords = [word.lower() for word in clean_query.split() if len(word) > 2 and word.lower() not in stop_words]
    
    snippets = []
    paragraphs = content.split('\n')
    for p in paragraphs:
        if any(kw in p.lower() for kw in keywords):
            snippets.append(p.strip())
            
    return "\n---\n".join(snippets) if snippets else "No relevant information found."

# 3. Global LLM Initialization
# This MUST be outside the node functions to prevent severe latency!
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=os.environ.get("GEMINI_API_KEY") 
)

# 4. Agent 1: Data Retriever Node
def data_retriever_node(state: AgentState) -> dict:
    query = state["query"]
    print(f"--> [Data Retriever] Searching knowledge base for: '{query}'")
    snippets = retrieve_information(query)
    print("--> [Data Retriever] Snippets found. Passing to Report Generator.")
    return {"retrieved_snippets": snippets}

# 5. Agent 2: Report Generator Node
def report_generator_node(state: AgentState) -> dict:
    print("--> [Report Generator] Synthesizing final answer...")
    
    sys_msg = SystemMessage(
        content="You are an expert report generator. Using ONLY the provided information snippets, "
                "synthesize a cohesive, non-redundant, and well-formatted answer to the user's query. "
                "Do not hallucinate."
    )
    human_msg = HumanMessage(
        content=f"User Query: {state['query']}\n\nInformation Snippets:\n{state['retrieved_snippets']}"
    )
    
    # Langchain might still print an AFC (Automatic Function Calling) warning in the terminal 
    # for 3.6-flash, but it will not slow down the execution.
    response = llm.invoke([sys_msg, human_msg])
    return {"final_report": response.content}

# 6. Graph Orchestration
def build_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("data_retriever", data_retriever_node)
    workflow.add_node("report_generator", report_generator_node)
    
    workflow.set_entry_point("data_retriever")
    workflow.add_edge("data_retriever", "report_generator")
    workflow.add_edge("report_generator", END)
    
    return workflow.compile()

# 7. Execution
if __name__ == "__main__":
    app = build_workflow()
    queries = [
        "What is the policy on international travel?",
        "Can I bring my cat to office?",
    ]
    
    for q in queries:
        print(f"\n==================================================")
        print(f"USER QUERY: {q}")
        print(f"==================================================")
        result = app.invoke({"query": q})
        print(f"\nFINAL OUTPUT:\n{result['final_report']}\n")
        
        # Pause for 15 seconds before the next query to avoid rate limits!
        print("--> Pausing to respect free tier API rate limits...")
        time.sleep(15)