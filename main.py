import os
import time
from typing import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. State Definition
class AgentState(TypedDict):
    query: str
    retrieved_snippets: str
    final_report: str

model = SentenceTransformer('all-MiniLM-L6-v2')

def index_knowledge_base():
    # Read paragraphs
    with open("knowledge_base.txt", "r", encoding="utf-8") as f:
        paragraphs = [p.strip() for p in f.read().split('\n') if p.strip()]
        
    # Generate embeddings once
    print("Embedding documents... Please wait.")
    doc_embeddings = model.encode(paragraphs, convert_to_numpy=True)
    
    # Save text and vectors to disk
    np.savez("knowledge_cache.npz", paragraphs=paragraphs, embeddings=doc_embeddings)
    print("Indexing complete. Saved to knowledge_cache.npz")

def retrieve_information_hybrid_fast(query: str, top_k: int = 3, semantic_weight: float = 0.5) -> str:
    """Performs hybrid search using pre-computed document embeddings."""
    # Load cached embeddings and text
    if not os.path.exists("knowledge_cache.npz"):
        return "Error: Index file not found. Please run the indexer script first."
        
    data = np.load("knowledge_cache.npz", allow_pickle=True)
    paragraphs = data['paragraphs']
    doc_embeddings = data['embeddings']

    # 1. KEYWORD SCORE (Jaccard Similarity)
    stop_words = {"what", "when", "where", "which", "who", "why", "how", "can", "could", "would", "should"}
    clean_query = query.replace('?', '').replace('.', '').replace(',', '')
    keywords = set(word.lower() for word in clean_query.split() if len(word) > 2 and word.lower() not in stop_words)
    
    keyword_scores = []
    for p in paragraphs:
        p_words = set(p.lower().split())
        match_count = len(keywords.intersection(p_words))
        keyword_scores.append(match_count / len(keywords) if keywords else 0.0)

    # 2. SEMANTIC SCORE (Embed only the single query string)
    query_embedding = model.encode(query, convert_to_numpy=True)
    
    # Calculate cosine similarities using pre-loaded doc_embeddings
    norm_query = query_embedding / np.linalg.norm(query_embedding)
    norm_docs = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    semantic_scores = np.dot(norm_docs, norm_query)

    # 3. HYBRID FUSION
    hybrid_results = []
    for idx, (k_score, s_score) in enumerate(zip(keyword_scores, semantic_scores)):
        combined_score = (1 - semantic_weight) * k_score + (semantic_weight * s_score)
        hybrid_results.append((combined_score, paragraphs[idx]))

    hybrid_results.sort(key=lambda x: x[0], reverse=True)
    top_snippets = [doc for score, doc in hybrid_results[:top_k] if score > 0.1]

    return "\n---\n".join(top_snippets) if top_snippets else "No relevant information found."

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
    snippets = retrieve_information_hybrid_fast(query)
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
    index_knowledge_base()
    app = build_workflow()
    queries = [
        "What is the policy on international travel?",
        "Can I bring Husky to workplace?",
    ]
    
    for q in queries:
        print(f"\n==================================================")
        print(f"USER QUERY: {q}")
        print(f"====================================================")
        result = app.invoke({"query": q})
        print(f"\nFINAL OUTPUT:\n{result['final_report']}\n")
        
        # Pause for 15 seconds before the next query to avoid rate limits!
        print("--> Pausing to respect free tier API rate limits...")
        time.sleep(15)