import os
import streamlit as st
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults

# 1. Setup your API Key
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE" # Paste your key here!

# 2. Define the State
class AgentState(TypedDict):
    topic: str
    research_notes: str
    final_post: str

# 3. Initialize the AI Model and Search Tool
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_retries=3,          # If Google returns a 503 error, automatically try again up to 3 times
    timeout=30              # Give up if the server takes longer than 30 seconds
)
search = DuckDuckGoSearchResults()

# --- NODE 1: The Researcher ---
def researcher(state: AgentState):
    topic = state["topic"]
    search_results = search.run(topic)
    
    prompt = f"Summarize these search results into key insights about {topic}. Focus on technical depth:\n{search_results}"
    response = llm.invoke(prompt)
    
    return {"research_notes": response.content}

# --- NODE 2: The LinkedIn Writer ---
def linkedin_writer(state: AgentState):
    topic = state["topic"]
    notes = state["research_notes"]
    
    prompt = f"""
    You are an expert technical content creator. Take the following research notes and write a professional LinkedIn post.
    
    Topic: {topic}
    Research Notes: {notes}
    
    Requirements:
    1. A catchy 'hook' line at the top to make people stop scrolling.
    2. A structured layout using bullet points.
    3. A conversational yet highly professional tone (ideal for a Senior Consultant).
    4. Relevant hashtags at the bottom.
    """
    response = llm.invoke(prompt)
    
    return {"final_post": response.content}

# 4. Build the Graph Workflow
workflow = StateGraph(AgentState)
workflow.add_node("researcher_node", researcher)
workflow.add_node("writer_node", linkedin_writer)
workflow.add_edge(START, "researcher_node")
workflow.add_edge("researcher_node", "writer_node")
workflow.add_edge("writer_node", END)
app = workflow.compile()


# ==========================================
# 5. STREAMLIT USER INTERFACE (THE NEW PART)
# ==========================================

st.set_page_config(page_title="AI Agent Creator", page_icon="🤖")

st.title("🚀 Multi-Agent LinkedIn Content Generator")
st.markdown("This app uses **LangGraph** to orchestrate an AI Researcher and an AI Writer working together.")

# The user input field
topic_input = st.text_input("What technical topic should the agents research today?", 
                            value="Latest trends in T24 banking software and Google Cloud API security")

# The Generate Button
if st.button("Deploy Agents"):
    
    # We use st.status to show a cool loading animation
    with st.status("Agents are working...", expanded=True) as status:
        
        st.write("🔍 Researcher Agent is browsing the live web...")
        
        # Run the engine
        initial_state = {"topic": topic_input}
        result = app.invoke(initial_state)
        
        st.write("✅ Research complete!")
        st.write("✍️ Writer Agent is formatting the final post...")
        
        status.update(label="Process Complete!", state="complete", expanded=False)

    # Display the final results in a nice green box
    st.success("Here is your ready-to-publish post!")
    st.markdown(result["final_post"])
    
    # Show the "behind the scenes" research in a toggle box
    with st.expander("View Raw Research Notes"):
        st.write(result["research_notes"])