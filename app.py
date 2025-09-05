import streamlit as st
import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List

# Load API keys from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Check for API keys
if not GROQ_API_KEY or not SERPER_API_KEY:
    st.error("Missing GROQ_API_KEY or SERPER_API_KEY in environment.")
    st.stop()

# Initialize the LLM
@st.cache_resource
def get_llm(model: str = "llama-3.3-70b-versatile") -> ChatGroq:
    """Initializes and returns a ChatGroq LLM instance with caching."""
    return ChatGroq(api_key=GROQ_API_KEY, model=model)

# Function to perform a general web search using the Serper API
@st.cache_data(show_spinner=False)
def serper_search(query: str) -> Dict:
    """Performs a general web search and returns results."""
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"q": query})
    if resp.status_code != 200:
        return {"error": resp.text}
    return resp.json()

# Function to process search results into a summary for the LLM
def summarize_search_results(search_results: Dict) -> str:
    """Extracts and formats key information from search results."""
    snippets = search_results.get("organic", [])
    summary_lines = []
    for snippet in snippets[:5]:  # Take the top 5 results
        title = snippet.get("title", "N/A")
        link = snippet.get("link", "N/A")
        snippet_text = snippet.get("snippet", "N/A")
        summary_lines.append(f"Title: {title}\nURL: {link}\nSummary: {snippet_text}\n---")
    return "\n".join(summary_lines)

# Function to build the final itinerary using the LLM
def build_itinerary(llm: ChatGroq, destination: str, duration: int, preferences: str, search_summary: str) -> str:
    """Uses the LLM to generate a day-by-day travel itinerary."""
    prompt_template = ChatPromptTemplate.from_template(
        "You are an expert travel agent. Create a detailed day-by-day travel itinerary "
        "for a {duration}-day trip to {destination}. "
        "The user has these preferences: {preferences}\n\n"
        "Here is some information I found from my research:\n"
        "{search_summary}\n\n"
        "Please use this information to create the itinerary. For each day, suggest "
        "2-3 activities, a food recommendation, and some practical travel tips. "
        "Keep the tone friendly and helpful. Do not mention that you used a search engine."
    )
    
    formatted_prompt = prompt_template.format(
        destination=destination,
        duration=duration,
        preferences=preferences,
        search_summary=search_summary
    )
    
    res = llm.invoke(formatted_prompt)
    return res.content

# Streamlit App Interface
st.set_page_config(page_title="AI Travel Itinerary Creator", page_icon="✈️", layout="wide")

st.title("✈️ AI Travel Itinerary Creator")
st.write("Plan your perfect trip with a little help from AI. Just enter your destination and preferences below!")

# User input forms
with st.form(key='itinerary_form'):
    destination_days = st.text_input("Destination and Duration (e.g., Ooty, 4 days)", key="dest_days")
    preferences = st.text_input("Travel Preferences (e.g., solo travel, budget-friendly)", key="prefs")
    submit_button = st.form_submit_button(label='Generate Itinerary ✨')

# Logic to handle form submission
if submit_button and destination_days:
    try:
        dest_parts = destination_days.split(',')
        place = dest_parts[0].strip()
        num_days = int(dest_parts[1].strip().split()[0])

        with st.spinner(f"Planning a {num_days}-day trip to {place}..."):
            llm = get_llm()

            # Step 1: Perform the search
            query = f"best things to do in {place} for {num_days} days {preferences} travel guide"
            search_results = serper_search(query)
            
            if "error" in search_results:
                st.error(f"Error during search: {search_results['error']}")
                st.stop()

            search_summary = summarize_search_results(search_results)
            
            if not search_summary:
                st.warning("Could not find relevant information. Please try a different query.")
                st.stop()

            # Step 2: Use the LLM to create the itinerary
            itinerary = build_itinerary(llm, place, num_days, preferences, search_summary)
            
            # Step 3: Display the final output
            st.markdown("---")
            st.header(f"Your {num_days}-Day Itinerary for {place}:")
            st.markdown(itinerary, unsafe_allow_html=True)

    except (IndexError, ValueError):
        st.error("Invalid input format. Please follow the example: 'Chennai, 3 days'.")