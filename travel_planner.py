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

if not GROQ_API_KEY or not SERPER_API_KEY:
    raise ValueError("Missing GROQ_API_KEY or SERPER_API_KEY in environment")

# Initialize the LLM
def get_llm(model: str = "llama-3.3-70b-versatile") -> ChatGroq:
    """Initializes and returns a ChatGroq LLM instance."""
    return ChatGroq(api_key=GROQ_API_KEY, model=model)

# Function to perform a general web search using the Serper API
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

def main():
    """Main function to run the travel planner."""
    print("Welcome to your AI Travel Itinerary Creator!")
    destination = input("Where do you want to go? Please enter the city and number of days (e.g., Chennai, 3 days): ")
    preferences = input("Any specific preferences? (e.g., history, budget, food): ")

    try:
        dest_parts = destination.split(',')
        place = dest_parts[0].strip()
        num_days = int(dest_parts[1].strip().split()[0])
        
        print(f"\nPlanning a {num_days}-day trip to {place}...")
        
        llm = get_llm()
        
        # Step 1: Perform the search
        query = f"best things to do in {place} for {num_days} days {preferences} travel guide"
        print(f"Searching the web for '{query}'...")
        search_results = serper_search(query)
        
        if "error" in search_results:
            print(f"Error during search: {search_results['error']}")
            return

        search_summary = summarize_search_results(search_results)
        
        if not search_summary:
            print("Could not find relevant information. Please try a different query.")
            return

        # Step 2: Use the LLM to create the itinerary
        print("Generating your custom itinerary...")
        itinerary = build_itinerary(llm, place, num_days, preferences, search_summary)
        
        # Step 3: Print the final output
        print("\n" + "="*50)
        print(f"Here is your {num_days}-day itinerary for {place}:")
        print("="*50)
        print(itinerary)
        
    except (IndexError, ValueError):
        print("Invalid input format. Please follow the example: 'Chennai, 3 days'.")
        
if __name__ == "__main__":
    main()