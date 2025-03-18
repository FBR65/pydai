import serpapi
from httpx import AsyncClient
from dataclasses import dataclass
import asyncio
from datetime import datetime
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, ValidationError
from typing import List
from pydantic_ai.models import Usage
from pydantic_ai import Agent, RunContext
import json

serpapi_key='cc92613598e0e46aaf2e30ec2c84daa3011d2790b1f3697cd164241c5aa13edd'

# Define a dataclass to encapsulate dependencies required by the agent.
# Includes an HTTP client for making requests and the Bing API key for authentication.
@dataclass
class Deps:
    client: AsyncClient

# Define a Pydantic model to represent individual search results.
# Each result contains a website URL and a summary of the content.
class SearchResult(BaseModel):
    website_url: str
    content_summary: str

# Define a Pydantic model for a collection of search results.
# This model wraps multiple SearchResult objects in a list.
class SearchResults(BaseModel):
    results: List[SearchResult]

# Initialize the web search agent with a specific model and system prompt.
# The system prompt instructs the LLM to process and summarize search results.

web_search_agent = Agent(
    model=OpenAIModel(model_name='qwen2.5:latest', provider=OpenAIProvider(base_url='http://localhost:11434/v1')),
    system_prompt=f"""
        You are an expert at summarizing search results. 
        Always provide the response as JSON in the following format:
    
        {{
            "results": [
                {{
                    "website_url": "string",
                    "content_summary": "string"
                }},
                ...
            ]
        }}
        Summarize the content from the search results provided, ensuring full sentences and cohesive summaries.
        If no valid results are provided, return:
        {{
            "results": []
        }}
        The current date is: {datetime.now().strftime("%Y-%m-%d")}
    """,
    deps_type=Deps,
    retries=2
)


# Define a tool that fetches web search results from the Bing API.
# The tool processes the Bing API response and formats it into text for further summarization.
@web_search_agent.tool
async def search_web(ctx: RunContext[Deps], web_query: str) -> str:
    """
    Perform a web search and return the raw results as a formatted string for LLM processing.
    """
    client = serpapi.Client(api_key=serpapi_key)

    params = {
    "engine": "duckduckgo",
    "q": web_query,
    "kl": "us-en",
    "api_key": serpapi_key
    }

    results = client.search(params)

    # Extract and format the search results into a readable text block.
    web_results = results.get('organic_results', []) # Changed this line
    formatted_results = []
    for item in web_results[:5]:  # Limit to the top 5 results
        title = item.get('title', 'No title provided') # Changed this line
        description = item.get('snippet', 'No description available')
        url = item.get('link', 'No URL provided') # Changed this line
        formatted_results.append(f"Title: {title}\nDescription: {description}\nURL: {url}\n")

    # Return the formatted results as a single string.
    return "\n".join(formatted_results)

# Define the main function to execute the agent and process search results.
# The main function handles communication between the Bing API and the LLM.
async def main():
    async with AsyncClient() as client:
        # Define the dependencies, including the HTTP client and Bing API key.
        deps = Deps(client=client)
        
        # Fetch search results using the search_web tool.
        search_query = """Search the web for articles talking about ManusAI."""
        #search_results = await search_web(RunContext(deps=deps, retry=None), search_query)
        dummy_run_context = RunContext(deps=deps, retry=None, model=web_search_agent.model, usage=Usage(), prompt="")
        search_results = await search_web(dummy_run_context, search_query)
        # Prepare a summarization query for the LLM using the fetched results.
        summarization_query = f"""
            Based on the following search results, summarize the key points about ManusAI in a cohesive manner:

            {search_results}
        """

        # Use the web_search_agent to generate structured summaries from the LLM.
        result = await web_search_agent.run(summarization_query, deps=deps)

        # Validate and print the structured response from the LLM.
        try:
            #Fix: LLM was adding text before the json. This removes it.
            json_start = result.data.find('{')
            if json_start != -1:
                json_string = result.data[json_start:]
                structured_response = SearchResults.model_validate_json(json_string)
                print("Structured Response:")
                print(structured_response.model_dump_json(indent=2))
            else:
                print("No JSON found in the response.")
                print("Raw Response:", result.data)
        except ValidationError as e:
            # Handle validation errors and display raw responses for debugging.
            print("Invalid structured response from the LLM:")
            print(e)
            print("Raw Response:", result.data)
        except json.JSONDecodeError as e:
            print("JSON Decode Error:")
            print(e)
            print("Raw Response:", result.data)

# Entry point for the script.
# Ensures the main function is executed asynchronously.
if __name__ == '__main__':
    asyncio.run(main())
