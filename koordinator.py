"""
Copyright 2025 Frank Reis
The use of this code or parts of it is permitted exclusively for private, educational, or non-commercial purposes. 
Any commercial use or use by governmental organizations is prohibited without prior written permission from the author.

Copyright 2025 Frank Reis
Die Nutzung dieses Codes oder Teile davon ist ausschließlich für private, Bildungs- oder nicht-kommerzielle Zwecke gestattet. 
Jegliche kommerzielle Nutzung oder Verwendung durch staatliche Organisationen ist ohne vorherige schriftliche Genehmigung des Autors untersagt.
"""
import asyncio
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent
from dotenv import load_dotenv

from bewerter import Berwerter

load_dotenv()

class ResultType(BaseModel):
    type: str = Field(
        ...,
        description="The type of the result. Possible values are 'problem' (for problem definitions) and 'requery' (for queries)."
    )
    value: str = Field(
        ...,
        description="The content of the result, which is a string provided by the corresponding agent."
    )


# Define a Pydantic model to capture both generated prompt and answer
class PromptAndAnswerResponse(BaseModel):
    generated_prompt: str  # The prompt generated using the technique
    answer: str = ""      # The final answer based on generated prompt, default empty string


class PromptGenerator:
    def __init__(self):
        # Initialize provider and model
        self.provider = OpenAIProvider(
            base_url=os.getenv('BASE_URL'),
            api_key=os.getenv('API_KEY')  # Ensure API_KEY is in .env
        )
        self.model = OpenAIModel(
            provider=self.provider,
            model_name=os.getenv('MODEL_NAME')
        )
        
        # Simplified system prompt with better error handling
        self.agent = Agent(
            model=self.model,
            system_prompt="""
                You are a helpful AI assistant. Please generate a prompt and answer based on the provided technique and problem.
                Always return a valid JSON response with the following structure:
                {
                    "generated_prompt": "The generated prompt",
                    "answer": "The answer based on the prompt"
                }
                If you cannot generate a response, return:
                {
                    "generated_prompt": "Error: Could not generate prompt",
                    "answer": "Error: Could not generate answer"
                }
            """,
            result_type=PromptAndAnswerResponse,
            retries=3
        )
    
    async def generiere_und_beantworte_prompts(self, problem):
        prompt_techniques = [
            "Few-Shot Prompting",
            "Chain-of-Thought (CoT) Prompting",
            "Self-Consistency",
            "Automatic Chain-of-Thought (Auto-CoT)",
            "Closed-Ended Prompt",
            "Open-Ended Prompt",
            "Rollenspiel (Persona Prompting)",
            "Tree-of-Thoughts Prompting",
            "Maieutic Prompting",
            "Complexity-based Prompting",
            "Generated Knowledge Prompting",
            "Least-to-Most Prompting",
            "Self-refine Prompting",
            "Role/Goal/Context Prompt",
            "Chain of Thoughts-Prompt",
            "Chain of Thoughts with Reflecion Prompt",
            "Role / Instructions / Steps / End goal / Narrowing",
            "Role / Objective / Details / Examples / Sense Check",
        ]
        
        all_prompts = [
            f"{technik}: {problem}" 
            for technik in prompt_techniques
        ]
        
        results = await asyncio.gather(
            *(self._beantworte_prompt(prompt, technik) 
              for technik, prompt in zip(prompt_techniques, all_prompts))
        )
        
        return results
    
    async def _beantworte_prompt(self, prompt, technique):
        try:
            response = await self.agent.run(prompt)
            if not response.data.generated_prompt or not response.data.answer:
                print(f"Warning: Empty response for technique: {technique}")
                return {
                    "original_prompt": prompt,
                    "technique": technique,
                    "generated_prompt": "Error: Empty response from model",
                    "answer": "Error: Empty response from model"
                }
            return {
                "original_prompt": prompt,
                "technique": technique,
                "generated_prompt": response.data.generated_prompt,
                "answer": response.data.answer
            }
        except Exception as e:
            print(f"Error processing technique {technique}: {str(e)}")
            return {
                "original_prompt": prompt,
                "technique": technique,
                "generated_prompt": f"Error: {str(e)}",
                "answer": f"Error: {str(e)}"
            }


class RequeryAgent():
    def __init__(self):
        # Initialize provider and model
        self.provider = OpenAIProvider(
            base_url=os.getenv('BASE_URL'),
            api_key=os.getenv('API_KEY')  # Ensure API_KEY is in .env
        )
        self.model = OpenAIModel(
            provider=self.provider,
            model_name=os.getenv('MODEL_NAME')
        )
        
        # Updated system prompt to request both generated_prompt and answer
        self.agent = Agent(
            model=self.model,
            system_prompt="""
                Rolle: Du bist ein analytischer Assistent zur Problemumformulierung. Deine Aufgabe ist es, die gegebene Problemstellung umzuformulieren, 
                um Unklarheiten zu beseitigen, das Problem in Teilprobleme zu zerlegen oder es für eine tiefere Analyse neu zu interpretieren. 
                Ziel:
                - Umformulierung: Analysiere die ursprüngliche Problemstellung und formuliere sie klarer und präziser um, ohne Lösungen oder Rückfragen zu präsentieren. 
                Gib ausschließlich das Ergebnis der Umformulierung aus.

                Beispiel:
                - Problem: Rede GreentIT 20 Minuten Laienpublikum
                - Umformulierung: Wie kann ich eine effektive, 20-minütige Präsentation über GreentIT halten, die ein Laienpublikum anspricht und verständlich ist, ohne dabei technische Details zu übergehen?
                
                Problemstellung:
                {problem}
            """,
            result_type=ResultType,  # Use our new model
            retries=2
        )
    async def requery_problem(self, problem):
        response = await self.agent.run(problem)
        return response
        
class Koordinator:
    def __init__(self, problem):
        self.problem = problem
        self.requery_agent = RequeryAgent()
        self.promptgenerator = PromptGenerator()
        self.berwerter = Berwerter()

    async def koordiniere(self):

        # Verfeinere den Input
        verfeinertes_input = await self.requery_agent.requery_problem(self.problem)
        print(verfeinertes_input.data.value)

        # Generiere und beantworte die Prompts
        prompts_und_antworten = await self.promptgenerator.generiere_und_beantworte_prompts(verfeinertes_input.data.value)
        print(prompts_und_antworten)

        # Bewerte die Prompts und Antworten
        bewertungen = await self.berwerter.bewerte_prompts_und_antworten(verfeinertes_input, prompts_und_antworten)
        #print(bewertungen)

        # Erstelle das Markdown-Dokument
        markdown = self.erstelle_markdown(self.problem, verfeinertes_input, prompts_und_antworten, bewertungen)
        return markdown

    def erstelle_markdown(self, problem, verfeinertes_input, prompts_und_antworten, bewertungen):
        markdown = f"# Multiagent-System Ergebnisse\n\n"
        markdown += f"## Problem\n{problem}\n\n"
        markdown += f"## Verfeinertes Input\n{verfeinertes_input}\n\n"
        markdown += "## Prompts und Antworten\n\n"
        for item in prompts_und_antworten:
            markdown += f"**Prompt Type:** {item['technique']}\n\n"
            markdown += f"**Prompt:** {item['generated_prompt']}\n\n"
            markdown += f"**Antwort:** {item['answer']}\n\n"
            # Filter bewertungen for this specific prompt type
            type_bewertungen = [b for b in bewertungen if b.data.type == item['technique']]
            for bewertung in type_bewertungen:
                markdown += f"### Bewertung\n\n"
                markdown += f"**Score:** {bewertung.data.score}\n\n"
                markdown += f"**Stärkste Argumente:**\n"
                for argument in bewertung.data.strongest_arguments:
                    markdown += f"- {argument}\n"
                markdown += f"\n**Update-Bedarf:** {bewertung.data.update_needs}\n\n"

        return markdown

# Beispielaufruf
if __name__ == "__main__":
    problem = "Rede Wehrhafte Demokratie 20 Minuten Laienpublikum"
    koordinator = Koordinator(problem)
    markdown = asyncio.run(koordinator.koordiniere())
    print(markdown)