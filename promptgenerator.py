import asyncio
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent
from dotenv import load_dotenv
import os

load_dotenv()

# Define a Pydantic model to capture both generated prompt and answer
class PromptAndAnswerResponse(BaseModel):
    generated_prompt: str  # The prompt generated using the technique
    answer: str           # The final answer based on the generated prompt

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
    
# Example usage
if __name__ == "__main__":
    generator = PromptGenerator()
    results = asyncio.run(
        generator.generiere_und_beantworte_prompts(
            problem="""**Umformulierte Problemstellung für einen 20-minütigen Vortrag über GreenIT für ein Laienpublikum:**

**Titel:** Einführung in GreenIT: Nachhaltigkeit in der digitalen Welt verständlich gemacht

**Ziel:** Ein Laienpublikum innerhalb von 20 Minuten über die Grundlagen und Bedeutung von GreenIT aufzuklären, hinweisend aufsimple, alltagsnahe Beispiele und praktische Umsetzungsmöglichkeiten.

**Schwerpunkte:**
1. **Verständliche Definition**: Was ist GreenIT und warum ist es wichtig?
2. **Alltagsszenarien**: Beispiele, wie GreenIT im täglichen Leben angewendet wird oder werden kann.
3. **Praktische Tipps**: Sofort umsetzbare Ratschläge für das Publikum, um nachhaltiger mit IT-Ressourcen umzugehen.
4. **Zukunftsperspektiven**: Kurze, inspirierende Ausblicke auf die Zukunft von GreenIT und dessen potenziellen positiven Auswirkungen.

**Erwartetes Ergebnis:** Das Publikum verlässt den Vortrag mit einem grundlegenden Verständnis von GreenIT, einer Wertschätzung für seine Bedeutung und kleinen, aber wirksamen Veränderungen, die es in seinem Alltag vornehmen kann, um zur Nachhaltigkeit beizutragen.""",
        )
    )

    for item in results:
        print(f"Technik: {item['technique']}")
        print(f"Original Prompt: {item['original_prompt']}")
        print(f"Generated Prompt: {item['generated_prompt']}")
        print(f"Answer: {item['answer']}\n{'='*40}")
