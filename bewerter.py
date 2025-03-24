"""
Copyright 2025 Frank Reis
The use of this code or parts of it is permitted exclusively for private, educational, or non-commercial purposes. 
Any commercial use or use by governmental organizations is prohibited without prior written permission from the author.

Copyright 2025 Frank Reis
Die Nutzung dieses Codes oder Teile davon ist ausschließlich für private, Bildungs- oder nicht-kommerzielle Zwecke gestattet. 
Jegliche kommerzielle Nutzung oder Verwendung durch staatliche Organisationen ist ohne vorherige schriftliche Genehmigung des Autors untersagt.
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()

class BewertResultType(BaseModel):
    type: str = Field(
        ...,
        description="The type of the result. Possible values are 'problem' (for problem definitions) and 'requery' (for queries)."
    )
    score: int = Field(
        ...,
        description="The score given to the prompt, on a scale from 0 to 10."
    )
    strongest_arguments: List[str] = Field(
        ...,
        description="A list of the strongest arguments (up to 3) supporting the evaluation."
    )
    update_needs: Optional[str] = Field(
        None,
        description="Critical remarks or suggestions for improvement, if relevant."
    )


# Definition der AgentRunResult-Klasse
class AgentRunResult:
    def __init__(self, data: BewertResultType):
        self.data = data


class Berwerter:
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
            Du wurdest beauftragt, die Qualität verschiedener vorgeschlagener Prompts für eine gegebene Problemstellung zu bewerten. Bitte folge den Anweisungen sorgfältig:

            1. Sichtung der Problemstellung:
            Analysiere die Problemstellung: {problem}. Identifiziere die zentralen Anforderungen, Ziele und Grenzen des Problems.

            2. Generierte Antwort:
            Die erzeugte eine Antwort auf den Prompt: {prompt} lautet {antwort}. Diese Antwort wird in die Bewertung einbezogen.

            3. Evaluation der Kandidaten-Prompts:
            Bewerte jeden {prompt} und die dazugehörige Antwort auf einer Skala von 1 (sehr unzureichend) bis 10 (perfekt). Berücksichtige dabei folgende Kriterien:

            Zielstellungsausrichtung (30%): Wie gut passt der Prompt zu den identifizierten Anforderungen der Problemstellung?
            Klarheit und Präzision (25%): Ist die Anweisung unmissverständlich formuliert?
            Effizienz (20%): Stellt der Prompt die Aufgabe effizient dar, ohne unnötige Informationen?
            Folgerichtigkeit (15%): Wird der Kernaussage des Problems gerecht?
            Kreativität/Neuheit (10%): Bringt der Prompt eine originelle Sichtweise oder spezifizierte Lösungsansätze?
            Qualität der Antwort (zusätzlich): Wie gut erfüllt die erzeugte Antwort die Anforderungen der Problemstellung?
            4. Dokumentation der Bewertung:
            Formatiere deine Auswertung folgendermaßen:

            Prompt X:

            Bewertung: [Score/10]
            Stärkste Argumente: (max. 3 inhaltsreiche Punkte)
            Aktualisierungsbedarf: (kritische Anmerkungen, falls relevant)

            5. Endbetrachtung:
            Strebe eine objektive, datenorientierte Analyse an. Vermeide subjektive Rhetorik (z. B. „Ich finde...“), sondern folge strukturierten Gedankengängen. Markiere alle zentralen Schlüsselbegriffe des Problems im ersten Schritt.

            Antworte bitte ausschließlich im Telegram-Format, z. B. kursiv, fett, Aufzählungen.
            """,
            result_type=BewertResultType,  # Use our new model
            retries=2
        )

    async def bewerte_prompts_und_antworten(self, problem, prompts_und_antworten):
        bewertungen = []
        for item in prompts_und_antworten:
            prompt = item['generated_prompt']
            antwort = item['answer']
            technique = item['technique']
            bewertung = await self.agent.run(f"Problem: {problem}\nPrompt: {prompt}\nAnswer: {antwort}\nTechnique: {technique}")
            bewertung.data.type = technique  # Add prompt type to the evaluation
            bewertungen.append(bewertung)
        return bewertungen
