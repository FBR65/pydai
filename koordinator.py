import asyncio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv
import os

from promptgenerator import PromptGenerator

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI provider with base URL and API key
provider = OpenAIProvider(base_url='https://openrouter.ai/api/v1', api_key=os.getenv('API_KEY'))

# Initialize OpenAI model
model = OpenAIModel(provider=provider, model_name="qwen/qwen-2.5-72b-instruct:free")


class Berwerter:
    async def bewerte_prompts_und_antworten(self, problem, prompts_und_antworten):
        # Hier könnte eine detailliertere Bewertung implementiert werden
        # Beispiel: Bewertung basierend auf der Länge der Antwort und der Relevanz
        bewertungen = []
        for prompt, antwort in zip(prompts_und_antworten['prompts'], prompts_und_antworten['antworten']):
            # Beispielbewertung: Länge der Antwort
            bewertung = len(antwort)
            bewertungen.append(bewertung)
        return bewertungen
    

class Problemdefinierer:
    async def pruefe_problem(self, input_data):
        problem = f"""Aufgabe:

                        Analysieren Sie den folgenden Eingabetext des Anwenders und erstellen Sie eine strukturierte Problemdefinition. Rückfragen an den Benutzer sind nicht möglich. 
                        Fehlende Informationen sind als "Nicht spezifiziert" zu markieren oder mit plausiblen Annahmen basierend auf Kontext zu ergänzen und als "Annahme" zu markieren.

                        Schritt-für-Schritt-Anweisungen:

                        Problemstellung

                        Identifizieren Sie den Kern des Problems (was genau ist fehlerhaft?).
                        Beispiel: „Die Website ist häufig nicht erreichbar.“

                        Kontext/Umfeld

                        Extrahieren Sie relevante Hintergrundinformationen (Wann, bei welchen Aktivitäten tritt das Problem auf?).
                        Beispiel: „Problem tritt nur während der Einkaufszeit auf.“

                        Ziele

                        Identifizieren Sie das gewünschte Ergebnis (Was soll erreicht werden?).
                        Beispiel: „Website soll 24/7 erreichbar sein.“

                        Beschränkungen

                        Suchen Sie nach einschränkenden Faktoren (Budget, Zeit, Ressourcen).
                        Beispiel: „Keine zusätzlichen Serverkosten zulässig.“

                        Betroffene Parteien/Stakeholder

                        Nennen Sie beteiligte Gruppen (Kunden, Teams, Partner).
                        Beispiel: „Kunden, IT-Team, Vertrieb.“

                        Auswirkungen

                        Beschreiben Sie negative Konsequenzen (z. B. Umsatzverlust, Kundenfrustration).
                        Beispiel: „Kunden verlieren Vertrauen.“

                        Bestehende Lösungsversuche

                        Notieren Sie bereits unternommene Maßnahmen.
                        Beispiel: „Load Balancer installiert, aber unzureichend.“

                        Hinweise für die LLM:

                        Priorisieren Sie relevante Informationen. Ignorieren Sie Nebensächlichkeiten.

                        Fehlende Daten:
                        Markieren Sie fehlende Punkte explizit mit "Nicht spezifiziert".
                        Ergänzen Sie plausibel, wenn Kontext Hinweise gibt (z. B. „Zielzeitraum: ~4 Wochen“).

                        Konzentration: Fokussieren Sie sich auf die im Eingabetext enthaltenen Details.

                        Formatierung: Fügen Sie einen kurzen Zusammenfassungssatz oben ein.

                        Benutzereingabe:
                        {input_data}
                        """
        if not problem or not input_data:
            return False
        return True

class RequeryAgent:
    async def verfeinere_input(self, problem):
        aufgabe = f"""Rolle:
        Du bist ein analytischer Assistent zur Problemrestrukturierung. Deine Aufgabe ist es, eine gegebene Problemstellung umzuformulieren, um Unklarheiten aufzuklären, das Problem in Teilprobleme zu zerlegen oder es für eine tiefere Analyse neu zu interpretieren.
        Ziel:
        Problem verstehen: Analysiere die ursprüngliche Problemstellung, um Hauptziele, Beschränkungen und Unsicherheiten zu identifizieren.
        Klarstellen: Entferne Fachjargon, formuliere vage Begriffe präziser oder vereinfache komplexe Konzepte.
        Umbauen: Zerlege das Problem in logische Teilprobleme oder Schritte.
        Problemstellung:
        {problem}"""

        verfeinertes_input = f"Bitte erkläre den Begriff '{problem}' in einfachen Worten."
        return verfeinertes_input
    
class Koordinator:
    def __init__(self, input_data):
        self.problem = problem
        self.input_data = input_data
        self.problemdefinierer = Problemdefinierer()
        self.requery_agent = RequeryAgent()
        self.promptgenerator = PromptGenerator()
        self.berwerter = Berwerter()

    async def koordiniere(self):
        # Überprüfe das Problem
        problem_gueltig = await self.problemdefinierer.pruefe_problem(self.problem, self.input_data)
        if not problem_gueltig:
            raise ValueError("Das Problem wurde nicht korrekt eingerichtet.")

        # Verfeinere den Input
        verfeinertes_input = await self.requery_agent.verfeinere_input(self.problem, self.input_data)

        # Generiere und beantworte die Prompts
        prompts_und_antworten = await self.promptgenerator.generiere_und_beantworte_prompts(self.problem, verfeinertes_input)

        # Bewerte die Prompts und Antworten
        bewertungen = await self.berwerter.bewerte_prompts_und_antworten(self.problem, prompts_und_antworten)

        # Erstelle das Markdown-Dokument
        markdown = self.erstelle_markdown(self.input_data, self.problem, verfeinertes_input, prompts_und_antworten, bewertungen)
        return markdown

    def erstelle_markdown(self, input_data, problem, verfeinertes_input, prompts_und_antworten, bewertungen):
        markdown = f"# Multiagent-System Ergebnisse\n\n"
        markdown += f"## Input\n{input_data}\n\n"
        markdown += f"## Problem\n{problem}\n\n"
        markdown += f"## Verfeinertes Input\n{verfeinertes_input}\n\n"
        markdown += "## Prompts und Antworten\n"
        for prompt, antwort, bewertung in zip(prompts_und_antworten['prompts'], prompts_und_antworten['antworten'], bewertungen):
            markdown += f"### Prompt\n{prompt}\n\n"
            markdown += f"### Antwort\n{antwort}\n\n"
            markdown += f"### Bewertung\n{bewertung}\n\n"
        return markdown

# Beispielaufruf
if __name__ == "__main__":
    problem = "Erkläre den Begriff Maschinelles Lernen in einfachen Worten."
    input_data = "Erkläre den Begriff Maschinelles Lernen in einfachen Worten."
    koordinator = Koordinator(problem, input_data)
    markdown = asyncio.run(koordinator.koordiniere())
    print(markdown)