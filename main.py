from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    model_name='qwen2.5:latest', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)
agent = Agent(model)

result = agent.run_sync('When was the first computer bug found?')  
print(result.data)