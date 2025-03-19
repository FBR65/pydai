import asyncio
from koordinator import Koordinator

async def main():
    input_data = "Erkläre den Begriff Maschinelles Lernen in einfachen Worten."
    koordinator = Koordinator(input_data)
    markdown = await koordinator.koordiniere()
    print(markdown)

if __name__ == "__main__":
    asyncio.run(main())