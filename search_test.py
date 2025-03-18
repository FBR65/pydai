import serpapi

serpapi_key='cc92613598e0e46aaf2e30ec2c84daa3011d2790b1f3697cd164241c5aa13edd'

client = serpapi.Client(api_key=serpapi_key)
search_query = """Search the web for articles talking about ManusAI."""

params = {
"engine": "duckduckgo",
"q": search_query,
"kl": "us-en",
"api_key": serpapi_key
}

results = client.search(params)
print(results)