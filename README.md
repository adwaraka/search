Command to run everything

```
docker compose down -v --rmi all && docker compose build --no-cache && docker compose up -d && docker exec -it ollama ollama pull llama3.2:3b && docker exec -it ollama ollama pull nomic-embed-text && docker compose run --rm rag-app
```