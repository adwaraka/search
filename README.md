Command to run everything

```
docker compose down -v --rmi all && docker compose build --no-cache && docker compose up -d && docker exec -it ollama ollama pull llama3.2:3b && docker exec -it ollama ollama pull nomic-embed-text && docker compose run --rm rag-app
```


Output Sample

```

0 :  The Hobbit.pdf
1 :  The Hound of the Baskervilles.pdf

Enter PDF filename or select the number (or 'exit'): 1
--- Building new index for The Hound of the Baskervilles.pdf (this may take a minute) ---

--- CHAT READY (Type 'exit' to switch files or quit) ---

You: Who accompanies Sir Henry and Dr Mortimer?

[DEBUG] Top Source: Page 82 (Score: 0.8885)

AI: Dr. Watson accompanies Sir Henry and Dr. Mortimer.

You: Quit

0 :  The Hobbit.pdf
1 :  The Hound of the Baskervilles.pdf

Enter PDF filename or select the number (or 'exit'): 0
--- Loading cached index for The Hobbit.pdf ---

--- CHAT READY (Type 'exit' to switch files or quit) ---

You: What is the name of the hobbit?

[DEBUG] Top Source: Page 4 (Score: 0.4787)

AI: The final answer is: Bilbo Baggins.
```