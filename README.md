# ai-text-embedding
Playground for AI text embedding

### SBERT
- Models useful for generating text embeddings (sbert.py)
- Free of cost
- Need to install sentence_transformers (py -m pip install sentence_transformers)

### Connect to Supabase Vector Database
- Install library (py -m pip install supabase)

### Save text embeddings to Pinecone
- Download Quora datasets (py -m pip install datasets)
- Pinecode Library (py -m pip install pinecone)
- Install dotenv (py -m pip install python-dotenv)
- DLAIUtils is a file provided by Pinecone. It contains the code for getting the API key details (dlai_utils.py)

### Experiment -1
- Use local models using Ollama to get vector embeddings instead of open AI models. This is to avoid the cost of OpenAI APIs 
- Note that embedding models cannot be run in ollama cli using ollama pull
- The sample csv provided has the vectors of size 1536. This is stored in the Pinecone vector DB.
- Using Ollama there are limited number of models for providing vector embeddings. List can be found here: https://ollama.com/search?c=embedding
  - nomic-embed-text
     - Vector size 768
  - mxbai-embed-large
     - Vector size 1024