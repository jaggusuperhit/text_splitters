from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
import os
import requests

# Load environment variables for OpenRouter API key
load_dotenv()

class OpenRouterEmbeddings:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "openai/text-similarity-ada-001"  # Free embedding model
    
    def embed_documents(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "LangChain Semantic Splitter",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": text} for text in texts],
            "encoding_format": "embed"  # Specify embedding format
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return [item['embedding'] for item in response.json()['data']]
        except Exception as e:
            print(f"Error details: {str(e)}")
            print(f"Response: {response.text if 'response' in locals() else 'None'}")
            raise

# Initialize the OpenRouter embeddings class
print("Testing OpenRouter connection...")
embeddings = OpenRouterEmbeddings()

try:
    test_result = embeddings.embed_documents("test")
    print("✓ Connection successful")
except Exception as e:
    print("Connection failed. Please verify your API key and model.")
    exit()

# Configure SemanticChunker with the working embeddings class
semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    buffer_size=3,
    breakpoint_threshold_type="percentile"
)

# Process sample text using SemanticChunker
sample_text = "LLMs are transforming AI. They understand and generate human-like text."
try:
    chunks = semantic_splitter.split_text(sample_text)
    print(f"\nSuccess! Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(chunk.strip())
except Exception as e:
    print(f"Processing error: {str(e)}")
