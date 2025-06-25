# SRE Framework

This is the main code for the SRE Framework project.

```python
import ollama
import neo4j
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
import json
import os
import time
import logging
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Any

# --- Configuration & Setup ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Load credentials and configuration from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
DEFAULT_DB = os.getenv("NEO4J_DATABASE", "neo4j") # Use environment variable or default to 'neo4j'
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434") # Default Ollama host
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3") # Or the specific llama3 model tag you pulled
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Or another suitable model

# --- Global Clients ---

# Initialize Ollama client (with error handling)
try:
    ollama_client = ollama.Client(host=OLLAMA_HOST)
    ollama_client.list()
    logging.info(f"Connected to Ollama at {OLLAMA_HOST}")
except Exception as e:
    logging.error(f"Failed to connect to Ollama: {e}")
    ollama_client = None

# Initialize Neo4j driver (with error handling)
try:
    neo4j_driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    logging.info(f"Connected to Neo4j at {NEO4J_URI}")
except Exception as e:
    logging.error(f"Failed to connect to Neo4j: {e}")
    neo4j_driver = None

# Initialize Sentence Transformer model (with error handling)
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logging.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logging.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
    embedding_model = None

# Initialize Tavily client (with error handling)
try:
    if TAVILY_API_KEY:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        logging.info("Tavily client initialized.")
    else:
        logging.warning("TAVILY_API_KEY not found in .env file. Web search functionality will be disabled.")
        tavily_client = None
except Exception as e:
    logging.error(f"Failed to initialize Tavily client: {e}")
    tavily_client = None

# --- Helper Functions ---

def execute_cypher_query(query: str, parameters: Optional[Dict[str, Any]] = None) -> List[neo4j.Record]:
    """Executes a Cypher query against the Neo4j database."""
    if not neo4j_driver:
        logging.error("Neo4j driver not initialized.")
        return []
    database_name = DEFAULT_DB
    try:
        with neo4j_driver.session(database=database_name) as session:
            result = session.run(query, parameters or {})
            records = list(result)
            summary = result.consume()
            return records
    except neo4j.exceptions.ServiceUnavailable:
        logging.error(f"Neo4j service is unavailable at {NEO4J_URI}. Ensure the database is running.")
        return []
    except neo4j.exceptions.AuthError:
        logging.error(f"Neo4j authentication failed for user '{NEO4J_USERNAME}'. Check credentials.")
        return []
    except Exception as e:
        if hasattr(e, 'code'):
             logging.error(f"Error executing Cypher query ({e.code}): {e.message}\nQuery: {query}\nParams: {parameters}")
        else:
             logging.error(f"Error executing Cypher query: {e}\nQuery: {query}\nParams: {parameters}")
        return []

def call_ollama(prompt: str, system_message: Optional[str] = None) -> str:
    """Calls the Ollama API with a given prompt."""
    if not ollama_client:
        logging.error("Ollama client not initialized.")
        return "Error: Ollama client not available."
    try:
        messages = []
        if system_message:
            messages.append({'role': 'system', 'content': system_message})
        messages.append({'role': 'user', 'content': prompt})

        response = ollama_client.chat(
            model=LLM_MODEL,
            messages=messages
        )
        if 'message' in response and 'content' in response['message']:
            return response['message']['content']
        else:
            logging.error(f"Unexpected response structure from Ollama: {response}")
            return "Error: Unexpected response structure from Ollama."

    except Exception as e:
        logging.error(f"Error calling Ollama: {e}")
        return f"Error: Failed to get response from Ollama - {e}"

def generate_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    """Generates embeddings for a list of texts using the sentence transformer model."""
    if not embedding_model:
        logging.error("Embedding model not initialized.")
        return None
    if not texts:
        return []
    try:
        # Normalize text before embedding (optional, but can help consistency)
        normalized_texts = [text.lower().strip() for text in texts]
        embeddings = embedding_model.encode(normalized_texts, show_progress_bar=False)
        return embeddings.tolist()
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return None

# --- Core Graph RAG Functions ---

def extract_knowledge_graph(text: str, document_id: str) -> List[Dict[str, str]]:
    """
    Extracts knowledge graph triples (Subject, Predicate, Object) from text using LLM.
    Encourages specific predicates and normalizes entity names.
    Adds document_id to each triple for tracking origin.
    """
    logging.info(f"Extracting knowledge graph from document: {document_id}")
    # ** MODIFICATION: Updated system prompt for dynamic predicates **
    system_message = """You are an expert knowledge graph extractor. Your task is to extract meaningful entities (nodes) and relationships (edges) from the provided text.
    Represent these as triples in the format: {"subject": "...", "predicate": "...", "object": "..."}.
    - Use specific and descriptive predicate names that capture the relationship accurately (e.g., "is capital of", "composed of", "orbits", "has mass", "called"). Avoid generic predicates like "related to" or "is".
    - Normalize entity names (e.g., convert to lowercase, consistent phrasing like "solar system").
    Focus on the core facts and relationships. Avoid trivial or overly complex extractions.
    Output *only* a valid JSON list of these triple dictionaries. Do not include any explanations, introductory text, or markdown formatting.
    Example: [{"subject": "paris", "predicate": "is capital of", "object": "france"}, {"subject": "eiffel tower", "predicate": "located in", "object": "paris"}, {"subject": "mars", "predicate": "called", "object": "red planet"}]
    """
    prompt = f"Extract knowledge graph triples from the following text:\n\n---\n{text}\n---\n\nOutput the result as a JSON list of triples:"

    response = call_ollama(prompt, system_message)

    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end != -1 and json_start < json_end:
            clean_response = response[json_start:json_end]
            triples = json.loads(clean_response)
            valid_triples = []
            for triple in triples:
                if isinstance(triple, dict) and all(k in triple for k in ['subject', 'predicate', 'object']):
                     # ** MODIFICATION: Normalize (lowercase, strip) subject/object here **
                     s = str(triple['subject']).strip().lower()
                     p = str(triple['predicate']).strip() # Predicate might be case-sensitive depending on desired modeling
                     o = str(triple['object']).strip().lower()
                     if s and p and o:
                         valid_triples.append({'subject': s, 'predicate': p, 'object': o, 'document_id': document_id})
                     else:
                         logging.warning(f"Skipping triple with empty fields after normalization: {triple}")
                else:
                    logging.warning(f"Skipping invalid triple structure: {triple}")

            logging.info(f"Extracted {len(valid_triples)} valid triples for document {document_id}.")
            return valid_triples
        else:
            logging.error(f"Failed to find valid JSON list brackets in LLM response for KG extraction: {response}")
            return []
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from LLM response for KG extraction: {e}\nResponse snippet: {response[:500]}")
        return []
    except Exception as e:
         logging.error(f"An unexpected error occurred during KG extraction parsing: {e}\nResponse: {response}")
         return []

def store_graph_in_neo4j(triples: List[Dict[str, str]], graph_id: str = "primary"):
    """
    Stores extracted triples in Neo4j, creates nodes and relationships.
    Uses normalized (lowercase, stripped) names for merging nodes.
    Generates and stores embeddings for nodes.
    Adds a 'graph_id' property to nodes and relationships for potential versioning/scoping.
    """
    if not neo4j_driver or not embedding_model:
        logging.error("Cannot store graph: Neo4j driver or embedding model not initialized.")
        return
    if not triples:
        logging.warning("No triples provided to store in Neo4j.")
        return
```
