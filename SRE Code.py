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

    logging.info(f"Storing {len(triples)} triples in Neo4j with graph_id: {graph_id}")

    # Node names are already normalized (lowercase, stripped) in extract_knowledge_graph
    node_names = set()
    for triple in triples:
        node_names.add(triple['subject'])
        node_names.add(triple['object'])
    node_list = list(node_names)

    logging.info(f"Generating embeddings for {len(node_list)} unique node names.")
    # Embeddings are generated based on the normalized names
    node_embeddings = generate_embeddings(node_list)

    if node_embeddings is None:
        logging.error("Failed to generate embeddings. Aborting graph storage.")
        return
    if len(node_embeddings) != len(node_list):
         logging.error(f"Mismatch between number of nodes ({len(node_list)}) and generated embeddings ({len(node_embeddings)}). Aborting.")
         return

    node_embedding_map = {name: emb for name, emb in zip(node_list, node_embeddings)}

    database_name = DEFAULT_DB

    def _create_graph_tx(tx, triples_batch, node_embeddings_map, graph_id_val):
        # ** MODIFICATION: Query uses normalized names (subj_name, obj_name) from triples **
        # The `type` property stores the potentially mixed-case predicate from the LLM
        query = """
        UNWIND $triples as triple
        // Use pre-normalized names from the triple dictionary
        WITH triple, triple.subject AS subj_name, triple.object AS obj_name
        MERGE (s:Entity {name: subj_name}) // MERGE based on normalized name
        ON CREATE SET s.embedding = coalesce($node_embeddings[subj_name], []), s.graph_id = $graph_id, s.document_id = triple.document_id
        ON MATCH SET s.embedding = coalesce($node_embeddings[subj_name], s.embedding), s.graph_id = $graph_id // Update embedding/graph_id

        MERGE (o:Entity {name: obj_name}) // MERGE based on normalized name
        ON CREATE SET o.embedding = coalesce($node_embeddings[obj_name], []), o.graph_id = $graph_id, o.document_id = triple.document_id
        ON MATCH SET o.embedding = coalesce($node_embeddings[obj_name], o.embedding), o.graph_id = $graph_id // Update embedding/graph_id

        // Store the original predicate in the 'type' property of a generic RELATED_TO relationship
        MERGE (s)-[r:RELATED_TO {type: triple.predicate}]->(o)
        ON CREATE SET r.graph_id = $graph_id, r.document_id = triple.document_id
        ON MATCH SET r.graph_id = $graph_id // Update graph_id
        """
        try:
            tx.run(query, triples=triples_batch, node_embeddings=node_embeddings_map, graph_id=graph_id_val)
        except Exception as tx_error:
             logging.error(f"Error within graph creation transaction: {tx_error}")
             raise

    try:
        with neo4j_driver.session(database=database_name) as session:
            batch_size = 5000
            for i in range(0, len(triples), batch_size):
                batch = triples[i:i + batch_size]
                session.execute_write(_create_graph_tx, batch, node_embedding_map, graph_id)
        logging.info(f"Successfully stored triples and embeddings in Neo4j for graph_id: {graph_id}.")
    except Exception as e:
        logging.error(f"Failed during Neo4j session/transaction for storing graph: {e}")
        return

    # Create vector index
    index_query = """
    CREATE VECTOR INDEX `entity_embeddings` IF NOT EXISTS
    FOR (e:Entity) ON (e.embedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: $embedding_dim,
        `vector.similarity_function`: 'cosine'
    }}
    """
    try:
        if node_embeddings and node_embeddings[0]:
             embedding_dim = len(node_embeddings[0])
             execute_cypher_query(index_query, {"embedding_dim": embedding_dim})
             logging.info("Ensured vector index 'entity_embeddings' exists.")
             time.sleep(5)
        else:
             logging.warning("No node embeddings found or embeddings are empty, skipping vector index creation.")
    except Exception as e:
        logging.error(f"Failed to create or verify vector index 'entity_embeddings': {e}")


def cluster_graph_neo4j(graph_id: str) -> Optional[Dict[int, List[str]]]:
    """
    Clusters the graph in Neo4j using the Louvain algorithm via GDS.
    Requires GDS plugin installed in Neo4j.
    Returns a dictionary mapping cluster ID to list of node names in that cluster.
    Uses corrected legacy GDS Cypher projection syntax.
    Note: Louvain optimizes modularity and does not guarantee a fixed number of clusters.
    """
    if not neo4j_driver:
        logging.error("Neo4j driver not initialized.")
        return None

    logging.info(f"Running Louvain clustering on graph_id: {graph_id}")
    projected_graph_name = f"clusterGraph_{graph_id.replace('-', '_')}"
    community_property = f'communityId_{graph_id.replace("-", "_")}'

    # 1. Drop previous projection if exists
    drop_graph_query = f"CALL gds.graph.drop('{projected_graph_name}', false) YIELD graphName;"
    execute_cypher_query(drop_graph_query)

    # 2. Build the node and relationship query strings
    node_query = f"""
    MATCH (n:Entity) WHERE n.graph_id = '{graph_id}'
    RETURN id(n) AS id
    """
    rel_query = f"""
    MATCH (s:Entity)-[r:RELATED_TO]->(o:Entity)
    WHERE s.graph_id = '{graph_id}' AND o.graph_id = '{graph_id}' AND r.graph_id = '{graph_id}'
    RETURN id(s) AS source, id(o) AS target
    """

    # 3. Project the graph using the legacy Cypher projection procedure
    project_graph_query = """
    CALL gds.graph.project.cypher(
      $graph_name,
      $node_query,
      $rel_query
    )
    YIELD graphName, nodeCount, relationshipCount
    RETURN graphName, nodeCount, relationshipCount
    """
    params = {
        "graph_name": projected_graph_name,
        "node_query": node_query,
        "rel_query": rel_query
    }
    result = execute_cypher_query(project_graph_query, params)

    if not result:
         logging.error(f"Graph projection query failed to return results for graph_id: {graph_id}.")
         execute_cypher_query(drop_graph_query)
         return None
    if not result or result[0]['nodeCount'] is None or result[0]['nodeCount'] == 0:
        logging.warning(f"Graph projection resulted in an empty graph (0 nodes) for graph_id: {graph_id}. Check node query and data in Neo4j.")
        execute_cypher_query(drop_graph_query)
        return None
    if result[0]['relationshipCount'] == 0:
         logging.warning(f"Graph projection resulted in 0 relationships for graph_id: {graph_id}. Clustering might produce trivial results (each node its own cluster).")

    logging.info(f"Projected graph '{result[0]['graphName']}' with {result[0]['nodeCount']} nodes and {result[0]['relationshipCount']} relationships.")

    # 4. Run Louvain and write results
    # Note: We cannot directly force 5 clusters with gds.louvain.write.
    # It optimizes modularity. Parameters like 'tolerance' could be experimented with,
    # but don't guarantee a specific cluster count.
    louvain_query = f"""
    CALL gds.louvain.write(
        '{projected_graph_name}',
        {{
            writeProperty: '{community_property}'
            //, tolerance: 0.001 // Example: Lower tolerance might lead to more clusters (experiment needed)
        }}
    ) YIELD communityCount, modularity
    RETURN communityCount, modularity
    """
    louvain_result = execute_cypher_query(louvain_query)
    if not louvain_result:
        logging.error(f"Louvain algorithm execution failed for graph '{projected_graph_name}'.")
        execute_cypher_query(drop_graph_query)
        return None
    found_clusters = louvain_result[0]['communityCount']
    logging.info(f"Louvain clustering completed: Found {found_clusters} communities with modularity {louvain_result[0]['modularity']:.4f}. Results written to '{community_property}'.")
    # Log if the number of clusters is different from the desired (though not enforced) 5
    if found_clusters != 5:
        logging.warning(f"Louvain algorithm found {found_clusters} clusters, not exactly 5.")


    # 5. Retrieve cluster results
    get_clusters_query = f"""
    MATCH (n:Entity) WHERE n.graph_id = $graph_id AND n.`{community_property}` IS NOT NULL
    RETURN n.name as nodeName, n.`{community_property}` as clusterId
    ORDER BY clusterId
    """
    cluster_data = execute_cypher_query(get_clusters_query, {"graph_id": graph_id})

    # 6. Group nodes by cluster ID
    clusters: Dict[int, List[str]] = {}
    if not cluster_data:
         logging.warning(f"No nodes found with community ID property '{community_property}' for graph_id '{graph_id}'.")
    else:
        for record in cluster_data:
            cluster_id = record['clusterId']
            node_name = record['nodeName']
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(node_name)
        logging.info(f"Retrieved {len(clusters)} clusters with data from Neo4j.")

    # 7. Clean up projected graph
    execute_cypher_query(drop_graph_query)
    logging.info(f"Cleaned up projected graph: {projected_graph_name}")

    return clusters


def summarize_cluster(cluster_nodes: List[str], graph_id: str) -> str:
    """
    Retrieves triples related to cluster nodes and asks LLM to summarize them.
    Filters relationships based on graph_id.
    """
    if not cluster_nodes:
        return "Cluster is empty."

    logging.info(f"Summarizing cluster with {len(cluster_nodes)} nodes.")

    # Retrieve triples involving nodes in the cluster, ensuring both nodes and rels match graph_id
    query = """
    MATCH (n:Entity)-[r:RELATED_TO]-(m:Entity)
    WHERE n.name IN $node_names AND r.graph_id = $graph_id AND n.graph_id = $graph_id AND m.graph_id = $graph_id
    RETURN n.name as subject, r.type as predicate, m.name as object // Use r.type to get the specific predicate
    LIMIT 100 // Limit context size for LLM
    """
    parameters = {"node_names": cluster_nodes, "graph_id": graph_id}
    triples_data = execute_cypher_query(query, parameters)

    if not triples_data:
        context = "Nodes in this cluster: " + ", ".join(cluster_nodes)
        logging.warning(f"No connecting relationships found within graph_id '{graph_id}' for cluster nodes: {cluster_nodes}. Summarizing based on node names only.")
    else:
        # Use the specific predicate from r.type in the context string
        context_triples = [f"({record['subject']})-[{record['predicate']}]->({record['object']})" for record in triples_data]
        context = "Nodes and relationships in this cluster:\n" + "\n".join(context_triples)

    system_message = "You are an expert summarizer. Based on the provided nodes and relationships (triples) from a knowledge graph cluster, provide a concise, coherent summary (1-2 sentences) of the main topic or theme of this cluster. Focus on capturing the essence of the information represented."
    prompt = f"Summarize the following knowledge graph cluster information:\n\n---\n{context}\n---\n\nSummary:"

    summary = call_ollama(prompt, system_message)
    logging.info(f"Generated summary for cluster.")
    return summary

def generate_reflection_questions(summary: str) -> List[str]:
    """Asks LLM to generate reflective questions based on a cluster summary."""
    logging.info("Generating reflection questions for summary.")
    system_message = """You are a critical thinking assistant. Based on the provided summary of a knowledge graph cluster, generate 2-3 insightful questions that probe deeper into the topic, identify potential gaps in knowledge, or suggest areas for further exploration.
    The questions should be open-ended and encourage enriching the graph.
    Output *only* a valid JSON list of question strings. Do not include any explanations, introductory text or markdown formatting.
    Example: ["What are the specific implications of X on Y?", "How does Z relate to the broader context of A?"]
    """
    prompt = f"Generate 2-3 reflective questions based on this summary:\n\n---\n{summary}\n---\n\nOutput the result as a JSON list of questions:"

    response = call_ollama(prompt, system_message)

    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end != -1 and json_start < json_end:
            clean_response = response[json_start:json_end]
            questions = json.loads(clean_response)
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                logging.info(f"Generated {len(questions)} reflection questions.")
                return questions
            else:
                logging.error(f"Parsed JSON is not a list of strings: {questions}")
                return []
        else:
            logging.error(f"Failed to find valid JSON list brackets in LLM response for reflection questions: {response}")
            return []
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from LLM response for reflection questions: {e}\nResponse snippet: {response[:500]}")
        return []
    except Exception as e:
         logging.error(f"An unexpected error occurred during reflection question parsing: {e}\nResponse: {response}")
         return []

def search_web(query: str) -> Optional[str]:
    """Performs web search using Tavily API."""
    if not tavily_client:
        logging.warning("Tavily client not available. Skipping web search.")
        return None
    logging.info(f"Performing web search for: {query}")
    try:
        response = tavily_client.search(query=query, search_depth="basic", include_answer=True, max_results=3)
        search_results_content = "\n".join([res.get('content', '') for res in response.get('results', [])])
        answer = response.get('answer', '')
        combined_context = ""
        if answer:
            combined_context += f"Web Answer: {answer}\n"
        if search_results_content:
             combined_context += f"Web Snippets:\n{search_results_content}"

        if not combined_context:
             logging.warning(f"Web search for '{query}' returned no results or answer.")
             return None
        logging.info(f"Web search successful for: {query}")
        return combined_context.strip()
    except Exception as e:
        logging.error(f"Error during Tavily web search for '{query}': {e}")
        return None

def enrich_graph_from_answers(answers: Dict[str, str], original_document_text: str, graph_id: str):
    """
    Processes answers to reflection questions, extracts new triples with specific predicates,
    normalizes entity names, and adds them to the Neo4j graph under the specified graph_id,
    attempting to connect to existing nodes via normalized names.
    """
    logging.info(f"Enriching graph '{graph_id}' based on {len(answers)} answers.")
    all_new_triples = []

    # Retrieve existing node names for context (optional, helps LLM)
    # existing_nodes_query = "MATCH (n:Entity) WHERE n.graph_id = $graph_id RETURN collect(n.name) as names"
    # existing_nodes_result = execute_cypher_query(existing_nodes_query, {"graph_id": graph_id})
    # existing_node_names = existing_nodes_result[0]['names'] if existing_nodes_result else []
    # logging.info(f"Providing {len(existing_node_names)} existing node names as context for enrichment.")

    for question, answer in answers.items():
        if not answer or answer.startswith("Could not find an answer"):
             logging.warning(f"Skipping enrichment for question due to no answer: '{question}'")
             continue

        logging.info(f"Extracting enrichment triples for answer to: '{question}'")
        # ** MODIFICATION: Updated system prompt for enrichment **
        system_message = """You are an expert knowledge graph extractor. Your task is to extract meaningful entities and relationships from the provided answer text, considering the original document context.
        Represent these as triples: {"subject": "...", "predicate": "...", "object": "..."}.
        - Use specific and descriptive predicate names (e.g., "has moon", "visited by", "composed of").
        - Normalize entity names (lowercase, consistent phrasing).
        - IMPORTANT: Try to use entity names for subjects/objects that likely already exist in the original document context provided below, where appropriate, to ensure the new information connects to the existing graph.
        - Focus on information that *enriches* or *expands* upon the original knowledge.
        Output *only* a valid JSON list of these triple dictionaries. No explanations or markdown.
        """
        context_limit = 2000
        prompt = f"""Original Document Context (for reference):
        ---
        {original_document_text[:context_limit]}...
        ---

        Answer to reflect on (source: LLM or Web Search):
        ---
        {answer}
        ---

        Extract new knowledge graph triples from the answer text as a JSON list, linking to existing entities from the context where possible:"""

        response = call_ollama(prompt, system_message)

        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end != -1 and json_start < json_end:
                clean_response = response[json_start:json_end]
                new_triples = json.loads(clean_response)
                valid_new_triples = []
                for triple in new_triples:
                    if isinstance(triple, dict) and all(k in triple for k in ['subject', 'predicate', 'object']):
                        # ** MODIFICATION: Normalize enrichment triples **
                        s = str(triple['subject']).strip().lower()
                        p = str(triple['predicate']).strip()
                        o = str(triple['object']).strip().lower()
                        if s and p and o:
                             valid_new_triples.append({
                                 'subject': s,
                                 'predicate': p,
                                 'object': o,
                                 'document_id': f"enrichment_{graph_id}" # Mark source
                             })
                        else:
                             logging.warning(f"Skipping enrichment triple with empty fields after normalization: {triple}")
                    else:
                        logging.warning(f"Skipping invalid enrichment triple structure: {triple}")

                logging.info(f"Extracted {len(valid_new_triples)} valid enrichment triples for one answer.")
                all_new_triples.extend(valid_new_triples)
            else:
                 logging.warning(f"Could not find valid JSON list brackets in enrichment extraction response: {response}")

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON from LLM response for enrichment extraction: {e}\nResponse snippet: {response[:500]}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during enrichment parsing: {e}\nResponse: {response}")


    if all_new_triples:
        logging.info(f"Storing a total of {len(all_new_triples)} enrichment triples.")
        # Store these new triples using the specified graph_id
        # The store function uses MERGE on normalized names, achieving the connection goal
        store_graph_in_neo4j(all_new_triples, graph_id)
        time.sleep(7)
    else:
        logging.info("No valid new triples extracted from enrichment answers.")


def query_graph_rag(user_query: str, graph_id: str, top_k: int = 5) -> str:
    """
    Performs RAG using the knowledge graph.
    Uses normalized query for embedding search.
    Retrieves subgraph and cluster summaries for context.
    """
    if not neo4j_driver or not embedding_model or not ollama_client:
        return "Error: Required components (Neo4j, Embeddings, LLM) are not initialized."

    logging.info(f"Performing RAG query for: '{user_query}' on graph_id: {graph_id}")

    # 1. Embed the user query (consider normalizing query too)
    normalized_query = user_query.lower().strip()
    query_embedding = generate_embeddings([normalized_query])
    if not query_embedding:
        return "Error: Failed to generate embedding for the query."

    # 2. Find relevant nodes via vector similarity search
    similarity_query = """
    CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $query_embedding) YIELD node, score
    WHERE node.graph_id = $graph_id // Filter results by graph_id
    RETURN node.name AS nodeName, score // node.name is already normalized (lowercase)
    ORDER BY score DESC
    """
    parameters = {
        "top_k": top_k,
        "query_embedding": query_embedding[0],
        "graph_id": graph_id
    }
    similar_nodes_result = execute_cypher_query(similarity_query, parameters)

    if not similar_nodes_result:
        logging.warning(f"No similar nodes found via vector search in graph '{graph_id}' for the query: '{user_query}'.")
        logging.info("Falling back to direct LLM call.")
        # Provide original query to LLM fallback
        return call_ollama(f"Answer the following question based on general knowledge: {user_query}")

    # Relevant node names are already normalized (lowercase)
    relevant_node_names = [record['nodeName'] for record in similar_nodes_result]
    logging.info(f"Found {len(relevant_node_names)} relevant nodes via vector search: {relevant_node_names}")

    # 3. Retrieve the local subgraph
    # Query uses normalized names from relevant_node_names
    subgraph_query = """
    MATCH (n:Entity)-[r:RELATED_TO]-(m:Entity)
    WHERE n.name IN $node_names
      AND n.graph_id = $graph_id
      AND m.graph_id = $graph_id
      AND r.graph_id = $graph_id
    RETURN n.name as subject, r.type as predicate, m.name as object // Use r.type for specific predicate
    LIMIT 50
    """
    subgraph_result = execute_cypher_query(subgraph_query, {"node_names": relevant_node_names, "graph_id": graph_id})
    # Format context using the specific predicate from r.type
    subgraph_context = "\n".join([f"({r['subject']})-[{r['predicate']}]->({r['object']})" for r in subgraph_result])
    logging.info(f"Retrieved {len(subgraph_result)} triples for subgraph context.")


    # 4. Retrieve summaries for relevant clusters
    cluster_summaries_context = ""
    community_property = f'communityId_{graph_id.replace("-", "_")}'
    get_community_ids_query = f"""
    MATCH (n:Entity)
    WHERE n.name IN $node_names AND n.graph_id = $graph_id AND n.`{community_property}` IS NOT NULL
    RETURN DISTINCT n.`{community_property}` as clusterId
    """
    community_ids_result = execute_cypher_query(get_community_ids_query, {"node_names": relevant_node_names, "graph_id": graph_id})
    relevant_cluster_ids = [record['clusterId'] for record in community_ids_result]

    if relevant_cluster_ids:
        logging.info(f"Relevant cluster IDs found: {relevant_cluster_ids}")
        # Fetch nodes for relevant clusters directly
        get_cluster_nodes_query = f"""
        MATCH (n:Entity)
        WHERE n.graph_id = $graph_id AND n.`{community_property}` IN $cluster_ids
        RETURN n.`{community_property}` as clusterId, collect(n.name) as nodeNames
        """
        cluster_nodes_data = execute_cypher_query(get_cluster_nodes_query, {"graph_id": graph_id, "cluster_ids": relevant_cluster_ids})
        cluster_nodes_map = {record['clusterId']: record['nodeNames'] for record in cluster_nodes_data}

        if cluster_nodes_map:
             summaries = []
             for cluster_id in relevant_cluster_ids:
                 if cluster_id in cluster_nodes_map:
                     summary = summarize_cluster(cluster_nodes_map[cluster_id], graph_id)
                     summaries.append(f"Cluster {cluster_id} Summary: {summary}")
                 else:
                      logging.warning(f"Cluster ID {cluster_id} node data not retrieved.")
             cluster_summaries_context = "\n\n".join(summaries)
             logging.info("Generated summaries for relevant clusters.")
        else:
            logging.warning("Could not retrieve cluster map to generate summaries.")
    else:
        logging.info("No relevant cluster IDs found for the retrieved nodes or clustering failed previously.")


    # 5. Synthesize answer using LLM (provide original query for clarity)
    system_message = "You are a helpful AI assistant. Answer the user's query based *only* on the provided context extracted from a knowledge graph. This context includes relevant triples (Subject-Predicate-Object) and potentially summaries of related clusters. If the context doesn't contain the answer, clearly state that the information is not available in the provided context."

    prompt = f"""User Query: {user_query} # Use original query for LLM

    === Relevant Knowledge Graph Context ===

    **Triples related to query:**
    {subgraph_context if subgraph_context else "No specific triples found matching the query."}

    **Relevant Cluster Summaries:**
    {cluster_summaries_context if cluster_summaries_context else "No relevant cluster summaries found or generated."}

    === End of Context ===

    Based *only* on the context above, answer the user query:"""

    final_answer = call_ollama(prompt, system_message)
    logging.info("Generated final answer using RAG.")
    return final_answer


# --- Main Workflow Example ---

def run_graph_rag_pipeline(file_path: str, document_id: str) -> Tuple[Optional[Dict[int, List[str]]], Dict[int, str], Optional[Dict[int, List[str]]], Dict[int, str]]:
    """
    Runs the full Graph RAG pipeline for a given file.
    Returns cluster and summary dictionaries for initial and enriched graphs.
    """
    initial_clusters = None
    initial_summaries = {}
    enriched_clusters = None
    enriched_summaries = {}

    if not all([ollama_client, neo4j_driver, embedding_model]):
         logging.error("One or more clients (Ollama, Neo4j, Embeddings) failed to initialize. Aborting pipeline.")
         return initial_clusters, initial_summaries, enriched_clusters, enriched_summaries

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
        logging.info(f"Successfully loaded document: {file_path}")
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return initial_clusters, initial_summaries, enriched_clusters, enriched_summaries
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return initial_clusters, initial_summaries, enriched_clusters, enriched_summaries

    initial_graph_id = f"{document_id}_initial"
    enriched_graph_id = f"{document_id}_enriched"

    logging.info(f"Optional: Cleaning up previous data for graph IDs '{initial_graph_id}' and '{enriched_graph_id}'...")
    db_name = DEFAULT_DB
    try:
        with neo4j_driver.session(database=db_name) as session:
            initial_community_prop = f"communityId_{initial_graph_id.replace('-', '_')}"
            enriched_community_prop = f"communityId_{enriched_graph_id.replace('-', '_')}"
            session.execute_write(lambda tx: tx.run(f"MATCH (n) WHERE n.graph_id = $id REMOVE n.`{initial_community_prop}`", id=initial_graph_id))
            session.execute_write(lambda tx: tx.run(f"MATCH (n) WHERE n.graph_id = $id REMOVE n.`{enriched_community_prop}`", id=enriched_graph_id))
            session.execute_write(lambda tx: tx.run("MATCH (n) WHERE n.graph_id = $initial_id OR n.graph_id = $enriched_id DETACH DELETE n",
                                                   initial_id=initial_graph_id, enriched_id=enriched_graph_id))
        logging.info("Cleanup finished.")
    except Exception as e:
         logging.error(f"Error during cleanup: {e}")

    time.sleep(2)

    initial_triples = extract_knowledge_graph(document_text, document_id)
    if not initial_triples:
        logging.error("Failed to extract initial knowledge graph. Aborting.")
        return initial_clusters, initial_summaries, enriched_clusters, enriched_summaries

    logging.info(f"\n--- Storing Initial Graph ({initial_graph_id}) ---")
    store_graph_in_neo4j(initial_triples, graph_id=initial_graph_id)
    logging.info("Waiting after initial storage for indexing...")
    time.sleep(10)

    logging.info(f"\n--- Clustering and Summarizing Initial Graph ({initial_graph_id}) ---")
    initial_clusters = cluster_graph_neo4j(graph_id=initial_graph_id)
    if initial_clusters:
        logging.info(f"Found {len(initial_clusters)} clusters in the initial graph.")
        for cluster_id, nodes in initial_clusters.items():
            summary = summarize_cluster(nodes, graph_id=initial_graph_id)
            initial_summaries[cluster_id] = summary
            logging.info(f"Cluster {cluster_id} Summary (Initial): {summary[:150]}...")
    else:
        logging.warning("Clustering did not produce results for the initial graph.")

    logging.info(f"\n--- Preparing Enriched Graph Space ({enriched_graph_id}) ---")
    copy_query = """
    MATCH (n:Entity) WHERE n.graph_id = $initial_id
    WITH n, $enriched_id AS enriched_graph_id
    MERGE (n_copy:Entity {name: n.name, graph_id: enriched_graph_id})
    SET n_copy = properties(n)
    SET n_copy.graph_id = enriched_graph_id

    WITH collect(n) as source_nodes, enriched_graph_id, $initial_id AS initial_id
    MATCH (s:Entity)-[r:RELATED_TO]->(o:Entity)
    WHERE s IN source_nodes AND o IN source_nodes AND r.graph_id = initial_id
    MATCH (s_copy:Entity {name: s.name, graph_id: enriched_graph_id})
    MATCH (o_copy:Entity {name: o.name, graph_id: enriched_graph_id})
    MERGE (s_copy)-[r_copy:RELATED_TO {type: r.type}]->(o_copy)
    SET r_copy = properties(r)
    SET r_copy.graph_id = enriched_graph_id
    RETURN count(r_copy) as copied_rels
    """
    copy_result = execute_cypher_query(copy_query, {"initial_id": initial_graph_id, "enriched_id": enriched_graph_id})
    copied_rels_count = copy_result[0]['copied_rels'] if copy_result else 0
    logging.info(f"Finished copying data ({copied_rels_count} relationships) to {enriched_graph_id}.")
    time.sleep(7)

    logging.info(f"\n--- Starting Reflection and Enrichment on {enriched_graph_id} ---")
    all_answers = {}
    if initial_summaries:
        logging.info(f"Generating reflection questions based on {len(initial_summaries)} initial cluster summaries.")
        for cluster_id, summary in initial_summaries.items():
            logging.info(f"--- Processing reflection for Cluster {cluster_id} ---")
            questions = generate_reflection_questions(summary)
            for question in questions:
                logging.info(f"  Reflective Question: {question}")
                answer_prompt = f"Based *only* on the following text, answer the question: '{question}'\n\nText:\n{document_text}"
                answer = call_ollama(answer_prompt)

                is_unsure = "don't know" in answer.lower() or \
                            "not mentioned" in answer.lower() or \
                            "no information" in answer.lower() or \
                            len(answer) < 40

                if is_unsure:
                    logging.info("  LLM unsure or answer too short, attempting web search...")
                    web_context = search_web(question)
                    if web_context:
                        web_answer_prompt = f"Based *only* on the following web search results, answer the question: '{question}'\n\nWeb Results:\n{web_context}"
                        answer = call_ollama(web_answer_prompt)
                        logging.info("  Generated answer from web search results.")
                    else:
                        answer = f"Could not find an answer via LLM or web search for '{question}'."
                        logging.warning(f"  Web search failed for question: {question}")
                else:
                     logging.info("  Generated answer from LLM context.")

                logging.info(f"  Answer: {answer[:150]}...")
                all_answers[question] = answer
    else:
        logging.warning("Skipping reflection phase as no initial cluster summaries were generated.")

    if all_answers:
        enrich_graph_from_answers(all_answers, document_text, graph_id=enriched_graph_id)
    else:
        logging.info("No answers generated from reflection, skipping enrichment storage.")

    logging.info(f"\n--- Clustering and Summarizing Enriched Graph ({enriched_graph_id}) ---")
    enriched_clusters = cluster_graph_neo4j(graph_id=enriched_graph_id)
    if enriched_clusters:
        logging.info(f"Found {len(enriched_clusters)} clusters in the enriched graph.")
        for cluster_id, nodes in enriched_clusters.items():
            summary = summarize_cluster(nodes, graph_id=enriched_graph_id)
            enriched_summaries[cluster_id] = summary
            logging.info(f"Cluster {cluster_id} Summary (Enriched): {summary[:150]}...")
    else:
        logging.warning("Clustering did not produce results for the enriched graph.")

    logging.info("\n--- Pipeline Finished ---")
    logging.info(f"Initial graph stored under graph_id: {initial_graph_id}")
    logging.info(f"Enriched graph stored under graph_id: {enriched_graph_id}")
    logging.info("You can now query the graphs using query_graph_rag(query, graph_id).")

    return initial_clusters, initial_summaries, enriched_clusters, enriched_summaries


# --- Example Usage ---
if __name__ == "__main__":
    DUMMY_FILE_PATH = "example_document.txt"
    DUMMY_DOCUMENT_ID = "doc001"
    # ** MODIFICATION: Expanded dummy text **
    DUMMY_TEXT = """
    The Solar System is the gravitationally bound system comprising the Sun and the objects that orbit it, either directly or indirectly. It formed approximately 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud. The vast majority of the system's mass (about 99.86%) resides in the Sun. Jupiter contains most of the remaining mass.

    The four inner planets are Mercury, Venus, Earth, and Mars. These are often called the terrestrial planets due to their composition, being primarily rock and metal. They are relatively small and dense. Mercury is the smallest and closest to the Sun. Venus possesses a thick, toxic atmosphere primarily composed of carbon dioxide, leading to a runaway greenhouse effect and making it the hottest planet. Earth, our home, has liquid water on its surface and supports a diverse biosphere. Mars, known as the Red Planet due to prevalent iron oxide on its surface, has a thin atmosphere and features such as volcanoes like Olympus Mons (the largest in the Solar System) and canyons like Valles Marineris. Evidence suggests Mars may have had liquid water in its past.

    Beyond Mars lies the asteroid belt, a region populated by numerous small bodies and minor planets, including the dwarf planet Ceres.

    The four outer planets are the gas giants Jupiter and Saturn, and the ice giants Uranus and Neptune. These planets are substantially larger than the terrestrials and have lower densities. Jupiter and Saturn are composed mainly of hydrogen and helium. Jupiter is the largest planet and possesses a strong magnetic field and numerous moons, including the four large Galilean moons (Io, Europa, Ganymede, Callisto) discovered by Galileo Galilei in 1610. Saturn is famous for its extensive ring system, composed mostly of ice particles.

    Uranus and Neptune, the ice giants, contain higher proportions of 'ices' like water, ammonia, and methane compared to the gas giants. Uranus has a unique axial tilt, orbiting the Sun almost on its side. Neptune, the outermost planet, is known for its strong winds. Beyond Neptune lies the Kuiper Belt, another region of small icy bodies, including dwarf planets like Pluto and Eris. Further out is the hypothesized Oort Cloud, a vast spherical cloud thought to be the origin of long-period comets.
    """
    try:
        with open(DUMMY_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(DUMMY_TEXT)
        logging.info(f"Created dummy document: {DUMMY_FILE_PATH}")

        initial_clusters, initial_summaries, enriched_clusters, enriched_summaries = run_graph_rag_pipeline(DUMMY_FILE_PATH, DUMMY_DOCUMENT_ID)

        print("\n\n" + "="*20 + " CLUSTER RESULTS " + "="*20)
        print("\n--- Initial Graph Clusters & Summaries ---")
        if initial_clusters:
            for cluster_id, nodes in initial_clusters.items():
                print(f"\nCluster ID: {cluster_id}")
                print(f"Nodes: {nodes}")
                print(f"Summary: {initial_summaries.get(cluster_id, 'N/A')}")
        else:
            print("No clusters found for the initial graph.")

        print("\n--- Enriched Graph Clusters & Summaries ---")
        if enriched_clusters:
             for cluster_id, nodes in enriched_clusters.items():
                print(f"\nCluster ID: {cluster_id}")
                print(f"Nodes: {nodes}")
                print(f"Summary: {enriched_summaries.get(cluster_id, 'N/A')}")
        else:
             print("No clusters found for the enriched graph.")
        print("\n" + "="*57)


        logging.info("Running example queries...")
        time.sleep(5)

        # --- Example Queries ---
        example_queries = [
            "What are the inner planets?",
            "Tell me about Mars.",
            "What is the atmosphere of Venus like?",
            "How many moons does Jupiter have?",
            "What is beyond Neptune?",
            "Describe the asteroid belt.",
        ]

        for query in example_queries:
            print("\n\n" + "="*10 + f" Query: {query} " + "="*10)

            print("\n--- Answer from Initial Graph ---")
            answer_initial = query_graph_rag(query, graph_id=f"{DUMMY_DOCUMENT_ID}_initial")
            print(answer_initial)

            print("\n--- Answer from Enriched Graph ---")
            answer_enriched = query_graph_rag(query, graph_id=f"{DUMMY_DOCUMENT_ID}_enriched")
            print(answer_enriched)

            print("\n" + "="*(22 + len(query)))


        # ** MODIFICATION: Add interactive query loop **
        print("\n\n=== Interactive Query Session ===")
        print("Enter your questions about the document. Type 'quit' or 'exit' to stop.")
        while True:
            try:
                user_query = input("Your Query: ")
                if user_query.lower() in ['quit', 'exit']:
                    break
                if not user_query:
                    continue

                print("\n--- Answer from Initial Graph ---")
                answer_initial = query_graph_rag(user_query, graph_id=f"{DUMMY_DOCUMENT_ID}_initial")
                print(answer_initial)

                print("\n--- Answer from Enriched Graph ---")
                answer_enriched = query_graph_rag(user_query, graph_id=f"{DUMMY_DOCUMENT_ID}_enriched")
                print(answer_enriched)
                print("-" * 40)

            except EOFError: # Handle Ctrl+D or end of input stream
                 break
            except KeyboardInterrupt: # Handle Ctrl+C
                 break

        print("\nExiting interactive query session.")


    except Exception as e:
        logging.exception(f"An error occurred during the main execution: {e}")
    finally:
        if neo4j_driver:
            neo4j_driver.close()
            logging.info("Neo4j driver closed.")

