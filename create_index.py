from llama_index.readers.file import CSVReader
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter

DATA_DIR = "./data"
OUTPUT_DIR = "./index"

parser = CSVReader()
file_extractor = {".csv": parser}  # Add other file formats as needed

# Load documents from the directory
documents = SimpleDirectoryReader(
    DATA_DIR,
    file_extractor=file_extractor,
).load_data()

# Initialize the embedding model
embedding_model = OpenAIEmbedding()
Settings.embed_model = embedding_model

# Parse documents into nodes
node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents)

# Create an index with the nodes
index = VectorStoreIndex(nodes, show_progress=True)

# Save index
index.storage_context.persist(OUTPUT_DIR)
