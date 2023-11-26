from main import load_chunk_persist_pdf, create_agent_chain, get_llm_response
import os 

import sys
sys.modules['sqlite3'] = __import__('pysqlite3')


# Load Tools
vectordb = load_chunk_persist_pdf()
chain = create_agent_chain()


# Local Testing
from local_secrets.key import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Query PDF Source
QUERY = "Which dashboard can I use for financial analysis by business location"
get_llm_response(QUERY)