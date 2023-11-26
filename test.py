from main import get_llm_response
import os 

# Local Testing
from local_secrets.key import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Query PDF Source
QUERY = "Describe sktime"
get_llm_response(QUERY)