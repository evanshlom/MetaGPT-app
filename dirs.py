import os

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/data"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/db"