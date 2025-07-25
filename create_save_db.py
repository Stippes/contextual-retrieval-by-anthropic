from src.contextual_retrieval import create_and_save_db
import os
from dotenv import load_dotenv
load_dotenv()

BASE_PATH = os.getenv("BASE_PATH", "")
data_dir = os.path.join(BASE_PATH, os.getenv("DATA_DIR", ""))
save_dir = os.path.join(BASE_PATH, os.getenv("SAVE_DIR", ""))
collection_name = os.getenv("COLLECTION_NAME")
db_name = "cook_book_db"

create_and_save_db(
    data_dir=data_dir, 
    save_dir=save_dir,
    collection_name=collection_name,
    db_name=db_name
    )