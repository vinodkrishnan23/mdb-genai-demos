import os
import json
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.operations import SearchIndexModel
from dotenv import load_dotenv
load_dotenv()

MONGODB_ATLAS_CLUSTER_URI = os.getenv('MONGODB_ATLAS_CLUSTER_URI')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
#client = MongoClient(MONGODB_ATLAS_CLUSTER_URI, server_api=ServerApi('1'))
collection_name = os.getenv('COLLECTION_NAME')
db_name = os.getenv('MDB_NAME')
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME')
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
mcollection=client[db_name][collection_name]

def load_json_files_to_mongodb(data_directory, mongodb_uri=MONGODB_ATLAS_CLUSTER_URI):
    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    
    # Iterate over all files in the data directory
    for filename in os.listdir(data_directory):
        if filename.endswith(".json"):
            # Extract database and collection names from the filename
            db_name, collection_name, _ = filename.split('.')
            
            # Select the database and collection
            db = client[db_name]
            collection = db[collection_name]
            
            file_path = os.path.join(data_directory, filename)
            with open(file_path, "r") as file:
                # Load JSON data
                data = json.load(file)
                final_data = []
                for document in data:
                        document.pop("_id", None)
                        final_data.append(document)

                
                # Insert JSON data into the appropriate MongoDB collection
                if isinstance(final_data, list):
                    collection.insert_many(final_data)  # If the JSON is an array of documents
                else:
                    collection.insert_one(final_data)  # If the JSON is a single document

            print(f"Loaded {filename} into {db_name}.{collection_name} collection in MongoDB.")

    # Close the MongoDB connection
    client.close()
def createVectorSearchIndex():
    search_index_model = SearchIndexModel(
    definition={
        "fields": [
        {
            "type": "vector",
            "numDimensions": 1536,
            "path": "description_embedding",
            "similarity": "cosine"
        }
        ]
    },
    name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    type="vectorSearch",
    )
    try:
        result = mcollection.create_search_index(model=search_index_model)
    except Exception as e:
        print("Vector search Index Exists")

if __name__ == "__main__":
    # Construct the path to the data directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_directory, "../data")

    # Load JSON files into MongoDB
    load_json_files_to_mongodb(data_directory)
    createVectorSearchIndex()