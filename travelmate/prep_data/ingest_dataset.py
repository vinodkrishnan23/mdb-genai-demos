# Prior to running the script, kindly update .env file which lies right besides this script for all env variables
# In Env file values need to be updated for OPENAI_API_KEY, MONGODB_ATLAS_CLUSTER_URI, MDB_NAME, COLLECTION_NAME, ATLAS_VECTOR_SEARCH_INDEX_NAME
# This script will pull data from MongoDB/airbnb_embeddings dataset hosted in hugging face into your mongodb Atlas Cluster.
# Dataset already has vector embeddings for description generated using Open AI's text-embedding-3-small model.
# The script also creates atlas vector search index on the text_embeddings field in the same collection

import os
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, ValidationError
from datetime import datetime

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MONGODB_ATLAS_CLUSTER_URI = os.getenv('MONGODB_ATLAS_CLUSTER_URI')


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

DB_NAME = os.getenv('MDB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME')

collection = client[DB_NAME][COLLECTION_NAME]

try:
    client[DB_NAME].drop_collection(COLLECTION_NAME)
except:
    print("Collection does not exist")


dataset = load_dataset("MongoDB/airbnb_embeddings", streaming=True, split="train")

dataset_df = pd.DataFrame(dataset)

print("Columns: ", dataset_df.columns)

class Host(BaseModel):
    host_id: str
    host_url: str
    host_name: str
    host_location: str
    host_about: str
    host_response_time: Optional[str] = None
    host_thumbnail_url: str
    host_picture_url: str
    host_response_rate: Optional[int] = None
    host_is_superhost: bool
    host_has_profile_pic: bool
    host_identity_verified: bool

class Location(BaseModel):
    type: str
    coordinates: List[float]
    is_location_exact: bool

class Address(BaseModel):
    street: str
    government_area: str
    market: str
    country: str
    country_code: str
    location: Location

class Review(BaseModel):
    _id: str
    date: Optional[datetime] = None
    listing_id: str
    reviewer_id: str
    reviewer_name: Optional[str] = None
    comments: Optional[str] = None

class Listing(BaseModel):
    _id: int
    listing_url: str
    name: str
    summary: str
    space: str
    description: str
    neighborhood_overview: Optional[str] = None
    notes: Optional[str] = None
    transit: Optional[str] = None
    access: str
    interaction: Optional[str] = None
    house_rules: str
    property_type: str
    room_type: str
    bed_type: str
    minimum_nights: int
    maximum_nights: int
    cancellation_policy: str
    last_scraped: Optional[datetime] = None
    calendar_last_scraped: Optional[datetime] = None
    first_review: Optional[datetime] = None
    last_review: Optional[datetime] = None
    accommodates: int
    bedrooms: Optional[float] = 0
    beds: Optional[float] = 0
    number_of_reviews: int
    bathrooms: Optional[float] = 0
    amenities: List[str]
    price: int
    security_deposit: Optional[float] = None
    cleaning_fee: Optional[float] = None
    extra_people: int
    guests_included: int
    images: dict
    host: Host
    address: Address
    availability: dict
    review_scores: dict
    reviews: List[Review]
    text_embeddings: List[float]

records = dataset_df.to_dict(orient='records')

for record in records:
    for key, value in record.items():
        # Check if the value is list-like; if so, process each element.
        if isinstance(value, list):
            processed_list = [None if pd.isnull(v) else v for v in value]
            record[key] = processed_list
        # For scalar values, continue as before.
        else:
            if pd.isnull(value):
                record[key] = None
                
try:
  # Convert each dictionary to a Movie instance
  listings = [Listing(**record).dict() for record in records]
  # Get an overview of a single datapoint
  print(listings[0].keys())
except ValidationError as e:
  print(e)

def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB."""

    # gateway to interacting with a MongoDB database cluster
    client = MongoClient(mongo_uri)
    print("Connection to MongoDB successful")
    return client

if not MONGODB_ATLAS_CLUSTER_URI:
    print("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(MONGODB_ATLAS_CLUSTER_URI)

# Pymongo client of database and collection
db = mongo_client.get_database(DB_NAME)
collection = db.get_collection(COLLECTION_NAME)

collection.delete_many({})

collection.insert_many(listings)
print("Data ingestion into MongoDB completed")

## Create a search index

search_index_model = SearchIndexModel(
  definition={
    "fields": [
      {
        "type": "vector",
        "numDimensions": 1536,
        "path": "text_embeddings",
        "similarity": "cosine"
      },
      {
        "type": "filter",
        "path": "bathrooms"
      },
      {
        "type": "filter",
        "path": "bedrooms"
      },
      {
        "type": "filter",
        "path": "security_deposit"
      }
    ]
  },
  name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
  type="vectorSearch",
)

try:
    result = collection.create_search_index(model=search_index_model)
except Exception as e:
    print("Vector search Index Exists")