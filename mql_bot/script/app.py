import streamlit as st
from pymongo import MongoClient
import datetime
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from pymongo.operations import SearchIndexModel
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

st.set_page_config(layout="wide")
# MongoDB connection setup
mclient = MongoClient(st.secrets["MONGODB_URI"])
mcollection = mclient[st.secrets["DB_NAME"]][st.secrets["METDATA_COLL_NAME"]]
ATLAS_VECTOR_SEARCH_INDEX_NAME = "default"
llm = ChatOpenAI(model="gpt-4", temperature=0,api_key=st.secrets["OPENAI_API_KEY"])

# Embedding Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Vectorstore
vector_store = MongoDBAtlasVectorSearch(
    collection=mcollection,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
    embedding_key="description_embedding",
    text_key="description"
)

relevant_docs=[]

# Page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Add Dataset", "View Datasets","Natural Language Query"])

# Page 1: Add Dataset
if page == "Add Dataset":
    st.title("Add Dataset Information")

    # Input fields for the dataset information
    dataset_name = st.text_input("Dataset Name")
    dataset_description = st.text_area("Dataset Description")
    dataset_usage = st.text_area("How the dataset can be used")

    if st.button("Submit"):
        if dataset_name and dataset_description and dataset_usage:
            # Create a document to insert into MongoDB
            dataset_document = {
                "name": dataset_name,
                "description": dataset_description,
                "usage": dataset_usage,
                "description_embedding":embeddings.embed_query(dataset_description),
                "created_at": datetime.datetime.utcnow()
            }
            
            # Insert the document into the MongoDB collection
            mcollection.insert_one(dataset_document)
            
            st.success("Dataset information has been stored successfully!")
        else:
            st.error("Please fill in all fields.")

# Page 2: Query Datasets
elif page == "View Datasets":
    st.title("View Datasets")
    # Simple query options
    query_type = st.selectbox("Select Query Type", ["View All Datasets", "Search by Name"])

    if query_type == "View All Datasets":
        st.header("All Stored Datasets")
        datasets = mcollection.find()
        for dataset in datasets:
            st.subheader(dataset["name"])
            st.write(f"**Description:** {dataset['description']}")
            st.write(f"**Usage:** {dataset['usage']}")
elif page == "Natural Language Query":
    chat_container, metadata_container = st.columns([1,1])
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    with chat_container:
        st.subheader("Natural Language Query")
        if user_prompt := st.chat_input("Ask me a question"):
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)
            with st.chat_message("assistant"):
                relevant_docs=vector_store.similarity_search(user_prompt,include_scores=True,post_filter_pipeline=[{"$match": {"score": {"$gt": 0.5}}},{"$project": {"_id": 0, "description": 1,"score": 1}}])
                refined_documents=[]
                schema_documents=[]
                #print(relevant_docs)
                for doc in relevant_docs:
                    document = Document(page_content=doc.page_content, metadata=doc.metadata)
                    refined_documents.append(document)
                
                promptT = PromptTemplate(
                input_variables = ["context","query"],
                template = """ 
                Given the following schema context:

                {context}
                
                Please generate a MongoDB aggregation pipeline that satisfies the user query described below:
                
                User Query: {query}
                
                """
                )
                document_chain = create_stuff_documents_chain(llm, promptT)
                response = st.write_stream(document_chain.stream({"query":user_prompt, "context":refined_documents}))
                st.session_state.messages.append({"role": "assistant", "content": response})
            
    with metadata_container:
        if relevant_docs:
            st.subheader("Chunk Information")
            data = []
            for doc in relevant_docs:
                data.append({
                    "vs_score":doc.metadata['score'],
                    "document_or_chunk":doc.page_content
                })
            st.write(data)