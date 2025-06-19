import streamlit as st
import os
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import boto3
from io import BytesIO
import base64
from voyageai import Client
import fitz  # PyMuPDF
from PIL import Image
import time
import numpy as np
from typing import List
import json


MODEL_NAME = "voyage-multimodal-3"
VOYAGEAI_API_KEY = st.secrets["VOYAGEAI_KEY"]
GENERATIVE_VLM = "anthropic.claude-3-5-sonnet-20240620-v1:0"
vo = Client(api_key=VOYAGEAI_API_KEY)
mclient = MongoClient(st.secrets["MONGODB_URI"])
mcollection = mclient[st.secrets["DB_NAME"]][st.secrets["COLLECTION_NAME"]]
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_search"
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

def pdf_to_screenshots(uploaded_file, zoom: float = 1.0) -> list[Image]:
    """
    Convert an UploadedFile object (PDF) to a list of PIL Images.
    
    Args:
        uploaded_file (UploadedFile): Streamlit UploadedFile object containing the PDF
        zoom (float): Zoom factor for the output images (default: 1.0)
    
    Returns:
        list[Image]: List of PIL Images, one for each page
    """
    # Ensure the file has a .pdf extension
    if not uploaded_file.name.endswith(".pdf"):
        raise ValueError("File must have .pdf extension")

    # Read the file contents into memory
    file_contents = uploaded_file.getvalue()

    # Open the PDF from the bytes stream
    pdf = fitz.open(stream=file_contents, filetype="pdf")
    images = []

    # Loop through each page, render as pixmap, and convert to PIL Image
    mat = fitz.Matrix(zoom, zoom)
    for n in range(pdf.page_count):
        pix = pdf[n].get_pixmap(matrix=mat)

        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    # Close the document
    pdf.close()

    return images

def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str

def batch_embed_documents(docs: List[str], vo, model_name: str, batch_size: int = 5, delay: float = 1.0):
    """
    Embed documents in batches with delay to avoid rate limits.
    
    Args:
        docs: List of documents to embed
        vo: Vector object for embedding
        model_name: Name of the model to use
        batch_size: Number of documents to process in each batch
        delay: Delay in seconds between batches
    
    Returns:
        numpy.ndarray: Array of document embeddings
    """
    all_embeddings = []
    
    # Process documents in batches
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        
        try:
            # Create batch input format
            batch_input = [[page] for page in batch]
            
            # Get embeddings for current batch
            batch_embeddings = vo.multimodal_embed(
                inputs=batch_input,
                model=model_name,
                input_type="document"
            )
            print(f"Number of vectors generated: {len(batch_embeddings.embeddings)}")
            print(f"Number of text tokens ingested: {batch_embeddings.text_tokens}")
            print(f"Number of image pixels processed: {batch_embeddings.image_pixels}")
            print(f"Total number of tokens (texts + images): {batch_embeddings.total_tokens}")
            print(f"Total number of dimensions (texts + images): {len(batch_embeddings.embeddings[0])}")
            
            all_embeddings.extend(batch_embeddings.embeddings)
            
            # Add delay between batches if not the last batch
            if i + batch_size < len(docs):
                time.sleep(delay)
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            # Increase delay and retry on error
            time.sleep(delay * 2)
            try:
                batch_embeddings = vo.multimodal_embed(
                    inputs=batch_input,
                    model=model_name,
                    input_type="document"
                )
                print(f"Number of vectors generated: {len(batch_embeddings.embeddings)}")
                print(f"Number of text tokens ingested: {batch_embeddings.text_tokens}")
                print(f"Number of image pixels processed: {batch_embeddings.image_pixels}")
                print(f"Total number of tokens (texts + images): {batch_embeddings.total_tokens}")
                print(f"Total number of dimensions (texts + images): {len(batch_embeddings.embeddings[0])}")
                all_embeddings.extend(batch_embeddings.embeddings)
            except Exception as e:
                print(f"Retry failed for batch {i//batch_size + 1}: {str(e)}")
                raise
    
    return np.array(all_embeddings)

def createVectorSearchIndex():
    search_index_model = SearchIndexModel(
        definition={
            "fields":
                [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 1024,
                        "similarity": "cosine"
                        },
                    {
                        "type": "filter",
                        "path": "page_num"
                        },
                    {
                        "type": "filter",
                        "path": "file_name"
                        }
                    ]
                },
        name="vector_search",
        type="vectorSearch"
        )
    try:
        result = mcollection.create_search_index(model=search_index_model)
        print(result)
    except Exception as e:
        print(f"Error creating search index: {str(e)}")


def orchestrateRAG(uploaded_file):
    file_name = uploaded_file.name
    print(file_name)
    if mcollection.find_one({"file_name":file_name}):
            print("Chunks already loaded")
            return file_name
    else:
        docs = pdf_to_screenshots(uploaded_file)
        document_vectors = batch_embed_documents(
            docs=docs,
            vo=vo,
            model_name=MODEL_NAME,
            batch_size=3,  # Adjust based on your rate limits
            delay=0      # Adjust based on your rate limits
            )
        documents = []
        for n, vec in enumerate(document_vectors):
            doc = {"file_name":file_name,"page_num":n+1, "embedding":vec.tolist(), "base64":im_2_b64(docs[n])}
            documents.append(doc)
        mcollection.insert_many(documents)
        createVectorSearchIndex()
        return file_name
    
def queryVectorStore(file_name, query, mcollection, index_name, path='embedding',num_candidates=10, limit=3,):
    query_vector = np.array(
        vo.multimodal_embed(
            inputs=[[query]],
            model=MODEL_NAME,
            input_type="query"
            ).embeddings[0]
        )
    qvector=query_vector.tolist()
    results = {}
    pipe = [
        {
            '$vectorSearch': 
                {
                    "filter":{ "file_name": file_name},
                    "index":  index_name,
                    "path": path,
                    "numCandidates": num_candidates,
                    "limit": limit,
                    "queryVector": qvector
                    }
                },
        {
            "$project": {
                "_id": 1,
                "page_num": 1,
                "file_name":1,
                "base64":1,
                "score": { "$meta": "vectorSearchScore" }
            }
        }
      ]
    cursor = mcollection.aggregate(pipe)
    results = list(cursor)
    return results

def configure_aws():
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name='ap-south-1',
        aws_access_key_id=st.secrets["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws_secret_access_key"],
        aws_session_token=st.secrets["aws_session_token"])
    return bedrock_runtime

def getAnswersAWSVLM(colpali_results, query):
    bedrock_runtime=configure_aws()
    relevant_context=[]
    for n, data in enumerate(colpali_results):
        relevant_context.append({"type": "text", "text": "Image "+str(n+1)+" :"})
        relevant_context.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": 'image/jpeg',
                "data": data['base64'].decode("utf-8"),
            }
        })
    relevant_context.append({"type": "text", "text": query})
    request_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": relevant_context
                }
            ]
        }
    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        body=json.dumps(request_payload)
        )
    response_body = json.loads(response['body'].read())
    print(response_body['content'][0]['text'])
    response_text = response_body['content'][0]['text']
    return response_text

def showPages(results):
    
    for result in results:
        # Decode the base64 string into bytes
        image_bytes = base64.b64decode(result['base64'])
        # Convert bytes to a PIL Image
        image = Image.open(BytesIO(image_bytes))
        st.image(image, use_container_width=True)
        st.write(f"Page Number: {result['page_num']}")
    

st.set_page_config(layout="wide")
st.title("Upload PDF and start research!!")
relevant_docs=""
#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

chat_container, metadata_container = st.columns([1,1])


with chat_container:
    uploaded_file = st.file_uploader("Choose a file")
    text=""
    documents=[]
    if uploaded_file is not None:
        file_name=orchestrateRAG(uploaded_file)
        st.write(file_name + " is loaded")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            final_prompt = prompt + " as described in pdf - " + file_name
            relevant_docs=queryVectorStore(file_name, final_prompt, mcollection, ATLAS_VECTOR_SEARCH_INDEX_NAME)
            response_text=getAnswersAWSVLM(relevant_docs, final_prompt)
            response = st.write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response})

with metadata_container:
    if relevant_docs:
        st.subheader("Chunk Information")
        showPages(relevant_docs)