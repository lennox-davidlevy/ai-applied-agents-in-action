import json
import os
from typing import Dict, List
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai import Credentials

from crewai import Agent, Task, Crew, Process, LLM
from langchain.tools import tool

from pydantic import BaseModel, Field

import tempfile
import chromadb

from routes.models import ModelRequest

from schemas import (
    ExamplesTemplate,
    PromptTemplateRequest,
)

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Global dictionaries to store models and prompt templates
models: Dict[str, ModelRequest] = {}
prompt_templates: Dict[str, PromptTemplateRequest] = {}
examples_templates: Dict[str, ExamplesTemplate] = {}

# Directory paths for models and prompt templates
MODELS_DIR_PATH = "data/models"
TEMPLATES_DIR_PATH = "data/prompt_templates"
EXAMPLES_DIR_PATH = "data/examples"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle the lifespan of the FastAPI application.
    Loads models and prompt templates from files at startup and
    clears them at shutdown.
    """
    try:
        load_models()
        load_prompt_templates()
        load_examples()
        yield
    finally:
        models.clear()
        prompt_templates.clear()
        examples_templates.clear()


def load_models():
    """
    Load models from JSON files.
    """
    for filename in os.listdir(MODELS_DIR_PATH):
        if filename.endswith(".json"):
            file_path = os.path.join(MODELS_DIR_PATH, filename)
            with open(file_path, "r") as f:
                model_data = json.load(f)
            template_model = filename[:-5]
            model_request = ModelRequest.from_json(model_data)
            models[template_model] = model_request


def load_prompt_templates():
    """
    Load prompt templates from text files.
    """
    for filename in os.listdir(TEMPLATES_DIR_PATH):
        if filename.endswith(".txt"):
            file_path = os.path.join(TEMPLATES_DIR_PATH, filename)
            with open(file_path, "r") as f:
                template = f.read()
            template_name = filename[:-4]
            prompt_templates[template_name] = PromptTemplateRequest(template=template)


def load_examples():
    """
    Load examples from text files.
    """
    for filename in os.listdir(EXAMPLES_DIR_PATH):
        if filename.endswith(".txt"):
            file_path = os.path.join(EXAMPLES_DIR_PATH, filename)
            with open(file_path, "r") as f:
                template = f.read()
            template_name = filename[:-4]
            examples_templates[template_name] = ExamplesTemplate(template=template)


# Initialize FastAPI app with custom lifespan
app = FastAPI(lifespan=lifespan)


# List of available models for watsonx
available_watsonx_models = {
    "models_available": [
        "codellama/codellama-34b-instruct-hf",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
        "google/flan-ul2",
        "ibm/granite-13b-instruct-v2",
        "ibm/granite-20b-code-instruct",
        "ibm/granite-20b-multilingual",
        "ibm/granite-3-2-8b-instruct-preview-rc",
        "ibm/granite-3-2b-instruct",
        "ibm/granite-3-8b-instruct",
        "ibm/granite-34b-code-instruct",
        "ibm/granite-3b-code-instruct",
        "ibm/granite-8b-code-instruct",
        "ibm/granite-guardian-3-2b",
        "ibm/granite-guardian-3-8b",
        "meta-llama/llama-2-13b-chat",
        "meta-llama/llama-3-1-70b-instruct",
        "meta-llama/llama-3-1-8b-instruct",
        "meta-llama/llama-3-2-11b-vision-instruct",
        "meta-llama/llama-3-2-1b-instruct",
        "meta-llama/llama-3-2-3b-instruct",
        "meta-llama/llama-3-2-90b-vision-instruct",
        "meta-llama/llama-3-3-70b-instruct",
        "meta-llama/llama-3-405b-instruct",
        "meta-llama/llama-guard-3-11b-vision",
        "mistralai/mistral-large",
        "mistralai/mixtral-8x7b-instruct-v01",
    ]
}


@app.get("/health")
async def health():
    """
    Health check endpoint to see if the API is up and running.
    """
    return {"message": "Fast API up!"}


chroma_client = chromadb.PersistentClient(path="./chroma_db")


@app.post("/process-documents-by-collection")
async def process_documents_by_collection(files: List[UploadFile] = File(...)):
    txt_files = []
    try:
        apikey = os.environ.get("IBM_APIKEY")
        project_id = os.environ.get("PROJECT_ID")
        url = os.environ.get("WATSON_URL")
        if not (apikey and project_id and url):
            raise ValueError("Missing one or more required environment variables.")

        credentials = Credentials(
            url=url,
            api_key=apikey,
        )
        embedding_model = Embeddings(
            model_id="intfloat/multilingual-e5-large",
            credentials=credentials,
            project_id=project_id,
        )

        for upload in files:
            if upload.filename.endswith(".txt"):
                content = await upload.read()
                txt_files.append((upload.filename, content))
            else:
                print(f"Skipping file {upload.filename} as it is not a .txt file.")

        with tempfile.TemporaryDirectory() as temp_dir:
            for filename, content in txt_files:
                file_path = os.path.join(temp_dir, filename)
                file_name = os.path.splitext(filename)[0]

                with open(file_path, "wb") as buffer:
                    buffer.write(content)

                loader = TextLoader(file_path)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100,
                    length_function=len,
                    separators=["\n\n", "\n", " "],
                    is_separator_regex=False,
                )
                documents = loader.load()
                splits = text_splitter.split_documents(documents)

                collection = chroma_client.get_or_create_collection(
                    name=file_name.lower(),
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:construction_ef": 400,
                        "hnsw:M": 128,
                    },
                )

                batch_size = 100
                for i in range(0, len(splits), batch_size):
                    batch = splits[i : i + batch_size]
                    texts = [doc.page_content for doc in batch]
                    metadatas = [doc.metadata for doc in batch]
                    embeddings = embedding_model.embed_documents(
                        texts=texts, concurrency_limit=5
                    )
                    collection.add(
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metadatas,
                        ids=[f"{file_name}_{i}_{j}" for j in range(len(batch))],
                    )

        return {
            "message": f"Successfully processed {len(txt_files)} files into their respective collections"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description=(
            "The user's input query that will be processed through a multi-agent pipeline "
            "to generate an intelligent answer."
        ),
    )


@app.post("/agentic-route")
async def agentic_route(query: QueryRequest):
    """
    Process a user query through a multi-agent pipeline to generate an intelligent answer.

    This endpoint orchestrates a three-step process:
      1. **Query Categorization**: An LLM-powered agent determines the query category from
         a fixed set ("technical", "billing", or "account").
      2. **Context Retrieval**: Based on the determined category, the system queries the
         corresponding ChromaDB collection using an embedding model to extract relevant documents.
      3. **Response Generation**: A dedicated agent generates a detailed answer using a structured
         prompt that incorporates the query and the retrieved document context.

    **Return Structure**:
      The response is a JSON object conforming to the `AIQueryAnswerResponse` model:
        {
            "response": {
                "category": <str>,  # one of "technical", "billing", or "account"
                "response": <str>   # the generated natural language answer
            }
        }

    **Raises**:
      - HTTPException: If an error occurs during processing.
    """
    try:

        crew_result = {
            "json_dict": {
                "response": "This WILL be generated by our multi agent RAG process",
                "category": "something cool",
            }
        }

        return {"response": crew_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
