# AI Applied: Agents in Action Tutorial

[![IBM Cloud](https://img.shields.io/badge/IBM%20Cloud-Watsonx.ai-blue)](https://cloud.ibm.com/watsonx)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blueviolet)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-18%2B-green)](https://nodejs.org/)
[![CrewAI](https://img.shields.io/badge/Framework-CrewAI-orange)](https://www.crewai.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/UI-React-blue)](https://reactjs.org/)

---

## Introduction

**Welcome to AI Applied: Agents in Action!**

Hi everyone, I'm David Levy, a Solution Architect from IBM. In this tutorial, I'll show you how to seamlessly integrate multiple AI agents into your application using the CrewAI framework. We'll walk through a practical example that covers query categorization, context retrieval from a ChromaDB vector database, and natural language response generation, all orchestrated using a multi agent approach connected to IBM watsonx.ai.

This session is designed to give you a clear, step by step guide on setting up the project and building the agentic pipeline. Let's dive in and explore how you can leverage these tools to build smarter applications!

---

## Table of Contents

1.  [Prerequisites](#prerequisites)
2.  [Repository Setup](#repository-setup)
3.  [UI Setup](#ui-setup)
4.  [API Setup](#api-setup)
5.  [Application Initialization](#application-initialization)
6.  [Tutorial Steps](#tutorial-steps)
    *   [Step 1: Categorization Agent (`01_Step`)](#step-1-categorization-agent-01_step)
    *   [Step 2: Retriever Agent (`02_Step`)](#step-2-retriever-agent-02_step)
    *   [Step 3: Generation Agent (`03_Step`)](#step-3-generation-agent-03_step)
7.  [Key Technologies](#key-technologies)
8.  [Conclusion and Next Steps](#conclusion-and-next-steps)

---

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Git:** For cloning the repository.
*   **Node.js and npm:** (v18 or later recommended) For the UI setup.
*   **Python:** (v3.12.10 or later) For the API setup.
*   **pyenv (Optional but Recommended):** For managing Python virtual environments.
*   **Access to IBM watsonx.ai:** You will need:
    *   An IBM Cloud Account.
    *   A watsonx.ai Project ID.
    *   An IBM Cloud API Key.
    *   The URL endpoint for your watsonx.ai instance.

---

## Repository Setup

First, clone the tutorial repository from GitHub to your local machine.

```bash
git clone https://github.com/lennox-davidlevy/ai-applied-agents-in-action.git
cd ai-applied-agents-in-action
```
---

## UI Setup

The user interface is built with React, TypeScript, and uses the Carbon Design System. We won't modify the UI in this tutorial, but here's how to set it up:

1. Navigate to the UI Directory
```bash
cd ui
```
2. Install root dependencies
```bash
npm i
npm run setup
```
3. Set up environment variables: copy the example environment file to create your local configuration
```bash
cp client/.env.example client/.env
cp server/.env.example server/.env
```
*Note: You may need to configure variables insde this .env file later depending on your specific setup, but for this tutoral, the defaults are ok.* 

---


## API Setup
The backend API is built using Python and FastAPI.

1. Navigate to the API Directory
```bash
# From root
cd api
```
2. Create and activate a Python virtual environment: (I used pyenv virtualenv, but there are many ways to do this)
```bash
pyenv virtualenv 3.12.10 aiagentic 
pyenv activate aiagentic
```
3. Install Python dependencies

***Important: use the following command due to dependency issues that are in process of being resolved*** 

```sh
pip install --use-deprecated=legacy-resolver -r requirements.txt
```
4. Set up environment variables
```sh
cp .env.example .env
```
5. Configure API Credentials
```bash
# .env (in api directory)
IBM_APIKEY="your_ibm_cloud_api_key"
PROJECT_ID="your_watsonx_ai_project_id"
WATSON_URL="your_watsonx_ai_instance_url"
```

## Application Initialization
Before running the main application, we need to populate the ChromaDB vector database with sample documents. The tutorial uses different branches to represent stages of development.

1. Checkout the first step's branch:
```
git checkout 01_Step
```
2. Populate the ChromaDB Database: Run the processing script from the **root** directory.
```sh
cd api/
# Run the script (ensure your 'aiagentic' virtual environment is active)
# Make sure you are in the /api directory
python scripts/process_documents.py
```
*Explanation*: The `process_documents.py` script reads files from the `api/docs` directory, chunks them, generates embeddings using a model form `watsonx.ai` and stores them in ChromaDB as their own collections based on the file names (e.g. `billing.txt` goes to `billing` collection).

3. **Start the services**
- **API Server**: Navigate to the `api` directory and star the FastAPI server (ensure your virtual environment is active)
    ```sh
    cd api
    uvicorn server:app --reload
    ```
- **UI**  
    ```sh
    cd ../ui
    npm run dev
    ```
You should now be able to access the initial UI in your browser at `http://localhost:3000` and if you want to see the API's swagger docs `http://localhost:8000/docs`

---

## Tutorial Steps

Lets build the mutli agent pipeline step by step!

### Step 1: Categorization Agent (01_Step)

**Goal:** Create the first agent responsible for categorizing the user's query into "technical", "billing", or "account".

**Branch:** Ensure you are on the `01_Step` branch.
```sh
git checkout 01_step
```

**Code Overview (`api/server.py` in `01_step`)** 

This step introduces the basic FastAPI route (`/agentic-route`) and the first CrewAI agent

```python
class CategoryResponse(BaseModel):
    category: str = Field(
        ...,
        description=(
            "The determined category for the query. It must be one of the following values: "
            "'technical', 'billing', or 'account', which help route the query to the correct domain."
        )
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
        apikey = os.environ.get("IBM_APIKEY")
        project_id = os.environ.get("PROJECT_ID")
        url = os.environ.get("WATSON_URL")

        categorization_llm = LLM(
            model="watsonx/ibm/granite-3-8b-instruct",
            base_url=url,
            project_id=project_id,
            max_tokens=50,
            temperature=0.7,
            api_key=apikey,
        )

        categorization_agent = Agent(
            role="Collection Selector",
            goal="Analyze user queries and determine the most relevant ChromaDB collection.",
            backstory="Expert in query classification. Routes questions to the correct domain.",
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            llm=categorization_llm,
        )

        categorization_task = Task(
            description=f"""
            Based on the user query below, determine the best category.
            You must return ONLY one of these exact values: "technical", "billing", or "account".
            
            Category Definitions:
            - technical: Issues with system access, errors, API integration
            - billing: Questions about pricing, payments, invoices
            - account: User management, roles, organization settings
            
            IMPORTANT: Respond with EXACTLY ONE WORD from the list above.
            
            User Query: "{query.query}"
            """,
            expected_output="A JSON object with a 'category' field that must be either 'technical', 'billing', or 'account'",
            agent=categorization_agent,
            output_json=CategoryResponse,
        )

        crew = Crew(
            agents=[categorization_agent],
            tasks=[categorization_task],
            process=Process.sequential,
            verbose=True
        )

        category_result = crew.kickoff()

        print(category_result)
        crew_result = {
            "json_dict": {
                "response": "This WILL be generated by our multi agent RAG process",
                "category": category_result['category'],
            }
        }

        return {"response": crew_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Explanation:** 

- **LLM Configuration**: Sets up the connection to watsonx.ai for the categorization model.
- **Agent Definition**: Defines the categorization_agent with its specific role, goal, and backstory. It's instructed to use the configured LLM.
- **Task Definition**: Defines categorization_task, providing the agent with instructions (the prompt) and the expected output format (CategoryResponse Pydantic model).
- **Crew**: Initializes the Crew with the single agent and task, set to run sequentially.
- **Kickoff**: crew.kickoff() executes the workflow. Since output_json is specified in the task, CrewAI attempts to parse the LLM's output into the CategoryResponse model.
- **Response**: The endpoint currently returns the determined category and a placeholder response string.

### Step 2: Retriever Agent (02_Step)

**Goal**: Add a second agent that uses the category from Step 1 to retrieve relevant documents from the corresponding ChromaDB collection.

**Branch**: Check out the `02_Step` branch.
```sh
git checkout 02_step
```

**Code Overview (`api/server.py` in `02_Step`)**:

This step builds upon the previous one by adding a ChromaDB querying tool and a new agent/task for retrieval.

```python
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
        apikey = os.environ.get("IBM_APIKEY")
        project_id = os.environ.get("PROJECT_ID")
        url = os.environ.get("WATSON_URL")

        retriever_llm = LLM(
            model="watsonx/ibm/granite-3-8b-instruct",
            base_url=url,
            project_id=project_id,
            max_tokens=1000,
            temperature=0.7,
            api_key=apikey,
        )

        @tool("query_collection_tool")
        def query_collection_tool(category: str, query: str) -> dict:
            """Tool to query ChromaDB based on category and return relevant documents"""

            credentials = Credentials(
                url=url,
                api_key=apikey,
            )

            embedding_model = Embeddings(
                model_id="intfloat/multilingual-e5-large",
                credentials=credentials,
                project_id=project_id,
                verify=True,
            )

            query_embedding = embedding_model.embed_query(query)
            collection = chroma_client.get_collection(category.lower())
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["documents", "metadatas", "distances"],
            )

            relevant_documents = []
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                similarity = 1 - distance
                if similarity > 0.8:  # should adjust? maybe?
                    metadata["collection"] = category.lower()
                    metadata["relevance_score"] = similarity
                    relevant_documents.append({"content": doc, "metadata": metadata})

            relevant_documents.sort(
                key=lambda x: x["metadata"]["relevance_score"], reverse=True
            )
            # lets see if 5 is enough
            relevant_documents = relevant_documents[:5]

            context = ""
            for doc in relevant_documents:
                score = doc["metadata"]["relevance_score"]
                content = doc["content"]
                context += f"\nRelevance Score: {score:.2f}\n{content}\n---\n"

            return {"category": category, "query": query, "context": context}

        retriever_agent = Agent(
            role="Category Retriever",
            goal="Query ChromaDB with the appropriate category and return results",
            backstory=(
                "You are responsible for taking the classified category and original query, "
                "querying the appropriate ChromaDB collection, and returning the results."
            ),
            verbose=True,
            allow_delegation=False,
            llm=retriever_llm,
            max_iter=3,
            tools=[query_collection_tool],
        )

        retriever_task = Task(
            description=(
                "Take the category from the categorization task and the original query, "
                "use them to query the appropriate ChromaDB collection, and return the results. "
                f"Current query: {query.query}"
            ),
            expected_output=(
                "An object containing the category, query, and context from ChromaDB"
            ),
            agent=retriever_agent,
            context=[categorization_task],
        )


        crew = Crew(
            agents=[categorization_agent, retriever_agent],
            tasks=[categorization_task, retriever_task],
            process=Process.sequential,
            verbose=True
        )

        category_result = crew.kickoff()

        print(category_result)
        crew_result = {
            "json_dict": {
                "response": "This WILL be generated by our multi agent RAG process",
                "category": "Bye for now" 
            }
        }

        return {"response": crew_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Explanation**:

- **ChromaDB Client**: Initializes connection to the vector database.
- **@tool Decorator**: The query_collection_tool function is decorated with @tool from crewai_tools, making it available for agents to use.
- **Tool Logic**: The tool function takes the category and query, generates an embedding for the query using WatsonxEmbeddings, queries the appropriate ChromaDB collection, filters results based on similarity, formats the documents into a context string, and returns a dictionary.
- **Retriever Agent**: A new retriever_agent is defined. Crucially, it's given tools=[query_collection_tool]. Its goal is specifically to use this tool.
- **Retriever Task**: The retriever_task instructs the agent to use the tool. It specifies context=[categorization_task], meaning it will receive the output from the first task (the CategoryResponse object or dictionary). The agent needs to extract the category string from this context to pass to the tool.
- **Updated Crew**: The Crew now includes both agents and tasks in sequential order.
- **Result**: crew.kickoff() now returns the output of the retriever_task, which should be the dictionary containing the context fetched by the query_collection_tool.


### Step 3: Generation Agent (`03_step`)

**Goal**: Add the final agent responsible for synthesizing a natural language answer using the original query and the context retrieved in Step 2.

**Branch**: Check out the `03_step` branch.
```sh
git checkout 03_step
```

**Code Overview (`api/server.py` in `03_step`)**

This step adds the generation agent, its task, and updates the final response structure.

```python
class FinalResponse(BaseModel):
    category: str = Field(
        ...,
        description=(
            "The category assigned to the query by the categorization agent. "
            "This value is one of 'technical', 'billing', or 'account'."
        ),
    )
    response: str = Field(
        ...,
        description=(
            "The final generated natural language answer. This response is produced after "
            "retrieving relevant documents and processing them via a dedicated generation agent."
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
        apikey = os.environ.get("IBM_APIKEY")
        project_id = os.environ.get("PROJECT_ID")
        url = os.environ.get("WATSON_URL")

        generation_llm = LLM(
            model="watsonx/ibm/granite-3-8b-instruct",
            # model="watsonx/mistralai/mistral-large",
            base_url=url,
            project_id=project_id,
            max_tokens=3000,
            temperature=0.7,
            api_key=apikey,
        )

        @tool("generate_response_tool")
        def generate_response_tool(context: str, query: str) -> dict:
            """Tool to generate a response using the specific prompt template"""
            prompt = f"""<|start_of_role|>system<|end_of_role|>
                        - You are a helpful, respectful, and honest assistant that can summarize long documents.
                        - Always respond as helpfully as possible, while being safe.
                        - Your responses should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
                        - Please ensure that your responses are socially unbiased and positive in nature.
                        - If a document does not make any sense, or is not factually coherent, explain why instead of responding something not correct.
                        - If you don't know the response to a query, please do not share false information.
                        <|start_of_role|>user<|end_of_role|>
                        You are an assistant for question-answering tasks. Generate a conversational response for the given question based on the given set of document context. Think step by step to answer in a crisp manner. Answer should not be more than 300 words. If you do not find any relevant answer in the given documents, please state you do not have an answer. Do not try to generate any information.
                        Context : {context}
                        Question : {query}
                        Answer: <|start_of_role|>assistant<|end_of_role|>"""

            return {"response": prompt}

        generation_agent = Agent(
            role="Response Generator",
            goal="Generate a comprehensive response using the specific prompt template",
            backstory=(
                "You are an expert at using structured prompts to generate precise, "
                "informative responses based on provided context."
            ),
            verbose=True,
            allow_delegation=False,
            llm=generation_llm,
            max_iter=3,
            tools=[generate_response_tool],
        )

        generation_task = Task(
            description=(
                "Using the context and query from the retriever task, generate a response using "
                "the specific prompt template via generate_response_tool. Return the complete response."
            ),
            expected_output="A JSON object with 'category' and 'response' fields, where 'response' contains the natural language answer",
            agent=generation_agent,
            context=[retriever_task],
            output_json=FinalResponse,
        )

        crew = Crew(
            agents=[categorization_agent,
                    retriever_agent, generation_agent],
            tasks=[categorization_task, retriever_task, generation_task],
            process=Process.sequential,
            verbose=True,
        )

        crew_result = crew.kickoff()

        return {"response": crew_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Explanation**:

- **FinalResponse Model**: A Pydantic model `FinalResponse` is defined to structure the final output JSON, containing both the category and the generated `response`.
- **Generation LLM**: A potentially distinct LLM instance (`generation_llm`) is configured, possibly with more max_tokens to accommodate longer answers.
- **Generation Agent**: The generation_agent is defined with the goal of synthesizing the final answer using its assigned `generation_llm`.
- **Generation Task**:
    - The `generation_task` takes the output (context dictionary) from the `retriever_task`.
    - Its description now includes the full prompt template. The `{context}` and `{query}` placeholders will be filled by CrewAI using the data received from the `retriever_task`.
    - Crucially, `output_json=FinalResponse` is set. This tells CrewAI to:
        - Append instructions to the prompt asking the LLM to generate a JSON object matching the `FinalResponse` schema.
        - Parse the LLM's response string into a `FinalResponse` Pydantic object.
- **Final Crew**: The Crew includes all three agents and tasks in sequence.
- **Final Result**: `crew.kickoff()` executes the entire pipeline. The return value should be the `FinalResponse` object generated by the last task. This object is then returned by the API endpoint.


## Key Technologies

- **CrewAI**: Framework for orchestrating autonomous AI agents.
- **IBM watsonx.ai**: Platform providing foundation models (LLMs) and embedding models.
- **FastAPI**: Modern Python web framework for building the API.
- **React & Carbon Design**: For the frontend user interface.
- **ChromaDB**: Open-source vector database for storing and retrieving document embeddings.
- **Python**: Backend language.
- **Node.js/TypeScript**: Frontend language/tooling.

---



## Conclusion and Next Steps

Awesome! We've built a sophisticated multi-agent pipeline using CrewAI and watsonx.ai.

**Recap**: We created a backend for an agentic RAG (Retrieval-Augmented Generation) chatbot that:

1. Identifies the query's category (billing, technical, account).
2. Targets the correct ChromaDB collection based on the category.
3. Retrieves relevant context documents.
4. Interpolates the query and context into a custom prompt.
5. Generates a natural language response using a dedicated agent.

**Explore Further:** 

- **Experiment**: Try different LLMs available on watsonx.ai for each agent.
- **Enhance**: Add error handling for cases where no relevant documents are found.
- **Extend**: Implement agent memory for conversational context.
- **New Features:**
    - Add an agent that performs a web search if the query falls outside the database's scope.
    - Create a dedicated "Formatting Agent" to ensure the final response adheres to specific style guidelines.
- **Customize**: Modify the UI or refactor the backend code.

Dive into the code, experiment, and build something cool! Thanks for following along!

---

