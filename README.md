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
6.  [Key Technologies](#key-technologies)
7.  [Conclusion and Next Steps](#conclusion-and-next-steps)

---

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Git:** For cloning the repository.
*   **Node.js and npm:** (v18 or later recommended) For the UI setup.
*   **Python:** (v3.12 or later recommended) For the API setup.
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
pyenv virtualenv 3.12 aiagentic 
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
