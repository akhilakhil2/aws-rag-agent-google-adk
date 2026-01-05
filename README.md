Agentic RAG Assistant: AWS Prescriptive Guidance

🚀 Overview

This project is an Agentic Retrieval-Augmented Generation (RAG) Assistant designed to answer complex architectural questions regarding AWS RAG options. It is grounded strictly in the AWS Prescriptive Guidance: "Retrieval Augmented Generation options and architectures on AWS".

Built with the Google Agent Development Kit (ADK), the system utilizes a multi-agent orchestration pattern to decompose queries, retrieve high-relevance semantic context, and synthesize grounded answers with explicit citations.

🏗️ Architecture & Agentic Design

The assistant follows a Sequential Workflow where specialized agents collaborate to fulfill user requests:

1. Planner Agent

Role: Strategic Analysis and Decomposition.

Function: Categorizes queries (e.g., 'comparison', 'recommendation') and breaks them into a structured research plan consisting of multiple sub-queries.

State Management: Outputs a PlannerPlan JSON schema used by the downstream agents.

2. Retrieval Agent (Tool-Integrated)

Role: Data Extraction.

Function: Queries a persistent ChromaDB vector store using all-MiniLM-L6-v2 embeddings. It implements deduplication to handle overlapping search results from the Planner's sub-queries.

3. Synthesis Agent

Role: Knowledge Synthesis and Citations.

Function: Combines retrieved document segments into a professional response.

Grounding: Enforces strict adherence to the source document. If information is missing, it triggers a mandatory refusal: "This information is not available in the provided AWS RAG guide."

📂 Project Structure

.
├── agents/
│   ├── planner.py        # Planner Agent definition
│   └── synthesizer.py    # Synthesizer Agent with tool access
├── tools/
│   └── retriever.py      # ChromaDB search & deduplication logic
├── schemas/
│   └── planner_schema.py # Pydantic schemas for agent state
├── data_ingestion/
│   └── ingestion.py      # PDF processing and vector indexing
├── main.py               # Application entry point & Orchestrator
├── requirements.txt      # Dependency manifest
├── .env                  # Environment configuration (API Keys)
└── .gitignore            # Git exclusion rules


🛠️ Tech Stack

Orchestration: Google Agent Development Kit (ADK)

LLM: Llama-3.3-70b-Versatile (via Groq/LiteLLM)

Vector DB: ChromaDB (Persistent)

Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)

Data Integrity: Pydantic v2

🚀 Setup & Installation

Clone the Repository

Environment Configuration
Create a .env file in the root directory:

GROQ_API_KEY=your_actual_api_key_here
ANONYMIZED_TELEMETRY=False


Install Dependencies

pip install -r requirements.txt


Prepare Document
Place the AWS Prescriptive Guidance PDF in the root folder as rag_pdf.pdf. The system will automatically build the vector store on the first run.

📖 Usage

Run the assistant via:

python main.py


Example Evaluation Queries

"Compare fully managed RAG options on AWS with custom architectures. When would you choose each?"

"List the retriever options described in the guide. What are their key characteristics for RAG?"

"How does the guide compare RAG vs. fine-tuning, and what are the trade-offs?"

🔍 Observability & Traceability
