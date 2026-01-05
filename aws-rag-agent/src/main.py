import os
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional
from google.adk.sessions import InMemorySessionService
from google.adk.agents import SequentialAgent
from google.adk.runners import Runner
from google.genai import types
from data_ingestion.data_ingestion import create_vector_store

# Importing refactored agents
from agents.planner import initialize_planner_agent
from agents.synthesizer import initialize_synthesizer_agent

# 1. GLOBAL CONFIGURATION & LOGGING
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MainRunner")

# 2. ORCHESTRATION SETUP
APP_NAME = "aws_rag_agent_team"
SESSION_ID = "session_001"
USER_ID = "user_001"

def setup_orchestrator() -> SequentialAgent:
    """Initializes and chains the agents into a sequential workflow."""
    planner = initialize_planner_agent()
    synthesizer = initialize_synthesizer_agent()
    
    return SequentialAgent(
        name="AWSRagRootAgent",
        sub_agents=[planner, synthesizer],
        description="Sequential pipeline for AWS RAG: Planning followed by Synthesized Answering."
    )

# 3. ASYNC RUNNER LOGIC
async def run_agentic_rag(query: str):
    """
    Handles session lifecycle and executes the agentic workflow.
    Ensures observability by printing state transitions.
    """
    session_service = InMemorySessionService()
    root_agent = setup_orchestrator()
    
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service
    )

    # Initialize Session
    try:
        await session_service.create_session(
            app_name=APP_NAME, 
            user_id=USER_ID, 
            session_id=SESSION_ID
        )
        logger.info(f"Session {SESSION_ID} initialized.")
    except Exception as e:
        logger.warning(f"Session already exists or could not be created: {e}")

    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    print("\n--- Starting Agentic Workflow ---\n")
    
    try:
        # run_async handles the transition between Planner and Synthesizer automatically
        async for event in runner.run_async(
            user_id=USER_ID, 
            session_id=SESSION_ID, 
            new_message=content
        ):
            # Useful for debugging raw events from Google ADK
            pass 
            
    except Exception as e:
        logger.error(f"Execution Error: {str(e)}", exc_info=True)
        return

    # 4. OBSERVABILITY & FINAL OUTPUT
    # Fetch final state to display the synthesized answer and citations
    final_session = await session_service.get_session(
        app_name=APP_NAME, 
        user_id=USER_ID, 
        session_id=SESSION_ID
    )

    if final_session and final_session.state:
        state = final_session.state
        
        print("\n" + "="*50)
        print("AGENTIC TRACE (OBSERVABILITY)")
        print("-" * 50)
        # Showing how the Planner decomposed the query
        print(f"Planner Result: {state.get('planner_plan')}") 
        print("="*50 + "\n")
        
        # Displaying the final grounded response
        print("FINAL ANSWER:")
        print(state.get("final_response", "No response generated."))
    else:
        logger.error("Failed to retrieve final session state.")

if __name__ == "__main__":

    current_dir = Path(__file__).parent.resolve()
    vector_db_path = current_dir.parent / "vectorstore" / "chroma_db"
    pdf_file = 'rag_pdf.pdf'

    # CONDITIONAL INGESTION LOGIC
    # Requirement Loading and indexing the PDF
    if not vector_db_path.exists():
        print(f"--- Vector store not found at {vector_db_path}. Starting ingestion... ---")
        try:
            # This handles PDF loading, chunking, and embedding
            create_vector_store(pdf_file_name=pdf_file)
            print("--- Ingestion completed successfully. ---")
        except Exception as e:
            print(f"Critical Error during ingestion: {e}")
            sys.exit(1)
    else:

        print(f"--- Vector store detected at {vector_db_path}. \n Skipping ingestion. ---")
    # Enter Query 
    query = input('enter query: ')
    
    try:
        asyncio.run(run_agentic_rag(query))
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"Fatal Startup Error: {e}")