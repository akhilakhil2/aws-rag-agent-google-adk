import logging
import sys
import os
from typing import Optional, Dict, Any
from pydantic import ValidationError
from dotenv import load_dotenv

# Google ADK & Schema Imports
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from schemas.planner_schema import PlannerPlan

# 1. ENHANCED LOGGING CONFIGURATION
# Ensuring we capture timestamps, levels, and specifically which agent is reporting.
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent_system.log") # Persistent logs for traceability 
    ]
)
logger = logging.getLogger("PlannerAgent")

def initialize_planner_agent() -> Agent:
    """
    Initializes the Planner Agent with robust error handling for environment 
    variables and model configuration.
    """
    # 2. ENVIRONMENT & API KEY VALIDATION
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        logger.critical("GROQ_API_KEY is missing from environment variables.")
        raise EnvironmentError("GROQ_API_KEY not found. Ensure your .env file is configured.")

    try:
        logger.info("Configuring LLM and output schema for Planner...")
        
        # 3. ROBUST AGENT INITIALIZATION
        planner = Agent(
            name="planner_agent_v1",
            model=LiteLlm(
                model='groq/llama-3.3-70b-versatile', 
                api_key=api_key
            ),
            description=(
                "Strategist responsible for query classification and decomposition "
                "to guide the Retrieval Agent." 
            ),
            instruction=(
                "### ROLE\n"
                "You are a Strategic Planning Agent for an AWS RAG system. [cite: 58]\n\n"
                "### TASKS\n"
                "1. CLASSIFY: Assign the query to exactly one category: 'comparison', 'definition', 'recommendation', or 'uses'. [cite: 61]\n"
                "2. DECOMPOSE: Generate 2-3 specific sub-queries optimized for semantic search in an AWS RAG guide. [cite: 63, 64]\n\n"
                "### CONSTRAINTS\n"
                "- Output ONLY valid JSON matching the provided schema.\n"
                "- No conversational filler or preamble.\n"
                "- If the query is unrelated to RAG on AWS, categorize as 'general' and suggest a refusal search term."
            ),
            output_schema=PlannerPlan, # Enforces strict structure 
            output_key="planner_plan"  # Key for state management in SequentialAgent flows
        )
        
        logger.info("Planner Agent successfully initialized.")
        return planner

    except Exception as e:
        logger.error(f"Failed to initialize Planner Agent component: {str(e)}", exc_info=True)
        raise RuntimeError("Agent initialization aborted due to configuration error.") from e

# 4. ERROR CORRECTION & VALIDATION UTILITY
def safe_validate_plan(raw_data: Dict[str, Any]) -> Optional[PlannerPlan]:
    """
    Safely validates raw LLM output against the PlannerPlan schema.
    Used for observability and debugging traces. 
    """
    try:
        plan = PlannerPlan(**raw_data)
        logger.info(f"Validation Success: Query classified as '{plan.query_type}'")
        return plan
    except ValidationError as ve:
        logger.error(f"Schema Mismatch: The LLM returned incompatible JSON. Errors: {ve.json()}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during plan validation: {str(e)}")
        return None

