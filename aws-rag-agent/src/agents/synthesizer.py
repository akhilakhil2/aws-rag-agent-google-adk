import logging
import sys
import os
from dotenv import load_dotenv

# Google ADK & Tool Imports
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from tools.retriever import retriever_tool

# 1. STANDARDIZED LOGGING CONFIGURATION
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent_system.log") # Shared log file for multi-agent tracing
    ]
)
logger = logging.getLogger("SynthesizerAgent")

def initialize_synthesizer_agent() -> Agent:
    """
    Initializes the Synthesizer Agent with strict grounding instructions 
    and mandatory tool access for RAG operations.
    """
    # 2. ENVIRONMENT VALIDATION
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        logger.critical("GROQ_API_KEY is missing. Synthesizer cannot function without LLM access.")
        raise EnvironmentError("GROQ_API_KEY not found in environment.")

    try:
        logger.info("Initializing Synthesizer Agent with tool-calling capabilities...")

        # 3. AGENT CONFIGURATION WITH ASSIGNMENT-SPECIFIC CONSTRAINTS
        synthesizer = Agent(
            name="synthesizer_agent_v1",
            model=LiteLlm(
                model='groq/llama-3.3-70b-versatile', 
                api_key=api_key
            ),
            description=(
                "A specialist that synthesizes retrieved document segments into "
                "structured, cited answers for the user." 
            ),
            # {planner_plan} is injected from the session state
            instruction=(
                "### ROLE\n"
                "You are a Technical Synthesis Expert for AWS Prescriptive Guidance. "
                "Your goal is to provide cited, accurate answers based ONLY on retrieved text.\n\n"
                
                "### EXECUTION PROTOCOL\n"
                "1. ANALYZE PLAN: Extract the 'sub_queries' list from the {planner_plan} object.\n"
                "2. RETRIEVE DATA: You MUST call the 'retriever_tool' using that list of sub_queries. "
                "Do not attempt to answer from memory.\n"
                "3. SYNTHESIZE: Review the 'retriever_content' returned by the tool.\n\n"
                
                "### RESPONSE FORMATTING RULES\n"
                "- STRUCTURE: Use clear sections and bullet points for readability[cite: 32, 81].\n"
                "- CITATIONS: Every factual claim MUST be followed by a citation, e.g., '(Source: [Section Name])'[cite: 29, 83].\n"
                "- GROUNDING: If the tool returns no data or the data does not answer the query, "
                "you MUST state: 'This information is not available in the provided AWS RAG guide'.\n"
                "- NO HALLUCINATIONS: Never use external technical knowledge[cite: 30, 106]."
            ),
            tools=[retriever_tool], # Essential for the agent to access the vector store
            output_key="final_response"
        )

        logger.info("Synthesizer Agent successfully initialized with retriever_tool.")
        return synthesizer

    except Exception as e:
        logger.error(f"Critical error during Synthesizer Agent setup: {str(e)}", exc_info=True)
        raise RuntimeError("Synthesizer Agent failed to initialize.") from e

