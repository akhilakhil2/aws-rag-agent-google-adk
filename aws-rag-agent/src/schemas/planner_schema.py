from pydantic import BaseModel,Field
from typing import List

# --- Schema Definition ---
class PlannerPlan(BaseModel):
    """
    Schema for the Planner Agent's output. 
    Decomposes user queries into structured steps for the Retriever.
    """
    query_type: str = Field(..., description="The category of the query (e.g., technical, general, comparison)")
    sub_queries: List[str] = Field(..., description="A list of specific search strings to be used in the vector database")
    
