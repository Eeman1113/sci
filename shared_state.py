from typing import List, Dict, Any, Set, Optional
from langchain_core.pydantic_v1 import BaseModel, Field

# Using BaseModel for better type checking and potential future serialization
class SectionData(BaseModel):
    title: str
    # Raw data collected (e.g., text snippets, URLs)
    raw_data: List[str] = Field(default_factory=list)
    # Summarized insights from Analysis Agent
    summary: Optional[str] = None
    # Drafted content from Writing Agent
    draft_content: Optional[str] = None
    # Feedback from Review Agent
    review_feedback: Optional[str] = None
    # Number of revision attempts for this section
    revision_attempts: int = 0
    # Follow-up questions identified by Analysis Agent for recursive research
    follow_up_questions: List[str] = Field(default_factory=list)
    # Current recursion depth for this section
    recursion_depth: int = 0

class ResearchState(BaseModel):
    topic: str
    # High-level outline (e.g., list of main section titles)
    initial_outline: Optional[List[str]] = None
    # Detailed plan for each section, could include sub-topics or specific questions
    detailed_plan: Dict[str, Any] = Field(default_factory=dict)
    # Data for each section, key is section title
    sections_data: Dict[str, SectionData] = Field(default_factory=dict)
    # Current section being processed
    current_section_title: Optional[str] = None
    # List of all URLs collected to avoid duplicates
    all_collected_urls: Set[str] = Field(default_factory=set)
    # List of all search queries made
    all_search_queries: Set[str] = Field(default_factory=set)
    # Overall iteration count for the main research loop
    main_loop_iterations: int = 0
    # Log of significant events or errors
    event_log: List[str] = Field(default_factory=list)
    # Final assembled report in Markdown
    final_report_md: Optional[str] = None
    # Generated Table of Contents
    table_of_contents: Optional[str] = None
    # Collected references
    references: List[str] = Field(default_factory=list)

    # Configuration settings (can be populated from Streamlit UI)
    ollama_model_general: str = "llama3" # Default model
    ollama_model_writing: str = "llama3" # Potentially a different model for writing
    max_searches_per_section: int = 5
    max_sources_per_search: int = 3 # How many search results to process
    max_revision_cycles_per_section: int = 2
    max_main_loop_iterations: int = 10 # To prevent infinite loops in overall process
    max_recursion_depth_per_section: int = 2 # Max depth for recursive research on a single section

    # Fields for tracking progress or errors
    error_message: Optional[str] = None
    current_status: str = "Initialized"

