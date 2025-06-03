from crewai import Agent
from langchain_community.llms import Ollama
from custom_tools import search_tool, web_fetcher_tool # Import your custom tools

# Centralized LLM configuration
def get_ollama_llm(model_name: str, temperature: float = 0.7):
    """Helper function to create an Ollama LLM instance."""
    return Ollama(model=model_name, temperature=temperature)

# Define Agents
# The prompts (role, goal, backstory) are crucial and will need refinement.

class ResearchAgents:
    def __init__(self, ollama_model_general: str = "llama3"):
        self.llm_general = get_ollama_llm(ollama_model_general)

    def planner_agent(self) -> Agent:
        return Agent(
            role="Lead Research Planner",
            goal=(
                "Given a research topic, develop a comprehensive, structured outline for a detailed report. "
                "The outline should consist of main section titles. "
                "Also, for each section, identify key questions or sub-topics that need to be investigated."
            ),
            backstory=(
                "You are an expert research strategist and planner. You excel at breaking down complex topics "
                "into manageable sections and formulating precise research questions to guide the investigation. "
                "Your outlines are logical, thorough, and form the backbone of high-quality research reports."
            ),
            llm=self.llm_general,
            tools=[], # This agent might not need external tools, relies on its LLM's planning capabilities
            allow_delegation=False,
            verbose=True,
            memory=False # Planners usually don't need long term memory for this specific task
        )

    def research_agent(self) -> Agent:
        return Agent(
            role="Senior Information Retriever",
            goal=(
                "Given a research question or sub-topic and a list of previously searched queries/URLs, "
                "conduct targeted web searches using DuckDuckGo to find relevant articles, data, and sources. "
                "Prioritize credible and informative sources. Avoid redundant searches."
            ),
            backstory=(
                "You are a master of information retrieval, skilled in crafting effective search queries "
                "and quickly identifying the most relevant and trustworthy information online. "
                "You are meticulous about avoiding previously covered ground."
            ),
            llm=self.llm_general,
            tools=[search_tool], # Equipped with the search tool
            allow_delegation=False, # Can be True if it needs to delegate fetching to another agent
            verbose=True,
            memory=True # Memory of past searches can be useful
        )

    def analysis_agent(self) -> Agent:
        # This agent will also fetch content from URLs found by the research agent
        return Agent(
            role="Principal Data Analyst and Synthesizer",
            goal=(
                "Given a collection of raw data (text snippets, article URLs) for a specific research question or section, "
                "fetch content from URLs, critically analyze the information, synthesize key insights, facts, and arguments. "
                "Identify any conflicting information or gaps. Extract citable sources/references if possible."
                "Determine if the current information is sufficient or if follow-up questions are needed for depth."
            ),
            backstory=(
                "You are a highly skilled analyst with a keen eye for detail and a talent for "
                "distilling complex information into concise, meaningful insights. You can identify patterns, "
                "evaluate source credibility, and determine the completeness of research for a given topic."
            ),
            llm=self.llm_general,
            tools=[web_fetcher_tool], # Tool to fetch content from URLs
            allow_delegation=False,
            verbose=True,
            memory=True
        )

    def writing_agent(self, ollama_model_writing: str = "llama3") -> Agent:
        llm_writing = get_ollama_llm(ollama_model_writing, temperature=0.7) # Potentially different model/temp for writing
        return Agent(
            role="Expert Academic Writer",
            goal=(
                "Given a specific section title, a summary of key insights, and supporting data/references, "
                "draft a comprehensive, well-structured, and coherent section for a research report. "
                "The writing style should be academic, clear, and engaging. Ensure proper attribution if source material is quoted or closely paraphrased."
                "Aim for a detailed and thorough coverage of the provided insights."
            ),
            backstory=(
                "You are a renowned academic writer, celebrated for your ability to transform complex analyses "
                "into eloquent and informative prose. Your writing is precise, well-organized, and adheres to "
                "high academic standards. You strive for depth and clarity in your explanations."
            ),
            llm=llm_writing,
            tools=[], # Primarily relies on its writing capabilities and provided context
            allow_delegation=False,
            verbose=True,
            memory=False # Usually writes based on current input context
        )

    def review_agent(self) -> Agent:
        return Agent(
            role="Meticulous Quality Reviewer",
            goal=(
                "Critically review a drafted section of a research report. Evaluate it for clarity, coherence, "
                "accuracy, completeness, grammar, and style. Provide specific, actionable feedback and "
                "suggestions for improvement. If the section is satisfactory, approve it."
            ),
            backstory=(
                "You are an exacting editor with an unwavering commitment to quality. No error or inconsistency "
                "escapes your notice. Your feedback is constructive and aimed at elevating the report to the highest standards."
            ),
            llm=self.llm_general,
            tools=[],
            allow_delegation=False,
            verbose=True,
            memory=False
        )

# Note: The "Revision Agent" and "Publishing Agent" from the prompt might be better handled
# as specific tasks or logic within the LangGraph workflow rather than separate LLM-based CrewAI agents.
# For instance, revision can be a re-invocation of the Writing Agent with feedback.
# Publishing is an aggregation and formatting step.

if __name__ == '__main__':
    agents_manager = ResearchAgents(ollama_model_general="mistral") # Example using mistral
    planner = agents_manager.planner_agent()
    researcher = agents_manager.research_agent()
    # print(planner.role)
    # print(researcher.tools[0].name)
    pass
