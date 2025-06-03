from crewai import Agent
from langchain_community.llms import Ollama
from custom_tools import search_tool, web_fetcher_tool # Import your custom tools

# Centralized LLM configuration
def get_ollama_llm(model_name: str, temperature: float = 0.7):
    """Helper function to create an Ollama LLM instance."""
    return Ollama(model=model_name, temperature=temperature)

# Define Agents
class ResearchAgents:
    def __init__(self, ollama_model_general: str = "llama3"):
        self.llm_general = get_ollama_llm(ollama_model_general)

    def planner_agent(self) -> Agent:
        return Agent(
            role="Lead Research Architect and Strategist",
            goal=(
                "Given a complex research topic, develop an exceptionally comprehensive and meticulously structured outline "
                "for an in-depth report (expected to be 30-50 pages in length). The outline must consist of detailed main section titles. "
                "For each section, identify a granular set of key questions, sub-topics, and specific areas of investigation "
                "required to ensure exhaustive coverage. The structure should be logical and facilitate a deep dive into the topic."
            ),
            backstory=(
                "You are an elite research strategist and architect, renowned for your ability to deconstruct multifaceted topics "
                "into highly organized and actionable research plans for substantial, publication-quality reports. "
                "You excel in formulating precise, probing questions that guide in-depth investigations. "
                "Your outlines are the blueprints for authoritative and extensive research papers."
            ),
            llm=self.llm_general,
            tools=[],
            allow_delegation=False,
            verbose=True,
            memory=False
        )

    def research_agent(self) -> Agent:
        return Agent(
            role="Chief Information Specialist",
            goal=(
                "For a given research question or sub-topic (part of a larger report), and a list of previously searched queries/URLs, "
                "conduct exhaustive and targeted web searches using DuckDuckGo. Your mission is to find a wide array of "
                "highly relevant articles, academic papers, datasets, primary sources (where applicable), and other credible information. "
                "Prioritize authoritative, diverse, and detailed sources to build a strong evidence base for a comprehensive report. Avoid redundant searches meticulously."
            ),
            backstory=(
                "You are a distinguished information specialist with extensive experience in deep-dive research for large-scale academic and scientific publications. "
                "You possess mastery in crafting advanced search queries and rapidly discerning the most valuable and trustworthy information from a vast sea of data. "
                "You are systematic in your approach to ensure thoroughness and avoid prior work."
            ),
            llm=self.llm_general,
            tools=[search_tool],
            allow_delegation=False,
            verbose=True,
            memory=True
        )

    def analysis_agent(self) -> Agent:
        return Agent(
            role="Senior Principal Analyst and Insight Weaver",
            goal=(
                "Given a collection of raw data (text snippets, article URLs, research notes) for a specific research section, "
                "fetch content from all provided URLs, then perform a profound and critical analysis of all information. "
                "Synthesize this into a rich tapestry of key insights, crucial facts, compelling arguments, and supporting evidence. "
                "Identify any conflicting information, nuanced perspectives, or significant gaps in the collected data. Extract and list all citable sources meticulously. "
                "Crucially, determine if the current information is sufficiently comprehensive for an in-depth report section and, if not, "
                "formulate specific follow-up questions or identify precise areas needing deeper investigation for recursive exploration."
            ),
            backstory=(
                "You are a preeminent analyst and sense-maker, possessing an extraordinary ability to delve into complex datasets and extract profound insights. "
                "With a meticulous eye for detail and a talent for synthesis, you weave disparate information into coherent, compelling narratives. "
                "You are adept at evaluating source credibility, identifying subtle patterns, uncovering hidden connections, and pinpointing areas requiring further rigorous exploration to achieve complete understanding."
            ),
            llm=self.llm_general,
            tools=[web_fetcher_tool],
            allow_delegation=False,
            verbose=True,
            memory=True
        )

    def writing_agent(self, ollama_model_writing: str = "llama3") -> Agent:
        llm_writing = get_ollama_llm(ollama_model_writing, temperature=0.7)
        return Agent(
            role="Distinguished Academic Author",
            goal=(
                "Given a specific section title, a comprehensive summary of synthesized insights, and a list of supporting data/references, "
                "draft an exceptionally detailed, well-structured, and coherent section for a significant research report (contributing to a final document of 30-50 pages). "
                "The writing style must be sophisticated, academic, clear, and engaging, suitable for a discerning audience. "
                "Ensure meticulous attribution for all sources. Each section should be a substantial piece of writing, reflecting deep understanding and thorough coverage of the provided insights."
            ),
            backstory=(
                "You are a distinguished academic author, highly regarded for your ability to transform intricate analyses and synthesized data "
                "into eloquent, compelling, and publishable-quality prose. Your work is characterized by its precision, organizational clarity, depth of content, "
                "and adherence to the highest academic standards. You specialize in crafting extensive, detailed chapters for comprehensive research monographs and reports."
            ),
            llm=llm_writing,
            tools=[],
            allow_delegation=False,
            verbose=True,
            memory=False
        )

    def review_agent(self) -> Agent:
        return Agent(
            role="Lead Editorial Reviewer",
            goal=(
                "Critically and meticulously review a drafted section of a major research report. Evaluate its clarity, coherence, "
                "logical flow, accuracy, and depth of analysis. Assess the comprehensiveness and completeness of the content relative to its intended role in a substantial report (30-50 pages). "
                "Check for rigorous argumentation, proper use of evidence, and academic integrity. Scrutinize grammar, style, and formatting. "
                "Provide specific, constructive, and actionable feedback to elevate the section to publication standards. If the section meets these exacting criteria, approve it."
            ),
            backstory=(
                "You are a lead editorial reviewer for a prestigious academic press, with an unwavering commitment to scholarly excellence. "
                "No error, inconsistency, or weakness in argument escapes your notice. Your feedback is insightful, constructive, and aimed at ensuring each component of a major research document achieves the highest possible quality and impact."
            ),
            llm=self.llm_general,
            tools=[],
            allow_delegation=False,
            verbose=True,
            memory=False
        )

if __name__ == '__main__':
    agents_manager = ResearchAgents(ollama_model_general="mistral")
    planner = agents_manager.planner_agent()
    researcher = agents_manager.research_agent()
    analyzer = agents_manager.analysis_agent()
    writer = agents_manager.writing_agent()
    reviewer = agents_manager.review_agent()

    # print(f"Planner Role: {planner.role}\nPlanner Goal: {planner.goal}\nPlanner Backstory: {planner.backstory}\n")
    # print(f"Research Role: {researcher.role}\nResearch Goal: {researcher.goal}\nResearch Backstory: {researcher.backstory}\n")
    # print(f"Analysis Role: {analyzer.role}\nAnalysis Goal: {analyzer.goal}\nAnalysis Backstory: {analyzer.backstory}\n")
    # print(f"Writer Role: {writer.role}\nWriter Goal: {writer.goal}\nWriter Backstory: {writer.backstory}\n")
    # print(f"Reviewer Role: {reviewer.role}\nReviewer Goal: {reviewer.goal}\nReviewer Backstory: {reviewer.backstory}\n")
    pass
