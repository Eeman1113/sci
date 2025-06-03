from crewai import Task
from agents_config import ResearchAgents # To access agent instances

# Note: The context for tasks will typically come from the LangGraph state.
# The 'expected_output' descriptions are crucial for the LLMs.

class ResearchTasks:
    def __init__(self, agents: ResearchAgents):
        self.agents = agents

    def plan_research_outline_task(self, topic: str) -> Task:
        return Task(
            description=(
                f"Develop a comprehensive, structured outline for a research report on the topic: '{topic}'. "
                "The outline should consist of main section titles. "
                "For each main section, also list 2-3 key questions or sub-topics that need to be investigated. "
                "The output should be a well-formatted list of section titles, each with its corresponding key questions."
                "Example Output Format:\n"
                "Section 1: Introduction\n"
                "  - Question 1.1: ...\n"
                "  - Question 1.2: ...\n"
                "Section 2: Background and History\n"
                "  - Question 2.1: ...\n"
                "  - Question 2.2: ...\n"
                "(and so on for at least 5-7 major sections, including Conclusion and References sections)"
            ),
            expected_output=(
                "A string containing a structured outline. It should list main section titles, "
                "and under each section, 2-3 key questions or sub-topics. "
                "Ensure 'Introduction', 'Conclusion', and 'References' are included as sections."
            ),
            agent=self.agents.planner_agent(),
            async_execution=False, # Synchronous for planning usually
        )

    def conduct_research_task(self, section_title: str, research_questions: list[str], existing_urls: list[str], existing_queries: list[str], max_searches: int) -> Task:
        # Format existing URLs and queries for the prompt to avoid re-searching
        existing_urls_str = ", ".join(existing_urls) if existing_urls else "None"
        existing_queries_str = ", ".join(existing_queries) if existing_queries else "None"
        
        return Task(
            description=(
                f"For the report section titled '{section_title}', investigate the following key questions/sub-topics: {'; '.join(research_questions)}. "
                f"Conduct targeted web searches to find relevant information. You are allowed up to {max_searches} distinct search queries for this task. "
                f"Prioritize credible sources. \n"
                f"IMPORTANT: Do NOT research or return results for URLs already in this list: {existing_urls_str}.\n"
                f"IMPORTANT: Do NOT use search queries similar to these already used: {existing_queries_str}.\n"
                "For each useful source found, provide its title, URL, and a brief snippet of its content."
            ),
            expected_output=(
                "A list of dictionaries, where each dictionary represents a found source and contains 'title', 'href' (URL), and 'snippet'. "
                "Return at most 3-4 top sources per research question. If no new relevant sources are found, return an empty list or a message indicating so."
                "Also output a list of the actual search queries you performed."
            ),
            agent=self.agents.research_agent(),
            async_execution=False, # Can be True if multiple research tasks run in parallel
            # context: This task might need context from previous tasks (e.g., overall topic)
        )

    def analyze_data_task(self, section_title: str, research_data: list[dict], research_questions: list[str]) -> Task:
        # research_data is a list of dicts like {'title': '...', 'href': '...', 'snippet': '...'}
        # The agent will use its web_fetcher_tool to get full content if needed.
        data_summary_for_prompt = "\n".join([f"- {d['title']} ({d['href']})" for d in research_data]) if research_data else "No initial data provided."

        return Task(
            description=(
                f"For the report section titled '{section_title}', you have been provided with the following initial research findings (titles and URLs):\n{data_summary_for_prompt}\n"
                f"The key questions for this section are: {'; '.join(research_questions)}.\n"
                "Your tasks are:\n"
                "1. If URLs are provided, use the 'Web Page Content Fetcher' tool to get the full text content for each relevant source. Focus on the most promising 2-3 sources.\n"
                "2. Critically analyze all gathered information (snippets and fetched full content).\n"
                "3. Synthesize the key insights, facts, arguments, and important data points relevant to the research questions.\n"
                "4. Identify any conflicting information or significant gaps in the current data.\n"
                "5. Extract any citable source details (URL, Title) for the insights you provide.\n"
                "6. Based on your analysis, determine if the information is sufficient to write a comprehensive section, or if specific follow-up research questions are needed for more depth. If so, list those new questions."
            ),
            expected_output=(
                "A structured response containing:\n"
                "1. 'summary_of_insights': A detailed summary of the synthesized information and key findings.\n"
                "2. 'gaps_and_conflicts': Notes on any identified gaps or conflicting information.\n"
                "3. 'cited_sources': A list of sources (URL, Title) that contributed to the insights.\n"
                "4. 'sufficiency_assessment': Your judgment on whether the information is sufficient.\n"
                "5. 'follow_up_questions': A list of new, specific research questions if more depth is needed (or an empty list if sufficient)."
                "The output should be a clear, well-organized text or JSON-like structure."
            ),
            agent=self.agents.analysis_agent(),
            async_execution=False,
        )

    def write_section_task(self, section_title: str, section_insights: str, cited_sources: list[dict]) -> Task:
        sources_str = "\n".join([f"- {s['title']} ({s['href']})" for s in cited_sources]) if cited_sources else "No specific sources cited for this section draft."
        return Task(
            description=(
                f"Draft a comprehensive and detailed report section titled '{section_title}'.\n"
                f"Base your writing on the following key insights and analysis summary:\n{section_insights}\n\n"
                f"Refer to these sources if applicable during your writing:\n{sources_str}\n"
                "The writing style should be academic, clear, objective, and well-structured. "
                "Ensure the section thoroughly covers the provided insights. "
                "Do not just list the insights; elaborate on them, explain them, and connect them logically. "
                "Aim for a substantial piece of writing for this section, as it will be part of a larger report. "
                "Use Markdown for formatting (e.g., headings, lists, bold text)."
            ),
            expected_output=(
                "A string containing the fully drafted report section in Markdown format. "
                "The section should be well-organized, coherent, and detailed, directly addressing the provided insights. "
                "It should start with a heading for the section title (e.g., `## {section_title}`)."
            ),
            agent=self.agents.writing_agent(), # Assuming writing_agent is configured
            async_execution=False,
        )

    def review_section_task(self, section_title: str, draft_content: str) -> Task:
        return Task(
            description=(
                f"Critically review the following drafted report section titled '{section_title}':\n\n---\n{draft_content}\n---\n\n"
                "Evaluate it for:\n"
                "1. Clarity and Coherence: Is the writing clear, logical, and easy to understand?\n"
                "2. Accuracy: Does the information seem factually correct (based on general knowledge, flag if uncertain)?\n"
                "3. Completeness: Does the section adequately cover the likely scope implied by its title and content? Are there obvious omissions?\n"
                "4. Grammar and Style: Are there any grammatical errors, typos, or awkward phrasing?\n"
                "5. Structure and Flow: Is the section well-organized with smooth transitions?\n\n"
                "Provide specific, actionable feedback. List bullet points for areas of improvement. "
                "If the section is excellent and requires no changes, state 'Approved as is'."
            ),
            expected_output=(
                "A string containing detailed feedback. This should be a list of specific suggestions for improvement, "
                "or the phrase 'Approved as is' if no changes are needed. If providing suggestions, be precise about what to change and why."
            ),
            agent=self.agents.review_agent(),
            async_execution=False,
        )

if __name__ == '__main__':
    # Example of how to instantiate and potentially test a task
    # This requires agents to be set up first.
    # from agents_config import ResearchAgents
    # agents_obj = ResearchAgents()
    # tasks_obj = ResearchTasks(agents_obj)
    #
    # topic_for_planning = "The Impact of AI on Climate Change Solutions"
    # plan_task = tasks_obj.plan_research_outline_task(topic_for_planning)
    # print(f"Task: {plan_task.description}")
    #
    # # To run a task, you'd typically do it within a Crew:
    # # from crewai import Crew
    # # planner_crew = Crew(
    # #     agents=[agents_obj.planner_agent()],
    # #     tasks=[plan_task],
    # #     verbose=2
    # # )
    # # results = planner_crew.kickoff()
    # # print(results)
    pass
