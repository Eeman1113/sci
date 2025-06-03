import json
from typing import Dict, Any, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # For persisting state if needed

from shared_state import ResearchState, SectionData
from agents_config import ResearchAgents
from tasks_config import ResearchTasks
from crewai import Crew, Process
import re # For parsing outline

# Helper to parse the outline from the planner agent
def parse_planner_output(planner_output: str) -> List[str]:
    """
    Parses the planner's output string to extract main section titles.
    Assumes planner output has lines like "Section X: Title" or similar.
    This will need to be robust based on the actual LLM output format.
    """
    sections = []
    # A more robust regex might be needed depending on LLM's consistency
    # This regex looks for lines starting with "Section" followed by a number/letter and colon, or just a title-like line.
    # It also tries to capture lines that are just titles if the "Section X:" pattern isn't strictly followed.
    # This is a common source of brittleness in LLM pipelines.
    # A better approach is to ask the LLM to output JSON.
    
    # Simple line-based parsing, assuming each main section is on a new line
    # and might be prefixed.
    # Let's assume the planner task asks for a list of section titles.
    # For now, we'll rely on a simpler split if the output is just a list of titles.
    
    # If the planner agent is prompted to return a list of section titles,
    # the output might be simpler. For the example task, it's more structured.
    # Let's try to extract lines that look like section headers.
    
    # Try to find lines that look like "Section X: Title" or just "Title" if it's a list
    # This is a heuristic and might need significant adjustment.
    # A better way is to ask the LLM for JSON output in the prompt for the planner_agent.
    
    # For now, let's assume the planner_output is a string where section titles are clearly identifiable.
    # The task `plan_research_outline_task` asks for:
    # "Section 1: Introduction", "  - Question 1.1: ..."
    # We only need "Introduction", "Background and History", etc.
    
    # Regex to find "Section X: Actual Title"
    matches = re.findall(r"Section\s*\d*[\w]*\s*:\s*([^\n]+)", planner_output, re.IGNORECASE)
    if matches:
        sections = [match.strip() for match in matches]
    else:
        # Fallback: if no "Section X:" format, try to split by lines and take non-empty ones
        # This is less reliable.
        potential_sections = [line.strip() for line in planner_output.split('\n') if line.strip()]
        # Filter out lines that look like questions
        sections = [s for s in potential_sections if not s.startswith(("- ", "* ", "  -")) and len(s) > 3] # Heuristic

    # Ensure standard sections are present if not explicitly planned, or add them
    required_sections = ["Introduction", "Conclusion", "References"]
    present_sections_lower = [s.lower() for s in sections]
    
    if "introduction" not in present_sections_lower and sections:
        sections.insert(0, "Introduction") # Add to beginning
    elif not sections: # If planner failed to produce anything
        sections.append("Introduction")

    # Remove duplicates while preserving order (if any from regex)
    seen = set()
    sections = [x for x in sections if not (x in seen or seen.add(x))]


    # Check for Conclusion and References, add if missing
    # This is a bit hacky; ideally, the planner includes them.
    if "conclusion" not in [s.lower() for s in sections]:
        sections.append("Conclusion")
    if "references" not in [s.lower() for s in sections]:
        sections.append("References")
        
    return sections if sections else ["Default Section 1", "Default Section 2", "Conclusion", "References"]


# --- Graph Nodes ---
# Each node will take the ResearchState and return a partial ResearchState update.

def planning_node(state: ResearchState, agents_cfg: ResearchAgents, tasks_cfg: ResearchTasks) -> Dict[str, Any]:
    """Generates the initial research outline."""
    state.current_status = f"Planning outline for topic: {state.topic}"
    state.event_log.append(state.current_status)
    
    planner_agent = agents_cfg.planner_agent()
    plan_task = tasks_cfg.plan_research_outline_task(topic=state.topic)
    
    # Using CrewAI to run this specific task
    crew = Crew(
        agents=[planner_agent],
        tasks=[plan_task],
        process=Process.sequential,
        # memory=False, # Handled by LangGraph state
        verbose=1 # 0 for no output, 1 for minimal, 2 for detailed
    )
    try:
        planner_output_str = crew.kickoff()
        if not planner_output_str or not isinstance(planner_output_str, str):
             raise ValueError("Planner agent did not return a valid string output.")

        parsed_outline = parse_planner_output(planner_output_str) # This function needs to be robust
        
        # Initialize sections_data based on the outline
        sections_data = {title: SectionData(title=title, raw_data=[], revision_attempts=0) for title in parsed_outline}
        
        state.current_status = "Outline planned."
        state.event_log.append(f"Planned outline: {parsed_outline}")
        return {
            "initial_outline": parsed_outline,
            "sections_data": sections_data,
            "current_status": state.current_status,
            "event_log": state.event_log
        }
    except Exception as e:
        error_msg = f"Error in planning node: {str(e)}"
        state.event_log.append(error_msg)
        return {"error_message": error_msg, "event_log": state.event_log, "current_status": "Error in Planning"}


def research_node(state: ResearchState, agents_cfg: ResearchAgents, tasks_cfg: ResearchTasks) -> Dict[str, Any]:
    """Conducts research for the current section."""
    section_title = state.current_section_title
    if not section_title or section_title not in state.sections_data:
        return {"error_message": "Research node: Current section not set or invalid."}

    state.current_status = f"Researching section: {section_title}"
    section_data_obj = state.sections_data[section_title]

    # Determine research questions based on recursion state
    if section_data_obj.recursion_depth > 0 and section_data_obj.follow_up_questions:
        state.current_status += f" (Recursion depth: {section_data_obj.recursion_depth}, using {len(section_data_obj.follow_up_questions)} follow-up questions)"
        research_questions_for_section = list(section_data_obj.follow_up_questions) # Make a copy to use
        # Clear the follow-up questions from the state immediately after they are copied for use
        state.sections_data[section_title].follow_up_questions = []
        state.event_log.append(f"Cleared follow-up questions for section '{section_title}' after setting them for recursive research.")
    else:
        # Initial research questions (could be from planner or default)
        # For now, using a generic question if not in recursion and no specific planned questions are loaded here.
        # This part could be enhanced to pull initial questions from state.detailed_plan if available.
        state.current_status += " (Initial research pass or no follow-up questions provided)"
        research_questions_for_section = [f"Key information about {section_title} related to {state.topic}"]

    state.event_log.append(state.current_status)

    research_agent_instance = agents_cfg.research_agent()
    research_task_instance = tasks_cfg.conduct_research_task(
        section_title=section_title,
        research_questions=research_questions_for_section, # Use dynamically determined questions
        existing_urls=list(state.all_collected_urls),
        existing_queries=list(state.all_search_queries),
        max_searches=state.max_searches_per_section
    )
    crew = Crew(agents=[research_agent_instance], tasks=[research_task_instance], verbose=1)
    
    try:
        # CrewAI's kickoff can return a string or sometimes structured data if the task is well-prompted.
        # The task is expected to return "A list of dictionaries... Also output a list of the actual search queries..."
        # This is tricky. CrewAI's output parsing can be basic.
        # We might need to prompt the LLM to return JSON string and then parse it.
        research_output = crew.kickoff() # This might be a string.

        # Attempt to parse the research_output. This is a common pain point.
        # For now, let's assume it's a string that we can try to process, or the agent returns structured list of dicts.
        # If it's a string, we might need another LLM call to structure it, or very careful prompting.
        
        # Let's assume for now the task's agent is well-prompted to return a string that can be loaded as JSON
        # or the CrewAI framework handles this. This part is often fragile.
        # A robust solution asks the LLM to output JSON.
        
        # Placeholder for parsing - this needs to be robust
        # Example: if research_output is a string "{'results': [], 'queries': []}"
        # For now, we'll assume the agent returns a list of dicts for sources if successful.
        # And we'll try to extract queries if they are part of a larger string.

        found_sources = []
        performed_queries = []

        if isinstance(research_output, str):
            # Try to find a list of dicts (for sources) and list of strings (for queries) in the output string.
            # This is highly heuristic. A JSON output from LLM is better.
            try:
                # A simple attempt if the output is a JSON string representing a list of dicts
                # Or a dict containing 'results' and 'queries'
                data = json.loads(research_output)
                if isinstance(data, dict):
                    found_sources = data.get("results", [])
                    performed_queries = data.get("queries", [])
                elif isinstance(data, list): # if it's just a list of sources
                    found_sources = data
            except json.JSONDecodeError:
                state.event_log.append(f"Warning: Research output for '{section_title}' was not valid JSON: {research_output[:200]}...")
                # Fallback: add the raw output as a single piece of data if it's not empty
                if research_output.strip():
                     state.sections_data[section_title].raw_data.append(research_output)


        elif isinstance(research_output, list): # If CrewAI managed to return a list of dicts
            found_sources = research_output
        elif isinstance(research_output, dict) and "results" in research_output: # If it's a dict with results
             found_sources = research_output.get("results", [])
             performed_queries = research_output.get("queries", [])


        updated_raw_data = state.sections_data[section_title].raw_data
        newly_added_urls_this_run = set()

        for source in found_sources:
            if isinstance(source, dict) and source.get("href"):
                url = source["href"]
                # Add source text (snippet or title) to raw_data
                source_text = f"Title: {source.get('title', 'N/A')}\nURL: {url}\nSnippet: {source.get('snippet', 'N/A')}"
                updated_raw_data.append(source_text)
                if url not in state.all_collected_urls:
                    state.all_collected_urls.add(url)
                    newly_added_urls_this_run.add(url)
            elif isinstance(source, str): # If sources are just strings
                updated_raw_data.append(source)


        for pq in performed_queries:
            if isinstance(pq, str) and pq not in state.all_search_queries:
                state.all_search_queries.add(pq)
        
        state.sections_data[section_title].raw_data = updated_raw_data
        state.current_status = f"Research complete for section: {section_title}. Found {len(found_sources)} potential sources."
        state.event_log.append(state.current_status + f" New URLs: {len(newly_added_urls_this_run)}. Queries used: {performed_queries}")

        return {
            "sections_data": state.sections_data,
            "all_collected_urls": state.all_collected_urls,
            "all_search_queries": state.all_search_queries,
            "current_status": state.current_status,
            "event_log": state.event_log
        }

    except Exception as e:
        error_msg = f"Error in research node for '{section_title}': {str(e)}"
        state.event_log.append(error_msg)
        return {"error_message": error_msg, "event_log": state.event_log, "current_status": f"Error in Research for {section_title}"}


def analysis_node(state: ResearchState, agents_cfg: ResearchAgents, tasks_cfg: ResearchTasks) -> Dict[str, Any]:
    """Analyzes researched data for the current section."""
    section_title = state.current_section_title
    if not section_title or section_title not in state.sections_data:
        return {"error_message": "Analysis node: Current section not set or invalid."}

    state.current_status = f"Analyzing data for section: {section_title}"
    state.event_log.append(state.current_status)
    
    section_data_obj = state.sections_data[section_title]
    
    # The analysis agent needs URLs or text. We pass raw_data which might contain URLs or text snippets.
    # The task prompt for analysis agent guides it to use WebFetcherTool for URLs.
    # We need to prepare the research_data argument for the task.
    # For now, let's pass the raw_data strings. The agent/task needs to be smart about it.
    # A better way: research_node should output structured URL list.
    
    # Let's try to extract URLs from raw_data for the analysis task if they are embedded.
    # This is again heuristic.
    urls_for_analysis = []
    for item_text in section_data_obj.raw_data:
        url_match = re.search(r"URL:\s*(https?://[^\s]+)", item_text)
        if url_match:
            urls_for_analysis.append({"title": "Source from research", "href": url_match.group(1), "snippet": item_text[:100]})

    if not urls_for_analysis and section_data_obj.raw_data: # If no explicit URLs, pass snippets
        urls_for_analysis = [{"title": "Data Snippet", "href": "N/A", "snippet": snippet[:200]} for snippet in section_data_obj.raw_data]


    analysis_task_instance = tasks_cfg.analyze_data_task(
        section_title=section_title,
        research_data=urls_for_analysis, # This should be list of dicts with 'href'
        research_questions=[f"Key insights for {section_title} regarding {state.topic}"] # Simplified
    )
    analysis_agent_instance = agents_cfg.analysis_agent()
    crew = Crew(agents=[analysis_agent_instance], tasks=[analysis_task_instance], verbose=1)

    try:
        analysis_output_str = crew.kickoff()
        # Expected output: A dict-like string with 'summary_of_insights', 'gaps_and_conflicts', etc.
        # This is another point where robust JSON prompting for the LLM is crucial.
        
        insights_summary = "Analysis output was not in the expected format or was empty."
        follow_up_questions_from_analysis = [] # Initialize

        if isinstance(analysis_output_str, str) and analysis_output_str.strip():
            try:
                # Attempt to parse if it's a JSON string (as per updated task config)
                analysis_result = json.loads(analysis_output_str)
                insights_summary = analysis_result.get("summary_of_insights", insights_summary)

                # Extract cited sources from analysis and add to global list
                cited_sources_from_analysis = analysis_result.get("cited_sources", [])
                if isinstance(cited_sources_from_analysis, list):
                    for source_dict in cited_sources_from_analysis:
                        if isinstance(source_dict, dict):
                            title = source_dict.get("title", "N/A")
                            href = source_dict.get("href", "N/A")
                            # Format as a string for state.references (List[str])
                            ref_string = f"{title} ({href})"
                            if ref_string not in state.references:
                                state.references.append(ref_string)
                else:
                    state.event_log.append(f"Warning: 'cited_sources' from analysis was not a list for section '{section_title}'.")

                # Extract follow-up questions
                follow_up_questions_from_analysis = analysis_result.get("follow_up_questions", [])
                if not isinstance(follow_up_questions_from_analysis, list):
                    state.event_log.append(f"Warning: 'follow_up_questions' from analysis was not a list for section '{section_title}'. Treating as empty.")
                    follow_up_questions_from_analysis = []

            except json.JSONDecodeError:
                # If not JSON, use the raw string as summary (less ideal)
                insights_summary = analysis_output_str # Keep the raw output for manual inspection if needed
                error_detail = f"Analysis output for '{section_title}' was not valid JSON. Content: {analysis_output_str[:200]}..."
                state.event_log.append(f"Error: {error_detail}")
                # Set error message to ensure it's caught by the router
                state.error_message = f"Error in analysis_node: {error_detail}"
        
        state.sections_data[section_title].summary = insights_summary
        state.sections_data[section_title].follow_up_questions = follow_up_questions_from_analysis

        status_message = f"Analysis complete for section: {section_title}."
        if follow_up_questions_from_analysis:
            status_message += f" Identified {len(follow_up_questions_from_analysis)} follow-up questions."
        state.current_status = status_message
        state.event_log.append(state.current_status)

        return {
            "sections_data": state.sections_data, # This now includes follow_up_questions in the section data
            "current_status": state.current_status,
            "event_log": state.event_log,
            "references": state.references # Pass updated references list
        }
    except Exception as e:
        error_msg = f"Error in analysis node for '{section_title}': {str(e)}"
        state.event_log.append(error_msg)
        return {"error_message": error_msg, "event_log": state.event_log, "current_status": f"Error in Analysis for {section_title}"}


def writing_node(state: ResearchState, agents_cfg: ResearchAgents, tasks_cfg: ResearchTasks) -> Dict[str, Any]:
    """Writes a draft for the current section based on analysis."""
    section_title = state.current_section_title
    if not section_title or section_title not in state.sections_data:
        return {"error_message": "Writing node: Current section not set or invalid."}
    
    section_data_obj = state.sections_data[section_title]
    if not section_data_obj.summary:
        state.event_log.append(f"Skipping writing for '{section_title}' due to missing summary.")
        # Potentially mark this section as problematic or needing re-analysis
        section_data_obj.draft_content = f"Content generation for '{section_title}' skipped due to missing analysis summary."
        return {"sections_data": state.sections_data, "event_log": state.event_log}

    state.current_status = f"Writing draft for section: {section_title}"
    state.event_log.append(state.current_status)

    writing_agent_instance = agents_cfg.writing_agent(state.ollama_model_writing)
    # TODO: Pass actual cited sources if extracted by analysis_node
    write_task_instance = tasks_cfg.write_section_task(
        section_title=section_title,
        section_insights=section_data_obj.summary,
        cited_sources=[] # Placeholder for now
    )
    crew = Crew(agents=[writing_agent_instance], tasks=[write_task_instance], verbose=1)
    
    try:
        draft_content = crew.kickoff()
        if isinstance(draft_content, str) and draft_content.strip():
            state.sections_data[section_title].draft_content = draft_content
            state.current_status = f"Draft complete for section: {section_title}"
        else:
            state.sections_data[section_title].draft_content = f"Draft generation failed or produced empty content for '{section_title}'."
            state.current_status = f"Draft generation failed for section: {section_title}"

        state.event_log.append(state.current_status)
        return {"sections_data": state.sections_data, "current_status": state.current_status, "event_log": state.event_log}
    except Exception as e:
        error_msg = f"Error in writing node for '{section_title}': {str(e)}"
        state.sections_data[section_title].draft_content = f"Error during draft generation for '{section_title}': {str(e)}"
        state.event_log.append(error_msg)
        return {"error_message": error_msg, "sections_data": state.sections_data, "event_log": state.event_log, "current_status": f"Error in Writing for {section_title}"}


def review_node(state: ResearchState, agents_cfg: ResearchAgents, tasks_cfg: ResearchTasks) -> Dict[str, Any]:
    """Reviews the drafted section."""
    section_title = state.current_section_title
    if not section_title or section_title not in state.sections_data:
        return {"error_message": "Review node: Current section not set or invalid."}

    section_data_obj = state.sections_data[section_title]
    if not section_data_obj.draft_content:
        state.event_log.append(f"Skipping review for '{section_title}' due to missing draft content.")
        # No feedback if no draft
        section_data_obj.review_feedback = "No draft content to review."
        return {"sections_data": state.sections_data, "event_log": state.event_log}

    state.current_status = f"Reviewing section: {section_title}"
    state.event_log.append(state.current_status)

    review_agent_instance = agents_cfg.review_agent()
    review_task_instance = tasks_cfg.review_section_task(
        section_title=section_title,
        draft_content=section_data_obj.draft_content
    )
    crew = Crew(agents=[review_agent_instance], tasks=[review_task_instance], verbose=1)

    try:
        feedback = crew.kickoff()
        if isinstance(feedback, str) and feedback.strip():
            state.sections_data[section_title].review_feedback = feedback
            state.current_status = f"Review complete for section: {section_title}."
        else:
            state.sections_data[section_title].review_feedback = "Reviewer provided no actionable feedback or an empty response."
            state.current_status = f"Review for section: {section_title} resulted in empty feedback."
        
        state.event_log.append(state.current_status + f" Feedback: {feedback[:100]}...") # Log snippet of feedback
        return {"sections_data": state.sections_data, "current_status": state.current_status, "event_log": state.event_log}

    except Exception as e:
        error_msg = f"Error in review node for '{section_title}': {str(e)}"
        state.sections_data[section_title].review_feedback = f"Error during review: {str(e)}"
        state.event_log.append(error_msg)
        return {"error_message": error_msg, "sections_data": state.sections_data, "event_log": state.event_log, "current_status": f"Error in Review for {section_title}"}


def revision_node(state: ResearchState, agents_cfg: ResearchAgents, tasks_cfg: ResearchTasks) -> Dict[str, Any]:
    """Revises the section based on feedback. (Effectively re-runs writing with feedback)."""
    section_title = state.current_section_title
    if not section_title or section_title not in state.sections_data:
        return {"error_message": "Revision node: Current section not set or invalid."}

    section_data_obj = state.sections_data[section_title]
    if not section_data_obj.review_feedback or "approved as is" in section_data_obj.review_feedback.lower():
        state.event_log.append(f"Skipping revision for '{section_title}' as it's approved or no feedback given.")
        return {} # No changes to state if no revision needed

    state.current_status = f"Revising section: {section_title} (Attempt: {section_data_obj.revision_attempts + 1})"
    state.event_log.append(state.current_status)

    # Use the Writing Agent again, but provide the original draft and feedback
    # The task needs to be adapted or a new one created for revision.
    # For simplicity, let's re-use write_section_task and prepend feedback to insights.
    
    insights_with_feedback = (
        f"REVISION INSTRUCTIONS:\n{section_data_obj.review_feedback}\n\n"
        f"ORIGINAL INSIGHTS FOR CONTEXT:\n{section_data_obj.summary}\n\n"
        f"PREVIOUS DRAFT (for reference, rewrite based on feedback and insights):\n{section_data_obj.draft_content[:500]}...\n\n" # Send snippet of prev draft
        "Please provide a new, revised draft of the section."
    )

    writing_agent_instance = agents_cfg.writing_agent(state.ollama_model_writing)
    # Re-using write_section_task, but the description is now for revision
    revise_task_instance = tasks_cfg.write_section_task( # This might need a dedicated revision task for better prompting
        section_title=section_title,
        section_insights=insights_with_feedback, # This is now a combined prompt
        cited_sources=[] # Placeholder
    )
    # Update task description for revision context
    revise_task_instance.description = (
        f"Revise the report section titled '{section_title}'.\n"
        f"Incorporate the following feedback: \n{section_data_obj.review_feedback}\n\n"
        f"The original summary of insights for this section was:\n{section_data_obj.summary}\n\n"
        f"The previous draft started with:\n{section_data_obj.draft_content[:500]}...\n\n"
        "Your goal is to produce an improved version of the section that addresses the feedback. "
        "Use Markdown for formatting."
    )


    crew = Crew(agents=[writing_agent_instance], tasks=[revise_task_instance], verbose=1)
    
    try:
        revised_draft_content = crew.kickoff()
        if isinstance(revised_draft_content, str) and revised_draft_content.strip():
            state.sections_data[section_title].draft_content = revised_draft_content
            state.sections_data[section_title].revision_attempts += 1
            state.current_status = f"Revision complete for section: {section_title}"
        else:
            state.current_status = f"Revision failed or produced empty content for section: {section_title}"
            # Optionally, keep the old draft or mark as failed revision
        
        state.event_log.append(state.current_status)
        return {"sections_data": state.sections_data, "current_status": state.current_status, "event_log": state.event_log}

    except Exception as e:
        error_msg = f"Error in revision node for '{section_title}': {str(e)}"
        # Keep previous draft if revision fails
        state.event_log.append(error_msg)
        return {"error_message": error_msg, "event_log": state.event_log, "current_status": f"Error in Revision for {section_title}"}


# --- Conditional Edges ---

def should_continue_overall_loop(state: ResearchState) -> Literal["process_next_section", "compile_report", "handle_error"]:
    # Allow specific node errors to be handled by their local routers first if error_message is set
    # Added "planner_node" to the exclusion list as planner errors are handled by its own router.
    if state.error_message and not any(tag in state.error_message for tag in ["analysis_node", "review_node", "revision_node", "planner_node"]):
        state.event_log.append(f"Overall Loop: Unhandled error detected: {state.error_message}. Halting.")
        return "handle_error"
    
    if state.main_loop_iterations >= state.max_main_loop_iterations:
        state.event_log.append(f"Max overall iterations ({state.max_main_loop_iterations}) reached. Moving to compile report.")
        return "compile_report"

    next_section_to_process = None
    if state.initial_outline:
        for sec_title in state.initial_outline:
            # Ensure section exists in sections_data; it should if planner worked correctly
            if sec_title not in state.sections_data:
                state.event_log.append(f"Warning: Section '{sec_title}' from outline not found in sections_data. Skipping.")
                continue

            sec_data = state.sections_data[sec_title]
            # Process if it's a content section (not intro/conclu/refs which are handled differently)
            # AND it doesn't have a draft yet.
            # The recursive handling (follow-up Qs) is managed by `decide_after_analysis`.
            # This loop focuses on picking up entirely new, un-drafted sections.
            if sec_title.lower() not in ["introduction", "conclusion", "references"] and not sec_data.draft_content:
                # Check if it's already being processed due to recursion. If so, don't pick it as "next new".
                # A section is "new" if its recursion depth is 0.
                if sec_data.recursion_depth == 0: # Check only recursion_depth, follow_up_questions might be from a previous failed run
                    next_section_to_process = sec_title
                    break
    
    if next_section_to_process:
        state.current_section_title = next_section_to_process
        # Ensure recursion_depth and follow_up_questions are reset for a truly new section pass
        state.sections_data[next_section_to_process].recursion_depth = 0
        state.sections_data[next_section_to_process].follow_up_questions = []
        state.main_loop_iterations += 1
        state.event_log.append(f"Overall Loop: Processing new section '{next_section_to_process}'. Iteration: {state.main_loop_iterations}.")
        return "process_next_section"
    else:
        state.event_log.append("Overall Loop: No new sections to process or all sections drafted/recursed. Moving to compile report.")
        return "compile_report"


def decide_to_revise_or_continue(state: ResearchState) -> Literal["revise_section", "continue_to_next_main_task", "handle_error"]:
    section_title = state.current_section_title
    if not section_title or section_title not in state.sections_data:
        state.event_log.append(f"Error: current_section_title ('{section_title}') missing or invalid in decide_to_revise_or_continue.")
        # Ensure error_message is set to trigger halt if not already set by a more specific error
        if not state.error_message: # Avoid overwriting a more specific error
            state.error_message = f"Critical error: current_section_title ('{section_title}') not set or invalid for review decision."
        return "handle_error"

    # Check for errors not originating from review/revise itself.
    # Errors from review_node or revision_node are part of their specific loop/feedback mechanism.
    if state.error_message and not any(err_tag in state.error_message.lower() for err_tag in ["review node", "revision node", "review_section_task", "revision_task"]): # more robust check
        state.event_log.append(f"Error detected before revision decision for section '{section_title}': {state.error_message}. Halting.")
        return "handle_error"

    section_data_obj = state.sections_data[section_title]
    feedback = section_data_obj.review_feedback
    
    if feedback and "approved as is" not in feedback.lower() and section_data_obj.revision_attempts < state.max_revision_cycles_per_section:
        state.event_log.append(f"Revision needed for '{section_title}'. Attempt {section_data_obj.revision_attempts + 1}")
        return "revise_section"
    else:
        if section_data_obj.revision_attempts >= state.max_revision_cycles_per_section:
            state.event_log.append(f"Max revisions reached for '{section_title}'. Continuing.")
        else:
            state.event_log.append(f"Section '{section_title}' approved or no actionable feedback. Continuing.")
        # This edge means we are done with this section's research-analyze-write-review cycle
        return "continue_to_next_main_task" # This will loop back to should_continue_overall_loop


def decide_after_analysis(state: ResearchState) -> Literal["researcher", "writer", "handle_error"]:
    """
    Decides the next step after the analysis node.
    Routes to 'researcher' for recursion if follow-up questions exist and depth limit not reached.
    Otherwise, proceeds to 'writer'.
    """
    section_title = state.current_section_title
    if not section_title or section_title not in state.sections_data:
        error_msg = f"Error in decide_after_analysis: current_section_title ('{section_title}') is missing or invalid."
        state.event_log.append(error_msg)
        state.error_message = error_msg # Ensure this is set for global error handling
        return "handle_error"

    # Check for pre-existing errors, especially from analysis_node
    if state.error_message and "analysis_node" in state.error_message:
        # Error already logged by analysis_node, just route to handle_error
        return "handle_error"

    section_data_obj = state.sections_data[section_title]

    if section_data_obj.follow_up_questions and \
       section_data_obj.recursion_depth < state.max_recursion_depth_per_section:

        state.sections_data[section_title].recursion_depth += 1
        status_msg = f"Recursion {section_data_obj.recursion_depth}/{state.max_recursion_depth_per_section}: Starting deeper research for section '{section_title}' on {len(section_data_obj.follow_up_questions)} new questions."
        state.current_status = status_msg
        state.event_log.append(status_msg)
        state.event_log.append(f"Recursive questions for '{section_title}': {section_data_obj.follow_up_questions}")
        return "researcher"
    else:
        if section_data_obj.follow_up_questions:
            status_msg = f"Max recursion depth ({state.max_recursion_depth_per_section}) reached for section '{section_title}', or no more recursion allowed. {len(section_data_obj.follow_up_questions)} follow-up questions will be cleared. Proceeding to writer."
            state.current_status = status_msg
            state.event_log.append(status_msg)
            state.sections_data[section_title].follow_up_questions = [] # Clear questions
        else:
            status_msg = f"No follow-up questions for section '{section_title}'. Proceeding to writer."
            state.current_status = status_msg
            state.event_log.append(status_msg)

        # Optionally reset recursion_depth here if desired when moving to writer.
        # For now, depth is kept to show its final state for that section's processing cycle.
        # state.sections_data[section_title].recursion_depth = 0
        return "writer"


# --- Final Report Compilation Node ---
from report_assembler import assemble_report_markdown, generate_placeholder_intro_conclusion

def compile_report_node(state: ResearchState) -> Dict[str, Any]:
    """Assembles the final report from all drafted sections."""
    state.current_status = "Compiling final report..."
    state.event_log.append(state.current_status)

    # Use placeholder intro/conclusion for now. These could be agent-generated too.
    intro_content, conclusion_content = generate_placeholder_intro_conclusion(state.topic)
    
    # Ensure Introduction and Conclusion sections exist in sections_data for assembly,
    # even if they are just placeholders or to be filled by these generated ones.
    if "Introduction" not in state.sections_data:
        state.sections_data["Introduction"] = SectionData(title="Introduction", draft_content=intro_content, raw_data=[])
    else: # If it exists but has no content, fill it
        if not state.sections_data["Introduction"].draft_content:
             state.sections_data["Introduction"].draft_content = intro_content
             
    if "Conclusion" not in state.sections_data:
        state.sections_data["Conclusion"] = SectionData(title="Conclusion", draft_content=conclusion_content, raw_data=[])
    else: # If it exists but has no content, fill it
        if not state.sections_data["Conclusion"].draft_content:
            state.sections_data["Conclusion"].draft_content = conclusion_content

    # References section might be empty if not explicitly populated
    if "References" not in state.sections_data:
        state.sections_data["References"] = SectionData(title="References", draft_content="_No specific references were compiled for this report version._", raw_data=[])
    elif not state.sections_data["References"].draft_content: # if exists but empty
        state.sections_data["References"].draft_content = "_No specific references were compiled for this report version._"


    # Reorder sections_data according to initial_outline for assembly
    ordered_sections_data = {}
    if state.initial_outline:
        for title in state.initial_outline:
            if title in state.sections_data:
                ordered_sections_data[title] = state.sections_data[title]
        # Add any sections in sections_data not in initial_outline (e.g. if added dynamically)
        for title, data in state.sections_data.items():
            if title not in ordered_sections_data:
                 ordered_sections_data[title] = data # Add at the end
    else: # Fallback if no outline
        ordered_sections_data = state.sections_data


    final_report = assemble_report_markdown(
        report_title=f"Research Report: {state.topic}",
        introduction=state.sections_data.get("Introduction", SectionData(title="Introduction", draft_content=intro_content, raw_data=[])).draft_content or intro_content,
        sections_data=ordered_sections_data, # Pass the main content sections
        conclusion=state.sections_data.get("Conclusion", SectionData(title="Conclusion", draft_content=conclusion_content, raw_data=[])).draft_content or conclusion_content,
        references_list=state.references, # state.references should be populated by analysis/writing agents
        topic=state.topic # Added topic for the subtitle in the report
    )
    
    state.final_report_md = final_report
    state.current_status = "Report compilation complete."
    state.event_log.append(state.current_status)
    
    return {"final_report_md": final_report, "current_status": state.current_status, "event_log": state.event_log}

def error_handling_node(state: ResearchState) -> Dict[str, Any]:
    """Handles errors and stops the graph."""
    state.current_status = f"Error occurred: {state.error_message}. Halting process."
    state.event_log.append(state.current_status)
    # Potentially save state or log more details
    return {"current_status": state.current_status, "event_log": state.event_log}


# --- Build the Graph ---
def build_graph(agents_cfg: ResearchAgents, tasks_cfg: ResearchTasks):
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("planner", lambda s: planning_node(s, agents_cfg, tasks_cfg))
    workflow.add_node("researcher", lambda s: research_node(s, agents_cfg, tasks_cfg))
    workflow.add_node("analyzer", lambda s: analysis_node(s, agents_cfg, tasks_cfg))
    workflow.add_node("writer", lambda s: writing_node(s, agents_cfg, tasks_cfg))
    workflow.add_node("reviewer", lambda s: review_node(s, agents_cfg, tasks_cfg))
    workflow.add_node("reviser", lambda s: revision_node(s, agents_cfg, tasks_cfg))
    workflow.add_node("compiler", compile_report_node) # No agent/task cfg needed for this one
    workflow.add_node("error_handler", error_handling_node)

    # Set entry and exit points
    workflow.set_entry_point("planner")
    workflow.add_edge("compiler", END) # Report compiled, end of process
    workflow.add_edge("error_handler", END) # Error, end of process

    # Define workflow logic (edges)
    # After planning, decide if we start processing sections or if planning failed
    workflow.add_conditional_edges(
        "planner",
        lambda s: "process_next_section" if s.initial_outline and not s.error_message else "handle_error",
        {
            "process_next_section": "researcher", # This should go to the overall loop condition first
            "handle_error": "error_handler"
        }
    )
    
    # This is the main loop controller. It's called after planning, and after a section is fully processed.
    # For the first run after 'planner', current_section_title will be None, so should_continue_overall_loop will set it.
    # This node itself doesn't perform an action, it's a routing hub.
    # To implement this, we need a dummy node or to make should_continue_overall_loop a node itself.
    # Let's make it simpler: planner directly goes to researcher IF outline is good.
    # Then the loop is managed after a section is "done" (written/reviewed/revised).

    # Simplified flow: Plan -> Research -> Analyze -> Write -> Review -> (Revise loop) -> (Next Section loop) -> Compile
    
    workflow.add_conditional_edges(
        "planner", # Source node
        # This function decides the next step after planning
        lambda state: "researcher" if state.initial_outline and not state.error_message and state.initial_outline[0] not in ["Introduction", "Conclusion", "References"] else ("compiler" if state.initial_outline else "handle_error"),
        {
            "researcher": "researcher", # Start with the first actual content section
            "compiler": "compiler",     # If outline is empty or only has boilerplate, go to compile
            "handle_error": "error_handler"
        }
    )
    # The first section_title needs to be set by the conditional edge logic or the first node.
    # Let's refine the planner conditional edge to set the first current_section_title.
    
    def planner_to_researcher_router(state: ResearchState):
        if state.error_message: return "handle_error"
        if not state.initial_outline: return "compiler" # Or error

        first_content_section = None
        for title in state.initial_outline:
            # This condition might need to be more robust if Intro/Conclusion are also agent-generated
            if title.lower() not in ["introduction", "conclusion", "references"]:
                first_content_section = title
                break
        
        if first_content_section:
            state.current_section_title = first_content_section # Set current section for the first run
            state.main_loop_iterations = 1 # Start count
            return "researcher"
        else: # Only intro/conclusion/refs in outline
            return "compiler"

    workflow.add_conditional_edges("planner", planner_to_researcher_router, {
        "researcher": "researcher",
        "compiler": "compiler",
        "handle_error": "error_handler"
    })


    # Core section processing flow
    workflow.add_edge("researcher", "analyzer")
    # workflow.add_edge("analyzer", "writer") # Replaced by conditional edge below

    # New conditional router after analysis to decide on recursion or writing
    workflow.add_node("decide_after_analysis_router", lambda state: state) # Dummy node for routing logic

    # Analyzer always goes to the decision router
    workflow.add_conditional_edges(
        "analyzer",
        lambda state: "decide_after_analysis_router",
        {"decide_after_analysis_router": "decide_after_analysis_router"}
    )

    # Conditional edges from the decision router
    workflow.add_conditional_edges(
        "decide_after_analysis_router",
        decide_after_analysis, # New conditional function to handle recursion logic
        {
            "researcher": "researcher", # Loop back to researcher for recursive depth
            "writer": "writer",         # Proceed to writer if no recursion needed
            "handle_error": "error_handler"
        }
    )

    workflow.add_edge("writer", "reviewer")

    # Review and Revision Loop
    workflow.add_conditional_edges(
        "reviewer",
        decide_to_revise_or_continue, # This function determines next step
        {
            "revise_section": "reviser",
            "continue_to_next_main_task": "research_loop_controller", # Go to check if more sections or compile
            "handle_error": "error_handler"
        }
    )
    workflow.add_edge("reviser", "reviewer") # After revision, review again (or go to next section if max revisions hit - handled by decide_to_revise)


    # Node to control the main loop (process next section or compile)
    # This node is reached after a section is fully processed (approved or max revisions)
    workflow.add_node("research_loop_controller", lambda state: state) # Dummy node for routing
    workflow.add_conditional_edges(
        "research_loop_controller",
        should_continue_overall_loop, # This function decides
        {
            "process_next_section": "researcher", # current_section_title is updated by should_continue
            "compile_report": "compiler",
            "handle_error": "error_handler"
        }
    )
    
    # memory = MemorySaver() # In-memory checkpointing
    # app = workflow.compile(checkpointer=memory)
    app = workflow.compile() # Compile without checkpointing for now for simplicity in Streamlit
    return app

if __name__ == '__main__':
    # This is for local testing of the graph structure, not a full run.
    # A full run needs Streamlit UI to provide topic and model choices.
    print("Workflow graph definition loaded.")
    # To visualize (if you have graphviz and relevant Python packages):
    # from IPython.display import Image, display
    # try:
    #     app = build_graph(ResearchAgents(), ResearchTasks(ResearchAgents())) # Dummy agents/tasks
    #     img_data = app.get_graph().draw_mermaid_png()
    #     with open("workflow_graph.png", "wb") as f:
    #         f.write(img_data)
    #     print("Graph image saved to workflow_graph.png")
    # except Exception as e:
    #     print(f"Could not draw graph: {e}")
    pass
