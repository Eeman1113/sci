import streamlit as st
import json
from dotenv import load_dotenv
import os
import time
from datetime import datetime

# Load environment variables if any (e.g., for API keys, though Ollama is local)
load_dotenv()

# Project modules
from shared_state import ResearchState, SectionData
from agents_config import ResearchAgents
from tasks_config import ResearchTasks
from workflow_graph import build_graph # The compiled LangGraph application

# --- Streamlit UI Configuration ---
st.set_page_config(layout="wide", page_title="Multi-Agent Research System")

st.title("📚 Autonomous Multi-Agent Research System")
st.markdown("""
    Enter a research topic, and the system will attempt to generate a structured report.
    This system uses multiple AI agents (powered by local Ollama models via CrewAI) 
    orchestrated by LangGraph to plan, research, analyze, write, and review content.
""")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    ollama_model_general = st.text_input(
        "Ollama Model (General Purpose)", 
        value="llama3", 
        help="Model for planning, research, analysis, review (e.g., 'llama3', 'mistral')"
    )
    ollama_model_writing = st.text_input(
        "Ollama Model (Writing)", 
        value="llama3", 
        help="Model for drafting content (e.g., 'llama3', 'gpt-4o-mini' if using a compatible Ollama setup or other LLM service)"
    )

    st.subheader("Research Parameters")
    max_searches_per_section = st.slider("Max Searches per Section", 1, 10, 3)
    # max_sources_per_search = st.slider("Max Sources per Search Result", 1, 5, 3) # This is handled in DuckDuckGoTool args
    max_revision_cycles = st.slider("Max Revision Cycles per Section", 0, 5, 1)
    # max_total_iterations = st.slider("Max Overall Loop Iterations (Safety)", 5, 20, 10) # For main loop

    research_topic = st.text_area("🔬 Enter Research Topic Here:", height=100, placeholder="e.g., The Impact of AI on Renewable Energy")

    start_button = st.button("🚀 Start Research Process")

    st.markdown("---")
    st.markdown("ℹ️ **Note:** Ensure Ollama is running locally with the specified models pulled.")
    st.markdown("Example: `ollama pull llama3`")


# --- Main Area for Output ---
# Placeholders for results and progress
progress_area = st.container()
results_area = st.container()
log_area = st.expander("📜 View Detailed Event Log", expanded=False)

if 'research_state' not in st.session_state:
    st.session_state.research_state = None
if 'graph_app' not in st.session_state:
    st.session_state.graph_app = None
if 'run_active' not in st.session_state:
    st.session_state.run_active = False


def stream_graph_events(graph_app, initial_state_dict):
    """Streams events from the LangGraph invocation."""
    # For streaming, LangGraph's `stream` method is used.
    # It yields intermediate states or events.
    # The `ResearchState` object needs to be converted to a dict for `invoke` or `stream`.
    # However, the nodes in our graph expect ResearchState objects.
    # This means we need to handle the dict-to-object conversion if using stream directly with dicts.
    # For simplicity with `invoke`, we'll update UI after each major step if possible,
    # or show final output. A true streaming UI with LangGraph is more involved.

    # The current graph is compiled without a checkpointer that supports streaming updates easily to UI.
    # So, we'll use `invoke` and update UI based on the final state or periodically if we add callbacks.
    # For a more responsive UI, LangGraph's streaming and checkpointer features would be key.

    # Let's simulate progress updates by inspecting the state if we were to run it step-by-step
    # or by having nodes update a shared status message that Streamlit can poll.
    # With `invoke`, it's a blocking call.
    
    # For now, we'll run `invoke` and then display the results.
    # True streaming of intermediate agent thoughts/actions requires deeper integration.
    
    st.session_state.run_active = True
    final_state_dict = {}
    try:
        # The `invoke` method takes a dictionary as input.
        # Our nodes are written to expect a ResearchState object.
        # LangGraph handles the conversion if the input to StateGraph is a Pydantic model.
        with st.spinner("Research process initiated. This may take a while..."):
            start_time = time.time()
            
            # The graph's `invoke` method will run the entire flow.
            # We need to pass the initial configuration into the state.
            # The `ResearchState` Pydantic model will be the input type for the graph.
            
            # The `input` to invoke should match the `ResearchState` structure.
            # Our graph nodes are defined as `lambda s: node_function(s, agents_cfg, tasks_cfg)`.
            # LangGraph will pass the state `s` to these lambdas.
            # The initial input to `graph_app.invoke` should be the initial state.
            
            # `initial_state_dict` is already a `ResearchState` object.
            # LangGraph's `StateGraph(ResearchState)` should handle Pydantic models correctly.
            
            for chunk in graph_app.stream(initial_state_dict):
                # `chunk` will be a dictionary where keys are node names
                # and values are the output of that node (updates to the state).
                # We need to merge these updates into our main state object to show progress.
                
                # The `stream` method yields the state *after* each node execution.
                # The key in the chunk is the node that just ran.
                # The value is its output (which is a dict of updates to ResearchState).
                
                # Let's get the latest full state from the stream.
                # The last item in the chunk is usually the most complete state after a node.
                if chunk:
                    latest_node_name = list(chunk.keys())[-1]
                    latest_node_output = chunk[latest_node_name] # This is the partial update
                    
                    # Update our Streamlit display based on this partial update or the implied full state
                    # This is complex because `latest_node_output` is just the *change*.
                    # To get the full state, we'd need a checkpointer or to reconstruct it.
                    
                    # For simplicity in this example, let's assume `latest_node_output` might contain
                    # enough info, or we update a running log.
                    # A more robust way is to use a checkpointer and load the state.
                    
                    # Let's update the display with current status if available in the output
                    if isinstance(latest_node_output, dict):
                        if "current_status" in latest_node_output:
                            progress_area.info(f"Status: {latest_node_output['current_status']}")
                        if "event_log" in latest_node_output:
                            # Append to a running log in st.session_state
                            if 'running_event_log' not in st.session_state:
                                st.session_state.running_event_log = []
                            if isinstance(latest_node_output['event_log'], list):
                                # This event_log from node output is the full log up to that point
                                st.session_state.running_event_log = latest_node_output['event_log'] 
                            elif isinstance(latest_node_output['event_log'], str): # if it's a single new event
                                st.session_state.running_event_log.append(latest_node_output['event_log'])
                            
                            with log_area:
                                st.empty() # Clear previous log
                                for log_entry in reversed(st.session_state.running_event_log[-20:]): # Show last 20
                                    st.text(log_entry)
                        
                        # Store the final state by accumulating updates (simplified)
                        # This is not a perfect way to get the final state from stream without checkpointer.
                        # The last yielded chunk for the END node would be the final state.
                        # Or, the output of the node connected to END.
                        if latest_node_name == END or latest_node_name == "compiler" or latest_node_name == "error_handler":
                             # If the graph is more complex, the actual final state might be in the value of the node that ran.
                             # For StateGraph, the value of the node is the update to the state.
                             # The stream output is { node_name: state_after_node_ran }
                             # So, latest_node_output here *is* the state after that node ran.
                             final_state_dict = latest_node_output # This should be the full state.

            end_time = time.time()
            progress_area.success(f"Research process finished in {end_time - start_time:.2f} seconds.")
            
            # After stream finishes, final_state_dict should hold the state from the last relevant node.
            # We need to ensure it's the complete ResearchState.
            # If using `invoke`, it directly returns the final state.
            # With `stream`, the last yielded item from the "compiler" or "error_handler" node
            # will be the final state.

            # Let's refine: the stream yields `Dict[str, Any]`, where the key is the node name
            # and the value is the *entire state object* after that node has executed, if the graph
            # is defined with a Pydantic model as its state.
            
            # So, `final_state_dict` should be the state after the "compiler" or "error_handler" node.
            # We need to find that specific chunk.
            # A simpler way for now: run invoke if streaming is too complex for this UI update model.
            # Let's switch to invoke for simplicity of getting final state, and accept less granular progress.

            # Reverting to invoke for simpler final state retrieval for this example
            # For true streaming UI, `graph_app.stream` needs careful handling of its output format.
            
            # final_state_obj = graph_app.invoke(initial_state_dict) # This returns the full final ResearchState object
            # st.session_state.research_state = final_state_obj

            # If using stream, the last element of the stream corresponding to the END node
            # or the node leading to END will contain the final state.
            # The variable `final_state_dict` should be the state after the "compiler" or "error_handler" node.
            # Let's assume `final_state_dict` correctly captured this.
            st.session_state.research_state = ResearchState(**final_state_dict) if final_state_dict else None


    except Exception as e:
        st.error(f"An error occurred during the research process: {str(e)}")
        st.exception(e)
        st.session_state.research_state = ResearchState(
            topic=initial_state_dict.topic, # Use initial topic
            error_message=str(e),
            current_status="Critical error occurred."
        )
    finally:
        st.session_state.run_active = False
        st.rerun() # Rerun to update UI based on new session_state


if start_button and not st.session_state.run_active:
    if not research_topic.strip():
        st.warning("Please enter a research topic.")
    else:
        # Initialize agents and tasks (they are lightweight)
        agents = ResearchAgents(ollama_model_general=ollama_model_general)
        tasks = ResearchTasks(agents=agents)
        
        # Build or get the graph app
        # We can cache this, but for simplicity, rebuild if not in session state
        # or if config changes (though we are not handling config changes mid-session here)
        if st.session_state.graph_app is None:
            st.session_state.graph_app = build_graph(agents, tasks)

        # Prepare initial state
        initial_state = ResearchState(
            topic=research_topic,
            ollama_model_general=ollama_model_general,
            ollama_model_writing=ollama_model_writing,
            max_searches_per_section=max_searches_per_section,
            max_revision_cycles_per_section=max_revision_cycles,
            # max_main_loop_iterations=max_total_iterations, # Set this in ResearchState defaults or here
            current_status="Initializing..."
        )
        st.session_state.research_state = initial_state # Store initial state
        st.session_state.running_event_log = [f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Process initiated for topic: {research_topic}"]


        # Clear previous results display areas
        progress_area.empty()
        results_area.empty()
        with log_area:
            st.empty()
        
        # Call the streaming function (which will use invoke for now)
        stream_graph_events(st.session_state.graph_app, initial_state)
        # The rerun will happen inside stream_graph_events after completion/error

elif st.session_state.run_active:
    st.info("Research process is currently active. Please wait...")
    # Potentially add a cancel button here (more complex)

# Display results if available
if st.session_state.research_state:
    current_state: ResearchState = st.session_state.research_state
    
    if current_state.current_status:
        if "error" in current_state.current_status.lower() or current_state.error_message:
            progress_area.error(f"Status: {current_state.current_status} | Error: {current_state.error_message}")
        else:
            progress_area.info(f"Status: {current_state.current_status}")

    if hasattr(st.session_state, 'running_event_log') and st.session_state.running_event_log:
        with log_area: # Ensure log is updated even if process ended elsewhere
            st.empty()
            for log_entry in reversed(st.session_state.running_event_log[-30:]): # Show last 30
                st.text(log_entry)
    elif current_state.event_log: # Fallback to state's event log
         with log_area:
            st.empty()
            for log_entry in reversed(current_state.event_log[-30:]):
                st.text(log_entry)


    if current_state.final_report_md:
        results_area.subheader("📄 Generated Report")
        results_area.markdown(current_state.final_report_md, unsafe_allow_html=True) # Allow HTML for anchors
        
        results_area.download_button(
            label="⬇️ Download Report (Markdown)",
            data=current_state.final_report_md,
            file_name=f"research_report_{current_state.topic[:20].replace(' ', '_')}.md",
            mime="text/markdown",
        )
    elif not st.session_state.run_active : # If not running and no report, show message
        if not current_state.error_message : # if no specific error, but no report
             results_area.markdown("The research process completed, but no final report was generated. Check the logs for details.")
        else:
             results_area.error(f"Report could not be generated due to an error: {current_state.error_message}")


    with st.expander("🔬 View Final State Object (for debugging)"):
        st.json(current_state.model_dump_json(indent=2) if isinstance(current_state, ResearchState) else str(current_state))

