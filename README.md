# Autonomous AI Research System (SciAgents)

## Overview

SciAgents is a sophisticated multi-agent AI system designed to automate the process of conducting in-depth research on complex topics. Leveraging cutting-edge technologies like CrewAI for agent orchestration and LangGraph for state management, this system can plan research, gather information from the web, analyze data, synthesize insights, write detailed reports, and even perform recursive exploration for deeper understanding.

The primary goal of SciAgents is to produce comprehensive, well-structured research reports (e.g., 30-50 pages) complete with citations and a clear narrative flow, mimicking the output of a human research team.

## Features

*   **Dynamic Outlining:** Generates a structured outline for the research topic.
*   **Targeted Web Research:** Conducts focused web searches using DuckDuckGo to find relevant articles, data, and sources.
*   **Content Fetching & Analysis:** Fetches full content from URLs and performs critical analysis to synthesize key insights, identify gaps, and extract citable sources.
*   **Recursive Exploration:** If initial analysis is insufficient or new questions arise, the system can trigger recursive research loops to delve deeper into specific sub-topics.
*   **Automated Report Writing:** Drafts comprehensive sections for the report based on synthesized insights.
*   **Review and Revision Cycle:** Includes a review process to evaluate drafted sections for clarity, coherence, accuracy, and completeness, followed by a revision cycle.
*   **Markdown Report Generation:** Assembles the final report in Markdown format, including a title page, table of contents, and formatted references.
*   **State Management:** Utilizes LangGraph to manage the complex state of the research process, allowing for robust error handling and conditional logic.
*   **Customizable Agents:** Employs specialized AI agents (Planner, Researcher, Analyst, Writer, Reviewer) built with CrewAI, each with distinct roles and goals.
*   **Extensible Toolset:** Agents are equipped with custom tools (e.g., web search, content fetching) that can be expanded.
*   **Configurable Parameters:** Key operational parameters like recursion depth, number of search results, and revision cycles can be configured.

## System Architecture

The system is built upon a graph-based workflow orchestrated by LangGraph. Each node in the graph represents a specific stage in the research process, executed by one or more AI agents.

1.  **Planning Node:** The Planner Agent devises a research outline.
2.  **Research Node:** The Research Agent gathers information based on the outline or follow-up questions.
3.  **Analysis Node:** The Analysis Agent processes the gathered data, extracts insights, identifies follow-up questions, and collects references.
4.  **Decision Node (after Analysis):** Determines if recursive research is needed based on follow-up questions and recursion depth limits.
    *   If **recursion needed**: Routes back to the Research Node with new questions.
    *   If **no recursion needed**: Proceeds to the Writing Node.
5.  **Writing Node:** The Writing Agent drafts a section of the report based on the analysis.
6.  **Review Node:** The Review Agent evaluates the drafted section.
7.  **Decision Node (after Review):** Determines if revisions are needed.
    *   If **revision needed**: Routes to the Revision Node.
    *   If **no revision needed**: Proceeds to the next main task (e.g., processing the next section or compiling the report).
8.  **Revision Node:** The Writing Agent revises the draft based on feedback. Routes back to Review Node.
9.  **Main Loop Controller:** Decides whether to process the next section or move to compilation.
10. **Compilation Node:** Assembles all drafted sections, introduction, conclusion, and references into the final Markdown report.
11. **Error Handling Node:** Manages any errors that occur during the process.

## Core Technologies

*   **CrewAI:** For creating and managing autonomous AI agents with specific roles and goals.
*   **LangGraph:** For building robust, stateful multi-agent applications with cyclical graph workflows.
*   **LangChain:** Provides core components for LLM interaction, tool creation, and prompt management.
*   **Ollama:** Used for running local LLMs (e.g., Llama 3, Mistral) that power the agents.
*   **DuckDuckGo Search:** For web search capabilities.
*   **BeautifulSoup & Requests:** For fetching and parsing web page content.
*   **Streamlit (Implied):** For user interface and interaction (based on typical project structure for such systems).

## Project Structure (Illustrative)

```
.
├── app.py                  # Main application (e.g., Streamlit UI)
├── workflow_graph.py       # Defines the LangGraph research workflow and nodes
├── shared_state.py         # Defines the Pydantic models for graph state (ResearchState, SectionData)
├── agents_config.py        # Configuration for CrewAI agents (roles, goals, backstories)
├── tasks_config.py         # Configuration for CrewAI tasks (descriptions, expected outputs)
├── custom_tools.py         # Custom tools for agents (e.g., WebPageContentFetcherTool)
├── report_assembler.py     # Logic for assembling the final Markdown report
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (e.g., API keys, model names)
└── README.md               # This file
```

## Setup and Usage

1.  **Prerequisites:**
    *   Python 3.8+
    *   Ollama installed and running with desired models (e.g., `ollama pull llama3`, `ollama pull mistral`). Refer to [Ollama's official website](https://ollama.ai/) for installation instructions.

2.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Environment Variables:**
    *   Create a `.env` file in the project root.
    *   Specify the Ollama models to be used, if different from defaults:
        ```env
        OLLAMA_MODEL_GENERAL=llama3
        OLLAMA_MODEL_WRITING=llama3
        # Add any other necessary API keys or configurations
        ```

6.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```
    Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

7.  **Using the System:**
    *   Enter the research topic in the Streamlit UI.
    *   Configure parameters like max recursion depth, search limits, etc., via the sidebar.
    *   Start the research process.
    *   Monitor the progress via status updates and event logs displayed in the UI.
    *   Once complete, the generated report will be available for download or viewing.

## Customization

*   **Agents:** Modify roles, goals, backstories, and LLMs in `agents_config.py`.
*   **Tasks:** Adjust task descriptions and expected outputs in `tasks_config.py`.
*   **Tools:** Add or modify tools in `custom_tools.py`.
*   **Workflow:** Alter the graph structure, nodes, or conditional logic in `workflow_graph.py`.
*   **Prompts:** Refine agent prompts for better performance or different output styles.
*   **LLMs:** Experiment with different local LLMs supported by Ollama or integrate other LLM providers.

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

Please ensure your code adheres to good practices and includes relevant documentation or tests where applicable.

## License

This project is licensed under the MIT License - see the LICENSE file for details (assuming one would be added).
```
