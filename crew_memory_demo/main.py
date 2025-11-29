import os
from pathlib import Path

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource


# 1. Load environment variables from .env
load_dotenv()

# 2. Configure CrewAI storage to use the current directory
PROJECT_ROOT = Path.cwd()
os.environ.setdefault("CREWAI_STORAGE_DIR", str(PROJECT_ROOT))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def show_memory_storage_location():
    """
    Print where CrewAI will store its memory and knowledge.
    This is now set to the current project directory.
    """
    storage_dir = os.getenv("CREWAI_STORAGE_DIR", str(PROJECT_ROOT))
    print(f"\n[CrewAI] Memory & knowledge storage path:\n  {storage_dir}\n")


def save_plan_to_markdown(user_query: str, result: str, output_filename: str = "learning_plan.md"):
    """
    Save the final learning plan to a markdown file in the current directory.
    """
    path = PROJECT_ROOT / output_filename
    header = (
        "# Learning Plan\n\n"
        f"**User query:** {user_query}\n\n"
        "---\n\n"
    )
    path.write_text(header + str(result), encoding="utf-8")
    print(f"\n[Saved] Learning plan written to: {path}\n")

def build_crew():
    # 3. Define a simple user-profile knowledge source (declarative memory)
    user_profile_text = """
    The user is a software engineer with experience in backend systems.
    They are comfortable with Python and interested in Generative AI,
    agent frameworks, and production-grade LLMOps.
    The user prefers hands-on, code-first learning with clear architecture diagrams.
    """

    user_profile_knowledge = StringKnowledgeSource(
        content=user_profile_text.strip(),
        metadata={"source": "user_profile", "type": "declarative_knowledge"},
    )


    preference_researcher = Agent(
        role="Preference Researcher",
        goal=(
            "Incrementally build an accurate, structured model of the user's "
            "learning preferences and goals over time."
        ),
        backstory=(
            "You specialize in interviewing developers, extracting their goals, "
            "constraints, and preferences, and turning that into clear summaries "
            "that can be reused later."
        ),
        allow_delegation=False,
        verbose=True,
        llm=OPENAI_MODEL,
    )
    
    learning_planner = Agent(
        role="Learning Planner",
        goal=(
            "Design the next learning steps for the user by combining their "
            "stored preference history with stable background knowledge."
        ),
        backstory=(
            "You are an expert learning architect for software engineers. "
            "You create small, incremental learning plans that build on "
            "what the user has already done."
        ),
        allow_delegation=False,
        verbose=True,
        llm=OPENAI_MODEL,
    )
    
    collect_preferences_task = Task(
        description=(
            "Interview the user based on their current request: '{user_query}'. "
            "Ask 2–3 concise follow-up questions if needed, then produce a "
            "short, structured summary of their learning goals and preferences. "
            "Focus on: topics, depth, time available per week, and preferred "
            "style of learning."
        ),
        expected_output=(
            "A JSON-like bullet summary of the user's preferences, suitable "
            "to be stored into long-term memory."
        ),
        agent=preference_researcher,
        verbose=True,
    )
    
    learning_plan_task = Task(
        description=(
            "Using all available memory and knowledge, design the next 3 "
            "learning sessions for the user. Each session must include: "
            "a title, concrete objective, and a short description of the "
            "hands-on work to be done. Make sure the plan clearly builds on "
            "prior preferences and past sessions if they exist."
        ),
        expected_output=(
            "A numbered list of 3 upcoming learning sessions tailored to the user."
        ),
        agent=learning_planner,
        verbose=True,
    )
    
    crew = Crew(
        agents=[preference_researcher, learning_planner],
        tasks=[collect_preferences_task, learning_plan_task],
        process=Process.sequential,
        memory=True,  # enables STM + LTM + entity memory
        knowledge_sources=[user_profile_knowledge],
        verbose=True,
    )

    return crew


def main():
    print("=== CrewAI Memory Demo ===")
    show_memory_storage_location()

    user_query = input(
        "\nDescribe what you want to learn next about AI agents: "
    ).strip()

    if not user_query:
        print("No query provided. Exiting.")
        return

    crew = build_crew()

    print("\n[Running crew with memory enabled...]\n")
    result = crew.kickoff(inputs={"user_query": user_query})

    print("\n=== Final Learning Plan ===\n")
    print(result)

    save_plan_to_markdown(user_query, str(result))


if __name__ == "__main__":
    main()