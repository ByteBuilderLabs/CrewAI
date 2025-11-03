from crewai import Agent, Crew, Task
from dotenv import load_dotenv


# Define agents
researcher = Agent(
    role="Researcher",
    goal="Collect key insights about multi-agent systems and CrewAI.",
    backstory=(
        "You are an AI researcher who studies large language model systems. "
        "You focus on summarizing complex ideas into factual, useful insights for developers."
    ),
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Transform the research findings into a clear, developer-friendly article.",
    backstory=(
        "You are a concise technical writer who explains technical concepts simply "
        "and formats them like a professional tech blog post."
    ),
    verbose=True
)

# Define tasks
research_task = Task(
    description=(
        "Research what multi-agent systems are, why single-agent LLMs have bottlenecks, "
        "and how CrewAI enables collaboration between agents. "
        "Focus on developer use cases and architectural clarity."
    ),
    expected_output=(
        "A numbered list of 5–8 concise bullet points explaining key concepts "
        "and one short takeaway summary."
    ),
    agent=researcher
)

writing_task = Task(
    description=(
        "Using the research bullet points, write a concise developer-oriented blog post "
        "titled 'Why a Single AI Agent Isn't Enough Anymore'. "
        "Keep it under 200 words with a short intro, two paragraphs, and one closing line."
    ),
    expected_output=(
        "A short markdown-formatted blog post with a clear title, body, and conclusion."
    ),
    agent=writer
)


# Build the crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process="sequential",
    verbose=True
)


# Run the workflow
result = crew.kickoff()

print("\n=== FINAL RESULT ===\n")
print(result)
