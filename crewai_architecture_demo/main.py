from crewai import Agent, Task, Crew

researcher = Agent(
    role="Researcher",
    goal="Gather concise, accurate notes on CrewAI architecture.",
    backstory="An AI researcher skilled at extracting key technical concepts and explaining them clearly.",
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Explain CrewAI architecture for beginners using the research notes.",
    backstory="A technical writer who communicates complex topics with clarity and structure.",
    verbose=True
)

task_research = Task(
    description="Research how CrewAI connects Agents, Tasks, Tools, and the Orchestrator.",
    expected_output="Bullet-point notes explaining architecture, components, and data flow.",
    agent=researcher
)

task_write = Task(
    description="Write a beginner-friendly summary of CrewAI’s architecture based on the research notes.",
    expected_output="A 5-paragraph explanation describing Agents, Tasks, Tools, and Orchestrator roles.",
    agent=writer
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task_research, task_write],
    verbose=True
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("=== Crew Output ===")
    print(result)