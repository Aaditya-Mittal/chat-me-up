from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class ChatWithMe():
    """ChatWithMe crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def persona_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['persona_agent'], # type: ignore[index]
            verbose=True
        )

    @task
    def respond_task(self) -> Task:
        return Task(
            config=self.tasks_config['respond_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ChatWithMe crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
