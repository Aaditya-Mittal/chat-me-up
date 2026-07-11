import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai import LLM
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource


@CrewBase
class ChatWithMe():
    """ChatWithMe crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    text_source = TextFileKnowledgeSource(
        file_paths=["about_me.md", "resume.md", "faq.yaml", "projects.md"],
        chunk_size=1000,
        chunk_overlap=200
    )

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def persona_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['persona_agent'], # type: ignore[index]
            verbose=True,
            llm=LLM(model=os.environ.get("MODEL", "gemini/gemini-2.5-flash"), api_key=os.environ.get("GOOGLE_API_KEY"))
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
            memory=False,
            embedder={"provider": "google", "config": {"model": "models/embedding-001", "api_key": os.environ.get('GOOGLE_API_KEY')}},
            knowledge_sources=[self.text_source]
        )
