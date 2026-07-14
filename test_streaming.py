import os
import threading
from queue import Queue
from crewai import LLM, Agent, Task, Crew
from langchain_core.callbacks import BaseCallbackHandler

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put(token)
        
    def on_llm_end(self, *args, **kwargs) -> None:
        self.queue.put(None) # EOF marker

def run_test():
    q = Queue()
    handler = StreamingCallbackHandler(q)
    
    # Needs stream=True for litellm to stream tokens
    llm = LLM(
        model=os.environ.get("MODEL", "gemini/gemini-2.5-flash"), 
        api_key=os.environ.get("GOOGLE_API_KEY"),
        callbacks=[handler],
        stream=True
    )
    
    agent = Agent(
        role="Test",
        goal="Test",
        backstory="Test",
        llm=llm
    )
    
    task = Task(
        description="Write a short 50 word poem about a cat.",
        expected_output="A poem",
        agent=agent
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    
    def kickoff_crew():
        crew.kickoff()
        
    t = threading.Thread(target=kickoff_crew)
    t.start()
    
    output = ""
    while True:
        token = q.get()
        if token is None:
            break
        output += token
        print(token, end="", flush=True)
        
    t.join()
    print("\nDONE")

if __name__ == "__main__":
    run_test()
