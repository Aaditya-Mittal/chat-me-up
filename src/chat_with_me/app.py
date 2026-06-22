#!/usr/bin/env python
"""
Gradio Chat Interface for the Perona Chat Bot.
Wraps the CrewAI crew in an interactive chat UI with conversation history.
"""

import warnings

import gradio as gr

from chat_with_me.crew import ChatWithMe

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize the crew once at startup (avoids re-embedding knowledge each call)
print("Initializing ChatWithMe crew...")
crew_instance = ChatWithMe().crew()
print("Crew ready!")


def format_history(history: list) -> str:
    """Convert Gradio chat history (list of tuples) into a readable string."""
    if not history:
        return ""

    lines = []
    for user_msg, bot_msg in history:
        if user_msg:
            lines.append(f"Visitor: {user_msg}")
        if bot_msg:
            lines.append(f"Aaditya: {bot_msg}")

    return "\n".join(lines)


def respond(message: str, history: list) -> str:
    """
    Handle a chat message by running the CrewAI crew with conversation context.

    Args:
        message: The visitor's new message.
        history: List of previous (user, bot) message tuples.

    Returns:
        The agent's response string.
    """
    # Build the full input with conversation history for context
    history_text = format_history(history)

    if history_text:
        full_message = (
            f"Previous conversation:\n{history_text}\n\n"
            f"New message from visitor: {message}"
        )
    else:
        full_message = message

    try:
        result = crew_instance.kickoff(inputs={"user_message": full_message})
        return result.raw
    except Exception as e:
        return f"Sorry, I ran into an issue. Could you try again? (Error: {str(e)[:100]})"


# --- Gradio UI ---

EXAMPLES = [
    "Tell me about yourself",
    "What's your experience at Deloitte?",
    "What tech stack do you use?",
    "Are you open to work?",
    "How did you win the hackathon?",
    "What projects have you built?",
]

CSS = """
.gradio-container {
    max-width: 800px !important;
    margin: auto !important;
}
footer { display: none !important; }
"""

demo = gr.ChatInterface(
    fn=respond,
    title="Chat with Aaditya",
    description="Ask me anything about my experience, projects, skills, or just say hi! I'm an AI representative trained on Aaditya's knowledge base.",
    examples=EXAMPLES,
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
