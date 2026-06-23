#!/usr/bin/env python
"""
Gradio Chat Interface for the Perona Chat Bot.
A premium chat UI wrapping the CrewAI persona agent.
"""

import warnings

import gradio as gr

from chat_with_me.crew import ChatWithMe

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize the crew once at startup
print("Initializing ChatWithMe crew...")
crew_instance = ChatWithMe().crew()
print("Crew ready!")


def format_history(history: list[dict]) -> str:
    """Convert Gradio messages-format history into a readable string."""
    if not history:
        return ""

    lines = []
    for msg in history:
        role = "Visitor" if msg["role"] == "user" else "Aaditya"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)


def respond(message: str, history: list[dict]) -> str:
    """Handle a chat message with conversation context."""
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


# --- Theme & Styling ---

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=("Inter", "system-ui", "sans-serif"),
    font_mono=("JetBrains Mono", "Fira Code", "monospace"),
).set(
    # Global
    body_background_fill="linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%)",
    body_background_fill_dark="linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%)",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",

    # Blocks / containers
    block_background_fill="rgba(30, 30, 60, 0.6)",
    block_background_fill_dark="rgba(30, 30, 60, 0.6)",
    block_border_color="rgba(99, 102, 241, 0.2)",
    block_border_color_dark="rgba(99, 102, 241, 0.2)",
    block_label_text_color="#a5b4fc",
    block_label_text_color_dark="#a5b4fc",
    block_shadow="0 8px 32px rgba(0, 0, 0, 0.3)",
    block_shadow_dark="0 8px 32px rgba(0, 0, 0, 0.3)",

    # Buttons
    button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_background_fill_hover="linear-gradient(135deg, #818cf8 0%, #a78bfa 100%)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #818cf8 0%, #a78bfa 100%)",

    # Inputs
    input_background_fill="rgba(15, 15, 40, 0.8)",
    input_background_fill_dark="rgba(15, 15, 40, 0.8)",
    input_border_color="rgba(99, 102, 241, 0.3)",
    input_border_color_dark="rgba(99, 102, 241, 0.3)",
    input_border_color_focus="rgba(129, 140, 248, 0.6)",
    input_border_color_focus_dark="rgba(129, 140, 248, 0.6)",
    input_placeholder_color="#64748b",
    input_placeholder_color_dark="#64748b",
)

CSS = """
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Container centering */
.gradio-container {
    max-width: 850px !important;
    margin: auto !important;
}

/* Hide footer */
footer { display: none !important; }

/* Header styling */
#chat-header {
    text-align: center;
    padding: 1.5rem 1rem 0.5rem;
}
#chat-header h1 {
    background: linear-gradient(135deg, #a5b4fc 0%, #c4b5fd 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
#chat-header p {
    color: #94a3b8;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* Chatbot area */
.chatbot-container .message {
    border-radius: 16px !important;
}
.chatbot-container .user {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
}
.chatbot-container .bot {
    background: rgba(30, 30, 60, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
}

/* Example chips */
.example-btn {
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    background: rgba(30, 30, 60, 0.5) !important;
    color: #c7d2fe !important;
    border-radius: 20px !important;
    transition: all 0.2s ease !important;
}
.example-btn:hover {
    border-color: rgba(129, 140, 248, 0.6) !important;
    background: rgba(99, 102, 241, 0.15) !important;
    transform: translateY(-1px) !important;
}

/* Links row */
#links-row {
    text-align: center;
    padding: 0.5rem;
}
#links-row a {
    color: #a5b4fc;
    text-decoration: none;
    margin: 0 0.75rem;
    font-size: 0.85rem;
    transition: color 0.2s;
}
#links-row a:hover {
    color: #c4b5fd;
}

/* Smooth scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.5);
}
"""

EXAMPLES = [
    ["Tell me about yourself"],
    ["What's your experience at Deloitte?"],
    ["What tech stack do you use?"],
    ["Are you open to work?"],
    ["How did you win the hackathon?"],
    ["What projects have you built?"],
]

# --- Build the UI ---

with gr.Blocks(title="Chat with Aaditya Mittal") as demo:

    # Header
    gr.HTML("""
        <div id="chat-header">
            <h1>Chat with Aaditya</h1>
            <p>
                AI-powered portfolio assistant &mdash; ask me anything about my
                experience, projects, skills, or just say hi!
            </p>
        </div>
    """)

    # Chat area
    chatbot = gr.Chatbot(
        height=480,
        avatar_images=(None, "https://api.dicebear.com/9.x/initials/svg?seed=AM&backgroundColor=6366f1"),
        elem_classes=["chatbot-container"],
    )

    # Input row
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your message here...",
            show_label=False,
            scale=8,
            container=False,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

    # Examples
    gr.Examples(
        examples=EXAMPLES,
        inputs=msg,
    )

    # Links footer
    gr.HTML("""
        <div id="links-row">
            <a href="https://portfolio-aaditya-mittal.vercel.app/" target="_blank">Portfolio</a>
            <a href="https://www.linkedin.com/in/aadityamittal01" target="_blank">LinkedIn</a>
            <a href="https://github.com/Aaditya-Mittal" target="_blank">GitHub</a>
        </div>
    """)

    # --- Event wiring ---
    def user_message(message, history):
        """Append user message to history and clear input."""
        history = history + [{"role": "user", "content": message}]
        return "", history

    def bot_response(history):
        """Generate bot response using the crew."""
        user_msg = history[-1]["content"]
        prior = history[:-1]  # everything before the latest message

        history_text = format_history(prior)
        if history_text:
            full_message = (
                f"Previous conversation:\n{history_text}\n\n"
                f"New message from visitor: {user_msg}"
            )
        else:
            full_message = user_msg

        try:
            result = crew_instance.kickoff(inputs={"user_message": full_message})
            reply = result.raw
        except Exception as e:
            reply = f"Sorry, I ran into an issue. Could you try again? (Error: {str(e)[:100]})"

        history = history + [{"role": "assistant", "content": reply}]
        return history

    # Submit on Enter or button click
    msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, chatbot, chatbot
    )
    send_btn.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, chatbot, chatbot
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=theme,
        css=CSS,
    )
