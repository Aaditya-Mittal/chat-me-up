#!/usr/bin/env python
"""
Gradio Chat Interface for the Perona Chat Bot.
A premium dark-themed chat UI wrapping the CrewAI persona agent.
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


# --- Helper functions ---

def format_history(history: list[dict]) -> str:
    """Convert Gradio messages-format history into a readable string."""
    if not history:
        return ""
    lines = []
    for msg in history:
        role = "Visitor" if msg["role"] == "user" else "Aaditya"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# --- Event handlers ---

def user_message(message, history):
    """Append user message to history and clear input."""
    if not message.strip():
        return "", history
    history = history + [{"role": "user", "content": message}]
    return "", history


def bot_response(history):
    """Generate bot response using the CrewAI crew."""
    if not history:
        return history

    user_msg = history[-1]["content"]
    prior = history[:-1]

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
        reply = f"Hmm, something went wrong on my end. Mind trying again? (Error: {str(e)[:120]})"

    history = history + [{"role": "assistant", "content": reply}]
    return history


# --- Theme ---

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=("Inter", "system-ui", "sans-serif"),
    font_mono=("JetBrains Mono", "Fira Code", "monospace"),
).set(
    # Background
    body_background_fill="#0b0d1a",
    body_background_fill_dark="#0b0d1a",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",

    # Containers
    block_background_fill="rgba(15, 18, 35, 0.85)",
    block_background_fill_dark="rgba(15, 18, 35, 0.85)",
    block_border_color="rgba(99, 102, 241, 0.12)",
    block_border_color_dark="rgba(99, 102, 241, 0.12)",
    block_label_text_color="#818cf8",
    block_label_text_color_dark="#818cf8",
    block_shadow="0 4px 24px rgba(0, 0, 0, 0.4)",
    block_shadow_dark="0 4px 24px rgba(0, 0, 0, 0.4)",

    # Buttons
    button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_background_fill_hover="linear-gradient(135deg, #818cf8 0%, #a78bfa 100%)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #818cf8 0%, #a78bfa 100%)",
    button_secondary_background_fill="rgba(30, 30, 60, 0.6)",
    button_secondary_background_fill_dark="rgba(30, 30, 60, 0.6)",
    button_secondary_text_color="#c7d2fe",
    button_secondary_text_color_dark="#c7d2fe",
    button_secondary_border_color="rgba(99, 102, 241, 0.25)",
    button_secondary_border_color_dark="rgba(99, 102, 241, 0.25)",

    # Inputs
    input_background_fill="rgba(10, 12, 28, 0.9)",
    input_background_fill_dark="rgba(10, 12, 28, 0.9)",
    input_border_color="rgba(99, 102, 241, 0.2)",
    input_border_color_dark="rgba(99, 102, 241, 0.2)",
    input_border_color_focus="rgba(129, 140, 248, 0.5)",
    input_border_color_focus_dark="rgba(129, 140, 248, 0.5)",
    input_placeholder_color="#475569",
    input_placeholder_color_dark="#475569",
)


# --- CSS ---

CSS = """
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Animated gradient background */
.gradio-container {
    max-width: 880px !important;
    margin: auto !important;
    background: linear-gradient(135deg, #0b0d1a 0%, #111336 50%, #0b0d1a 100%) !important;
    min-height: 100vh;
}

/* Hide footer */
footer { display: none !important; }

/* ── Header ── */
#hero-section {
    text-align: center;
    padding: 2rem 1rem 1rem;
    position: relative;
}

#hero-section::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

.hero-avatar {
    width: 72px; height: 72px;
    border-radius: 50%;
    border: 2px solid rgba(129, 140, 248, 0.4);
    margin: 0 auto 0.75rem;
    display: block;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);
}

.hero-title {
    background: linear-gradient(135deg, #c7d2fe 0%, #a78bfa 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0 0 0.3rem;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    color: #64748b;
    font-size: 0.9rem;
    margin: 0 0 0.2rem;
    font-weight: 400;
}

.hero-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #22c55e;
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 0.5rem;
}

.hero-status .pulse {
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse-glow 2s ease-in-out infinite;
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.4); }
    50% { box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); }
}

/* ── Chat messages ── */
.chat-area .message-wrap {
    padding: 0.5rem 0 !important;
}

.chat-area .bot .message-content {
    background: rgba(20, 22, 45, 0.9) !important;
    border: 1px solid rgba(99, 102, 241, 0.1) !important;
    border-radius: 18px 18px 18px 4px !important;
    color: #e2e8f0 !important;
}

.chat-area .user .message-content {
    background: linear-gradient(135deg, #4338ca 0%, #6366f1 100%) !important;
    border-radius: 18px 18px 4px 18px !important;
    color: #ffffff !important;
}

/* ── Input area ── */
.input-row {
    margin-top: 0.5rem !important;
}

.input-row textarea {
    border-radius: 14px !important;
    padding: 0.85rem 1rem !important;
    font-size: 0.92rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.input-row textarea:focus {
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15) !important;
}

.send-btn {
    border-radius: 14px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    min-height: 46px !important;
}

.send-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
}

/* ── Quick questions ── */
#quick-questions {
    text-align: center;
    padding: 0.75rem 0 0.25rem;
}

#quick-questions .label {
    color: #475569;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    font-weight: 600;
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
}

.chip {
    display: inline-block;
    padding: 8px 16px;
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 24px;
    background: rgba(20, 22, 45, 0.6);
    color: #a5b4fc;
    font-size: 0.82rem;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
    font-weight: 500;
}

.chip:hover {
    border-color: rgba(129, 140, 248, 0.5);
    background: rgba(99, 102, 241, 0.1);
    color: #c7d2fe;
    transform: translateY(-1px);
}

/* ── Links footer ── */
#footer-links {
    text-align: center;
    padding: 1rem 0 0.5rem;
    border-top: 1px solid rgba(99, 102, 241, 0.08);
    margin-top: 0.5rem;
}

#footer-links a {
    color: #64748b;
    text-decoration: none;
    margin: 0 1rem;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    transition: color 0.2s ease;
}

#footer-links a:hover {
    color: #a5b4fc;
}

.footer-divider {
    color: #1e293b;
    margin: 0 0.25rem;
}

#footer-credit {
    text-align: center;
    color: #334155;
    font-size: 0.7rem;
    padding: 0.25rem 0 1rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.2);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.4);
}

/* ── Responsive ── */
@media (max-width: 640px) {
    .hero-title { font-size: 1.5rem; }
    .chip { font-size: 0.78rem; padding: 6px 12px; }
    #footer-links a { margin: 0 0.5rem; }
}
"""


# --- Quick question chips (handled via JS) ---

QUICK_QUESTIONS = [
    "Tell me about yourself",
    "Deloitte experience?",
    "Tech stack",
    "Open to work?",
    "Hackathon win",
    "Projects",
]


# --- Build the UI ---

with gr.Blocks(title="Chat with Aaditya Mittal") as demo:

    # Hero header
    gr.HTML(f"""
        <div id="hero-section">
            <img class="hero-avatar"
                 src="https://api.dicebear.com/9.x/initials/svg?seed=AM&backgroundColor=4338ca&textColor=ffffff"
                 alt="AM" />
            <h1 class="hero-title">Chat with Aaditya</h1>
            <p class="hero-subtitle">
                AI-powered portfolio assistant — trained on my real background,
                projects, and experience
            </p>
            <div class="hero-status">
                <span class="pulse"></span> Online
            </div>
        </div>
    """)

    # Chatbot
    chatbot = gr.Chatbot(
        height=460,
        avatar_images=(
            None,
            "https://api.dicebear.com/9.x/initials/svg?seed=AM&backgroundColor=4338ca&textColor=ffffff",
        ),
        elem_classes=["chat-area"],
        layout="bubble",
        placeholder="Start a conversation — say hi, ask about my work, or pick a question below!",
        buttons=["copy"],
    )

    # Input row
    with gr.Row(elem_classes=["input-row"]):
        msg = gr.Textbox(
            placeholder="Type your message...",
            show_label=False,
            scale=8,
            container=False,
            autofocus=True,
        )
        send_btn = gr.Button(
            "Send ↗",
            variant="primary",
            scale=1,
            min_width=90,
            elem_classes=["send-btn"],
        )

    # Quick question chips (pure HTML + JS)
    chips_html = "".join(
        f'<span class="chip" onclick="'
        f"document.querySelector('.input-row textarea').value = '{q}';"
        f"document.querySelector('.input-row textarea').dispatchEvent(new Event('input', {{bubbles: true}}));"
        f'">{q}</span>'
        for q in QUICK_QUESTIONS
    )
    gr.HTML(f"""
        <div id="quick-questions">
            <div class="label">Quick Questions</div>
            <div class="chip-row">{chips_html}</div>
        </div>
    """)

    # Footer links
    gr.HTML("""
        <div id="footer-links">
            <a href="https://portfolio-aaditya-mittal.vercel.app/" target="_blank">Portfolio</a>
            <span class="footer-divider">·</span>
            <a href="https://www.linkedin.com/in/aadityamittal01" target="_blank">LinkedIn</a>
            <span class="footer-divider">·</span>
            <a href="https://github.com/Aaditya-Mittal" target="_blank">GitHub</a>
        </div>
        <div id="footer-credit">
            Built with CrewAI + Gemini · Powered by RAG
        </div>
    """)

    # Event wiring
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
