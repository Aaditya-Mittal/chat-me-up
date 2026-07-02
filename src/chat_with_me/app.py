#!/usr/bin/env python
"""
Gradio Chat Interface for the Perona Chat Bot.
A premium dark-themed chat UI wrapping the CrewAI persona agent.
"""

import warnings
import os
import datetime
import tempfile
import google.generativeai as genai

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

def transcribe_audio(audio_path):
    if not audio_path:
        return ""
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("Transcription error: GOOGLE_API_KEY is not set.")
        return "I'm sorry, voice input is currently unavailable due to missing API configuration."
        
    try:
        genai.configure(api_key=api_key)
        audio_file = genai.upload_file(path=audio_path)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([
            "Transcribe this audio exactly as spoken. Only return the transcribed text, nothing else.",
            audio_file
        ])
        return response.text.strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return "I couldn't quite hear that clearly. Could you try typing it?"


def user_message(text_msg, audio_path, history):
    """Append user message (or transcribed audio) to history and clear input."""
    message = text_msg
    if audio_path:
        message = transcribe_audio(audio_path)
        
    if not message or not message.strip():
        return "", None, history
        
    # Security: Limit input length to prevent token exhaustion or buffer overflow attempts
    message = message.strip()[:1000]
        
    history = history + [{"role": "user", "content": message}]
    return "", None, history


def bot_response(history):
    """Generate bot response using the CrewAI crew."""
    if not history:
        return history

    user_msg = history[-1]["content"]
    prior = history[:-1]

    history_text = format_history(prior)
    
    # Inject current date context
    now = datetime.datetime.now()
    date_str = now.strftime("%B %d, %Y")
    system_context = f"[System Context: The current date is {date_str}. Keep this timeline in mind.]"

    if history_text:
        full_message = (
            f"{system_context}\n\n"
            f"Previous conversation:\n{history_text}\n\n"
            f"New message from visitor: {user_msg}"
        )
    else:
        full_message = f"{system_context}\n\nNew message from visitor: {user_msg}"

    try:
        result = crew_instance.kickoff(inputs={"user_message": full_message})
        reply = result.raw
    except Exception as e:
        reply = f"Hmm, something went wrong on my end. Mind trying again? (Error: {str(e)[:120]})"

    history = history + [{"role": "assistant", "content": reply}]
    return history


def export_chat(history):
    text = format_history(history)
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="chat_history_")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def undo_last(history):
    if len(history) >= 2:
        return history[:-2]
    return []


# --- Theme ---

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=(gr.themes.GoogleFont("Inter"), "system-ui", "-apple-system", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "Fira Code", "monospace"),
).set(
    body_background_fill="#0a0b14",
    body_background_fill_dark="#0a0b14",
    body_text_color="#cbd5e1",
    body_text_color_dark="#cbd5e1",

    block_background_fill="rgba(15, 17, 30, 0.7)",
    block_background_fill_dark="rgba(15, 17, 30, 0.7)",
    block_border_color="rgba(99, 102, 241, 0.08)",
    block_border_color_dark="rgba(99, 102, 241, 0.08)",
    block_label_text_color="#818cf8",
    block_label_text_color_dark="#818cf8",
    block_shadow="none",
    block_shadow_dark="none",

    button_primary_background_fill="#6366f1",
    button_primary_background_fill_dark="#6366f1",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_background_fill_hover="#818cf8",
    button_primary_background_fill_hover_dark="#818cf8",

    input_background_fill="rgba(15, 17, 30, 0.95)",
    input_background_fill_dark="rgba(15, 17, 30, 0.95)",
    input_border_color="rgba(99, 102, 241, 0.15)",
    input_border_color_dark="rgba(99, 102, 241, 0.15)",
    input_border_color_focus="rgba(99, 102, 241, 0.4)",
    input_border_color_focus_dark="rgba(99, 102, 241, 0.4)",
    input_placeholder_color="#3f4565",
    input_placeholder_color_dark="#3f4565",
)


# --- CSS ---

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ─── Reset & Base ─── */
* { transition: background-color 0.3s ease, border-color 0.3s ease, color 0.2s ease, transform 0.2s ease, box-shadow 0.3s ease, opacity 0.3s ease; }

.gradio-container {
    max-width: 820px !important;
    margin: 0 auto !important;
    background: #0a0b14 !important;
    padding: 0 1rem !important;
}

footer { display: none !important; }

/* ─── Hero Header ─── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    animation: fadeInDown 0.6s ease-out;
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-12px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero-avatar-ring {
    width: 78px; height: 78px;
    border-radius: 50%;
    padding: 3px;
    background: linear-gradient(135deg, #6366f1, #a78bfa, #6366f1);
    background-size: 200% 200%;
    animation: ringShift 4s ease infinite;
    margin: 0 auto 1rem;
    display: block;
}

@keyframes ringShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.hero-avatar-ring img {
    width: 100%; height: 100%;
    border-radius: 50%;
    border: 3px solid #0a0b14;
    display: block;
}

.hero h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 0.35rem;
    letter-spacing: -0.03em;
}

.hero p {
    color: #475569;
    font-size: 0.88rem;
    margin: 0;
    font-weight: 400;
    line-height: 1.6;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 0.75rem;
    padding: 4px 12px;
    border-radius: 20px;
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.15);
    font-size: 0.72rem;
    color: #4ade80;
    font-weight: 500;
    letter-spacing: 0.03em;
}

.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4ade80;
    animation: softPulse 2.5s ease-in-out infinite;
}

@keyframes softPulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.3); }
    50% { opacity: 0.7; box-shadow: 0 0 0 4px rgba(74, 222, 128, 0); }
}

/* ─── Chatbot ─── */
.chat-wrap {
    animation: fadeIn 0.5s ease-out 0.2s both;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.chat-wrap .chatbot {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.chat-wrap .message-wrap .message {
    animation: msgSlide 0.35s ease-out;
}

@keyframes msgSlide {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ─── Input Row ─── */
.input-row {
    animation: fadeIn 0.5s ease-out 0.3s both;
    margin-top: 0.25rem !important;
}

.input-row textarea {
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
    font-size: 0.9rem !important;
    background: rgba(15, 17, 30, 0.95) !important;
    border: 1px solid rgba(99, 102, 241, 0.12) !important;
    color: #e2e8f0 !important;
    resize: none !important;
}

.input-row textarea:focus {
    border-color: rgba(99, 102, 241, 0.35) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.08) !important;
    outline: none !important;
}

.send-btn {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    min-height: 44px !important;
    background: #6366f1 !important;
    border: none !important;
    color: #fff !important;
    cursor: pointer !important;
}

.send-btn:hover {
    background: #818cf8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25) !important;
}

.send-btn:active {
    transform: translateY(0) !important;
}

/* ─── Quick Questions ─── */
.quick-section {
    animation: fadeIn 0.5s ease-out 0.4s both;
    padding: 1rem 0 0.5rem;
}

.quick-label {
    color: #334155;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    text-align: center;
    margin-bottom: 0.6rem;
}

.quick-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
}

.quick-card {
    padding: 10px 14px;
    border-radius: 10px;
    border: 1px solid rgba(99, 102, 241, 0.1);
    background: rgba(15, 17, 30, 0.6);
    color: #94a3b8;
    font-size: 0.8rem;
    line-height: 1.4;
    cursor: pointer;
    text-align: left;
    font-weight: 400;
}

.quick-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    background: rgba(99, 102, 241, 0.06);
    color: #c7d2fe;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

.quick-card:active {
    transform: translateY(0);
}

.quick-card .card-icon {
    font-size: 0.9rem;
    margin-right: 4px;
}

/* ─── Footer ─── */
.footer-section {
    animation: fadeIn 0.5s ease-out 0.5s both;
    text-align: center;
    padding: 1.25rem 0 0.5rem;
    border-top: 1px solid rgba(99, 102, 241, 0.06);
    margin-top: 0.75rem;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-bottom: 0.75rem;
}

.footer-link {
    color: #475569;
    text-decoration: none;
    font-size: 0.78rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

.footer-link:hover {
    color: #a5b4fc;
}

.footer-credit {
    color: #1e293b;
    font-size: 0.65rem;
    letter-spacing: 0.04em;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.15); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.3); }

/* ─── Responsive ─── */
@media (max-width: 640px) {
    .hero h1 { font-size: 1.4rem; }
    .quick-grid { grid-template-columns: 1fr; }
    .footer-links { gap: 1rem; }
}
"""


# --- Quick questions ---

QUICK_QUESTIONS = [
    {"icon": "👋", "text": "Tell me about yourself and your background"},
    {"icon": "💼", "text": "What's your experience at Deloitte like?"},
    {"icon": "🛠", "text": "What tech stack and tools do you work with?"},
    {"icon": "🏆", "text": "How did you win the HackNow hackathon?"},
    {"icon": "🚀", "text": "What are some projects you've built?"},
    {"icon": "📬", "text": "Are you open to new opportunities?"},
]


# --- Build the UI ---

with gr.Blocks(title="Chat with Aaditya Mittal") as demo:

    # Hero
    gr.HTML("""
        <div class="hero">
            <div class="hero-avatar-ring">
                <img src="https://api.dicebear.com/9.x/initials/svg?seed=AM&backgroundColor=4338ca&textColor=ffffff"
                     alt="AM" />
            </div>
            <h1>Chat with Aaditya</h1>
            <p>AI portfolio assistant — ask about my work, skills, or experience</p>
            <div class="status-badge">
                <span class="status-dot"></span>
                Online &middot; Powered by Gemini
            </div>
        </div>
    """)

    # Chatbot
    chatbot = gr.Chatbot(
        height=440,
        avatar_images=(
            None,
            "https://api.dicebear.com/9.x/initials/svg?seed=AM&backgroundColor=4338ca&textColor=ffffff",
        ),
        elem_classes=["chat-wrap"],
        layout="bubble",
        placeholder="Say hi or pick a question below to get started →",
        buttons=["copy"],
    )

    # Input
    with gr.Row(elem_classes=["input-row"]):
        msg = gr.Textbox(
            placeholder="Ask me anything...",
            show_label=False,
            scale=7,
            container=False,
            autofocus=True,
        )
        audio = gr.Audio(
            sources=["microphone"], 
            type="filepath", 
            scale=1, 
            show_label=False, 
            container=False,
            min_width=60,
            waveform_options=gr.WaveformOptions(
                waveform_color="#6366f1",
                waveform_progress_color="#818cf8",
                skip_length=2,
                show_controls=False,
            )
        )
        send_btn = gr.Button(
            "Send",
            variant="primary",
            scale=1,
            min_width=80,
            elem_classes=["send-btn"],
        )

    with gr.Row(elem_classes=["tools-row"]):
        clear_btn = gr.Button("🗑️ Clear Chat", size="sm", min_width=60)
        undo_btn = gr.Button("↩️ Undo Last", size="sm", min_width=60)
        download_btn = gr.DownloadButton("💾 Export Chat", size="sm", min_width=60)

    # Quick questions grid
    cards_html = "".join(
        f'<div class="quick-card" onclick="'
        f"document.querySelector('.input-row textarea').value = '{q['text']}';"
        f"document.querySelector('.input-row textarea').dispatchEvent(new Event('input', {{bubbles: true}}));"
        f"document.querySelector('.input-row textarea').focus();"
        f'">'
        f'<span class="card-icon">{q["icon"]}</span> {q["text"]}'
        f'</div>'
        for q in QUICK_QUESTIONS
    )
    gr.HTML(f"""
        <div class="quick-section">
            <div class="quick-label">Suggested Questions</div>
            <div class="quick-grid">{cards_html}</div>
        </div>
    """)

    # Footer
    gr.HTML("""
        <div class="footer-section">
            <div class="footer-links">
                <a class="footer-link" href="https://portfolio-aaditya-mittal.vercel.app/" target="_blank">
                    🌐 Portfolio
                </a>
                <a class="footer-link" href="https://www.linkedin.com/in/aadityamittal01" target="_blank">
                    💼 LinkedIn
                </a>
                <a class="footer-link" href="https://github.com/Aaditya-Mittal" target="_blank">
                    🐙 GitHub
                </a>
            </div>
            <div class="footer-credit">Built with CrewAI + Gemini · RAG-powered</div>
        </div>
    """)

    # Event wiring
    msg.submit(user_message, [msg, audio, chatbot], [msg, audio, chatbot], queue=False).then(
        bot_response, chatbot, chatbot
    )
    send_btn.click(user_message, [msg, audio, chatbot], [msg, audio, chatbot], queue=False).then(
        bot_response, chatbot, chatbot
    )
    clear_btn.click(lambda: [], None, chatbot, queue=False)
    undo_btn.click(undo_last, chatbot, chatbot, queue=False)
    download_btn.click(export_chat, inputs=[chatbot], outputs=[download_btn])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=theme,
        css=CSS,
    )
