#!/usr/bin/env python
"""
Gradio Chat Interface for the Perona Chat Bot.
A premium dark-themed chat UI wrapping the CrewAI persona agent.
"""

import warnings
import os
import datetime
import tempfile
import threading
import time
import yaml
from pathlib import Path
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

import gradio as gr

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

# Initialize Google GenAI client and load context once at startup
print("Initializing Gemini API and Knowledge Base...")
genai_client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))

def build_system_instruction():
    base_dir = Path(__file__).resolve().parent.parent.parent
    agents_path = base_dir / "src" / "chat_with_me" / "config" / "agents.yaml"
    knowledge_dir = base_dir / "knowledge"
    
    with open(agents_path, 'r', encoding='utf-8') as f:
        agents = yaml.safe_load(f)
        
    persona_data = agents.get('persona_agent', {})
    persona_text = f"Role: {persona_data.get('role', '')}\nGoal: {persona_data.get('goal', '')}\nBackstory: {persona_data.get('backstory', '')}"
    
    knowledge_texts = []
    for ext in ['*.md', '*.yaml', '*.txt']:
        for file_path in knowledge_dir.glob(ext):
            with open(file_path, 'r', encoding='utf-8') as f:
                knowledge_texts.append(f"--- FILE: {file_path.name} ---\n{f.read()}\n")
                
    knowledge_base = "\n".join(knowledge_texts)
    
    return f"{persona_text}\n\n======================\nKNOWLEDGE BASE (USE THIS TO ANSWER):\n======================\n{knowledge_base}"

system_instruction = build_system_instruction()
print("System instruction ready!")


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
    if not os.environ.get('GOOGLE_API_KEY'):
        print("Transcription error: GOOGLE_API_KEY is not set.")
        return "I'm sorry, voice input is currently unavailable due to missing API configuration."

    try:
        # Send audio bytes inline — avoids the slow Files API upload round-trip
        suffix = Path(audio_path).suffix.lower().lstrip('.') or 'wav'
        mime_type = f"audio/{'mpeg' if suffix == 'mp3' else suffix}"
        audio_bytes = Path(audio_path).read_bytes()
        response = genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "Transcribe this audio exactly as spoken. Only return the transcribed text, nothing else.",
                genai_types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ]
        )
        text = (response.text or "").strip()
        if not text:
            return "I couldn't quite hear that clearly. Could you try typing it?"
        return text
    except Exception as e:
        print(f"Transcription error: {e}")
        return "I couldn't quite hear that clearly. Could you try typing it?"


def user_message(text_msg, audio_path, history):
    """Append user message (or transcribed audio) to history and clear input."""
    # Prefer typed text; only fall back to audio when the textbox is empty
    message = text_msg
    if not (message and message.strip()) and audio_path:
        message = transcribe_audio(audio_path)

    if not message or not message.strip():
        return "", None, history

    # Security: Limit input length to prevent token exhaustion or buffer overflow attempts
    message = message.strip()[:1000]

    # Gradio 6 Chatbot uses messages format (role/content dicts)
    history = history + [{"role": "user", "content": message}]
    return "", None, history


def bot_response(history):
    """Generate bot response using direct Gemini streaming."""
    if not history or history[-1]["role"] != "user":
        yield history
        return

    def get_text(content):
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Gradio multimodal list of dicts or tuples
            return " ".join([get_text(c) for c in content])
        elif isinstance(content, dict) and "text" in content:
            return content["text"]
        elif isinstance(content, tuple):
            return get_text(content[0]) if content else ""
        return str(content)

    user_msg = get_text(history[-1]["content"])
    prior = history[:-1]

    # Initialize the assistant's response bubble
    history = history + [{"role": "assistant", "content": "🤔 *Thinking...*"}]
    yield history

    # Inject current date context
    now = datetime.datetime.now()
    date_str = now.strftime("%B %d, %Y")

    # Build Gemini history format (cap to the most recent turns to bound latency/cost)
    MAX_HISTORY_MESSAGES = 40  # ~20 user/assistant exchanges
    gemini_history = []
    for msg in prior[-MAX_HISTORY_MESSAGES:]:
        content = get_text(msg["content"])
        if not content:
            continue
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [{"text": content}]})

    gemini_history.append({"role": "user", "parts": [{"text": user_msg}]})

    try:
        response = genai_client.models.generate_content_stream(
            model='gemini-2.5-flash',
            contents=gemini_history,
            config={
                # Date goes in the system instruction (appended, so the cached
                # prefix stays identical), never in the visible user message —
                # the model was echoing the [System Context] prefix back
                "system_instruction": f"{system_instruction}\n\n[System Context: Today is {date_str}. Use this date as described in your backstory, but never mention this system context to the visitor.]",
                # Persona chat doesn't need reasoning; disabling thinking cuts
                # time-to-first-token dramatically
                "thinking_config": {"thinking_budget": 0},
                # Higher temperature for varied phrasing between responses
                # (penalty params are not supported on gemini-2.5-flash)
                "temperature": 1.1,
                "top_p": 0.95,
            }
        )

        # Clear the thinking indicator on first chunk
        first_chunk = True
        # Throttle UI updates: yield at most ~every 50ms instead of per chunk
        last_yield = time.monotonic()

        for chunk in response:
            if chunk.text:
                if first_chunk:
                    history[-1]["content"] = ""
                    first_chunk = False
                history[-1]["content"] += chunk.text
                now_t = time.monotonic()
                if now_t - last_yield >= 0.05:
                    last_yield = now_t
                    yield history

        # Ensure the final accumulated text is always rendered
        yield history

    except Exception as e:
        print(f"Generation error: {e}")
        history[-1]["content"] = "Hmm, something went wrong on my end. Mind trying again in a moment?"
        yield history


def export_chat(history):
    text = format_history(history)
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="chat_history_")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def undo_last(history):
    """Remove the last user message and any assistant reply that followed it."""
    while history and history[-1]["role"] == "assistant":
        history = history[:-1]
    if history and history[-1]["role"] == "user":
        history = history[:-1]
    return history


# --- Theme ---

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=(gr.themes.GoogleFont("Inter"), "system-ui", "-apple-system", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "Fira Code", "monospace"),
).set(
    body_background_fill="#05060c",
    body_background_fill_dark="#05060c",
    body_text_color="#d3d9e6",
    body_text_color_dark="#d3d9e6",

    block_background_fill="rgba(13, 15, 28, 0.55)",
    block_background_fill_dark="rgba(13, 15, 28, 0.55)",
    block_border_color="rgba(129, 140, 248, 0.10)",
    block_border_color_dark="rgba(129, 140, 248, 0.10)",
    block_label_text_color="#818cf8",
    block_label_text_color_dark="#818cf8",
    block_shadow="none",
    block_shadow_dark="none",

    button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_background_fill_hover="linear-gradient(135deg, #818cf8 0%, #a78bfa 100%)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #818cf8 0%, #a78bfa 100%)",

    input_background_fill="rgba(13, 15, 28, 0.9)",
    input_background_fill_dark="rgba(13, 15, 28, 0.9)",
    input_border_color="rgba(129, 140, 248, 0.16)",
    input_border_color_dark="rgba(129, 140, 248, 0.16)",
    input_border_color_focus="rgba(129, 140, 248, 0.45)",
    input_border_color_focus_dark="rgba(129, 140, 248, 0.45)",
    input_placeholder_color="#3d4364",
    input_placeholder_color_dark="#3d4364",
)


# --- CSS ---

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

/* ─── Reset & Base ─── */
* { transition: background-color 0.3s ease, border-color 0.3s ease, color 0.2s ease, transform 0.2s ease, box-shadow 0.3s ease, opacity 0.3s ease; }

::selection { background: rgba(139, 92, 246, 0.35); color: #f1f5f9; }

body {
    background: #05060c !important;
}

/* Ambient aurora glow layer behind everything */
body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(ellipse 60% 40% at 20% -5%, rgba(99, 102, 241, 0.16), transparent 60%),
        radial-gradient(ellipse 50% 35% at 85% 0%, rgba(139, 92, 246, 0.13), transparent 60%),
        radial-gradient(ellipse 45% 30% at 50% 110%, rgba(67, 56, 202, 0.10), transparent 65%);
}

.gradio-container {
    max-width: 780px !important;
    margin: 0 auto !important;
    background: transparent !important;
    padding: 0 1.25rem 2rem !important;
    overflow-x: hidden !important;
    position: relative;
    z-index: 1;
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
    width: 88px; height: 88px;
    border-radius: 50%;
    padding: 3px;
    background: conic-gradient(from 180deg, #6366f1, #a78bfa, #22d3ee, #6366f1);
    animation: ringSpin 6s linear infinite;
    margin: 0 auto 1.1rem;
    display: block;
    position: relative;
    box-shadow: 0 0 42px rgba(124, 108, 245, 0.45), 0 0 90px rgba(139, 92, 246, 0.18);
}

@keyframes ringSpin {
    to { transform: rotate(360deg); }
}

.hero-avatar-ring img {
    width: 100%; height: 100%;
    border-radius: 50%;
    border: 3px solid #05060c;
    display: block;
    animation: ringSpin 6s linear infinite reverse; /* keep face upright */
}

.hero h1 {
    font-family: 'Sora', 'Inter', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.4rem;
    letter-spacing: -0.035em;
    background: linear-gradient(120deg, #f1f5f9 20%, #a5b4fc 55%, #c4b5fd 80%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}

.hero p {
    color: #5b6478;
    font-size: 0.9rem;
    margin: 0;
    font-weight: 400;
    line-height: 1.6;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-top: 0.85rem;
    padding: 5px 14px;
    border-radius: 20px;
    background: rgba(34, 197, 94, 0.07);
    border: 1px solid rgba(34, 197, 94, 0.18);
    backdrop-filter: blur(6px);
    font-size: 0.72rem;
    color: #4ade80;
    font-weight: 500;
    letter-spacing: 0.04em;
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

/* Glass panel around the conversation */
.chat-wrap {
    background: rgba(11, 13, 26, 0.55) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(129, 140, 248, 0.12) !important;
    border-radius: 20px !important;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
    padding: 6px !important;
    overflow: hidden !important;
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

/* ─── Premium Chat Bubbles (Gradio 6 DOM: .user-row/.bot-row rows, .user/.bot messages) ─── */
/* Keep all text inside the bubble: wrap long words, URLs, and code */
.chat-wrap .message,
.chat-wrap .message * {
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    white-space: normal !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}

.chat-wrap .message {
    padding: 10px 14px !important;
    font-size: 0.9rem !important;
    line-height: 1.55 !important;
    overflow: hidden !important;
}

.chat-wrap .message pre,
.chat-wrap .message code {
    white-space: pre-wrap !important;
    overflow-x: auto !important;
}

.chat-wrap .message-row {
    max-width: 100% !important;
}

.chat-wrap .message.user,
.chat-wrap .user-row .message {
    background: linear-gradient(135deg, #5458e8 0%, #7c4deb 100%) !important;
    border: none !important;
    border-radius: 18px 18px 5px 18px !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.32) !important;
    max-width: 82% !important;
    margin-left: auto !important;
}

.chat-wrap .message.user,
.chat-wrap .message.user *,
.chat-wrap .user-row .message,
.chat-wrap .user-row .message * {
    color: #ffffff !important;
}

.chat-wrap .message.bot,
.chat-wrap .bot-row .message {
    background: rgba(32, 36, 60, 0.55) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 18px 18px 18px 5px !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
    max-width: 88% !important;
    color: #d3d9e6 !important;
}

.chat-wrap .message a {
    color: #a5b4fc !important;
    text-decoration: underline !important;
    text-underline-offset: 2px;
}

/* Placeholder text before first message */
.chat-wrap .placeholder-content, .chat-wrap .placeholder {
    color: #4a5170 !important;
}

/* ─── Input Row ─── */
.input-row {
    animation: fadeIn 0.5s ease-out 0.3s both;
    margin-top: 0.25rem !important;
}

.input-row {
    gap: 8px !important;
    align-items: stretch !important;
}

.input-row textarea {
    border-radius: 999px !important;
    padding: 0.85rem 1.35rem !important;
    font-size: 0.92rem !important;
    background: rgba(13, 15, 28, 0.85) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(129, 140, 248, 0.18) !important;
    color: #e8ecf4 !important;
    resize: none !important;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35) !important;
}

.input-row textarea:focus {
    border-color: rgba(139, 92, 246, 0.55) !important;
    box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.12), 0 8px 28px rgba(0, 0, 0, 0.35) !important;
    outline: none !important;
}

.send-btn {
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    min-height: 48px !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: #fff !important;
    cursor: pointer !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 6px 22px rgba(99, 102, 241, 0.38) !important;
}

.send-btn:hover {
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 10px 28px rgba(129, 140, 248, 0.5) !important;
}

.send-btn:active {
    transform: translateY(0) scale(0.98) !important;
}

/* ─── Voice Accordion ─── */
.voice-accordion {
    background: rgba(11, 13, 26, 0.55) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(129, 140, 248, 0.12) !important;
    border-radius: 14px !important;
    margin-top: 0.6rem !important;
    overflow: hidden !important;
}

.voice-accordion .label-wrap {
    color: #818cf8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ─── Tools Row Buttons ─── */
.tools-row {
    justify-content: center !important;
    gap: 10px !important;
    margin-top: 0.6rem !important;
}

.tools-row button {
    background: rgba(13, 15, 28, 0.5) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(129, 140, 248, 0.16) !important;
    color: #7b86b8 !important;
    border-radius: 999px !important;
    padding: 6px 16px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease;
    white-space: nowrap !important;
    flex-grow: 0 !important;
    /* Gradio's min_width was squashing these into each other */
    width: auto !important;
    min-width: fit-content !important;
    overflow: visible !important;
}

/* Hide any stray component labels inside the chat panel */
.chat-wrap .label-wrap, .chat-wrap > label, .chat-wrap span[data-testid="block-info"] {
    display: none !important;
}

.tools-row button:hover {
    background: rgba(99, 102, 241, 0.12) !important;
    border-color: rgba(129, 140, 248, 0.4) !important;
    color: #c7d2fe !important;
    transform: translateY(-1px);
}

/* ─── Quick Questions ─── */
.quick-section {
    animation: fadeIn 0.5s ease-out 0.4s both;
    padding: 1rem 0 0.5rem;
}

.quick-label {
    color: #3d4364;
    font-size: 0.67rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-weight: 600;
    text-align: center;
    margin-bottom: 0.7rem;
}

/* Force a 2-column grid on the Gradio Row (which is flex by default) */
.quick-grid {
    display: grid !important;
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 10px !important;
}

.quick-card {
    width: 100% !important;
    height: auto !important;
    min-height: 50px !important;
    padding: 12px 16px !important;
    border-radius: 14px !important;
    border: 1px solid rgba(129, 140, 248, 0.12) !important;
    background: rgba(11, 13, 26, 0.55) !important;
    backdrop-filter: blur(10px) !important;
    color: #8a93b5 !important;
    font-size: 0.8rem !important;
    line-height: 1.45 !important;
    cursor: pointer !important;
    text-align: left !important;
    font-weight: 450 !important;
    justify-content: flex-start !important;
    /* Gradio buttons default to nowrap — long questions were spilling out */
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    overflow: hidden !important;
    position: relative !important;
}

.quick-card:hover {
    border-color: rgba(139, 92, 246, 0.45) !important;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.10), rgba(139, 92, 246, 0.07)) !important;
    color: #d4dcfb !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(99, 102, 241, 0.18) !important;
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
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-bottom: 0.9rem;
}

.footer-link {
    color: #5b6478;
    text-decoration: none;
    font-size: 0.78rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 999px;
    border: 1px solid rgba(129, 140, 248, 0.12);
    background: rgba(11, 13, 26, 0.5);
    backdrop-filter: blur(8px);
}

.footer-link:hover {
    color: #c7d2fe;
    border-color: rgba(139, 92, 246, 0.4);
    background: rgba(99, 102, 241, 0.1);
    transform: translateY(-1px);
}

.footer-credit {
    color: #2a3050;
    font-size: 0.66rem;
    letter-spacing: 0.06em;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(129, 140, 248, 0.22); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(139, 92, 246, 0.45); }

/* ─── Responsive ─── */
@media (max-width: 768px) {
    .gradio-container { padding: 0 0.9rem 1.5rem !important; }
    .hero { padding: 1.8rem 0 1.1rem; }
    .hero h1 { font-size: 1.6rem; }
    .hero-avatar-ring { width: 74px; height: 74px; }
    .chat-wrap { border-radius: 16px !important; }
    .quick-grid { gap: 8px !important; }
}

@media (max-width: 640px) {
    .hero h1 { font-size: 1.45rem; }
    .hero p { font-size: 0.84rem; }
    .quick-grid { grid-template-columns: 1fr !important; }
    .quick-card { min-height: 44px !important; padding: 10px 14px !important; }
    .footer-links { gap: 0.5rem; }
    .footer-link { padding: 5px 12px; font-size: 0.74rem; }
    .chat-wrap .message.user,
    .chat-wrap .user-row .message,
    .chat-wrap .message.bot,
    .chat-wrap .bot-row .message { max-width: 94% !important; }
    .input-row textarea { padding: 0.75rem 1.1rem !important; font-size: 16px !important; } /* 16px prevents iOS zoom-on-focus */
    .send-btn { min-height: 44px !important; min-width: 70px !important; }
    .tools-row button { padding: 5px 12px !important; font-size: 0.72rem !important; }
}

@media (max-width: 400px) {
    .hero h1 { font-size: 1.3rem; }
    .hero-avatar-ring { width: 64px; height: 64px; }
    .status-badge { font-size: 0.66rem; }
}

/* Respect users who prefer less motion */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
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
                Online &nbsp;·&nbsp; Replies instantly
            </div>
        </div>
    """)

    # Chatbot
    chatbot = gr.Chatbot(
        height=440,
        show_label=False,
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
        send_btn = gr.Button(
            "Send",
            variant="primary",
            scale=1,
            min_width=80,
            elem_classes=["send-btn"],
        )

    # Voice input in its own collapsible section — a full waveform widget doesn't
    # fit cleanly inside the input row
    with gr.Accordion("🎙️ Voice input", open=False, elem_classes=["voice-accordion"]):
        audio = gr.Audio(
            sources=["microphone"],
            type="filepath",
            show_label=False,
            waveform_options=gr.WaveformOptions(
                waveform_color="#6366f1",
                waveform_progress_color="#818cf8",
            )
        )

    with gr.Row(elem_classes=["tools-row"]):
        clear_btn = gr.Button("🗑️ Clear Chat", size="sm", min_width=60)
        undo_btn = gr.Button("↩️ Undo Last", size="sm", min_width=60)
        download_btn = gr.DownloadButton("💾 Export Chat", size="sm", min_width=60)

    # Quick questions grid
    gr.HTML("""
        <div class="quick-section">
            <div class="quick-label">Suggested Questions</div>
        </div>
    """)
    with gr.Row(elem_classes=["quick-grid"]):
        for q in QUICK_QUESTIONS:
            btn = gr.Button(f"{q['icon']} {q['text']}", elem_classes=["quick-card"])
            # Clicking a suggestion sends it immediately instead of just filling the box
            btn.click(
                lambda history, text=q['text']: user_message(text, None, history),
                inputs=[chatbot],
                outputs=[msg, audio, chatbot],
                queue=False,
            ).then(bot_response, chatbot, chatbot)

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
            <div class="footer-credit">Powered by Gemini 2.5 Flash · Real-time RAG</div>
        </div>
    """)

    # Event wiring (send events use the queue: audio transcription can take seconds)
    msg.submit(user_message, [msg, audio, chatbot], [msg, audio, chatbot]).then(
        bot_response, chatbot, chatbot
    )
    send_btn.click(user_message, [msg, audio, chatbot], [msg, audio, chatbot]).then(
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
