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


# --- Theme: "Emerald Noir" — green-tinted black, emerald primaries, mint accents ---

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    font=(gr.themes.GoogleFont("Inter"), "system-ui", "-apple-system", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "Fira Code", "monospace"),
).set(
    body_background_fill="#040807",
    body_background_fill_dark="#040807",
    body_text_color="#e2ece8",
    body_text_color_dark="#e2ece8",

    block_background_fill="rgba(13, 23, 20, 0.55)",
    block_background_fill_dark="rgba(13, 23, 20, 0.55)",
    block_border_color="rgba(52, 211, 153, 0.10)",
    block_border_color_dark="rgba(52, 211, 153, 0.10)",
    block_label_text_color="#34d399",
    block_label_text_color_dark="#34d399",
    block_shadow="none",
    block_shadow_dark="none",

    button_primary_background_fill="linear-gradient(135deg, #10b981 0%, #059669 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #10b981 0%, #059669 100%)",
    button_primary_text_color="#04110c",
    button_primary_text_color_dark="#04110c",
    button_primary_background_fill_hover="linear-gradient(135deg, #34d399 0%, #10b981 100%)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #34d399 0%, #10b981 100%)",

    input_background_fill="rgba(13, 23, 20, 0.9)",
    input_background_fill_dark="rgba(13, 23, 20, 0.9)",
    input_border_color="rgba(52, 211, 153, 0.16)",
    input_border_color_dark="rgba(52, 211, 153, 0.16)",
    input_border_color_focus="rgba(52, 211, 153, 0.45)",
    input_border_color_focus_dark="rgba(52, 211, 153, 0.45)",
    input_placeholder_color="#3d4f47",
    input_placeholder_color_dark="#3d4f47",
)


# --- CSS ---

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

/* ─── Design tokens: Emerald Noir ─── */
:root {
    --bg: #040807;
    --panel: rgba(13, 23, 20, 0.60);
    --panel-heavy: rgba(10, 18, 15, 0.75);
    --border: rgba(52, 211, 153, 0.12);
    --border-strong: rgba(52, 211, 153, 0.40);
    --primary: #10b981;
    --primary-deep: #059669;
    --accent: #6ee7b7;
    --text: #e2ece8;
    --text-bright: #f2f8f5;
    --muted: #7f948c;
    --faint: #46564f;
    /* Vertical space reserved for everything that isn't the chat panel.
       The chatbot's inline height is calc(100dvh - var(--chat-offset)). */
    --chat-offset: 460px;
}

/* ─── Reset & Base ─── */
* { transition: background-color 0.3s ease, border-color 0.3s ease, color 0.2s ease, transform 0.2s ease, box-shadow 0.3s ease, opacity 0.3s ease; }

::selection { background: rgba(16, 185, 129, 0.35); color: var(--text-bright); }

body {
    background: var(--bg) !important;
}

/* Ambient aurora glow layer behind everything */
body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(ellipse 60% 40% at 20% -5%, rgba(16, 185, 129, 0.14), transparent 60%),
        radial-gradient(ellipse 50% 35% at 85% 0%, rgba(45, 212, 191, 0.10), transparent 60%),
        radial-gradient(ellipse 45% 30% at 50% 110%, rgba(5, 150, 105, 0.10), transparent 65%);
}

.gradio-container {
    max-width: 840px !important;
    margin: 0 auto !important;
    background: transparent !important;
    padding: 0 1.25rem 0.75rem !important;
    overflow-x: hidden !important;
    position: relative;
    z-index: 1;
}

footer { display: none !important; }

/* ─── Hero Header (compact — the shell must fit in one viewport) ─── */
.hero {
    text-align: center;
    padding: 1.4rem 0 0.9rem;
    animation: fadeInDown 0.6s ease-out;
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-12px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero-avatar-ring {
    width: 64px; height: 64px;
    border-radius: 50%;
    padding: 3px;
    background: conic-gradient(from 180deg, #10b981, #6ee7b7, #2dd4bf, #10b981);
    animation: ringSpin 6s linear infinite;
    margin: 0 auto 0.8rem;
    display: block;
    position: relative;
    box-shadow: 0 0 36px rgba(16, 185, 129, 0.40), 0 0 80px rgba(45, 212, 191, 0.15);
}

@keyframes ringSpin {
    to { transform: rotate(360deg); }
}

.hero-avatar-ring img {
    width: 100%; height: 100%;
    border-radius: 50%;
    border: 3px solid var(--bg);
    display: block;
    animation: ringSpin 6s linear infinite reverse; /* keep face upright */
}

.hero h1 {
    font-family: 'Sora', 'Inter', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0 0 0.3rem;
    letter-spacing: -0.035em;
    background: linear-gradient(120deg, #f2f8f5 20%, #6ee7b7 55%, #5eead4 80%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}

.hero p {
    color: var(--muted);
    font-size: 0.86rem;
    margin: 0;
    font-weight: 400;
    line-height: 1.5;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-top: 0.6rem;
    padding: 4px 13px;
    border-radius: 20px;
    background: rgba(52, 211, 153, 0.07);
    border: 1px solid rgba(52, 211, 153, 0.20);
    backdrop-filter: blur(6px);
    font-size: 0.7rem;
    color: #34d399;
    font-weight: 500;
    letter-spacing: 0.04em;
}

.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #34d399;
    animation: softPulse 2.5s ease-in-out infinite;
}

@keyframes softPulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.3); }
    50% { opacity: 0.7; box-shadow: 0 0 0 4px rgba(52, 211, 153, 0); }
}

/* ─── Chatbot ─── */
.chat-wrap {
    animation: fadeIn 0.5s ease-out 0.2s both;
    /* Guard for tiny viewports — the page scrolls instead of crushing the chat */
    min-height: 300px !important;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Glass panel around the conversation */
.chat-wrap {
    background: var(--panel) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid var(--border) !important;
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
    background: linear-gradient(135deg, #0ea371 0%, #047857 100%) !important;
    border: none !important;
    border-radius: 18px 18px 5px 18px !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.28) !important;
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
    background: rgba(23, 37, 32, 0.55) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 18px 18px 18px 5px !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
    max-width: 88% !important;
    color: var(--text) !important;
}

.chat-wrap .message a {
    color: var(--accent) !important;
    text-decoration: underline !important;
    text-underline-offset: 2px;
}

/* Placeholder text before first message */
.chat-wrap .placeholder-content, .chat-wrap .placeholder {
    color: var(--faint) !important;
}

/* Hide any stray component labels inside the chat panel */
.chat-wrap .label-wrap, .chat-wrap > label, .chat-wrap span[data-testid="block-info"] {
    display: none !important;
}

/* ─── Quick Question Chips (horizontal scroll strip) ─── */
.quick-strip {
    animation: fadeIn 0.5s ease-out 0.3s both;
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    gap: 8px !important;
    margin-top: 0.6rem !important;
    padding: 2px 2px 6px !important;
    scrollbar-width: thin;
    -webkit-overflow-scrolling: touch;
    /* fade hint that the strip scrolls */
    mask-image: linear-gradient(to right, black 92%, transparent 100%);
    -webkit-mask-image: linear-gradient(to right, black 92%, transparent 100%);
}

.quick-strip button {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: fit-content !important;
    height: auto !important;
    border-radius: 999px !important;
    border: 1px solid var(--border) !important;
    background: var(--panel) !important;
    backdrop-filter: blur(10px) !important;
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 7px 15px !important;
    white-space: nowrap !important;
    cursor: pointer !important;
}

.quick-strip button:hover {
    border-color: var(--border-strong) !important;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(45, 212, 191, 0.08)) !important;
    color: #b9f4dd !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(16, 185, 129, 0.18) !important;
}

.quick-strip button:active {
    transform: translateY(0);
}

/* ─── Input Row ─── */
.input-row {
    animation: fadeIn 0.5s ease-out 0.3s both;
    margin-top: 0.5rem !important;
    gap: 8px !important;
    align-items: stretch !important;
}

.input-row textarea {
    border-radius: 999px !important;
    padding: 0.85rem 1.35rem !important;
    font-size: 0.92rem !important;
    background: var(--panel-heavy) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(52, 211, 153, 0.18) !important;
    color: var(--text-bright) !important;
    resize: none !important;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35) !important;
}

.input-row textarea:focus {
    border-color: rgba(52, 211, 153, 0.55) !important;
    box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.12), 0 8px 28px rgba(0, 0, 0, 0.35) !important;
    outline: none !important;
}

.send-btn {
    border-radius: 999px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    min-height: 48px !important;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    border: none !important;
    color: #04110c !important;
    cursor: pointer !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 6px 22px rgba(16, 185, 129, 0.35) !important;
}

.send-btn:hover {
    background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 10px 28px rgba(52, 211, 153, 0.45) !important;
}

.send-btn:active {
    transform: translateY(0) scale(0.98) !important;
}

/* ─── Voice Accordion ─── */
.voice-accordion {
    background: var(--panel) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    margin-top: 0.5rem !important;
    overflow: hidden !important;
}

.voice-accordion .label-wrap {
    color: #34d399 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ─── Tools Row Buttons ─── */
.tools-row {
    justify-content: center !important;
    gap: 10px !important;
    margin-top: 0.5rem !important;
}

.tools-row button {
    background: rgba(10, 18, 15, 0.5) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
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

.tools-row button:hover {
    background: rgba(16, 185, 129, 0.12) !important;
    border-color: var(--border-strong) !important;
    color: #b9f4dd !important;
    transform: translateY(-1px);
}

/* ─── Footer (compact single block) ─── */
.footer-section {
    animation: fadeIn 0.5s ease-out 0.5s both;
    text-align: center;
    padding: 0.8rem 0 0.3rem;
    border-top: 1px solid rgba(16, 185, 129, 0.06);
    margin-top: 0.6rem;
}

.footer-links {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-bottom: 0.6rem;
}

.footer-link {
    color: var(--muted);
    text-decoration: none;
    font-size: 0.76rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 13px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(10, 18, 15, 0.5);
    backdrop-filter: blur(8px);
}

.footer-link:hover {
    color: #b9f4dd;
    border-color: var(--border-strong);
    background: rgba(16, 185, 129, 0.1);
    transform: translateY(-1px);
}

.footer-credit {
    color: #24352d;
    font-size: 0.64rem;
    letter-spacing: 0.06em;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(52, 211, 153, 0.22); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(52, 211, 153, 0.45); }

/* ─── Responsive ───
   The chat panel height is calc(100dvh - var(--chat-offset)); each breakpoint
   only needs to re-declare --chat-offset to match its chrome height. */
@media (max-width: 768px) {
    .gradio-container { padding: 0 0.9rem 0.6rem !important; }
    .hero { padding: 1.1rem 0 0.7rem; }
    .hero h1 { font-size: 1.45rem; }
    .hero-avatar-ring { width: 56px; height: 56px; }
    .chat-wrap { border-radius: 16px !important; }
}

@media (max-width: 640px) {
    :root { --chat-offset: 440px; }
    .hero h1 { font-size: 1.35rem; }
    .hero p { font-size: 0.8rem; }
    .footer-links { gap: 0.45rem; }
    .footer-link { padding: 4px 11px; font-size: 0.72rem; }
    .chat-wrap .message.user,
    .chat-wrap .user-row .message,
    .chat-wrap .message.bot,
    .chat-wrap .bot-row .message { max-width: 94% !important; }
    .input-row textarea { padding: 0.75rem 1.1rem !important; font-size: 16px !important; } /* 16px prevents iOS zoom-on-focus */
    .send-btn { min-height: 44px !important; min-width: 70px !important; }
    .tools-row button { padding: 5px 12px !important; font-size: 0.72rem !important; }
    .quick-strip button { padding: 6px 13px !important; font-size: 0.74rem !important; }
}

@media (max-width: 400px) {
    .hero h1 { font-size: 1.25rem; }
    .hero-avatar-ring { width: 50px; height: 50px; }
    .status-badge { font-size: 0.64rem; }
}

/* Short viewports (landscape phones, small laptops): shrink the chrome so the
   chat still gets usable height without forcing a page scroll */
@media (max-height: 760px) {
    :root { --chat-offset: 360px; }
    .hero { padding: 0.8rem 0 0.5rem; }
    .hero p, .status-badge { display: none; }
    .hero-avatar-ring { width: 44px; height: 44px; margin-bottom: 0.5rem; }
    .hero h1 { font-size: 1.25rem; }
    .footer-section { display: none; }
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

AVATAR_URL = "https://api.dicebear.com/9.x/initials/svg?seed=AM&backgroundColor=047857&textColor=ffffff"


# --- Build the UI ---

with gr.Blocks(title="Chat with Aaditya Mittal", theme=theme, css=CSS) as demo:

    # Hero
    gr.HTML(f"""
        <div class="hero">
            <div class="hero-avatar-ring">
                <img src="{AVATAR_URL}" alt="AM" />
            </div>
            <h1>Chat with Aaditya</h1>
            <p>AI portfolio assistant — ask about my work, skills, or experience</p>
            <div class="status-badge">
                <span class="status-dot"></span>
                Online &nbsp;·&nbsp; Replies instantly
            </div>
        </div>
    """)

    # Chatbot — height is viewport-driven so the whole shell fits one screen;
    # --chat-offset is tuned per breakpoint in the CSS
    chatbot = gr.Chatbot(
        height="calc(100dvh - var(--chat-offset, 460px))",
        show_label=False,
        avatar_images=(None, AVATAR_URL),
        elem_classes=["chat-wrap"],
        layout="bubble",
        placeholder="Say hi or pick a question below to get started →",
        buttons=["copy"],
    )

    # Quick questions — horizontally scrollable chip strip between chat and input
    with gr.Row(elem_classes=["quick-strip"]):
        quick_buttons = [
            gr.Button(f"{q['icon']} {q['text']}", size="sm", min_width=40)
            for q in QUICK_QUESTIONS
        ]

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
                waveform_color="#10b981",
                waveform_progress_color="#34d399",
            )
        )

    with gr.Row(elem_classes=["tools-row"]):
        clear_btn = gr.Button("🗑️ Clear Chat", size="sm", min_width=60)
        undo_btn = gr.Button("↩️ Undo Last", size="sm", min_width=60)
        download_btn = gr.DownloadButton("💾 Export Chat", size="sm", min_width=60)

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

    # Quick-question wiring — clicking a chip sends it immediately
    for q, btn in zip(QUICK_QUESTIONS, quick_buttons):
        btn.click(
            lambda history, text=q['text']: user_message(text, None, history),
            inputs=[chatbot],
            outputs=[msg, audio, chatbot],
            queue=False,
        ).then(bot_response, chatbot, chatbot)

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
    )
