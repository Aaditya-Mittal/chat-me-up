<div align="center">

# 💬 Chat Me Up

**An AI-powered portfolio chatbot that represents me in my own voice. Customize it to create your own chat-bot for your portfolio.**

Built with [CrewAI](https://crewai.com) · Powered by [Gemini](https://deepmind.google/technologies/gemini/) · Served via [Gradio](https://gradio.app)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![Gradio](https://img.shields.io/badge/Gradio-3.x%20%7C%204.x-F97316?style=flat-square)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 🤔 What is this?

Portfolio websites are static — visitors browse, read, and leave. **Chat Me Up** changes that by letting visitors have a real conversation with an AI version of me.

It's not a generic chatbot. It's trained on my actual background, experiences, projects, and personality — so it responds the way I would: conversationally, authentically, with real stories and genuine enthusiasm.

> **Ask it about my hackathon wins, my full-time Analyst role at Deloitte, my tech stack, or even how I learned Flutter in 2 days — it'll tell you the story, not just the facts.**

---

## ✨ Features

- **🎤 Voice Input** — Speak your questions directly to the bot using native Gemini audio transcription
- **🎭 Authentic Persona** — Speaks in first person with my real communication style, humor, and personality quirks
- **📚 Real-Time Knowledge** — Compiles a curated knowledge base (about me, resume, deep-dive projects, and FAQs) into an optimized system context at startup
- **⚡ Native Streaming** — Zero-latency token streaming directly from the Gemini API for ultra-fast, human-like typing
- **🧠 Conversation Context & Time** — Maintains chat history and knows the current real-world date and time
- **🛡️ Enterprise-Grade Security** — Built-in protection against prompt injection, jailbreaks, and token-exhaustion
- **💾 Export Chat** — Recruiters can download a `.txt` transcript of their entire conversation with 1-click
- **💅 Premium UI** — Glassmorphism dark-themed Gradio interface with native quick-question buttons, Clear/Undo controls, and smooth micro-animations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  Gradio UI (app.py)              │
│  ┌───────────┐  ┌───────────┐  ┌─────────────┐  │
│  │ Chat Area │  │ Input Box │  │ Quick Cards │  │
│  └─────┬─────┘  └─────┬─────┘  └──────┬──────┘  │
│        │              │               │          │
│        └──────────────┼───────────────┘          │
│                       │                          │
│              ┌────────▼────────┐                 │
│              │ History Context │                 │
│              └────────┬────────┘                 │
└───────────────────────┼──────────────────────────┘
                        │
              ┌─────────▼─────────┐
              │ Gemini 2.5 Flash  │
              │  (Direct Client)  │
              │                   │
              │  ┌─────────────┐  │
              │  │  Streaming  │  │
              │  │  Response   │  │
              │  └──────┬──────┘  │
              │         │         │
              │  ┌──────▼──────┐  │
              │  │ In-Memory   │  │
              │  │ System ctx  │  │
              │  │ (Compiled)  │  │
              │  └──────┬──────┘  │
              │         │         │
              │  ┌──────▼──────┐  │
              │  │ about_me.md │  │
              │  │ agents.yaml │  │
              │  └─────────────┘  │
              └───────────────────┘
```

---

## 📁 Project Structure

```
chat_with_me/
├── src/chat_with_me/
│   ├── app.py                 # Gradio UI + Audio transcription + Chat Export
│   ├── crew.py                # CrewAI crew definition + knowledge sources
│   ├── main.py                # CLI entry point (crewai run)
│   └── config/
│       ├── agents.yaml        # Persona, storytelling rules, and ethical boundaries
│       └── tasks.yaml         # Response strategies and XML prompt injection defenses
├── knowledge/
│   ├── about_me.md            # Personal background, journey, personality
│   ├── resume.md              # Experience, skills, achievements
│   ├── projects.md            # Deep technical dives into key projects
│   └── faq.yaml               # Common recruiter/visitor questions
├── pyproject.toml             # Dependencies & build config
├── .env                       # API keys (not committed)
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10 - 3.13
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **Google API Key** — for Gemini LLM and embeddings ([Get one here](https://aistudio.google.com/apikey))

### Installation

```bash
# Clone the repo
git clone https://github.com/Aaditya-Mittal/chat-me-up.git
cd chat-me-up

# Install dependencies
uv sync
```

### Configuration

Create a `.env` file in the project root:

```env
MODEL=gemini/gemini-2.5-flash
GOOGLE_API_KEY=your_google_api_key_here
EMBEDDINGS_GOOGLE_GENERATIVE_AI_MODEL_NAME=gemini-embedding-001
```

### Running

**Gradio Chat UI (recommended):**

```bash
uv run python -m chat_with_me.app
# Opens at http://localhost:7860
```

**CLI mode (single query):**

```bash
crewai run
```

---

## 🧩 How It Works

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Gemini 1.5/2.5 Flash | Generates conversational responses and natively transcribes audio |
| **Streaming** | Google GenAI SDK | Bypasses orchestration latency for zero-delay token streaming directly to UI |
| **Knowledge** | In-Memory Compilation | Parses `.md`/`.yaml` files at startup into an optimized dynamic system prompt |
| **Frontend** | Gradio 3.x/4.x | Interactive chat interface with premium glassmorphism theme and audio capture |
| **History** | In-context injection | Full conversation and dynamic date/time passed directly in chat history |
| **Security** | XML Wrapping & Limits | Prevents prompt injections, jailbreaks, and token exhaustion |

### Agent Configuration

The agent's personality is defined in `agents.yaml` with:

- **7 signature stories** — go-to narratives the agent tells naturally (ISRO spark, Flutter in 2 days, pandemic reset, etc.)
- **Emotional intelligence** — adapts tone based on visitor mood (enthusiastic, skeptical, casual, etc.)
- **Good vs bad examples** — 3 side-by-side comparisons teaching natural vs robotic responses
- **Anti-patterns** — 10 things the agent must never do

### Task Configuration

The task in `tasks.yaml` follows a 3-step pipeline:

1. **Analyze** — classify message type, read tone, check history
2. **Craft** — apply the right strategy from 16 message type handlers
3. **Quality Check** — 10-point checklist before responding

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.10+ |
| **LLM SDK** | `google-genai` (Native Client) |
| **LLM** | Google Gemini 2.5 Flash |
| **Frontend** | Gradio 3.x/4.x |
| **Package Manager** | uv |
| **Legacy/CLI** | CrewAI 1.14+ (Available via `crew.py` fallback) |

---

## 🔧 Customization

Want to build your own persona chatbot? Here's what to change:

1. **`knowledge/about_me.md` & `resume.md`** — Replace with your personal story, journey, and experience
2. **`knowledge/projects.md` & `faq.yaml`** — Add your technical deep dives and common questions
3. **`config/agents.yaml`** — Update the backstory, signature stories, and ethical boundaries
4. **`config/tasks.yaml`** — Adjust response strategies and injection defenses
5. **`app.py`** — Update the header, avatar, quick questions, and social links

---

## 🤝 Contributing

This is a personal project, but ideas and feedback are welcome! Feel free to:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/cool-idea`)
3. Commit your changes (`git commit -m "Add cool idea"`)
4. Push to the branch (`git push origin feature/cool-idea`)
5. Open a Pull Request

---

<div align="center">

**Built by [Aaditya Mittal](https://portfolio-aaditya-mittal.vercel.app/)** · [LinkedIn](https://www.linkedin.com/in/aadityamittal01) · [GitHub](https://github.com/Aaditya-Mittal)

</div>
