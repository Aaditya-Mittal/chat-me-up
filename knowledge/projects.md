# Projects Deep Dive

## Social Media Engagement Tracker
**Role:** Architect & Lead Developer
**Problem Statement:** Tracking engagement metrics across multiple social media platforms was a manual, time-consuming process with no centralized view, leading to delayed reporting and missed insights.
**Solution:** I architected a centralized tracking portal on ServiceNow to aggregate cross-platform engagement data via automated REST API ingestion. I engineered backend pipelines using Flow Builder and scheduled data-fetching jobs, eliminating manual tracking efforts and enabling centralized monitoring alerts for end-users.
**Tech Stack:** ServiceNow, Flow Builder, UI Builder, REST APIs, SQL
**Demo:** https://drive.google.com/file/d/1X0vmCL21cFZ1kd6JnXrabO13jPaEmcmy/view
**Key Takeaways / What I Learned:** This project deepened my understanding of ServiceNow beyond basic administration. Building the REST API pipeline taught me how to design reliable data ingestion systems and manage scheduled jobs effectively.

---

## Anime Face Image Generator (DCGAN)
**Role:** Lead Researcher & Developer
**Problem Statement:** Generating high-fidelity synthetic anime character faces using deep learning requires careful tuning to avoid mode collapse and generate realistic features.
**Solution:** I designed and trained a Deep Convolutional Generative Adversarial Network (DCGAN) from scratch. I fine-tuned hyperparameters across the generator and discriminator networks to synthesize anime character faces successfully.
**Publications:** 
- Published and presented a technical review paper as first author at the ILIPS International Conference detailing optimal training methodologies, hyperparameter tuning, and architectural insights for generative models.
- Published in the book: "Future Trends in Smart Libraries: The Role of Artificial Intelligence, IoT and Empowerment"
**Tech Stack:** Python, PyTorch, Deep Learning, GANs
**Paper:** https://drive.google.com/file/d/1kMUeQTaXS15V1wdXfxTTvmZ4lgWwsN1Q/view
**Key Takeaways / What I Learned:** This was my first formal research experience. Training GANs taught me a lot about hyperparameter sensitivity, the balance between generator and discriminator loss, and the immense patience required for deep learning model training. Writing the paper improved my technical writing skills significantly.

---

## AI Website Summarizer
**Role:** Full Stack Developer
**Problem Statement:** Traditional web scrapers (like simple requests + BeautifulSoup) fail to extract content from modern, JavaScript-heavy single-page applications, making it hard to generate summaries for modern websites.
**Solution:** I engineered a robust web scraping pipeline utilizing Playwright for headless browser automation to render JS-heavy sites, paired with BeautifulSoup4 to parse and extract clean HTML text content. I integrated Gemini 2.5 Flash for intelligent summaries and served the entire system through a high-performance FastAPI backend using streaming responses to reduce perceived latency.
**Tech Stack:** Python, FastAPI, Playwright, BeautifulSoup4, Gemini 2.5 Flash
**Live Demo:** https://website-summarizer-livid.vercel.app/
**Key Takeaways / What I Learned:** Playwright was new to me - I chose it specifically because traditional scrapers couldn't handle JS-rendered content. Implementing the FastAPI streaming response pattern was a fun challenge to get right and drastically improved the user experience.

---

## Chat With Me (AI Portfolio Assistant)
**Role:** AI Engineer & Developer
**Problem Statement:** Portfolio websites are static and visitors can't easily explore or ask questions about a candidate's background interactively.
**Solution:** I built an AI chatbot using CrewAI that acts as my personal representative on my portfolio website. It answers questions in my authentic voice using a curated knowledge base (RAG). It maintains conversational history, uses an advanced multi-step task configuration to adapt to different visitor types (recruiters vs developers), and features a premium Gradio UI.
**Tech Stack:** CrewAI, Gemini 2.5 Flash, Gradio 6.x, Python, Markdown
**Key Takeaways / What I Learned:** Working with CrewAI's agent/task architecture, RAG optimization, handling conversational memory through context injection, and crafting prompt engineering for strong persona adherence.
