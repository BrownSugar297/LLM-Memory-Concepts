# 🧠 LangChain & LangGraph Memory — Groq & Gemini

A hands-on project to understand **conversational memory** in LLM applications using two popular frameworks — LangChain and LangGraph — with support for both **Groq** and **Google Gemini** as LLM providers.

---

## 📁 Project Structure

```
langchain_langgraph_memory/
│
├── langchain_memory_groq_gemini.py   # LangChain memory demo
├── langgraph_memory_groq_gemini.py   # LangGraph memory demo
├── requirements.txt                  # All dependencies
├── .env                              # API keys (never commit this!)
└── README.md
```

---

## 🧩 Core Concept: What is Memory?

Memory in LLM apps means the model can **remember previous messages** in a conversation. Without it, every message is treated as a fresh conversation.

| | LangChain | LangGraph |
|---|---|---|
| **Memory lives in** | `ConversationBufferMemory` object | `ChatState` dict |
| **How it works** | Auto-injected by `ConversationChain` | Passed explicitly through graph nodes |
| **Visibility** | Hidden / automatic | Explicit / inspectable |
| **Best for** | Simple chatbots | Agents, pipelines, multi-step flows |

---

## ⚙️ Setup

### 1. Clone / download the project

```bash
cd your-project-folder
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your free keys here:
- **Groq** → https://console.groq.com
- **Gemini** → https://aistudio.google.com/app/apikey

### 5. Choose your provider

At the top of either file, set:

```python
USE_PROVIDER = "groq"    # or "gemini"
```

---

## 🚀 Run

```bash
# LangChain version
python langchain_memory_groq_gemini.py

# LangGraph version
python langgraph_memory_groq_gemini.py
```

---

## 🔬 How Memory Works — Side by Side

### LangChain

```python
memory = ConversationBufferMemory()

conversation = ConversationChain(llm=llm, memory=memory)

# Each call automatically:
#   1. Loads history from memory
#   2. Sends history + new message to LLM
#   3. Saves the reply back into memory
response = conversation.predict(input="Hello!")
```

Memory is managed **for you** — you never touch the message list directly.

---

### LangGraph

```python
class ChatState(TypedDict):
    messages: List[BaseMessage]   # This IS the memory

def chatbot_node(state: ChatState):
    response = llm.invoke(state["messages"])  # full history passed in
    return {"messages": state["messages"] + [response]}
```

Memory is **your responsibility** — it's just a list you build and pass around. Full control, full visibility.

---

## 🧪 Test Memory is Working

Run either script and try this conversation:

```
You: my name is Rahim
You: I am from Barishal
You: what is my name?         ← should remember
You: where am I from?         ← should remember
You: history                  ← inspect stored messages
You: exit
```

Both scripts support the `history` command to print what's currently stored in memory.

---

## 📦 Dependencies

```
langchain
langgraph
langchain-core
langchain-groq
langchain-google-genai
langchain-community
python-dotenv
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## 🔐 Security Note

**Never commit your `.env` file to Git.**

Add this to your `.gitignore`:

```
.env
venv/
__pycache__/
*.pyc
```

---

## 💡 What to Explore Next

- `ConversationSummaryMemory` — summarizes old turns to save tokens
- `ConversationBufferWindowMemory` — keeps only the last N turns
- LangGraph `checkpointer` — persists memory to disk across sessions
- Multi-agent graphs with shared state

---

## 👤 Author

Built for learning LangChain & LangGraph memory concepts.
