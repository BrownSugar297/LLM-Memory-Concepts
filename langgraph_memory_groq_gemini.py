import os
import os
from dotenv import load_dotenv 
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END

load_dotenv() 

USE_PROVIDER = "groq"  

class ChatState(TypedDict):
    messages: List[BaseMessage]  


if USE_PROVIDER == "groq":
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    print("✅ Using Groq (llama-3.3-70b-versatile)\n")

elif USE_PROVIDER == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )
    print("✅ Using Gemini (gemini-2.5-flash)\n")

else:
    raise ValueError("USE_PROVIDER must be 'groq' or 'gemini'")


def chatbot_node(state: ChatState) -> dict:
    """
    Core node: takes all messages from state, calls LLM, appends the reply.
    
    Because we pass ALL messages (state["messages"]) to the LLM,
    it naturally has full conversation memory — no magic required.
    """

    all_messages = state["messages"]

    ai_response: AIMessage = llm.invoke(all_messages)

    return {
        "messages": all_messages + [ai_response]
    }

builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot_node)
builder.set_entry_point("chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()


print("LangGraph Memory Demo  |  type 'exit' to quit  |  type 'history' to inspect state\n")
print("-" * 60)

state: ChatState = {"messages": []}

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    if user_input.lower() == "history":
        print("\n── State contents ──")
        for i, msg in enumerate(state["messages"]):
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            print(f"  [{i}] {role}: {msg.content[:80]}...")
        print("────────────────────\n")
        continue

    state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
    state = graph.invoke(state)

    ai_reply = state["messages"][-1]
    print(f"AI: {ai_reply.content}\n")
