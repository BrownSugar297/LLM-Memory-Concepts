import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

USE_PROVIDER = "groq"   

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


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),  
    ("human", "{input}"),
])

chain = prompt | llm
store = ChatMessageHistory()

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda session_id: store,  
    input_messages_key="input",
    history_messages_key="history",
)

print("LangChain Memory Demo  |  type 'exit' to quit  |  type 'history' to inspect memory\n")
print("-" * 60)

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "exit":
        print("Goodbye!")
        break


    if user_input.lower() == "history":
        print("\n── Memory contents ──")
        for msg in store.messages:
            role = "Human" if msg.type == "human" else "AI"
            print(f"  [{role}]: {msg.content[:80]}")
        print("─────────────────────\n")
        continue

    response = conversation.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "default"}},
    )

    print(f"AI: {response.content}\n")
