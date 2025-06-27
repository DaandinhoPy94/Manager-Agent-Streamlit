# ==============================================================================
# STABIELE WERKENDE CODE MET GEHEUGEN
# ==============================================================================

import streamlit as st
import os
import pandas as pd
import sys
import platform

# --- PADEN CONFIGURATIE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')

# --- SQLITE FIX VOOR STREAMLIT CLOUD ---
if platform.system() == "Linux":
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# --- LANGCHAIN IMPORTS ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain import hub

# --- PAGINA CONFIGURATIE ---
st.set_page_config(page_title="AgentManagerGPT", page_icon="🧑‍💼", layout="wide")

# --- DATABASE & VECTORSTORE PADEN ---
DB_PATH = os.path.join(ROOT_DIR, 'data', 'portfolio.db')
VS_PATH = os.path.join(ROOT_DIR, 'vectorstore')

# --- CUSTOM PROMPT MET GEHEUGEN ---
REACT_PROMPT_WITH_MEMORY = """You are a helpful assistant that helps analyze a real estate portfolio database.

You have access to the following tools:
{tools}

Previous conversation:
{chat_history}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Remember to use the conversation history to understand context from previous questions.

Question: {input}
Thought: {agent_scratchpad}"""

@st.cache_resource
def setup_agent(_groq_api_key):
    """
    Zet een werkende agent op met geheugen.
    """
    print("Agent wordt opgezet met geheugen (stabiele versie)...")

    # --- LLM ---
    llm = ChatGroq(
        temperature=0,
        model_name="llama3-8b-8192",
        groq_api_key=_groq_api_key
    )

    # --- SQL Tools ---
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = sql_toolkit.get_tools()

    # --- RAG Tool ---
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=VS_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    rag_tool = create_retriever_tool(
        retriever,
        "portfolio_search",
        "Zoek semantisch in de portfolio database voor algemene vragen en analyses."
    )
    
    all_tools = sql_tools + [rag_tool]

    # --- Prompt met geheugen ---
    prompt = PromptTemplate.from_template(REACT_PROMPT_WITH_MEMORY)

    # --- Geheugen ---
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=False,  # Gebruik string format voor simpliciteit
        output_key="output"
    )

    # --- Agent ---
    agent = create_react_agent(llm, all_tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=False
    )

    print("Agent succesvol opgezet!")
    return agent_executor

# --- HOOFD APPLICATIE ---
st.title("🧑‍💼 AgentManagerGPT")
st.markdown("Stel een vraag aan de Manager. Hij onthoudt nu je eerdere vragen!")

# --- API KEY HANDLING ---
from dotenv import load_dotenv

groq_api_key = ""

if os.environ.get("STREAMLIT_SERVER_ENABLED"):
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        st.sidebar.success("✅ API Key geladen!")
    except KeyError:
        st.sidebar.error("❌ Geen API Key gevonden!")
        st.stop()
else:
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        st.sidebar.success("✅ API Key geladen!")
    else:
        st.sidebar.warning("Voer je API key in:")
        groq_api_key = st.sidebar.text_input("Groq API Key:", type="password")

if not groq_api_key:
    st.info("API Key is vereist.")
    st.stop()

# Setup agent
agent_executor = setup_agent(groq_api_key)

# --- PORTFOLIO DATA VIEWER ---
with st.expander("📊 Portfolio Data", expanded=False):
    try:
        df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'portfolio.csv'))
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        st.dataframe(df, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Panden", len(df))
        col2.metric("Totale waarde", f"€{df['value'].sum():,.0f}")
        col3.metric("Gem. leegstand", f"{df['vacancyrate'].mean()*100:.1f}%")
        col4.metric("Jaarinkomsten", f"€{df['anualincome'].sum():,.0f}")
    except Exception as e:
        st.error(f"Fout: {e}")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("💡 Tips")
    st.info("""
    **Met geheugen!** Probeer:
    1. "Hoeveel panden zijn er in Amsterdam?"
    2. "Wat is de totale waarde daarvan?"
    3. "En de gemiddelde leegstand?"
    
    De agent onthoudt nu je context!
    """)
    
    if st.button("🔄 Reset Geheugen"):
        st.session_state.messages = []
        st.cache_resource.clear()
        st.rerun()

# Toon berichten
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Vraag iets over de portfolio..."):
    # Voeg toe aan geschiedenis
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Aan het nadenken..."):
            try:
                response = agent_executor.invoke({"input": prompt})
                answer = response.get("output", "Geen antwoord gevonden.")
            except Exception as e:
                answer = f"Er ging iets mis: {str(e)}"
                st.error(f"Debug: {e}")
        
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})