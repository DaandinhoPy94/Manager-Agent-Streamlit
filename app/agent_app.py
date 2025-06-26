# ==============================================================================
# VOLLEDIGE, CORRECTE CODE VOOR app/agent_app.py
# ==============================================================================

import streamlit as st
import os
import pandas as pd
import time
import re

# --- PADEN CONFIGURATIE ---
# Dit zorgt ervoor dat de paden altijd kloppen, zowel lokaal als op Streamlit.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')

# --- SQLITE FIX VOOR STREAMLIT CLOUD ---
import sys
import platform

# Alleen op Linux-systemen (zoals Streamlit Cloud) de sqlite3 module vervangen.
# Op Windows en macOS wordt dit blok overgeslagen.
if platform.system() == "Linux":
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# --- LANGCHAIN IMPORTS ---
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.agents.structured_chat.prompt import render_structured_chat_prompt
from langchain.memory import ConversationBufferMemory
from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- PAGINA CONFIGURATIE ---
st.set_page_config(page_title="AgentManagerGPT", page_icon="🧑‍💼", layout="wide")

# --- DATABASE & VECTORSTORE PADEN ---
DB_PATH = os.path.join(ROOT_DIR, 'data', 'portfolio.db')
VS_PATH = os.path.join(ROOT_DIR, 'vectorstore')

@st.cache_resource
def setup_agent(_groq_api_key):
    """
    Zet de volledige agent op met beide tools EN geheugen.
    """
    print("Agent wordt opgezet...")

    # --- De Intelligentie ---
    llm = ChatGroq(
        temperature=0,
        model_name="llama3-8b-8192",
        groq_api_key=_groq_api_key
    )

    # --- TOOL 1: SQL Specialist ---
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = sql_toolkit.get_tools()

    # Voeg tool beschrijvingen toe
    for tool in sql_tools:
        if tool.name == "sql_db_query":
            tool.description = "Gebruik deze tool voor SQL queries op de 'portfolio' tabel. Kolommen: id, address, type, value, vacancyrate, anualincome, endlease."
        elif tool.name == "sql_db_list_tables":
            tool.description = "Lijst alle tabellen. Er is één tabel: 'portfolio'"
        elif tool.name == "sql_db_schema":
            tool.description = "Schema van de 'portfolio' tabel bekijken"

    # --- TOOL 2: RAG Specialist ---
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=VS_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
    rag_tool = create_retriever_tool(
        retriever,
        "portfolio_analyst_tool",
        "Gebruik deze tool voor semantische zoekopdrachten, open vragen, risico-analyses, en meningen over specifieke groepen panden."
    )
    all_tools = sql_tools + [rag_tool]

    # --- DE PROMPT ---
    # We bouwen de prompt dynamisch op basis van de tools.
    # Dit zorgt ervoor dat de 'tools' en 'tool_names' variabelen correct worden gevuld.
    prompt = render_structured_chat_prompt(
        tools=all_tools,
        # Voeg de conversationele placeholders toe
        input_variables=["input", "chat_history"],
        memory_prompts=[MessagesPlaceholder("chat_history", optional=True)],
)
    # --- HET GEHEUGEN ---
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # --- De "Manager" (De Agent) ---
    # We gebruiken create_structured_chat_agent, die beter omgaat met complexe tools en prompts.
    agent = create_structured_chat_agent(llm, all_tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        llm=llm
    )

    print("Agent succesvol opgezet met geheugen!")
    return agent_executor

# --- HOOFD APPLICATIE ---
st.title("🧑‍💼 AgentManagerGPT")
st.markdown("Stel een vraag aan de Onderzoeksmanager. Hij kiest de juiste specialist voor de klus.")

# --- DEFINITIEVE, ROBUUSTE API KEY HANDLING ---

# Voeg deze imports toe bovenaan je script, bij de andere imports
from dotenv import load_dotenv
import os

# ... (in de body van je script, na de imports) ...

groq_api_key = ""

# EERSTE POGING: Streamlit Cloud Secrets (voor productie)
# os.environ.get("STREAMLIT_SERVER_ENABLED") is een betrouwbare manier om te checken of we op de cloud draaien.
if os.environ.get("STREAMLIT_SERVER_ENABLED"):
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        st.sidebar.success("✅ API Key geladen via Streamlit Secrets!")
    except KeyError:
        st.sidebar.error("❌ Geen API Key gevonden in Streamlit Secrets!")
        st.stop()

# TWEEDE POGING: Lokaal .env bestand (voor ontwikkeling)
else:
    load_dotenv() # Simpele aanroep, laadt .env uit de hoofdmap
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        st.sidebar.success("✅ API Key geladen via lokaal .env bestand!")
    else:
        # LAATSTE REDMIDDEL: Handmatige invoer
        st.sidebar.warning("Geen .env bestand gevonden. Voer key handmatig in.")
        groq_api_key = st.sidebar.text_input(
            "Voer je Groq API Key in:",
            type="password",
            key="local_api_key"
        )

# Stop de app als er na alle pogingen nog steeds geen key is.
if not groq_api_key:
    st.info("API Key is vereist. Voer deze in via de sidebar of een .env bestand.")
    st.stop()



# Belangrijk: De setup_agent functie moet NU pas worden aangeroepen
agent_executor = setup_agent(groq_api_key)

# --- PORTFOLIO TABEL WEERGEVEN ---
with st.expander("📊 Bekijk Portfolio Data", expanded=False):
    try:
        csv_path = os.path.join(ROOT_DIR, 'data', 'portfolio.csv')
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        st.dataframe(df, use_container_width=True, height=400)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Totaal aantal panden", len(df))
        with col2:
            st.metric("Totale waarde", f"€{df['value'].sum():,.0f}")
        with col3:
            st.metric("Gem. leegstand", f"{df['vacancyrate'].mean()*100:.1f}%")
        with col4:
            st.metric("Totale jaarinkomsten", f"€{df['anualincome'].sum():,.0f}")

    except Exception as e:
        st.error(f"Kon portfolio data niet laden: {e}")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar met voorbeeldvragen en controls
with st.sidebar:
    st.header("💡 Voorbeeldvragen")
    st.code("Wat is de totale waarde van alle panden in Arnhem?")
    st.code("Welke panden in Amsterdam hebben risico's?")
    
    st.divider()
    
    if st.button("🔄 Clear Chat & Memory"):
        st.session_state.messages = []
        # Dit is een truc om de @st.cache_resource te resetten
        st.cache_resource.clear()
        st.rerun()

# Toon oude berichten
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ontvang en verwerk nieuwe input
if prompt := st.chat_input("Stel een vraag over de portefeuille..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("De manager is aan het overleggen met zijn team..."):
            try:
                # Roep de agent aan met de input en de chat geschiedenis
                response = agent_executor.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.messages
                })
                answer = response.get("output", "Sorry, ik kon geen antwoord vinden.")

            except Exception as e:
                answer = f"Er is een onverwachte fout opgetreden: {e}"
            
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})