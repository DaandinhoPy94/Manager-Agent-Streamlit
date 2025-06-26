import streamlit as st
import os
import pandas as pd  # Toegevoegd voor het lezen van de portfolio tabel
import time  # Voor retry mechanisme
# --- SQLite fix voor Streamlit Cloud ----------------------------------------
import sys, pysqlite3               # vervangt oude sqlite op Linux
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# ---------------------------------------------------------------------------
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub # Nieuwe import voor de prompt

# --- PAGINA CONFIGURATIE ---
st.set_page_config(page_title="AgentManagerGPT", page_icon="🧑‍💼", layout="wide")

# --- DATABASE & VECTORSTORE VERBINDING ---
DB_PATH = os.path.join('data', 'portfolio.db')
VS_PATH = "vectorstore"

@st.cache_resource
def setup_agent(_groq_api_key):
    """
    Zet de volledige agent op met beide tools. Deze functie wordt eenmalig uitgevoerd.
    """
    print("Agent wordt opgezet...")
    
    # --- De Intelligentie ---
    # Terug naar het slimmere model voor betere redenatie
    llm = ChatGroq(
        temperature=0, 
        model_name="llama3-8b-8192",  # Terug naar het slimmere model
        groq_api_key=_groq_api_key,
        max_tokens=1000  # Iets meer ruimte voor redeneren
    )

    # --- TOOL 1: De "Archivaris" (SQL Specialist) ---
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = sql_toolkit.get_tools()
    
    # Voeg beschrijvingen toe aan de belangrijkste SQL tools
    for tool in sql_tools:
        if tool.name == "sql_db_query":
            tool.description = """Gebruik deze tool voor SQL queries op de 'portfolio' tabel.
            De tabel heeft deze kolommen: id, address, type, value, vacancyrate, anualincome, endlease.
            Voorbeelden:
            - SELECT COUNT(*) FROM portfolio
            - SELECT SUM(value) FROM portfolio WHERE address LIKE '%Amsterdam%'
            - SELECT AVG(vacancyrate) FROM portfolio
            BELANGRIJK: De tabel heet 'portfolio', NIET 'panden'!"""
        elif tool.name == "sql_db_list_tables":
            tool.description = "Lijst alle tabellen. Er is één tabel: 'portfolio'"
        elif tool.name == "sql_db_schema":
            tool.description = "Schema van de 'portfolio' tabel bekijken"

    # --- TOOL 2: De "Analist" (RAG Specialist) ---
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=VS_PATH, embedding_function=embeddings)
    
    # Adaptieve retriever: 12 documenten (balans voor het zwaardere model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

    # Maak de RAG-tool
    # Deze tool kan de vectorstore doorzoeken en een antwoord formuleren
    rag_tool = create_retriever_tool(
        retriever,
        "portfolio_analyst_tool",
        """Gebruik deze tool voor semantische zoekopdrachten, open vragen, risico-analyses, en meningen over specifieke groepen panden.
        Voor totale portfolio samenvattingen of statistieken over ALLE panden, gebruik liever SQL tools.
        Deze tool geeft maximaal 15 relevante panden terug.""",
    )

    # Combineer de tools van beide specialisten
    all_tools = sql_tools + [rag_tool]

    # --- De "Manager" (De Agent) ---
    # We halen een standaard "ReAct" prompt op
    prompt = hub.pull("hwchase17/react-chat")
    
    agent = create_react_agent(llm, all_tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=all_tools, 
        verbose=True, # Laat de gedachtegang zien in de terminal
        handle_parsing_errors=True,
        max_iterations=10  # Verhoogd van 5 naar 10 om timeouts te voorkomen
    )
    
    print("Agent succesvol opgezet!")
    return agent_executor

# --- HOOFD APPLICATIE ---
st.title("🧑‍💼 AgentManagerGPT")
st.markdown("Stel een vraag aan de Onderzoeksmanager. Hij kiest de juiste specialist voor de klus.")

# --- API KEY HANDLING ---
if "api_key_entered" not in st.session_state:
    st.session_state.api_key_entered = False

if not st.session_state.api_key_entered:
    groq_api_key = st.text_input("Voer je Groq API Key in (begint met gsk_...):", type="password")
    if groq_api_key:
        st.session_state.groq_api_key = groq_api_key
        st.session_state.api_key_entered = True
        st.rerun()
else:
    groq_api_key = st.session_state.groq_api_key
    st.success("✅ API Key is ingesteld")

if not st.session_state.api_key_entered:
    st.info("Voer alsjeblieft je Groq API key in om de app te starten.")
    st.stop()

# --- PORTFOLIO TABEL WEERGEVEN ---
with st.expander("📊 Bekijk Portfolio Data", expanded=False):
    try:
        df = pd.read_csv(os.path.join('data', 'portfolio.csv'))
        st.dataframe(df, use_container_width=True, height=400)
        
        # Basis statistieken
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Totaal aantal panden", len(df))
        with col2:
            st.metric("Totale waarde", f"€{df['Value'].sum():,.0f}")
        with col3:
            st.metric("Gem. leegstand", f"{df['VacancyRate'].mean()*100:.1f}%")
        with col4:
            st.metric("Totale jaarinkomsten", f"€{df['AnualIncome'].sum():,.0f}")
    except Exception as e:
        st.error(f"Kon portfolio data niet laden: {e}")

agent_executor = setup_agent(groq_api_key)

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar met voorbeeldvragen
with st.sidebar:
    st.header("💡 Voorbeeldvragen")
    st.markdown("**SQL vragen (Archivaris):**")
    st.code("Wat is de totale waarde van alle panden in Arnhem?", language=None)
    st.code("Hoeveel panden hebben een leegstand boven 10%?", language=None)
    st.code("Geef een overzicht van de hele portfolio", language=None)
    st.code("Wat is de gemiddelde waarde per type pand?", language=None)
    
    st.markdown("**RAG vragen (Analist):**")
    st.code("Welke panden in Amsterdam hebben risico's?", language=None)
    st.code("Geef advies over panden met hoge leegstand", language=None)
    st.code("Welke panden zijn interessant voor renovatie?", language=None)
    
    st.markdown("**Combinatie vragen:**")
    st.code("Hoeveel panden in Den Haag hebben hoge leegstand en wat zijn de risico's?", language=None)
    
    st.divider()
    
    with st.expander("🔥 Token Bespaar Tips", expanded=True):
        st.markdown("""
        **Gratis Groq limiet: 6000 tokens/minuut**
        
        ✅ **Doe dit:**
        - Specifieke vragen ("panden in Utrecht")
        - Gebruik "top 5" of "top 10"
        - Één vraag per keer
        - SQL vragen voor statistieken
        
        ❌ **Vermijd dit:**
        - "Analyseer alle 50 panden"
        - Lange, complexe vragen
        - Meerdere vragen tegelijk
        
        💡 **Model:** llama3-8b-8192 (slimmer maar meer tokens)
        """)
    
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.info("💡 Tip: Voor portfolio samenvattingen gebruikt de Agent nu automatisch SQL voor betere performance!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Stel een vraag over de portefeuille..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("De manager is aan het overleggen met zijn team..."):
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = agent_executor.invoke({"input": prompt, "chat_history": []})
                    answer = response.get("output", "Sorry, ik kon geen antwoord vinden.")
                    
                    # Check of de agent gestopt is door iteration limit
                    if "Agent stopped" in str(response) or answer == "Agent stopped due to iteration limit or time limit.":
                        answer = "⚠️ De Agent heeft te veel stappen nodig. Probeer je vraag specifieker te maken of check de terminal voor het antwoord."
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e)
                    
                    if "no healthy upstream" in error_str:
                        retry_count += 1
                        if retry_count < max_retries:
                            st.warning(f"Groq API tijdelijk niet beschikbaar. Poging {retry_count}/{max_retries}...")
                            time.sleep(2)  # Wacht 2 seconden
                            continue
                        else:
                            answer = """⚠️ **Groq API tijdelijk niet beschikbaar**
                            
De Groq servers hebben momenteel problemen. Dit komt soms voor bij gratis accounts.

**Opties:**
1. Wacht een paar minuten en probeer opnieuw
2. Check https://status.groq.com/ voor updates
3. Probeer een andere API key als je die hebt

Dit is een tijdelijk probleem aan Groq's kant, niet met onze app!"""
                    
                    elif "rate_limit_exceeded" in error_str:
                        # Extract wachttijd uit de error message
                        import re
                        wait_match = re.search(r'try again in (\d+m)?(\d+\.?\d*s)?', error_str)
                        wait_time = wait_match.group(0) if wait_match else "1 minuut"
                        
                        answer = f"""⚠️ **Rate limit bereikt!**
                        
Het gratis Groq plan heeft een limiet van 6000 tokens per minuut.

**Opties:**
1. Wacht {wait_time} en probeer opnieuw
2. Stel kortere/specifiekere vragen
3. Upgrade naar Groq Dev Tier ($10/maand) voor 30x meer tokens

**Tips voor minder tokens:**
- Vraag naar specifieke steden/types in plaats van "hele portfolio"
- Gebruik "top 5" in plaats van "alle panden"
- Stel één vraag tegelijk"""
                    else:
                        answer = f"Er is een fout opgetreden: {e}"
                    break
            
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})