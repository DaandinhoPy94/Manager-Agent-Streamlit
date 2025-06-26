import pandas as pd
import sqlite3
import os

# Definieer de paden
CSV_PATH = os.path.join('data', 'portfolio.csv')
DB_PATH = os.path.join('data', 'portfolio.db')
TABLE_NAME = 'portfolio'

def create_sql_database():
    """
    Leest de portfolio CSV, voegt een ID toe, en slaat deze op
    als een SQLite database.
    """
    try:
        print(f"Lezen van het CSV-bestand: {CSV_PATH}")
        # Lees de CSV en voeg programmatisch een ID-kolom toe
        df = pd.read_csv(CSV_PATH)
        # Maak kolomnamen 'schoon' voor de zekerheid
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        df.insert(0, 'id', df.index)
        
        # Maak verbinding met de SQLite database (deze wordt aangemaakt als hij niet bestaat)
        print(f"Verbinding maken met en schrijven naar de database: {DB_PATH}")
        # De 'with' statement zorgt ervoor dat de verbinding netjes wordt gesloten
        with sqlite3.connect(DB_PATH) as conn:
            # Schrijf de data uit het pandas DataFrame naar een SQL-tabel
            # if_exists='replace' betekent: als de tabel al bestaat, overschrijf hem dan.
            df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        
        print("\nDatabase succesvol aangemaakt! De eerste 5 rijen zijn:")
        
        # Verbindings- en query-voorbeeld om te testen
        with sqlite3.connect(DB_PATH) as conn:
            # Voer een simpele SQL query uit om de eerste 5 rijen te tonen
            test_df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} LIMIT 5", conn)
            print(test_df)

    except FileNotFoundError:
        print(f"FOUT: Het bestand {CSV_PATH} kon niet worden gevonden.")
    except Exception as e:
        print(f"Er is een onverwachte fout opgetreden: {e}")

if __name__ == "__main__":
    create_sql_database()