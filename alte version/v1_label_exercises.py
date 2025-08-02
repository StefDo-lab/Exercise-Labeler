# label_exercises.py
# Dieses Skript liest eine Liste von Übungsnamen, lässt sie von einer KI 
# mit unserem definierten Tagging-System kategorisieren und speichert das Ergebnis als CSV.

import os
import json
import time
from openai import OpenAI

# --- KONFIGURATION ---
API_KEY_FILE = "openai_api_key.txt"  # Der Name deiner Textdatei mit dem Schlüssel
INPUT_FILE = "exercises_to_label.txt"  # Textdatei mit einem Übungsnamen pro Zeile
OUTPUT_FILE = "labeled_exercises.csv" # Ergebnis-CSV-Datei
BATCH_SIZE = 50  # Wie viele Übungen pro KI-Anfrage verarbeitet werden sollen

def get_api_key_from_file(filename):
    """Liest den API-Schlüssel aus der lokalen Textdatei."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"FEHLER: Die Schlüssel-Datei '{filename}' wurde nicht gefunden.")
        print("Bitte stelle sicher, dass die Datei im selben Ordner wie das Skript liegt und deinen OpenAI API-Schlüssel enthält.")
        return None

# Versuche, den API-Schlüssel aus der Datei zu laden
api_key = get_api_key_from_file(API_KEY_FILE)

if not api_key:
    exit() # Beendet das Skript, wenn kein Schlüssel gefunden wurde

try:
    client = OpenAI(api_key=api_key)
    print("OpenAI Client erfolgreich initialisiert.")
except Exception as e:
    print(f"Fehler bei der Initialisierung des OpenAI Clients: {e}")
    exit()


# Das Tagging-System, das wir der KI beibringen (KORRIGIERTE VERSION)
TAGGING_SYSTEM_PROMPT = """
Du bist ein Weltklasse-Sportwissenschaftler und Datenanalyst. Deine Aufgabe ist es, eine Liste von Fitness-Übungen zu analysieren und sie mit präzisen, funktionalen Tags zu versehen.

Antworte IMMER NUR mit einem validen JSON-Objekt. Dieses Objekt muss einen einzigen Schlüssel namens "exercises" enthalten. Der Wert dieses Schlüssels muss ein Array von Objekten sein.
Jedes Objekt im Array repräsentiert eine Übung und muss exakt die folgenden Felder haben:
- "name": Der exakte Name der Übung aus der Input-Liste.
- "tags": Ein Array von Text-Tags, die die Übung beschreiben.

Wähle passende Tags aus den folgenden Kategorien aus:

1.  **TIER-System (Wähle genau EINS):**
    * `tier1`: Standard-Übungen, die in fast jedem Fitnessstudio standardmäßig und häufig ausgeführt werden (z.B. Bankdrücken, Kniebeugen, Bizeps Curls, Schrägbankdrücken).
    * `tier2`: Spezielle, aber sinnvolle Variationen für Abwechslung (z.B. Zercher Squats, Spider Curls).
    * `tier3`: Exotische oder hochspezifische Übungen für Fortgeschrittene.

2.  **Übungstyp (Wähle genau EINS):**
    * `compound`: Mehrgelenksübung.
    * `isolation`: Isolationsübung.

3.  **Trainingsziel (Wähle EINS oder MEHRERE):**
    * `strength`, `hypertrophy`, `power`, `endurance`, `stability`, `mobility`.

4.  **Bewegungsmuster (Wähle EINS, falls zutreffend):**
    * `horizontal_press`, `vertical_press`, `horizontal_pull`, `vertical_pull`, `squat`, `hinge`, `lunge`, `carry`, `rotation`.

5.  **Besondere Merkmale (Wähle beliebig viele):**
    * `low_impact`, `beginner_friendly`, `plateau_breaker`, `rehab`, `prehab`, `bodyweight`.

---
**BEISPIELE FÜR KORREKTES LABELING:**

* **Input:** `["Barbell Bench Press", "Incline Dumbbell Curl", "Zercher Squat"]`
* **Output:**
    ```json
    {
      "exercises": [
        {
          "name": "Barbell Bench Press",
          "tags": ["tier1", "compound", "strength", "hypertrophy", "horizontal_press"]
        },
        {
          "name": "Incline Dumbbell Curl",
          "tags": ["tier2", "isolation", "hypertrophy", "beginner_friendly"]
        },
        {
          "name": "Zercher Squat",
          "tags": ["tier2", "compound", "strength", "squat", "plateau_breaker"]
        }
      ]
    }
    ```
---

Hier ist die Liste der Übungen, die du jetzt labeln sollst:
"""

def get_exercises_from_file(filename):
    """Liest Übungsnamen aus einer Textdatei."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Fehler: Die Datei '{filename}' wurde nicht gefunden.")
        return []

def label_batch(batch):
    """Sendet einen Batch von Übungen an die KI und erhält die Labels."""
    prompt = TAGGING_SYSTEM_PROMPT + "\n" + json.dumps(batch)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        content = response.choices[0].message.content
        try:
            # KORRIGIERT: Wir parsen das JSON und greifen direkt auf den Schlüssel "exercises" zu.
            data = json.loads(content)
            return data.get("exercises", []) # Gibt die Liste zurück oder eine leere Liste, falls der Schlüssel fehlt
        except (json.JSONDecodeError, TypeError):
            print(f"Warnung: Konnte JSON nicht korrekt parsen. Inhalt: {content}")
            return []

    except Exception as e:
        print(f"Ein Fehler ist bei der API-Anfrage aufgetreten: {e}")
        return None

def main():
    """Hauptfunktion zum Steuern des Labeling-Prozesses."""
    all_exercises = get_exercises_from_file(INPUT_FILE)
    if not all_exercises:
        return

    print(f"Insgesamt {len(all_exercises)} Übungen gefunden. Starte Labeling in Batches von {BATCH_SIZE}...")

    # Schreibe den Header in die CSV-Datei
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("name,tags\n")

    for i in range(0, len(all_exercises), BATCH_SIZE):
        batch = all_exercises[i:i + BATCH_SIZE]
        print(f"Verarbeite Batch {i//BATCH_SIZE + 1}/{(len(all_exercises) + BATCH_SIZE - 1)//BATCH_SIZE} (Übungen {i+1}-{i+len(batch)})...")
        
        labeled_data = label_batch(batch)
        
        if labeled_data:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                for item in labeled_data:
                    # Stellt sicher, dass die Tags als kommaseparierter String im korrekten Array-Format für Supabase geschrieben werden
                    tags_str = ",".join(item.get("tags", []))
                    # Format: "Übungsname","{tag1,tag2,tag3}"
                    f.write(f"\"{item['name']}\",\"{{{tags_str}}}\"\n")
            print(f"Batch erfolgreich verarbeitet und in '{OUTPUT_FILE}' gespeichert.")
        else:
            print("Fehler beim Verarbeiten des Batches. Überspringe...")
        
        time.sleep(5) # Pause, um API-Rate-Limits zu vermeiden

    print("\nLabeling-Prozess abgeschlossen!")

if __name__ == "__main__":
    main()
