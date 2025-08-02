# label_exercises.py
# Dieses Skript liest eine Liste von Übungsnamen, lässt sie von einer KI 
# mit unserem definierten Tagging-System kategorisieren und speichert das Ergebnis als CSV.

import os
import json
import time
import pandas as pd
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


# --- FINALER, STRUKTURIERTER PROMPT ---
TAGGING_SYSTEM_PROMPT = """
Du bist ein Weltklasse-Sportwissenschaftler und Datenanalyst. Deine Aufgabe ist es, eine Liste von Fitness-Übungen zu analysieren und sie mit präzisen, funktionalen Daten zu klassifizieren.

Antworte IMMER NUR mit einem validen JSON-Objekt. Dieses Objekt muss einen einzigen Schlüssel namens "exercises" enthalten. Der Wert dieses Schlüssels muss ein Array von Objekten sein.
Jedes Objekt im Array repräsentiert eine Übung und muss exakt die folgenden Felder haben:
- "name": Der exakte Name der Übung aus der Input-Liste.
- "tier": Ein einzelner Integer-Wert (`0`, `1`, `2` oder `3`).
- "type": Ein einzelner Text-Wert aus der vordefinierten Liste.
- "category": Ein Array von Text-Tags, das die primären Trainingsziele definiert.
- "pattern": Ein einzelner Text-Wert für das Bewegungsmuster, oder `null`, falls nicht zutreffend.
- "tags": Ein Array von Text-Tags für alle weiteren besonderen Merkmale.

Wähle die Werte für die Felder basierend auf den folgenden Kategorien aus:

### 1. TIER-System (Feld: `tier`)
*(Wähle genau EINEN Wert und gib ihn als Zahl aus)*
- `0`: **"Butter & Brot"-Übungen.** Fundamentale, mehrgelenkige Grundübungen (z.B. Langhantel Kniebeuge, Bankdrücken, Kreuzheben, Schulterdrücken, Rudern, Klimmzüge, Latzug, Chest Press, Beinpresse).
- `1`: **Standard-Zusatzübungen.** Sehr häufige und wichtige Übungen, die Tier0 ergänzen (z.B. Schrägbankdrücken, Leg Extensions, Seitheben, Bizeps Curls, Face Pulls).
- `2`: **Sinnvolle Variationen.** Speziellere Übungen für Abwechslung (z.B. Zercher Squats, Spider Curls, Pistol Squats, Diamond Push up, Wall Ball).
- `3`: **Exotische/Spezialübungen.** Hochspezifische Übungen, die sehr selten ausgeführt werden, und sehr spezielle Bewegungsabläufe, Methoden oder Ziele haben. (z.B. One Arm twisted Clean and Press, Barbell Deadlift with chains, wheel yoga pose )

### 2. Übungstyp (Feld: `type`)
*(Wähle genau EINEN Wert aus dieser Liste)*
- `compound`: Mehrgelenks-Kraftübung.
- `isolation`: Eingelenks-Kraftübung.
- `isometric`: Statische Halteübung (z.B. Plank, Wandsitz, viele Yoga-Posen).
- `stretch`: Jede Art von Dehnübung.
- `cardio`: Klassische Ausdauerübungen (z.B. Laufen, Radfahren).
- `ballistic`: Explosive Übungen (z.B. Kettlebell Swings, Box Jumps).

### 3. Primäre Trainingsziele (Feld: `category`)
*(Wähle EINEN oder MEHRERE passende Werte und füge sie zum `category`-Array hinzu)*
- `Strength`, `Hypertrophy`, `Power`, `Endurance`, `Stability`, `Mobility`.

### 4. Bewegungsmuster (Feld: `pattern`)
*(Wähle genau EINEN zutreffenden Wert oder gib `null` aus)*
- `horizontal_press`, `vertical_press`, `horizontal_pull`, `vertical_pull`, `squat`, `hinge`, `lunge`, `carry`, `rotation`, `crunch`, `plank`.

### 5. Besondere Merkmale (Feld: `tags`)
*(Wähle beliebig viele passende Werte und füge sie zum `tags`-Array hinzu)*
- `low_impact`, `beginner_friendly`, `plateau_breaker`, `rehab`, `prehab`, `bodyweight`, `unilateral`, `explosive`, `yoga`, `pilates`, `stretching`, `calisthenics`, `balance`, `coordination`, `agility`.

---
**BEISPIEL FÜR KORREKTES LABELING:**
* **Input:** `["Barbell Bench Press", "Bosu Ball Squat"]`
* **Output:**
    ```json
    {
      "exercises": [
        {
          "name": "Barbell Bench Press",
          "tier": 0,
          "type": "compound",
          "category": ["Strength", "Hypertrophy"],
          "pattern": "horizontal_press",
          "tags": []
        },
        {
          "name": "Bosu Ball Squat",
          "tier": 2,
          "type": "compound",
          "category": ["Stability"],
          "pattern": "squat",
          "tags": ["balance", "coordination", "prehab", "bodyweight"]
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
            temperature=0.1 # Sehr niedrige Temperatur für konsistente, regelbasierte Ergebnisse
        )
        content = response.choices[0].message.content
        try:
            data = json.loads(content)
            return data.get("exercises", [])
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

    # --- CSV-ERSTELLUNG FÜR DIE NEUE STRUKTUR ---
    all_labeled_data = []

    for i in range(0, len(all_exercises), BATCH_SIZE):
        batch = all_exercises[i:i + BATCH_SIZE]
        print(f"Verarbeite Batch {i//BATCH_SIZE + 1}/{(len(all_exercises) + BATCH_SIZE - 1)//BATCH_SIZE} (Übungen {i+1}-{i+len(batch)})...")
        
        labeled_data = label_batch(batch)
        
        if labeled_data:
            all_labeled_data.extend(labeled_data)
            print(f"Batch erfolgreich verarbeitet.")
        else:
            print("Fehler beim Verarbeiten des Batches. Überspringe...")
        
        time.sleep(5) # Pause, um API-Rate-Limits zu vermeiden

    if all_labeled_data:
        # Konvertiere die gesammelten Daten in einen DataFrame
        output_df = pd.DataFrame(all_labeled_data)
        
        # Stelle sicher, dass alle Spalten vorhanden sind
        expected_columns = ["name", "tier", "type", "category", "pattern", "tags"]
        for col in expected_columns:
            if col not in output_df.columns:
                output_df[col] = None
        
        # Funktion zum Formatieren von Arrays für Supabase
        def format_array_for_supabase(arr):
            if isinstance(arr, list):
                return f"{{{','.join(arr)}}}"
            return '{}'

        # Formatieren der Array-Spalten für den Supabase-Import
        output_df['category'] = output_df['category'].apply(format_array_for_supabase)
        output_df['tags'] = output_df['tags'].apply(format_array_for_supabase)

        # Speichere den DataFrame als CSV
        output_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"\nAlle Daten erfolgreich in '{OUTPUT_FILE}' gespeichert.")

    print("\nLabeling-Prozess abgeschlossen!")

if __name__ == "__main__":
    main()
