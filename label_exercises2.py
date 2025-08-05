# label_exercises.py
# Dieses Skript liest eine Liste von Übungsnamen, lässt sie von einer KI 
# mit unserem definierten Tagging-System kategorisieren und speichert das Ergebnis als CSV.

import os
import json
import time
from openai import OpenAI

# --- KONFIGURATION ---
# Stelle sicher, dass dein OpenAI API Key als Umgebungsvariable gesetzt ist
# z.B. export OPENAI_API_KEY='dein_schlüssel'
# Dann Visual Studio neu starten
try:
    client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    print(f"Fehler: OpenAI API-Schlüssel nicht gefunden. Bitte als Umgebungsvariable setzen. {e}")
    exit()



INPUT_FILE = "exercises_to_label.txt"  # Textdatei mit einem Übungsnamen pro Zeile
INPUT_FILE = "C:\\Data\\Fitness\\util\\exercises_complete_rows_test50.csv"
INPUT_FILE = "C:\\Data\\Fitness\\util\\exercises_complete_rows.csv"
OUTPUT_FILE = "labeled_exercises.csv" # Ergebnis-CSV-Datei

INPUT_FILE = "Youtube_Videos_Genemedics.csv"



INPUT_FILE = "exercises_genemedics.csv"
OUTPUT_FILE = f"labeled_{INPUT_FILE}" # Ergebnis-CSV-Datei
BATCH_SIZE = 50  # Wie viele Übungen pro KI-Anfrage verarbeitet werden sollen
TEST_SIZE = 3 # Es wird nur diese Zahl von Übungen verarbeitet, bei -1 werden alle verarbeitet.

# Das Tagging-System, das wir der KI beibringen (KORRIGIERTE VERSION)
TAGGING_SYSTEM_PROMPT = """
Du bist ein Weltklasse-Sportwissenschaftler und Datenanalyst. Deine Aufgabe ist es, eine Liste von Fitness-Übungen zu analysieren und sie mit präzisen, funktionalen Tags zu versehen.

Antworte IMMER NUR mit einem validen JSON-Objekt. Dieses Objekt muss einen einzigen Schlüssel namens "exercises" enthalten. Der Wert dieses Schlüssels muss ein Array von Objekten sein.
Jedes Objekt im Array repräsentiert eine Übung und muss exakt die folgenden Felder haben:
- "name": Der exakte Name der Übung aus der Input-Liste.
- "tags": Ein Array von Text-Tags, die die Übung beschreiben.

Wähle passende Tags aus den folgenden Kategorien aus:

1.  **Sportwissenschaftlicher Score (Wert zwischen 1 und 100 in Integer):**
    * `95-100`: Brot&Butter Übungen. Fundamentale, mehrgelenkige Grundübungen, die das Fundamnt typischer Krafttrainingspläne darstellen (z.B. Langhantel Kniebeuge, Bankdrücken, Kreuzheben, Schulterdrücken, Klimmzüge, aber auch Maschinenübungen wie Brustpresse, Latzug, Beinpresse).
    * `70-94`: Standard-Zusatzübungen. Sehr häufige und wichtige Übungen, die Score 90-100 ergänzen (z.B. Leg Extensions, Schrägbankdrücken, Trizeps drücken, Seitheben, Bizeps Curls, Face Pulls).
    * `40-69`: Spezielle, aber sinnvolle Variationen für Abwechslung (z.B. Zercher Squats, Spider Curls, Larsen Press, Narrow Grip Bench Press).
    * `1-39`: Hochspezifische Übungen die hin und wieder gewählt werden, um für Abwechslung zu sorgen oder sehr speziellen Trainingszielen dienen  (z.B. Dumbbell Standing Zottman Preacher Curl, Lying T-Spine Mobility Streching, Reverse Grip Bench Press, Squat Variationen mit Bändern oder Ketten).

2.  **Type (Wähle genau EINS):**
    * `compound`: Mehrgelenks-Kraftübung.
    * `isolation`: Eingelenks-Kraftübung.
    * `isometric`: Statische Halteübung (z.B. Plank, Wandsitz, viele Yoga-Posen).
    * `stretch`: Jede Art von Dehnübung.
    * `cardio`: Klassische Ausdauerübungen (z.B. Laufen, Radfahren).
    * `ballistic`: Explosive Übungen (z.B. Kettlebell Swings, Box Jumps).
    * `dynamic`: Übungen mit Schwung (z.B. Leg Swings, Front Kicks).

3.  **Goal (Wähle EINS oder MEHRERE):**
    * `strength`, `hypertrophy`, `power`, `endurance`, `stability`, `mobility`, `balance`, `agility`, `coordination`.

4.  **Motion pattern (Wähle EINS, falls zutreffend):**
    * `horizontal_press`, `vertical_press`, `horizontal_pull`, `vertical_pull`, `squat`, `hinge`, `lunge`, `carry`, `rotation`, `crunch`, `plank`.

5.  **Additional (Wähle beliebig viele):**
    * `low_impact`, `high_impact`, `beginner_friendly`, `plateau_breaker`, `bodyweight`, `assisted`, `at_home`, `on_travel`, `every_day`, `warm_up`, `cool_down`.

6. **Popularity men (mit einem Wert zwischen 0 und 100):**
    * ein hoher Wert steht für bekannte, sehr gerne ausgeführte Übungen durch Männer mit einer großen Zahl von Youtube Anleitungsvideos, ein kleiner für seltene, einem spezifischen Ziel dienende Übung.

7. **Popularity women (mit einem Wert zwischen 0 und 100):**
    * ein hoher Wert steht für bekannte, sehr gerne ausgeführte Übungen durch Fraune mit einer großen Zahl von Youtube Anleitungsvideos, ein kleiner für seltene, einem spezifischen Ziel dienende Übung.

8. **Summary (aus einem bis drei Wörtern, mit der die Übung in einer einfachen Kommunikation zB in einer lauten Umgebung ausreichend beschrieben wird.)

9. **Fitness philosophy (Wähle eins oder mehrere falls zutreffend):**
    * `yoga`, `pilates`, `lifting`, `calisthenics`, `reha`, `prehab`, `cardio`, `full_body`, `stretching`, `plyometricss`, `explosives`.

---
**BEISPIELE FÜR KORREKTES LABELING:**

* **Input:** `["Barbell Bench Press", "Incline Dumbbell Curl", "Zercher Squat"]`
* **Output:**
    ```json
    {
      "exercises": [
        {
            "name": "Barbell Bench Press",
            "tags": {
                "score": 95,
                "type": "compound",
                "goal": [
                    "strength",
                    "hypertrophy"
                ],
                "motion_pattern": "horizontal_press",
                "additional": [],
                "popularity_men":,
                "popularity_women":,
                "summary":,
                "fitness_philosophy": ["lifting"]
            }
        },
        {
            "name": "Incline Dumbbell Curl",
            "tags": {
                "score": 50,
                "type": "isolation",
                "goal": ["hypertrophy"],
                "motion_pattern": "",
                "additional": ["beginner_friendly"],
                "popularity_men":,
                "popularity_women":,
                "summary":,
                "fitness_philosophy": ["lifting"]
            }
        },
        {
            "name": "Zercher Squat",
            "tags": {
                "score": 45,
                "type": "compound",
                "goal": ["strength"],
                "motion_pattern": "squat",
                "additional": ["plateau_breaker"],
                "popularity_men":,
                "popularity_women":,
                "summary":,
                "fitness_philosophy": ["lifting"]
            }
        }
      ]
    }
    ```
---

Hier ist die Liste der Übungen, die du jetzt labeln sollst:
"""

def get_exercises_from_file(filename):
    """Liest Übungsnamen aus einer Textdatei."""
    colNr = 2
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            #return [line.strip() for line in f if line.strip()]
            return [line.split(";")[colNr-1] for line in f if line.strip()]
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
    if TEST_SIZE > 0 and len(all_exercises) > TEST_SIZE:
        all_exercises = all_exercises[:TEST_SIZE]


    print(f"Insgesamt {len(all_exercises)} Übungen gefunden. Starte Labeling in Batches von {BATCH_SIZE}...")

    # Schreibe den Header in die CSV-Datei
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("name;score;type;goal;motion_pattern;additional;popularity_men;popularity_women;summary;philosophy\n")

    # Erste Zeile auslassen
    for i in range(1, len(all_exercises), BATCH_SIZE):
        batch = all_exercises[i:i + BATCH_SIZE]
        print(batch)
        print(f"Verarbeite Batch {i//BATCH_SIZE + 1}/{(len(all_exercises) + BATCH_SIZE - 1)//BATCH_SIZE} (Übungen {i+1}-{i+len(batch)})...")
        
        labeled_data = label_batch(batch)
        
        if labeled_data:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                for item in labeled_data:
                    # Stellt sicher, dass die Tags als kommaseparierter String im korrekten Array-Format für Supabase geschrieben werden
                    test = item.get("tags", [])
                    tags_str = str(test)
                    tags = item.get("tags", [])
                    goals = ", ".join(tags.get("goal", []))
                    additional = ", ".join(tags.get("additional", []))
                    fitness_philosophy = ", ".join(tags.get("fitness_philosophy", []))
                    # Format: "Übungsname","{tag1,tag2,tag3}"
                    #f.write(f"\"{item['name']}\";\"{item['tags']}\"\n")
                    f.write(f"\"{item['name']}\";\"{tags['score']}\";\"{tags['type']}\";\"{goals}\";\"{tags['motion_pattern']}\";\"{additional}\";\"{tags['popularity_men']}\";\"{tags['popularity_women']}\";\"{tags['summary']}\";\"{fitness_philosophy}\"\n")
            print(f"Batch erfolgreich verarbeitet und in '{OUTPUT_FILE}' gespeichert.")
        else:
            print("Fehler beim Verarbeiten des Batches. Überspringe...")
        
        time.sleep(5) # Pause, um API-Rate-Limits zu vermeiden

    print("\nLabeling-Prozess abgeschlossen!")

if __name__ == "__main__":
    main()

