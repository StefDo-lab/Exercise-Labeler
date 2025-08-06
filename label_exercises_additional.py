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


# Input file has to be a ';'-separated file with the name of the exercise in the 2nd row
INPUT_FILE = "exercises_genemedics.csv"
OUTPUT_FILE = f"labeled_additional_{INPUT_FILE}" # Ergebnis-CSV-Datei
BATCH_SIZE = 50  # Wie viele Übungen pro KI-Anfrage verarbeitet werden sollen
TEST_SIZE = 5 # Es wird nur diese Zahl von Übungen verarbeitet, bei -1 werden alle verarbeitet.



# Das Tagging-System, das wir der KI beibringen (KORRIGIERTE VERSION)
N = 1
TAGGING_SYSTEM_PROMPT_1 = f"""
Du bist ein Weltklasse-Sportwissenschaftler und Datenanalyst. Deine Aufgabe ist es, eine Liste von Fitness-Übungen zu analysieren und sie mit präzisen, funktionalen Tags zu versehen.

Antworte IMMER NUR mit einem validen JSON-Objekt. Dieses Objekt muss einen einzigen Schlüssel namens "exercises" enthalten. Der Wert dieses Schlüssels muss ein Array von Objekten sein.
Jedes Objekt im Array repräsentiert eine Übung und muss exakt die folgenden Felder haben:
- "name": Der exakte Name der Übung aus der Input-Liste.
- "tags": Ein Array von Text-Tags, die die Übung beschreiben.

Wähle passende Tags aus den folgenden Kategorien aus:

{(N := N+1)}.  **Exercise_type (genau EINEN der nachfolgenden Werte):**
    * Strength, Stretching, Aerobic

{(N := N+1)}.  **BodyPart (Einen oder mehrere der nachfolgenden Werte):**
    * Hips, back, neck, chest, thighs, calves, forearms, shoulders, upper arms, waist, hands, feet
    
{(N := N+1)}.  **Equipment (Einen oder mehrere der nachfolgenden Werte):**
    * Assisted, Band, Barbell, Battling Rope, Body weight, Bosu ball, Cable, Dumbbell, EZ Barbell, Hammer, Kettlebell, Leverage machine, Medicine Ball, Olympic barbell, Power Sled, Resistance Band, Roll, Rollball, Rope, Sled machine, Smith machine, Stability ball, Stick, Suspension, Trap bar, Vibrate Plate, Weighted, Wheel roller
   
{(N := N+1)}.  **Muscle group (die ursächlich betroffene Muskelgruppe aus den nachfolgenden Werten):**
    * Biceps brachii, Pectoralis major, Rectus abdominis, Gluteus maximus, Obliquus externus abdominis, Transverse abdominis, Quadriceps, Gluteus medius, Gastrocnemius, Adductor magnus, Adductor muscles, Adductor longus, Adductor group, Hip Adductors, Erector spinae, Deltoid anterior, Triceps brachii, Deltoid lateral, Anconeus, Rhomboids, Latissimus dorsi, Brachioradialis, Deltoid, Deltoid posterior, Pronator teres, Flexor carpi ulnaris, Forearm supinator, Forearm flexors, Trapezius, Hamstrings, Deltoid (all heads), External oblique, Iliopsoas, Brachialis, Diaphragm, Soleus, Semispinalis capitis, Pectoralis major (clavicular head), Longus colli, Deep cervical flexors, Transversus abdominis, Posterior deltoid, Infraspinatus, Rotator cuff muscles, Pectoralis major (sternocostal head), Supraspinatus, Extensor carpi radialis brevis, Extensor carpi radialis longus, Rectus femoris, Flexor digitorum profundus, Flexor digitorum superficialis, Extensor digitorum, Flexor carpi radialis, Serratus anterior, Intrinsic foot muscles, Peroneus longus, Fibularis longus, Tibialis posterior, Supinator, Forearm extensors, Forearm flexor muscles, Deltoid (middle fibers), Deltoid middle, Adductor pollicis, Adductors, Forearm flexor group, Piriformis, Tensor fasciae latae, Hip External Rotators, Hip rotators, Iliacus, Subscapularis, External Obliques, Quadriceps femoris, Hip flexors, Quadratus lumborum, Deltoid (lateral head), Deltoid (lateral fibers), Deltoid (middle head), Cervical extensors, Cervical paraspinal muscles, Cervical extensor muscles, Cervical Erector Spinae, Deep cervical flexors (Longus colli), Sternocleidomastoid, Suboccipital muscles, Upper trapezius, Splenius capitis, Extensor digitorum communis, Deltoid medial, Pectineus, Peroneal muscles, Peroneus longus and brevis, Psoas major, Extensor carpi radialis, Deltoid (Anterior and Lateral fibers)

{(N := N+1)}.  **Secondary muscles (Einen oder mehrere der nachfolgenden Muskeln):**
    * Brachialis, Triceps brachii, Obliquus externus abdominis, Gluteus medius, Transversus abdominis, Iliopsoas, Obliquus internus abdominis, Rectus abdominis, Gluteus maximus, Tensor fasciae latae, Gluteus minimus, Soleus, Adductor longus, Pectineus, Adductor magnus, Gracilis, Adductor brevis, Pectoralis major, Gastrocnemius, Hamstrings, Quadriceps, Obliques, Brachioradialis, Deltoid anterior, Anconeus, Anterior deltoid, Trapezius, Erector spinae, Trapezius (middle fibers), Biceps brachii, Deltoid medial, Deltoid lateral, Supraspinatus, Rotator cuff muscles, Supinator, Pectoralis minor, Rhomboids, Flexor carpi radialis, Adductors, Rotator cuff, Multifidus, Middle Trapezius, Quadratus lumborum, Latissimus dorsi, Tibialis anterior, Internal oblique, Rectus femoris, Forearm flexors, Intercostal muscles, Splenius capitis, Longus capitis, Sternocleidomastoid, Hip flexors, Transverse abdominis, Infraspinatus, Teres minor, Deltoid, Extensor digitorum, Extensor carpi radialis brevis, Vastus lateralis, Piriformis, Flexor digitorum superficialis, Flexor digitorum profundus, Extensor indicis, Extensor carpi radialis longus, Extensor carpi radialis, Pectoralis major (clavicular head), Flexor carpi ulnaris, Pronator quadratus, Peroneus brevis, Fibularis brevis, Forearm extensor muscles, Forearm extensors, Clavicular head of Pectoralis major, Clavicular portion of Pectoralis Major, Lateral deltoid, Pectoralis major (clavicular), Deltoid middle, Intrinsic hand muscles, Tensor fascia latae, Obturator internus, Quadriceps (Rectus femoris), Hip adductors, Quadriceps femoris, Psoas major, Iliacus, Internal Obliques, Posterior deltoid, Deltoid posterior, Spinal erectors, Teres major, Trapezius (upper fibers), Hip abductors, Vastus medialis, External obliques, Diaphragm, Upper trapezius, Scalenes, Scalene muscles, Scalene, Levator scapulae, Semispinalis capitis, Longus colli, Suboccipital muscles, Scalenus anterior, Core stabilizers, Extensor carpi ulnaris, Palmaris longus, Subscapularis, Serratus anterior, Adductor pollicis, Extensor digitorum longus, Extensor hallucis longus, Peroneus longus, Flexor hallucis longus, External oblique, Wrist flexor muscles, Wrist extensors, Forearm flexor muscles,  Brachioradialis,  Deltoid anterior,  Transversus abdominis,  Piriformis,  Obliquus externus abdominis,  Iliopsoas,  Erector spinae,  Obliquus internus abdominis,  Rectus abdominis,  Transverse abdominis,  Hamstrings,  Gluteus minimus,  Quadriceps,  Tensor fasciae latae,  Tensor fascia latae,  Gracilis,  Adductor brevis,  Gluteus medius,  Pectineus,  Gluteus maximus,  Gastrocnemius,  Hip flexors,  Adductor magnus,  Brachialis,  Triceps brachii,  Deltoid lateral,  Anconeus,  Serratus anterior,  Deltoid medial,  Supraspinatus,  Posterior deltoid,  Tibialis anterior,  Pectoralis major,  Biceps brachii,  Trapezius,  Infraspinatus,  Rotator cuff muscles,  Rhomboids,  Flexor carpi radialis,  Flexor digitorum superficialis,  Rotator cuff,  Quadratus lumborum,  Multifidus,  Latissimus dorsi,  Levator scapulae,  Soleus,  Anterior deltoid,  Sartorius,  Core stabilizers,  Calves,  Adductors,  Suboccipital muscles,  Sternocleidomastoid,  Deltoid posterior,  Adductor longus,  Obliques,  Adductor group,  Trapezius (middle fibers),  Pectoralis minor,  Psoas major,  Teres minor,  Extensor carpi radialis longus,  Extensor carpi ulnaris,  Forearm flexors,  Vastus medialis,  Hip adductors,  Flexor carpi ulnaris,  Extensor digiti minimi,  intrinsic hand muscles,  Extensor carpi radialis brevis,  Flexor pollicis longus,  Palmaris longus,  Peroneus longus,  Fibularis longus,  Flexor digitorum longus,  Tibialis posterior,  Supinator,  Obturator externus,  Upper trapezius,  Pectoralis major (clavicular head),  Flexor digitorum profundus,  wrist flexors,  Middle Trapezius,  Tensor fascia lata,  Gemellus superior,  Obturator internus,  Hip external rotators,  Teres major,  Deltoid posterior head,  External obliques,  Paraspinal muscles,  Trapezius (middle),  Semispinalis capitis,  Scalenes,  Splenius capitis,  Longissimus capitis,  Longus capitis,  Anterior scalenes,  Scalenus medius,  Posterior deltoids,  Lower trapezius,  Extensor digitorum communis,  Extensor digitorum,  Trapezius (upper fibers),  Core muscles,  Intercostals,  Shoulder stabilizers,  Extensor carpi radialis,  Extensor pollicis brevis,  Extensor digitorum longus,  Extensor hallucis longus,  Peroneus brevis,  Forearm extensor muscles,  Forearm extensors,  Thoracic erector spinae,  Pronator teres,  Wrist extensor muscles,  Extensor muscles of the forearm,  Fibularis brevis,  Thoracic paraspinals,  wrist extensors,  Hip abductors,  Gemellus inferior,  Triceps brachii (long head),  Vastus intermedius,  External oblique,  Subscapularis,  Quadratus femoris
  
{(N := N+1)}.  **difficulty (Einen der nachfolgenden Werte):**
    * Intermediate, Beginner, Advanced

{(N := N+1)}.  **remarks (falls notwendig eine Bemerkung zur korrekten Ausführung der Übung):**
    
{(N := N+1)}.  **exID (eine unique exercise identifier analog einer uuid):**

---
"""

TAGGING_SYSTEM_PROMPT_2 = """

**BEISPIELE FÜR KORREKTES LABELING:**

* **Input:** `["Barbell Bench Press", "Incline Dumbbell Curl", "Zercher Squat"]`
* **Output:**
    ```json
    {
      "exercises": [
        {
            "name": "Barbell Bench Press",
            "tags": {
                "exercise_type": "Strength",
                "bodyPart": ["Chest"],
                "Equipment": ["Barbell"],
                "muscle_group": ["Pectoralis major"],
                "secondary_muscles": ["Deltoid anterior", "Triceps brachii"],
                "difficulty": "Intermediate",
                "remarks": "Perform the exercise with controlled movement and proper scapular retraction to protect the shoulders and maximize pectoral engagement.",
                "exID": "0e52b180-92b9-4313-acaf-5f449c0945f5"
            }
        },
        {
            "name": "Incline Dumbbell Curl",
            "tags": {
                "exercise_type": ,
                "bodyPart": ["lifting"],
                "Equipment": [],
                "muscle_group": [],
                "secondary_muscles": [],
                "difficulty": ,
                "remarks": ,
                "exID":
            }
        },
        {
            "name": "Zercher Squat",
            "tags": {
                "exercise_type": ,
                "bodyPart": ["lifting"],
                "Equipment": [],
                "muscle_group": [],
                "secondary_muscles": [],
                "difficulty": ,
                "remarks": ,
                "exID":
            }
        }
      ]
    }
    ```
---

Hier ist die Liste der Übungen, die du jetzt labeln sollst:
"""

# Prompt zusammensetzen
TAGGING_SYSTEM_PROMPT = TAGGING_SYSTEM_PROMPT_1 + TAGGING_SYSTEM_PROMPT_2


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
        f.write("name;exercise_type;bodyPart;Equipment;muscle_group;secondary_muscles;difficulty;remarks;exID\n")


    # Erste Zeile auslassen
    for i in range(1, len(all_exercises), BATCH_SIZE):
        batch = all_exercises[i:i + BATCH_SIZE]
        print(f"Verarbeite Batch {i//BATCH_SIZE + 1}/{(len(all_exercises) + BATCH_SIZE - 1)//BATCH_SIZE} (Übungen {i+1}-{i+len(batch)})...")
        
        labeled_data = label_batch(batch)
        
        

        if labeled_data:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                for item in labeled_data:
                    # Stellt sicher, dass die Tags als kommaseparierter String im korrekten Array-Format für Supabase geschrieben werden
                    test = item.get("tags", [])
                    tags_str = str(test)
                    tags = item.get("tags", [])
                    bodyPart = ", ".join(tags.get("bodyPart", []))
                    Equipment = ", ".join(tags.get("Equipment", []))
                    muscle_group = ", ".join(tags.get("muscle_group", []))
                    secondary_muscles = ", ".join(tags.get("secondary_muscles", []))
                    # Format: "Übungsname","{tag1,tag2,tag3}"
                    #f.write(f"\"{item['name']}\";\"{item['tags']}\"\n")
                    f.write(f"\"{item['name']}\";\"{tags['exercise_type']}\";\"{bodyPart}\";\"{Equipment}\";\"{muscle_group}\";\"{secondary_muscles}\";\"{tags['difficulty']}\";\"{tags['remarks']}\";\"{tags['exID']}\"\n")
            print(f"Batch erfolgreich verarbeitet und in '{OUTPUT_FILE}' gespeichert.")
        else:
            print("Fehler beim Verarbeiten des Batches. Überspringe...")
        
        time.sleep(5) # Pause, um API-Rate-Limits zu vermeiden

    print("\nLabeling-Prozess abgeschlossen!")

if __name__ == "__main__":
    main()

