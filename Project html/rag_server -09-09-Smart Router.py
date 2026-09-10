import os
import re
import sys
import warnings
import time
import shutil
from threading import Thread, Event

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

SKRIPT_VERSJON = "v38-web - oppdatert smart router med robust mappenavn- og stihåndtering"
print(f"[System] Kjører skriptversjon: {SKRIPT_VERSJON}")

if torch.cuda.is_available():
    ENHET = "cuda"
    PRESISJON = torch.float16
    print(f"[System] GPU funnet ({torch.cuda.get_device_name(0)}) - bruker CUDA.")
else:
    ENHET = "cpu"
    PRESISJON = torch.float32
    torch.set_num_threads(os.cpu_count())
    print(f"[System] Ingen GPU funnet - kjører på CPU med {os.cpu_count()} tråder.")

from transformers import pipeline, GenerationConfig
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import chromadb
from chromadb.utils import embedding_functions
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter

from flask import Flask, request, jsonify, render_template

MAPPESTI = r"C:\working\python\dokumenter"
DB_STI = r"C:\working\python\chroma"
MODUS_MAPPE = os.path.join(os.path.dirname(__file__), "MODUS")

chroma_client = chromadb.PersistentClient(path=DB_STI)

embedding_modell = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

COLLECTION_NAVN = "mine_dokumenter"

try:
    chroma_client.delete_collection(name=COLLECTION_NAVN)
except Exception:
    pass

db_samling = chroma_client.get_or_create_collection(
    name=COLLECTION_NAVN,
    embedding_function=embedding_modell
)

tekst_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

def del_tekst_i_chunks(tekst):
    return tekst_splitter.split_text(tekst)

def oppdater_fil_i_database(full_sti):
    if not full_sti.endswith(".txt"):
        return

    try:
        relativ_kilde = os.path.relpath(full_sti, MAPPESTI)
    except Exception:
        relativ_kilde = os.path.basename(full_sti)

    try:
        db_samling.delete(where={"kilde": relativ_kilde})
    except Exception:
        pass

    if not os.path.exists(full_sti):
        print(f"[Database] Fjernet {relativ_kilde} (filen ble slettet).")
        return

    try:
        with open(full_sti, "r", encoding="utf-8") as fil:
            tekst = fil.read().strip()
    except UnicodeDecodeError:
        try:
            with open(full_sti, "r", encoding="cp1252") as fil:
                tekst = fil.read().strip()
        except Exception as feil:
            print(f"[Feil] Klarte ikke å lese {relativ_kilde}: {feil}")
            return
    except Exception as feil:
        print(f"[Feil] Uventet feil: {feil}")
        return

    if not tekst:
        return

    chunks = del_tekst_i_chunks(tekst)
    dokumenter, metadatas, ids = [], [], []

    for indeks, chunk in enumerate(chunks):
        dokumenter.append(chunk)
        metadatas.append({
            "kilde": relativ_kilde, 
            "full_sti": full_sti, 
            "chunk_nr": indeks
        })
        safe_id = relativ_kilde.replace(os.sep, "_")
        ids.append(f"{safe_id}_chunk_{indeks}")

    db_samling.add(documents=dokumenter, metadatas=metadatas, ids=ids)
    print(f"[Database] Suksess! Indekserte {relativ_kilde} ({len(chunks)} biter).")

class DokumentLytter(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            oppdater_fil_i_database(event.src_path)
    def on_created(self, event):
        if not event.is_directory:
            oppdater_fil_i_database(event.src_path)
    def on_deleted(self, event):
        if not event.is_directory:
            oppdater_fil_i_database(event.src_path)

forste_indeksering_ferdig = Event()

def start_mappe_overvaking():
    try:
        print("[System] Skanner rotmappe og undermapper rekursivt...")
        if not os.path.exists(MAPPESTI):
            os.makedirs(MAPPESTI)

        for rot, _, filer in os.walk(MAPPESTI):
            for filnavn in filer:
                oppdater_fil_i_database(os.path.join(rot, filnavn))

        handler = DokumentLytter()
        observer = Observer()
        observer.schedule(handler, path=MAPPESTI, recursive=True)
        observer.start()
        print(f"[Watchdog] Lytter rekursivt i: {MAPPESTI}")
    except Exception as feil:
        print(f"[Feil] Mappeovervåking feilet: {feil}")
        forste_indeksering_ferdig.set()
        return

    forste_indeksering_ferdig.set()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

print("[System] Laster inn Qwen-modellen...")
ai_modell = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    device=ENHET,
    torch_dtype=PRESISJON,
)

generasjons_konfig = GenerationConfig(
    max_new_tokens=250,
    do_sample=True,
    temperature=0.05,
    top_p=0.9,
)

MODUS_INSTRUKSER = {
    "høflig": "Formuler SVAR-linjen formelt og høflig, med fullstendige setninger.",
    "morsom": "Formuler SVAR-linjen på en munter og humoristisk måte.",
    "kort": "Formuler SVAR-linjen så kort og direkte som mulig.",
    "vennlig": "Formuler SVAR-linjen uformelt og vennlig.",
}

def last_inn_moduser():
    global MODUS_INSTRUKSER
    if os.path.exists(MODUS_MAPPE):
        fant_filer = False
        for filnavn in os.listdir(MODUS_MAPPE):
            if filnavn.endswith(".md"):
                modus_navn = os.path.splitext(filnavn)[0]
                fil_sti = os.path.join(MODUS_MAPPE, filnavn)
                try:
                    with open(fil_sti, "r", encoding="utf-8") as f:
                        MODUS_INSTRUKSER[modus_navn] = f.read().strip()
                        fant_filer = True
                except Exception as e:
                    print(f"[Advarsel] Klarte ikke å lese modus-fil {filnavn}: {e}")
        if fant_filer:
            print(f"[System] Lastet inn egendefinerte moduser fra {MODUS_MAPPE}")

last_inn_moduser()
STANDARD_MODUS = "høflig"
print(f"[System] Aktive toner: {list(MODUS_INSTRUKSER.keys())} (Standard: {STANDARD_MODUS})")

def smart_router_handling(bruker_sporsmal):
    """Smart router som automatisk gjenkjenner handlinger basert på tekst, uavhengig av tonemenyen."""
    sporsmal_lower = bruker_sporsmal.lower()
    
    # 1. Sjekk om det gjelder Outlook / e-post
    if "e-post" in sporsmal_lower or "innboks" in sporsmal_lower or "siste e-post" in sporsmal_lower:
        try:
            import win32com.client
            outlook = win32com.client.Dispatch("Outlook.Application")
            namespace = outlook.GetNamespace("MAPI")
            inbox = namespace.GetDefaultFolder(6)
            messages = inbox.Items
            messages.Sort("[ReceivedTime]", True)
            
            eposter_tekst = []
            for msg in list(messages)[:3]:
                eposter_tekst.append(f"Fra: {msg.SenderName} | Emne: {msg.Subject}")
            
            lokal_info = "\n".join(eposter_tekst)
            return f"[Smart Router - Outlook]: Hentet de siste e-postene lokalt:\n{lokal_info}"
        except ImportError:
            return "[Smart Router]: pywin32 er ikke installert."
        except Exception as e:
            return f"[Smart Router]: Klarte ikke koble til Outlook lokalt: {e}"

    # 2. Sjekk om det gjelder mapper / filrydding
    elif "lag en mappe" in sporsmal_lower or "lag arkiv" in sporsmal_lower:
        try:
            mappenavn = "ny_mappe"
            if "med navn" in sporsmal_lower:
                mappenavn = bruker_sporsmal.split("med navn")[1].strip().split()[0].strip("'\"")
            elif "kalt" in sporsmal_lower:
                mappenavn = bruker_sporsmal.split("kalt")[1].strip().split()[0].strip("'\"")

            if "undere denne stien:" in sporsmal_lower:
                base_sti = bruker_sporsmal.split("undere denne stien:")[1].strip().strip('"\'')
            elif "under denne stien:" in sporsmal_lower:
                base_sti = bruker_sporsmal.split("under denne stien:")[1].strip().strip('"\'')
            elif "stien:" in bruker_sporsmal:
                base_sti = bruker_sporsmal.split("stien:")[1].strip().strip('"\'')
            elif " i " in sporsmal_lower:
                base_sti = bruker_sporsmal.split(" i ")[-1].strip().strip('"\'')
            else:
                base_sti = MAPPESTI
            
            mal_sti = os.path.join(base_sti, mappenavn)
            os.makedirs(mal_sti, exist_ok=True)
            return f"[Smart Router - Filrydder]: Opprettet mappe '{mappenavn}' vellykket på: {mal_sti}"
        except Exception as e:
            return f"[Smart Router]: Klarte ikke å opprette mappe: {e}"
            
    return None

def besvar_sporsmal(bruker_sporsmal, gjeldende_modus):
    if gjeldende_modus not in MODUS_INSTRUKSER:
        gjeldende_modus = STANDARD_MODUS

    sporsmal_lower = bruker_sporsmal.lower()
    
    ignorerte_ord = {"vis", "alle", "filer", "med", "ordet", "hvor", "finnes", "eller", "fra", "i", "på", "av", "til"}
    aktive_sokeord = [w.strip(".,?!") for w in sporsmal_lower.split() if len(w) > 3 and w not in ignorerte_ord]

    dokument_kontekster = []
    metadata_liste = []
    undersokte_biter = []

    if aktive_sokeord:
        alle_data = db_samling.get()
        if alle_data and "documents" in alle_data and "metadatas" in alle_data:
            for dok, meta in zip(alle_data["documents"], alle_data["metadatas"]):
                dok_lower = dok.lower()
                kilde_lower = meta.get("kilde", "").lower()
                if any(ord in dok_lower or ord in kilde_lower for ord in aktive_sokeord):
                    dokument_kontekster.append(dok)
                    metadata_liste.append(meta)
                    undersokte_biter.append({
                        "kilde": meta.get("kilde"),
                        "full_sti": meta.get("full_sti"),
                        "chunk": meta.get("chunk_nr"),
                        "avstand": 0.0,
                        "nokkelord": len(aktive_sokeord),
                        "tekst": dok
                    })

    if not dokument_kontekster:
        sok_resultat = db_samling.query(query_texts=[bruker_sporsmal], n_results=4)
        if sok_resultat["documents"] and sok_resultat["documents"][0]:
            dokument_kontekster = sok_resultat["documents"][0]
            metadata_liste = sok_resultat["metadatas"][0]
            avstander = sok_resultat.get("distances", [[None] * len(dokument_kontekster)])[0]
            for tekst_bit, meta, avstand in zip(dokument_kontekster, metadata_liste, avstander):
                undersokte_biter.append({
                    "kilde": meta.get("kilde"),
                    "full_sti": meta.get("full_sti"),
                    "chunk": meta.get("chunk_nr"),
                    "avstand": avstand,
                    "nokkelord": 0,
                    "tekst": tekst_bit
                })

    if not dokument_kontekster:
        return {"svar": "Fant ingen relevante dokumenter.", "fakta": "", "sammenligning": "",
                "kilde": "", "chunk": None, "advarsel": None, "kandidater": []}

    kombinert_kontekst = "\n\n".join(
        f"KILDEFIL: {meta.get('kilde')} | FULL STI: {meta.get('full_sti')} (Chunk {meta.get('chunk_nr')})\nTEKST:\n{tekst}" 
        for tekst, meta in zip(dokument_kontekster, metadata_liste)
    )
    
    meldinger = [
        {
            "role": "system",
            "content": (
                "Du er en presis assistent som svarer basert KUN på tekstbitene og metadataen under. "
                "Hver bit inneholder 'KILDEFIL' og 'FULL STI'. "
                "Svar ALLTID i nøyaktig dette formatet med tre linjer:\n"
                "FAKTA: <en nøyaktig setning fra tekstbitene eller tilhørende filinformasjon>\n"
                "SAMMENLIGNING: <hvis brukerens input er en påstand, vurder om det stemmer. Hvis rent spørsmål, skriv 'ikke aktuelt'>\n"
                "1. Du er en konsis dokumentassistent. "
                "2. Svar utelukkende basert på de oppgitte kildene. "
                "3. Du skal alltid prioritere informasjonen i de oppgitte kildene. "
                "4. Hvis du er i tvil om informasjonen finnes, gå grundig gjennom alle kilde-utdragene en ekstra gang før du konkluderer. "
                "5. Svar direkte, korrekt og uten unødvendige høflighetsfraser, med naturlig, flytende norsk språk.\n\n"
                "REGLER FOR SVAR:\n"
                "6. Svar direkte på det som blir spurt om, uten å kommentere spørsmålsstillingen, brukerens antakelser eller hvorfor de spør.\n"
                "7. Hvis brukeren oppgir en verdi eller påstand som skal bekreftes eller avkreftes mot kildene, skal du ALLTID starte svaret med et eksplisitt 'Nei,' (hvis påstanden er feil) eller 'Ja,' (hvis påstanden stemmer), og deretter oppgi den korrekte informasjonen fra kilden i samme setning. Ikke bare korriger implisitt - det eksplisitte Ja/Nei skal alltid stå først.\n"
                "8. Hvis kildene ikke inneholder svaret, si tydelig at informasjonen ikke finnes i kildene du har tilgjengelig. "
                "9. Ikke gjett eller fyll inn med generell kunnskap. "
                "10. Konkrete verdier som datoer, klokkeslett, tall og navn skal alltid gjengis EKSAKT slik de står i kildeteksten. "
                "11. Ikke rund av, ikke generaliser, og ikke legg til årstall, klokkeslett eller andre detaljer som ikke eksplisitt står i kilden. "
                "12. Hvis du er usikker på om en verdi stemmer med kildene, skal du heller si at du er usikker enn å gjette. "
                "13. Ikke bruk faste innledningsfraser, maler eller unødvendige høflighetsfraser. "
                "14. Gå rett på svaret. "
                "15. Hold svarene korte og presise – ikke lengre enn nødvendig for å svare fullstendig. "
                "16. Hvis spørsmålet er tvetydig og kildene inneholder flere mulige svar, spesifiser hvilket du svarer ut ifra, uten å gjette på hva brukeren mente. "
                "17. Ikke kom med selvmotsigende konklusjoner eller ugyldige logiske slutninger. "
                "18. Ikke dikt opp datoer, årstall eller verdier som ikke står eksplisitt i kilden. "
                "19. Hvis informasjonen ikke finnes, si klart at informasjonen mangler i kildene. "
                "20. Ikke gjett. "
                "21. Svar direkte og korrekt uten unødvendige høflighetsfraser."
                "SVAR: <et naturlig, direkte svar på norsk, én setning>\n\n"
                
                f"----- EKTE DOKUMENTINNHOLD -----\n{kombinert_kontekst}\n\n"
                f"TONE FOR SVAR-LINJEN: {MODUS_INSTRUKSER[gjeldende_modus]}"
            ),
        },
    ]

    meldinger.append({"role": "user", "content": bruker_sporsmal})

    prompt = ai_modell.tokenizer.apply_chat_template(
        meldinger, tokenize=False, add_generation_prompt=True
    )

    resultat = ai_modell(
        prompt,
        generation_config=generasjons_konfig,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
    )

    full_generert_tekst = resultat[0]["generated_text"]
    svar_linjer = full_generert_tekst.split("\n")

    svar, fakta_linje, sammenligning_linje = "", "", ""
    for linje in svar_linjer:
        linje = linje.strip()
        if linje.upper().startswith("SVAR:"):
            svar = linje.split(":", 1)[1].strip()
        elif linje.upper().startswith("FAKTA:"):
            fakta_linje = linje.split(":", 1)[1].strip()
        elif linje.upper().startswith("SAMMENLIGNING:"):
            sammenligning_linje = linje.split(":", 1)[1].strip()

    if not svar:
        svar = full_generert_tekst.strip()

    def finn_kilde_for_fakta(fakta, kontekst_biter, metadata):
        if not fakta or fakta.strip().lower() in ("(ingen)", "ingen", ""):
            return metadata[0]["kilde"], metadata[0]["chunk_nr"]
        fakta_ord = set(w.lower().strip(".,?!") for w in fakta.split() if len(w) > 3)
        beste_indeks, beste_overlapp = 0, -1
        for i, bit in enumerate(kontekst_biter):
            bit_ord = set(w.lower().strip(".,?!") for w in bit.split())
            overlapp = len(fakta_ord & bit_ord)
            if overlapp > beste_overlapp:
                beste_overlapp, beste_indeks = overlapp, i
        return metadata[beste_indeks]["kilde"], metadata[beste_indeks]["chunk_nr"]

    kilde_fil, chunk_nr = finn_kilde_for_fakta(fakta_linje, dokument_kontekster, metadata_liste)

    return {
        "svar": svar,
        "fakta": fakta_linje,
        "sammenligning": sammenligning_linje,
        "kilde": kilde_fil,
        "chunk": chunk_nr,
        "advarsel": None,
        "kandidater": undersokte_biter
    }

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route("/")
def index():
    return render_template("index.html", moduser=list(MODUS_INSTRUKSER.keys()), standard_modus=STANDARD_MODUS)

@app.route("/api/sporsmal", methods=["POST"])
def api_sporsmal():
    data = request.get_json(force=True, silent=True) or {}
    bruker_sporsmal = (data.get("sporsmal") or "").strip()
    gjeldende_modus = data.get("modus") or STANDARD_MODUS

    if not bruker_sporsmal:
        return jsonify({"feil": "Tomt spørsmål."}), 400

    try:
        smart_melding = smart_router_handling(bruker_sporsmal)
        resultat = besvar_sporsmal(bruker_sporsmal, gjeldende_modus)
        
        if smart_melding:
            resultat["svar"] = f"{smart_melding}\n\n{resultat['svar']}"
            
        return jsonify(resultat)
    except Exception as feil:
        print(f"[Feil] Uventet feil: {feil}")
        return jsonify({"feil": f"Noe gikk galt på serveren: {feil}"}), 500

if __name__ == "__main__":
    bakgrunns_lytter = Thread(target=start_mappe_overvaking, daemon=True)
    bakgrunns_lytter.start()

    if not forste_indeksering_ferdig.wait(timeout=60):
        print("[Advarsel] Indeksering tok lengre tid enn forventet.")

    print("[System] Serveren kjører på http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)