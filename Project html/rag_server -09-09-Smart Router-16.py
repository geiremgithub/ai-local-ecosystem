# ============================================================
# GEM. 10 septemeber 2026. 13:52 release
# ============================================================
# 
# ============================================================
# STANDARDBIBLIOTEKER
#
# Hensikt:
# Laster inn Python-biblioteker som brukes til
# filhåndtering, tekstbehandling, systemfunksjoner,
# tidsmåling og kjøring av bakgrunnsprosesser.
#
# ============================================================

import os # Arbeider med filer, mapper og filstier.
import re # Brukes til søk og behandling av tekst med regulære uttrykk.
import sys # Gir tilgang til Python-miljøet og systeminnstillinger.
import warnings # Brukes til å vise eller skjule advarsler.
import time # Måler tid og håndterer pauser i programmet.
import shutil # Kopierer, flytter og sletter filer og mapper.

from threading import Thread, Event

# Thread:
# Thread = kjør kode i bakgrunnen
# Kjører oppgaver parallelt i bakgrunnen, for eksempel
# mappeovervåking mens webserveren kjører.
#
# Event:
# Brukes til synkronisering mellom tråder, slik at
# programmet kan vente til en oppgave er ferdig før
# neste steg starter
# Event = vent på at en oppgave skal bli ferdig

sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

SKRIPT_VERSJON = "v54-web - viser relevante chunks ved hvert søk"

print(f"[System] Kjører skriptversjon: {SKRIPT_VERSJON}")

# ============================================================
# MASKINVARE
# ============================================================
#
# Hensikt:
# Kontrollerer hvilken maskinvare som er tilgjengelig
# for kjøring av AI-modellen.
# Kontrollerer om maskinen har et NVIDIA-grafikkort (GPU)
# tilgjengelig. Dersom GPU finnes brukes CUDA for raskere
# AI-beregninger. Dersom ikke kjøres modellen på CPU.
#
## GPU:
# Brukes dersom CUDA er tilgjengelig.
#
# CPU:
# Brukes dersom ingen kompatibel GPU finnes.
# Hvis ingen GPU er tilgjengelig, kjøres modellen
# på CPU og antall prosessortråder tilpasses
# automatisk maskinen.
#
# Dette gjør at programmet automatisk tilpasser seg
# maskinvaren det kjører på.
# Resultat:
# Programmet velger automatisk den beste
# tilgjengelige maskinvaren.
# AI-modellen bruker tilgjengelige ressurser på
# best mulig måte uten manuell konfigurering.
#
# ============================================================
#
if torch.cuda.is_available():
    ENHET = "cuda"
    PRESISJON = torch.float16

    print(
        f"[System] GPU funnet "
        f"({torch.cuda.get_device_name(0)}) - bruker CUDA."
    )

else:
    ENHET = "cpu"
    PRESISJON = torch.float32

    torch.set_num_threads(os.cpu_count())

    print(
        f"[System] Ingen GPU funnet - "
        f"kjører på CPU med {os.cpu_count()} tråder."
    )


# ============================================================
# IMPORTER
# ============================================================
#
# Hensikt:
# Laster inn alle bibliotekene som brukes videre i systemet.
# Laster inn eksterne biblioteker som brukes av
# RAG-løsningen.
#
#
# Viktige komponenter:
# - Transformers: kjører språkmodellen Qwen.
# - ChromaDB: lagrer dokument-chunks og embeddings.
# - Watchdog: overvåker filer og mapper.
# - Flask: lager webgrensesnittet.
# - LangChain Text Splitter: deler dokumenter i chunks.
## Programmet trenger blant annet:
# - En AI-modell (Qwen)
# - En vektordatabase (ChromaDB)
# - Dokumentovervåking (Watchdog)
# - Et webgrensesnitt (Flask)
# - Tekstsplitter for chunks
#
# Disse bibliotekene utgjør hovedkomponentene i
# dokumentsøk, AI-generering og webgrensesnittet.
#
# Uten disse importene vil ikke de ulike delene
# av løsningen kunne brukes senere i scriptet.
#
# ============================================================

from transformers import pipeline, GenerationConfig
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

import chromadb
from chromadb.utils import embedding_functions

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from langchain_text_splitters import RecursiveCharacterTextSplitter

from flask import Flask, request, jsonify, render_template


# ============================================================
# KONFIGURASJON
# ============================================================
#
# Hensikt:
# Samler alle viktige innstillinger på ett sted.
# Definerer hvilke mapper og innstillinger som
# skal brukes i RAG-løsningen.
#
# Her defineres:
# - Hvor dokumentene ligger.
# - Hvor ChromaDB lagres.
# - Hvor modus-filer ligger.
# - Hvor mange dokument-chunks som skal hentes ved søk.
#
# MAPPESTI
# Mappe som inneholder dokumentene som skal
# indekseres og gjøres søkbare.
#
# DB_STI
# Plassering for ChromaDB-databasen.
#
# MODUS_MAPPE
# Mappe som inneholder egne AI-moduser og
# instruksjoner.
#
# TOP_K_CHUNKS
# Antall relevante chunks som hentes fra
# ChromaDB ved hvert søk.
#
# ============================================================

MAPPESTI = r"C:\working\python\dokumenter"

DB_STI = r"C:\working\python\chroma"

MODUS_MAPPE = os.path.join(
    os.path.dirname(__file__),
    "MODUS"
)

# Antall chunks som hentes fra ChromaDB
TOP_K_CHUNKS = 7


# ============================================================
# CHROMADB
# ============================================================
# Hensikt:
# Oppretter og kobler til ChromaDB, som er
# vektordatabasen i RAG-løsningen.
# Oppretter forbindelsen til vektordatabasen ChromaDB.
#
# Dokumentbiter lagres her som embeddings slik at systemet
# kan finne relevante tekstavsnitt basert på mening
# i stedet for eksakte ord.
#
# ChromaDB lagrer:
# - Dokument-chunks
# - Embeddings for hver chunk
# - Metadata som filnavn og chunknummer
#
# Når brukeren stiller et spørsmål, brukes
# ChromaDB til å finne de chunkene som er mest
# relevante før informasjonen sendes videre
# til AI-modellen.
#
# Resultat:
# Dokumentene blir søkbare ved hjelp av
# semantisk søk og ikke bare nøkkelord.
# 
# Databasen fungerer som "hukommelsen" til RAG-løsningen.
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=DB_STI
)

embedding_modell = (
    embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
)

COLLECTION_NAVN = "mine_dokumenter"


try:
    chroma_client.delete_collection(
        name=COLLECTION_NAVN
    )
except Exception:
    pass


db_samling = chroma_client.get_or_create_collection(
    name=COLLECTION_NAVN,
    embedding_function=embedding_modell
)


# ============================================================
# TEKSTSPLITTING
# ============================================================
# Hensikt:
# Deler lange dokumenter opp i mindre tekstbiter
# (chunks) før lagring i ChromaDB.
#
# Hvorfor:
# Vektorsøk og AI-modeller fungerer bedre når de
# arbeider med mindre tekstutdrag i stedet for
# hele dokumenter.
# Språkmodeller fungerer best når de får mindre og mer
# relevante tekstutdrag.
#
# Chunk-overlap brukes for å bevare sammenhengen
# mellom tekstbiter slik at viktig informasjon
# ikke går tapt i overgangen mellom to chunks.
## Deler lange dokumenter opp i mindre tekstbiter
# (chunks).
# Chunkene brukes senere i vektorsøk mot ChromaDB.
#
# Resultat:
# Dokumentet blir delt opp i søkbare tekstbiter
# som senere kan hentes som kontekst ved spørsmål.
#
# ============================================================

tekst_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ],
    length_function=len,
)


def del_tekst_i_chunks(tekst):
    return tekst_splitter.split_text(tekst)


# ============================================================
# INDEKSERING AV DOKUMENTER
#
# Hensikt:
# Gjør tekstfiler søkbare i RAG-løsningen.
#
# Arbeidsflyt:
# 1. Leser inn dokumentet fra disk.
# 2. Deler teksten opp i mindre chunks.
# 3. Oppretter metadata for hver chunk.
# 4. Lagrer chunks og metadata i ChromaDB.
#
# Resultat:
# Dokumentets innhold kan senere gjenfinnes
# gjennom semantisk søk når brukeren stiller
# spørsmål til systemet.
#
# ============================================================

def oppdater_fil_i_database(full_sti):

    if not full_sti.endswith(".txt"):
        return

    try:
        relativ_kilde = os.path.relpath(
            full_sti,
            MAPPESTI
        )

    except Exception:
        relativ_kilde = os.path.basename(
            full_sti
        )

    # Fjern eksisterende chunks for filen
    try:
        db_samling.delete(
            where={
                "kilde": relativ_kilde
            }
        )

    except Exception:
        pass

    # Filen finnes ikke lenger
    if not os.path.exists(full_sti):

        print(
            f"[Database] Fjernet {relativ_kilde} "
            f"(filen ble slettet)."
        )

        return
    
    
    # --------------------------------------------------------
    # LES FIL
    ##
    # Hensikt:
    # Leser inn innholdet fra tekstfilen som skal
    # indekseres og lagres i ChromaDB.
    #
    # Leser inn tekstfilen fra disk.
    # Scriptet prøver først UTF-8(som er standard tegnsett i moderne
    # tekstfiler.) ) siden dette er
    # standard tegnsett. Hvis det feiler forsøkes
    # CP1252 som ofte brukes av eldre Windows-filer.
    #
    # Hvis filen ikke kan leses stoppes behandlingen.
    ## Hvis det ikke lykkes, prøves CP1252 som ofte
    # brukes av eldre Windows-filer.
    # Resultat:
    # Filinnholdet lagres i variabelen "tekst"
    # og brukes videre til chunking og indeksering.
    #--------------------------------------------------------

    
    try:

        with open(
            full_sti,
            "r",
            encoding="utf-8"
        ) as fil:

            tekst = fil.read().strip()

    except UnicodeDecodeError:

        try:

            with open(
                full_sti,
                "r",
                encoding="cp1252"
            ) as fil:

                tekst = fil.read().strip()

        except Exception as feil:

            print(
                f"[Feil] Klarte ikke å lese "
                f"{relativ_kilde}: {feil}"
            )

            return

    except Exception as feil:

        print(
            f"[Feil] Uventet feil: {feil}"
        )

        return

    if not tekst:
        return
    # --------------------------------------------------------
    # LAG CHUNKS
    #
    # Hensikt:
    # Deler dokumentteksten opp i mindre tekstbiter
    # (chunks) før lagring i ChromaDB.
    #
    # Hvorfor:
    # AI-modeller og vektorsøk fungerer bedre på
    # mindre tekstutdrag enn på hele dokumenter.
    ## AI-modeller og vektordatabaser arbeider mer
    # effektivt med mindre tekstutdrag enn med hele
    # dokumenter.
    #
    # Hver chunk får:
    # - Eget chunknummer
    # - Kobling til filen den kommer fra
    # - Metadata som brukes ved søk
    ## Resultat:
    # Dokumentet blir omgjort til søkbare tekstbiter
    # som senere kan hentes som kontekst til AI-modellen.
    #
    # --------------------------------------------------------
        

    chunks = del_tekst_i_chunks(tekst)

    dokumenter = []
    metadatas = []
    ids = []

    for indeks, chunk in enumerate(chunks):

        dokumenter.append(chunk)

        metadatas.append(
            {
                "kilde": relativ_kilde,
                "full_sti": full_sti,
                "chunk_nr": indeks
            }
        )

        safe_id = relativ_kilde.replace(
            os.sep,
            "_"
        )

        ids.append(
            f"{safe_id}_chunk_{indeks}"
        )

   
    # --------------------------------------------------------
    # LAGRE I CHROMADB
    #
    # Hensikt:
    # Lagrer alle genererte chunks i ChromaDB slik
    # at de kan brukes senere ved dokumentsøk.
    # Etter at dokumentet er delt opp i chunks,
    # lagres alle chunkene i ChromaDB.
    #
    # For hver chunk lagres:
    # - Tekstinnholdet
    # - Metadata (filnavn, filsti og chunknummer)
    # - En unik ID
    #
    # ChromaDB fungerer som hukommelsen til
    # RAG-løsningen.
    #
    # Når brukeren senere stiller et spørsmål,
    # søker systemet i disse lagrede chunkene
    # for å finne den mest relevante informasjonen.
    # Når brukeren stiller et spørsmål, søker
    # ChromaDB blant disse lagrede chunkene for å
    # finne de mest relevante tekstutdragene.
    ## Resultat:
    # Dokumentet blir tilgjengelig for semantisk
    # søk og kan brukes som kunnskapsgrunnlag for
    # AI-modellen.
    #
    # --------------------------------------------------------
    # --------------------------------------------------------
    
    db_samling.add(
        documents=dokumenter,
        metadatas=metadatas,
        ids=ids
    )

    print(
        f"[Database] Suksess! "
        f"Indekserte {relativ_kilde} "
        f"({len(chunks)} biter)."
    )


# ============================================================
# WATCHDOG
# ============================================================
#
# Hensikt:
# Overvåker dokumentmappen kontinuerlig.
# Overvåker dokumentmappen kontinuerlig mens
# systemet kjører.
#
# Hvis en fil:
# - opprettes
# - endres
# - slettes
#
# så oppdateres ChromaDB automatisk.
#
# Når en fil opprettes, endres eller slettes,
# oppdager Watchdog hendelsen automatisk og
# starter oppdatering av databasen.
#
# Dette gjør at nye dokumenter blir tilgjengelige
# uten at serveren må stoppes og startes på nytt.
#
# Arbeidsflyt:
# Fil endres
# ↓
# Watchdog registrerer endringen
# ↓
# oppdater_fil_i_database()
# ↓
# ChromaDB oppdateres
#
# Resultat:
# ChromaDB holdes automatisk synkronisert med
# innholdet i dokumentmappen.
#
# ============================================================
#
# ============================================================


class DokumentLytter(FileSystemEventHandler):

    def on_modified(self, event):

        if not event.is_directory:

            oppdater_fil_i_database(
                event.src_path
            )

    def on_created(self, event):

        if not event.is_directory:

            oppdater_fil_i_database(
                event.src_path
            )

    def on_deleted(self, event):

        if not event.is_directory:

            oppdater_fil_i_database(
                event.src_path
            )


forste_indeksering_ferdig = Event()


def start_mappe_overvaking():

    try:

        print(
            "[System] Skanner rotmappe og "
            "undermapper rekursivt..."
        )

        if not os.path.exists(MAPPESTI):

            os.makedirs(MAPPESTI)

        # ----------------------------------------------------
        # FØRSTE INDEKSERING
        #
        # Hensikt:
        # Ved oppstart går systemet gjennom alle filer
        # som allerede finnes i dokumentmappen.
        # skannes alle eksisterende filer
        # i dokumentmappen og alle undermapper.
        #
        #
        # Hver fil:
        # - leses inn
        # - deles opp i chunks
        # - lagres i ChromaDB
        #
        # Dette sikrer at databasen inneholder
        # dokumentene før brukeren begynner å søke.#
        # Dette kjøres én gang ved oppstart slik at
        # databasen inneholder alle dokumentene før
        # brukeren begynner å stille spørsmål.
        #
        # Resultat:
        # ChromaDB bygges opp og er klar for søk.
        #
        #----------------------------------------------------
      

        for rot, _, filer in os.walk(MAPPESTI):

            for filnavn in filer:

                oppdater_fil_i_database(
                    os.path.join(
                        rot,
                        filnavn
                    )
                )

        # ----------------------------------------------------
        # ----------------------------------------------------
        # START WATCHDOG
        #
        # Hensikt:
        # Starter Watchdog etter at første indeksering
        # er fullført. Overvåker dokumentmappen og alle undermapper i sanntid.
        #
        # Watchdog overvåker dokumentmappen og alle
        # undermapper for endringer i filer.
        # ## Starter Watchdog-overvåkingen etter at første
        # indeksering er fullført.
        #
        # Hendelser som overvåkes:
        # - Ny fil opprettes
        # - Eksisterende fil endres
        # - Fil slettes
        #
        # Når en endring oppdages, kalles
        # oppdater_fil_i_database() automatisk slik at
        # ChromaDB holdes oppdatert.
        #
        # Resultat:
        ## Dette gjør at ChromaDB alltid er synkronisert
        # med innholdet i dokumentmappen.
        # Nye eller endrede dokumenter blir søkbare
        # uten at serveren må startes på nytt.
        #
        # ----------------------------------------------------

        handler = DokumentLytter()

        observer = Observer()

        observer.schedule(
            handler,
            path=MAPPESTI,
            recursive=True
        )

        observer.start()

        print(
            f"[Watchdog] Lytter rekursivt i: "
            f"{MAPPESTI}"
        )

    except Exception as feil:

        print(
            f"[Feil] Mappeovervåking feilet: "
            f"{feil}"
        )

        forste_indeksering_ferdig.set()

        return

    forste_indeksering_ferdig.set()

    try:

        while True:

            time.sleep(1)

    except KeyboardInterrupt:

        observer.stop()

    observer.join()


# ============================================================
# QWEN-MODELL
#
# Hensikt:
# Laster inn språkmodellen Qwen som skal
# analysere dokumentinnhold og generere svar.
#
# Modellen mottar:
# - Brukerens spørsmål
# - Relevante chunks fra ChromaDB
# - Systeminstruksjoner
#
# Deretter genererer modellen et svar basert
# på informasjonen som ble funnet i dokumentene.
#
# Resultat:
# Systemet kan svare på spørsmål ved hjelp av
# innholdet som er lagret i RAG-løsningen.
#
# ============================================================

print(
    "[System] Laster inn Qwen-modellen..."
)

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

# ============================================================
# MODUSER
#
# Hensikt:
# Definerer hvordan AI-modellen skal formulere
# svaret til brukeren.
#
# Moduser påvirker tonen i svaret, men ikke
# hvilke fakta som brukes.
#
# Standardmoduser:
# - høflig
# - morsom
# - kort
# - vennlig
#
# Systemet kan også laste inn egne moduser fra
# MODUS-mappen uten at koden må endres.
#
# Resultat:
# Samme spørsmål kan besvares med ulik tone,
# men basert på de samme dokumentkildene.
#
# ============================================================

MODUS_INSTRUKSER = {

    "høflig":
        "Formuler SVAR-linjen formelt og høflig, "
        "med fullstendige setninger.",

    "morsom":
        "Formuler SVAR-linjen på en munter "
        "og humoristisk måte.",

    "kort":
        "Formuler SVAR-linjen så kort og "
        "direkte som mulig.",

    "vennlig":
        "Formuler SVAR-linjen uformelt "
        "og vennlig.",
}


def last_inn_moduser():

    global MODUS_INSTRUKSER

    if os.path.exists(MODUS_MAPPE):

        fant_filer = False

        for filnavn in os.listdir(MODUS_MAPPE):

            if filnavn.endswith(".md"):

                modus_navn = os.path.splitext(
                    filnavn
                )[0]

                fil_sti = os.path.join(
                    MODUS_MAPPE,
                    filnavn
                )

                try:

                    with open(
                        fil_sti,
                        "r",
                        encoding="utf-8"
                    ) as f:

                        MODUS_INSTRUKSER[
                            modus_navn
                        ] = f.read().strip()

                        fant_filer = True

                except Exception as e:

                    print(
                        f"[Advarsel] Klarte ikke å "
                        f"lese modus-fil {filnavn}: {e}"
                    )

        if fant_filer:

            print(
                f"[System] Lastet inn "
                f"egendefinerte moduser fra "
                f"{MODUS_MAPPE}"
            )


last_inn_moduser()


STANDARD_MODUS = "høflig"


print(
    f"[System] Aktive toner: "
    f"{list(MODUS_INSTRUKSER.keys())} "
    f"(Standard: {STANDARD_MODUS})"
)

# ============================================================
# SMART ROUTER
#
# Hensikt:
# Kontrollerer om brukerens spørsmål skal
# behandles av en lokal funksjon i stedet for
# AI-modellen.
#
# Smart Router fungerer som en trafikkdirigent
# som analyserer spørsmålet og velger riktig
# behandlingsmåte.
#
# Eksempler:
# - Hente informasjon fra Outlook
# - Opprette mapper på disken
# - Utføre lokale systemoperasjoner
#
# Hvis spørsmålet ikke matcher en kjent
# funksjon, sendes det videre til den vanlige
# RAG-prosessen med ChromaDB og Qwen.
#
# Resultat:
# Systemet kan kombinere AI-søk med lokale
# handlinger i samme løsning.
#
# ============================================================

def smart_router_handling(bruker_sporsmal):

    sporsmal_lower = bruker_sporsmal.lower()

    # --------------------------------------------------------
    # OUTLOOK
    #
    # Hensikt:
    # Behandler spørsmål som gjelder Outlook
    # og e-post.
    #
    # Hvis brukerens spørsmål inneholder ord som
    # "e-post", "innboks" eller "siste e-post",
    # forsøker systemet å hente informasjon direkte
    # fra Outlook på den lokale PC-en.
    #
    # Funksjonen leser de nyeste e-postene og
    # returnerer avsender og emne som lokal
    # informasjon til brukeren.
    #
    # Resultat:
    # Brukeren kan spørre om nylig mottatte
    # e-poster uten å åpne Outlook manuelt.
    #
    # --------------------------------------------------------
    if (
        "e-post" in sporsmal_lower
        or "innboks" in sporsmal_lower
        or "siste e-post" in sporsmal_lower
    ):

        try:

            import win32com.client

            outlook = win32com.client.Dispatch(
                "Outlook.Application"
            )

            namespace = outlook.GetNamespace(
                "MAPI"
            )

            inbox = namespace.GetDefaultFolder(6)

            messages = inbox.Items

            messages.Sort(
                "[ReceivedTime]",
                True
            )

            eposter_tekst = []

            for msg in list(messages)[:3]:

                eposter_tekst.append(
                    f"Fra: {msg.SenderName} | "
                    f"Emne: {msg.Subject}"
                )

            lokal_info = "\n".join(
                eposter_tekst
            )

            return (
                "[Smart Router - Outlook]: "
                "Hentet de siste e-postene lokalt:\n"
                f"{lokal_info}"
            )

        except ImportError:

            return (
                "[Smart Router]: "
                "pywin32 er ikke installert."
            )

        except Exception as e:

            return (
                "[Smart Router]: "
                f"Klarte ikke koble til "
                f"Outlook lokalt: {e}"
            )

    # --------------------------------------------------------
    # OPPRETT MAPPE
    # --------------------------------------------------------
    #
    # Hensikt:
    # Behandler forespørsler om å opprette nye
    # mapper på disken.
    #
    # Smart Router forsøker å hente både
    # mappenavn og ønsket plassering direkte
    # fra brukerens spørsmål.
    #
    # Dersom ingen plassering er angitt,
    # brukes standard dokumentmappe (MAPPESTI).
    #
    # Arbeidsflyt:
    # Brukerspørsmål
    # ↓
    # Finn mappenavn og sti
    # ↓
    # Opprett mappe
    # ↓
    # Returner bekreftelse
    #
    # Resultat:
    # Brukeren kan opprette mapper gjennom
    # naturlig språk uten å bruke Filutforsker.
    #
    # --------------------------------------------------------
    elif (
        "lag en mappe" in sporsmal_lower
        or "lag arkiv" in sporsmal_lower
    ):

        try:

            mappenavn = "ny_mappe"

            if "med navn" in sporsmal_lower:

                mappenavn = (
                    bruker_sporsmal
                    .split("med navn")[1]
                    .strip()
                    .split()[0]
                    .strip("'\"")
                )

            elif "kalt" in sporsmal_lower:

                mappenavn = (
                    bruker_sporsmal
                    .split("kalt")[1]
                    .strip()
                    .split()[0]
                    .strip("'\"")
                )

            if "undere denne stien:" in sporsmal_lower:

                base_sti = (
                    bruker_sporsmal
                    .split("undere denne stien:")[1]
                    .strip()
                    .strip('"\'')
                )

            elif "under denne stien:" in sporsmal_lower:

                base_sti = (
                    bruker_sporsmal
                    .split("under denne stien:")[1]
                    .strip()
                    .strip('"\'')
                )

            elif "stien:" in bruker_sporsmal:

                base_sti = (
                    bruker_sporsmal
                    .split("stien:")[1]
                    .strip()
                    .strip('"\'')
                )

            elif " i " in sporsmal_lower:

                base_sti = (
                    bruker_sporsmal
                    .split(" i ")[-1]
                    .strip()
                    .strip('"\'')
                )

            else:

                base_sti = MAPPESTI

            mal_sti = os.path.join(
                base_sti,
                mappenavn
            )

            os.makedirs(
                mal_sti,
                exist_ok=True
            )

            return (
                "[Smart Router - Filrydder]: "
                f"Opprettet mappe '{mappenavn}' "
                f"vellykket på: {mal_sti}"
            )

        except Exception as e:

            return (
                "[Smart Router]: "
                f"Klarte ikke å opprette mappe: {e}"
            )

    return None


# ============================================================
# BESVAR SPØRSMÅL
# ============================================================
#
# Hensikt:
# Dette er hovedmotoren i RAG-løsningen.
#
# Funksjonen mottar et spørsmål fra brukeren
# og utfører hele prosessen fra søk til svar.
#
# Arbeidsflyt:
# 1. Mottar brukerens spørsmål.
# 2. Utfører vektorsøk i ChromaDB.
# 3. Henter relevante chunks.
# 4. Bygger opp kontekst fra dokumentene.
# 5. Oppretter systemprompt.
# 6. Sender kontekst og spørsmål til Qwen.
# 7. Tolker svaret fra modellen.
# 8. Finner hvilken kilde svaret kom fra.
# 9. Returnerer svar, fakta og kildeinformasjon.
#
# Resultat:
# Brukeren får et svar som er basert på
# informasjon hentet fra de indekserte
# dokumentene.
#
# ============================================================
def besvar_sporsmal(
    bruker_sporsmal,
    gjeldende_modus,
    temperatur=None
):

    # --------------------------------------------------------
    # MODUS
    # --------------------------------------------------------
    # Hensikt:
    # Kontrollerer hvilken skrivestil AI-modellen
    # skal bruke når svaret formuleres.
    #
    # Modusen påvirker hvordan svaret skrives,
    # men ikke hvilke fakta som hentes fra
    # dokumentene.
    #
    # Eksempler:
    # - høflig
    # - vennlig
    # - kort
    # - morsom
    #
    # Hvis brukeren velger en ugyldig modus,
    # brukes standardmodusen automatisk.
    #
    # Resultat:
    # Samme informasjon kan presenteres med
    # ulike formuleringer og tonefall.
    #
    # --------------------------------------------------------
    if gjeldende_modus not in MODUS_INSTRUKSER:

        gjeldende_modus = STANDARD_MODUS

    # --------------------------------------------------------
    # TEMPERATUR
    # --------------------------------------------------------
    # Hensikt:
    # Styrer hvor kreativ eller variert AI-modellen
    # skal være når den genererer svar.
    #
    # Lav temperatur:
    # - Mer presise og forutsigbare svar
    # - Mindre variasjon
    # - Bedre for faktabaserte spørsmål
    #
    # Høy temperatur:
    # - Mer kreative svar
    # - Større variasjon
    # - Kan gi mindre konsistente svar
    #
    # Systemet kontrollerer også at verdien ligger
    # innenfor et gyldig område før spørsmålet
    # sendes til modellen.
    #
    # Resultat:
    # Brukeren kan justere balansen mellom
    # nøyaktighet og kreativitet.
    #
    # --------------------------------------------------------
    try:

        valgt_temperatur = (
            float(temperatur)
            if temperatur is not None
            else generasjons_konfig.temperature
        )

    except (TypeError, ValueError):

        valgt_temperatur = (
            generasjons_konfig.temperature
        )

    valgt_temperatur = max(
        0.01,
        min(1.0, valgt_temperatur)
    )

    # --------------------------------------------------------
    # VARIABLER
    # --------------------------------------------------------
    #
    # Hensikt:
    # Oppretter tomme datastrukturer som skal brukes
    # gjennom resten av spørsmålsbehandlingen.
    #
    # dokument_kontekster:
    # Lagrer tekstbitene (chunks) som hentes fra
    # ChromaDB.
    #
    # metadata_liste:
    # Lagrer informasjon om hver chunk, for eksempel
    # filnavn, filsti og chunknummer.
    #
    # undersokte_biter:
    # Lagrer detaljer om chunkene som ble funnet,
    # slik at de kan vises til brukeren eller brukes
    # til feilsøking.
    #
    # Resultat:
    # Scriptet får datastrukturer som fylles med
    # søketreff før svaret genereres.
    #
    # --------------------------------------------------------
    dokument_kontekster = []

    metadata_liste = []

    undersokte_biter = []

    # ========================================================
    # VEKTORSØK I CHROMADB
    # ========================================================

    sok_resultat = db_samling.query(
        query_texts=[
            bruker_sporsmal
        ],
        n_results=TOP_K_CHUNKS
    )

    # ========================================================
    # HENT RELEVANTE CHUNKS
    # ========================================================
    # Hensikt:
    # Søker etter de dokument-chunkene som er mest
    # relevante for brukerens spørsmål.
    #
    # I stedet for å søke etter eksakte ord,
    # sammenlignes betydningen (embeddings) av
    # spørsmålet mot embeddings lagret i ChromaDB.
    #
    # ChromaDB returnerer de chunkene som har
    # høyest semantisk likhet med spørsmålet.
    #
    # Resultat:
    # Systemet finner de mest relevante
    # tekstutdragene som senere brukes som
    # kontekst for AI-modellen.
    #
    # ========================================================
    if (
        sok_resultat
        and sok_resultat.get("documents")
        and sok_resultat["documents"][0]
    ):

        dokument_kontekster = (
            sok_resultat["documents"][0]
        )

        metadata_liste = (
            sok_resultat["metadatas"][0]
        )

        avstander = sok_resultat.get(
            "distances",
            [
                [None] *
                len(dokument_kontekster)
            ]
        )[0]

        # ----------------------------------------------------
        # LAG INFORMASJON OM HVER CHUNK
        # ----------------------------------------------------
        # Hensikt:
        # Oppretter nødvendig informasjon (metadata)
        # for hver chunk som er laget fra dokumentet.
        #
        # For hver chunk registreres:
        # - Filnavn (kilde)
        # - Full filsti
        # - Chunknummer
        # - Unik ID
        #
        # Metadataene gjør det mulig å spore hvilken
        # fil og hvilken del av dokumentet informasjonen
        # kommer fra.
        #
        # Resultat:
        # Hver chunk får både tekstinnhold og tilhørende
        # informasjon som senere brukes ved søk,
        # feilsøking og kildevisning.
        #
        # ----------------------------------------------------
        for indeks, (
            tekst_bit,
            meta,
            avstand
        ) in enumerate(
            zip(
                dokument_kontekster,
                metadata_liste,
                avstander
            ),
            start=1
        ):

            undersokte_biter.append(
                {
                    "rangering": indeks,
                    "kilde": meta.get("kilde"),
                    "full_sti": meta.get("full_sti"),
                    "chunk": meta.get("chunk_nr"),
                    "avstand": avstand,
                    "tekst": tekst_bit
                }
            )

        # ====================================================
        # VIS RELEVANTE CHUNKS
        # ====================================================
        # Hensikt:
        # Skriver ut alle chunkene som ble valgt av
        # vektorsøket i ChromaDB.
        #
        # Informasjonen brukes til å kontrollere og
        # forstå hvorfor disse chunkene ble valgt som
        # grunnlag for svaret.
        #
        # For hver chunk vises:
        # - Rangering
        # - Filnavn
        # - Full filsti
        # - Chunknummer
        # - Avstand (likhetsscore)
        # - Tekstinnhold
        #
        # Dette er spesielt nyttig under feilsøking,
        # testing og undervisning av RAG-prinsippet.
        #
        # Resultat:
        # Det blir mulig å se nøyaktig hvilke
        # dokumentutdrag AI-modellen mottar før
        # svaret genereres.
        #
        # ====================================================
        print("\n" + "=" * 80)

        print(
            "[RELEVANTE CHUNKS FRA VEKTORSØK]"
        )

        print(
            f"[Spørsmål]: {bruker_sporsmal}"
        )

        print(
            f"[Antall hentet]: "
            f"{len(undersokte_biter)}"
        )

        print(
            f"[TOP_K_CHUNKS]: "
            f"{TOP_K_CHUNKS}"
        )

        print("=" * 80)

        for chunk in undersokte_biter:

            print(
                f"\n--- Rangering "
                f"#{chunk['rangering']} ---"
            )

            print(
                f"Fil       : "
                f"{chunk['kilde']}"
            )

            print(
                f"Full sti  : "
                f"{chunk['full_sti']}"
            )

            print(
                f"Chunk nr. : "
                f"{chunk['chunk']}"
            )

            print(
                f"Avstand   : "
                f"{chunk['avstand']}"
            )

            print(
                "Tekst:"
            )

            print(
                chunk["tekst"]
            )

            print(
                "-" * 80
            )

        print("=" * 80)

        print(
            "[SLUTT PÅ HENTEDE CHUNKS]"
        )

        print(
            "=" * 80 + "\n"
        )

    # ========================================================
    # INGEN RELEVANTE DOKUMENTER
    # ========================================================
    # Hensikt:
    # Håndterer situasjonen der ChromaDB ikke finner
    # noen relevante chunks som passer til brukerens
    # spørsmål.
    #
    # Dette kan skje hvis:
    # - Informasjonen ikke finnes i dokumentene
    # - Dokumentene ikke er indeksert
    # - Spørsmålet ikke ligner innholdet i databasen
    #
    # I stedet for å gjette eller finne på et svar,
    # returnerer systemet en tydelig melding om at
    # ingen relevante dokumenter ble funnet.
    #
    # Resultat:
    # Brukeren får beskjed om at svaret ikke finnes
    # i tilgjengelige dokumenter, og AI-modellen
    # unngår å hallusinere informasjon.
    #
    # ========================================================
    if not dokument_kontekster:

        print(
            "[AI-Statistikk] "
            "Ingen relevante dokumenter funnet. "
            "(0 tokens, 0.00 sek)"
        )

        return {
            "svar":
                "Fant ingen relevante dokumenter.",

            "fakta":
                "",

            "sammenligning":
                "",

            "kilde":
                "",

            "chunk":
                None,

            "advarsel":
                None,

            "kandidater":
                [],

            "statistikk":
                {
                    "brukt_tid_sekund": 0.0,
                    "antall_tokens": 0,
                    "tokens_per_sekund": 0.0
                }
        }

    # ========================================================
    # KOMBINER CHUNKS TIL KONTEKST
    # ========================================================
    #
    # Hensikt:
    # Samler alle relevante chunks fra ChromaDB til
    # én felles kontekst som kan sendes til AI-modellen.
    #
    # Hver chunk inneholder:
    # - Filnavn
    # - Full filsti
    # - Chunknummer
    # - Tekstinnhold
    #
    # De valgte chunkene settes sammen til én
    # sammenhengende tekstblokk som blir en del av
    # prompten til Qwen.
    #
    # Dette er "Augmentation"-delen i RAG:
    #
    # Retrieval -> Finn relevante chunks
    # Augmentation -> Legg chunkene inn i prompten
    # Generation -> Generer svar
    #
    # Resultat:
    # AI-modellen får tilgang til relevant
    # dokumentinnhold før svaret genereres.
    #
    # ========================================================
    kombinert_kontekst = "\n\n".join(

        f"KILDEFIL: {meta.get('kilde')} | "
        f"FULL STI: {meta.get('full_sti')} "
        f"(Chunk {meta.get('chunk_nr')})\n"
        f"TEKST:\n{tekst}"

        for tekst, meta in zip(
            dokument_kontekster,
            metadata_liste
        )
    )

    # ========================================================
    # SYSTEMPROMPT
    # ========================================================
    # Hensikt:
    # Oppretter instruksjonene som styrer hvordan
    # Qwen skal analysere dokumentene og formulere
    # svaret.
    #
    # Systemprompten fungerer som regelsettet for
    # AI-modellen og sendes sammen med:
    # - Brukerens spørsmål
    # - Relevante chunks fra ChromaDB
    #
    # Viktige regler:
    # - Bruk kun informasjon fra dokumentene
    # - Ikke gjett eller finn på informasjon
    # - Oppgi svar i et fast format
    # - Prioriter dokumentkildene fremfor generell kunnskap
    #
    # Dette bidrar til å redusere hallusinasjoner
    # og sikrer at svarene er basert på faktiske
    # dokumentkilder.
    #
    # Resultat:
    # AI-modellen får klare instrukser om hvordan
    # spørsmålene skal besvares før genereringen starter.
    #
    # ========================================================
    meldinger = [

        {
            "role": "system",

            "content": (

                "1. Du er en konsis dokumentassistent. "

                "2. Svar utelukkende basert på "
                "de oppgitte kildene. "

                "3. Du skal alltid prioritere "
                "informasjonen i de oppgitte kildene. "

                "4. Hvis du er i tvil om informasjonen "
                "finnes, gå grundig gjennom alle "
                "kilde-utdragene en ekstra gang "
                "før du konkluderer. "

                "5. Svar direkte, korrekt og uten "
                "unødvendige høflighetsfraser, "
                "med naturlig, flytende norsk språk.\n\n"

                "REGLER FOR SVAR:\n"

                "6. Svar direkte på det som blir spurt om, "
                "uten å kommentere spørsmålsstillingen, "
                "brukerens antakelser eller hvorfor "
                "de spør.\n"

                "7. AVGJØR FØRST HVILKEN TYPE "
                "HENVENDELSE DETTE ER:\n"

                "   a) PÅSTAND: brukeren fremsetter "
                "en konkret påstand som kan være sann "
                "eller usann "
                "(f.eks. 'starter skolen klokken 10?', "
                "'stemmer det at biblioteket er stengt "
                "på lørdager?'). "

                "For denne typen SKAL SVAR-LINJEN "
                "ALLTID starte med 'Nei, ' etterfulgt "
                "av den korrekte informasjonen hvis "
                "påstanden er feil, eller 'Ja, ' "
                "etterfulgt av bekreftelsen hvis "
                "påstanden stemmer.\n"

                "   b) ÅPENT SPØRSMÅL: brukeren spør "
                "etter informasjon uten å fremsette "
                "noen påstand å vurdere "
                "(f.eks. 'når starter skolen', "
                "'hvor holder biblioteket til', "
                "'hvem er rektor'). "

                "For denne typen skal du IKKE bruke "
                "ordet 'Ja' eller 'Nei' i det hele tatt "
                "— ordene har ingen betydning her. "

                "Svar direkte og naturlig på spørsmålet "
                "med informasjonen fra kilden.\n"

                "8. Hvis kildene ikke inneholder svaret, "
                "si tydelig at informasjonen ikke finnes "
                "i kildene du har tilgjengelig.\n"

                "9. Ikke gjett eller fyll inn med "
                "generell kunnskap.\n"

                "10. Konkrete verdier som datoer, "
                "klokkeslett, tall og navn skal alltid "
                "gjengis EKSAKT slik de står i "
                "kildeteksten.\n"

                "11. Ikke rund av, ikke generaliser, "
                "og ikke legg til årstall, klokkeslett "
                "eller andre detaljer som ikke "
                "eksplisitt står i kilden.\n"

                "12. Hvis du er usikker på om en verdi "
                "stemmer med kildene, skal du heller "
                "si at du er usikker enn å gjette.\n"

                "13. Svar ALLTID i nøyaktig dette "
                "formatet med tre linjer:\n"

                "FAKTA: <en nøyaktig setning fra "
                "tekstbitene eller tilhørende "
                "filinformasjon>\n"

                "SAMMENLIGNING: <si eksplisitt om dette "
                "er en PÅSTAND eller et ÅPENT SPØRSMÅL, "
                "og begrunn kort ut fra kilden>\n"

                "SVAR: <Hvis PÅSTAND: MÅ starte med "
                "'Nei, ' eller 'Ja, ' etterfulgt av "
                "fasiten. Hvis ÅPENT SPØRSMÅL: svar "
                "direkte på spørsmålet UTEN å starte "
                "med 'Ja' eller 'Nei'>.\n\n"

                f"----- EKTE DOKUMENTINNHOLD -----\n"
                f"{kombinert_kontekst}\n\n"

                f"TONE FOR SVAR-LINJEN: "
                f"{MODUS_INSTRUKSER[gjeldende_modus]}"
            )
        }

    ]

    # ========================================================
    # BRUKERSPØRSMÅL
    # ========================================================
    # Hensikt:
    # Legger brukerens spørsmål inn i meldingslisten
    # som sendes til AI-modellen.
    #
    # Meldingslisten består nå av:
    # - Systemprompt (regler og instruksjoner)
    # - Brukerspørsmål
    #
    # Spørsmålet kombineres senere med de relevante
    # dokument-chunkene slik at modellen kan generere
    # et svar basert på tilgjengelig kontekst.
    #
    # Resultat:
    # AI-modellen mottar både instruksjoner og
    # brukerens spørsmål som grunnlag for svaret.
    #
    # ========================================================
    meldinger.append(
        {
            "role": "user",
            "content": bruker_sporsmal
        }
    )

    # ========================================================
    # LAG PROMPT
    # ========================================================
    # Hensikt:
    # Setter sammen alle meldingene til én ferdig
    # prompt som kan sendes til Qwen-modellen.
    #
    # Prompten består av:
    # - Systemprompt (regler og instrukser)
    # - Dokumentkontekst (relevante chunks)
    # - Brukerens spørsmål
    #
    # Prompten formateres til et format som
    # Qwen-modellen forstår før genereringen starter.
    #
    # Resultat:
    # Modellen mottar all nødvendig informasjon i
    # én samlet forespørsel og kan generere et
    # dokumentbasert svar.
    #
    # ========================================================
    prompt = (
        ai_modell.tokenizer.apply_chat_template(
            meldinger,
            tokenize=False,
            add_generation_prompt=True
        )
    )

    # ========================================================
    # GENERASJONSKONFIGURASJON
    # ========================================================
    #
    # Hensikt:
    # Definerer hvordan Qwen-modellen skal generere
    # svaret etter at prompten er opprettet.
    #
    # Konfigurasjonen styrer blant annet:
    # - Maksimalt antall tokens i svaret
    # - Temperatur (kreativitet)
    # - Om modellen skal bruke sampling
    # - Top-p filtrering av neste ord
    #
    # Disse innstillingene påvirker hvordan
    # modellen skriver svaret, men ikke hvilke
    # dokumenter eller chunks som brukes.
    #
    # Parametrene kan tilpasses for hver
    # forespørsel uten å endre den globale
    # standardkonfigurasjonen.
    #
    # Resultat:
    # Qwen mottar en egen konfigurasjon som
    # bestemmer hvordan svaret skal genereres.
    #
    # ========================================================
    per_forsporsel_konfig = GenerationConfig(

        max_new_tokens=
            generasjons_konfig.max_new_tokens,

        do_sample=
            generasjons_konfig.do_sample,

        temperature=
            valgt_temperatur,

        top_p=
            generasjons_konfig.top_p,
    )

    # ========================================================
    # KJØR QWEN
    # ========================================================
    # Hensikt:
    # Sender den ferdige prompten til Qwen-modellen
    # for behandling og svargenerering.
    #
    # På dette tidspunktet er alle forberedelser
    # fullført:
    # - Relevante chunks er hentet fra ChromaDB
    # - Kontekst er bygget opp
    # - Systemprompt er opprettet
    # - Brukerspørsmålet er lagt til
    # - Generasjonsinnstillinger er valgt
    #
    # Qwen analyserer deretter all informasjonen og
    # genererer et svar basert på dokumentinnholdet.
    #
    # Resultat:
    # Modellen returnerer generert tekst som senere
    # skal analyseres, valideres og sendes tilbake
    # til brukeren.
    #
    # ========================================================
    start_tid = time.time()

    resultat = ai_modell(

        prompt,

        generation_config=
            per_forsporsel_konfig,

        return_full_text=False,

        clean_up_tokenization_spaces=False,
    )

    slutt_tid = time.time()

    brukt_tid = (
        slutt_tid - start_tid
    )

    # ========================================================
    # TOKEN-STATISTIKK
    # ========================================================
    # Hensikt:
    # Måler hvor mye tekst Qwen genererte og hvor
    # lang tid svargenereringen tok.
    #
    # Det registreres:
    # - Antall genererte tokens
    # - Brukt tid i sekunder
    # - Tokens per sekund
    #
    # Denne informasjonen brukes til overvåking,
    # testing og ytelsesmåling av AI-modellen.
    #
    # Resultat:
    # Det blir mulig å vurdere hvor raskt modellen
    # svarer og hvor store svar som genereres.
    #
    # ========================================================
    full_generert_tekst = (
        resultat[0]["generated_text"]
    )

    genererte_tokens = len(
        ai_modell.tokenizer.encode(
            full_generert_tekst
        )
    )

    tokens_per_sekund = (
        genererte_tokens / brukt_tid
        if brukt_tid > 0
        else 0
    )

    print(
        "\n=========================================="
    )

    print(
        f"[AI-SPØRSMÅL]: "
        f"{bruker_sporsmal}"
    )

    print(
        f"[STATISTIKK] Antall tokens: "
        f"{genererte_tokens}"
    )

    print(
        f"[STATISTIKK] Brukt tid: "
        f"{brukt_tid:.2f} sekunder"
    )

    print(
        f"[STATISTIKK] Hastighet: "
        f"{tokens_per_sekund:.2f} tokens/sekund"
    )

    print(
        "==========================================\n"
    )

    # ========================================================
    # PARSE SVARET
    # ========================================================
    # Hensikt:
    # Leser og tolker svaret som kommer tilbake fra
    # Qwen-modellen.
    #
    # Modellen er instruert til å returnere svaret
    # i et fast format med tre deler:
    #
    # - FAKTA
    # - SAMMENLIGNING
    # - SVAR
    #
    # Denne seksjonen finner de ulike delene og
    # lagrer dem i separate variabler slik at de
    # senere kan vises i brukergrensesnittet.
    #
    # Resultat:
    # Det genererte svaret deles opp i strukturert
    # informasjon som kan behandles videre av
    # programmet.
    #
    # ========================================================
    svar_linjer = (
        full_generert_tekst.split("\n")
    )

    svar = ""

    fakta_linje = ""

    sammenligning_linje = ""

    for linje in svar_linjer:

        linje = linje.strip()

        if linje.upper().startswith("SVAR:"):

            svar = (
                linje.split(
                    ":",
                    1
                )[1].strip()
            )

        elif linje.upper().startswith("FAKTA:"):

            fakta_linje = (
                linje.split(
                    ":",
                    1
                )[1].strip()
            )

        elif linje.upper().startswith(
            "SAMMENLIGNING:"
        ):

            sammenligning_linje = (
                linje.split(
                    ":",
                    1
                )[1].strip()
            )

    if not svar:

        svar = full_generert_tekst.strip()

    # ========================================================
    # KONTROLLER SVAR
    # ========================================================
    # Hensikt:
    # Kontrollerer at svaret fra Qwen er gyldig og
    # inneholder nyttig informasjon.
    #
    # Noen ganger kan modellen returnere kun:
    # - "Ja"
    # - "Nei"
    #
    # uten noen forklaring. Dette gir liten verdi
    # for brukeren.
    #
    # Dersom svaret er for kort eller mangler
    # innhold, forsøker systemet å bygge opp et
    # bedre svar ved å bruke informasjonen som ble
    # funnet i FAKTA-linjen.
    #
    # Dette fungerer som en ekstra kvalitetskontroll
    # før svaret sendes tilbake til brukergrensesnittet.
    #
    # Resultat:
    # Brukeren får et mer komplett og forklarende
    # svar selv om modellen skulle generere et
    # ufullstendig resultat.
    #
    # ========================================================
    def svar_mangler_innhold(tekst):

        kjerne = re.sub(

            r"^(ja|nei)\b[,.:]?\s*",

            "",

            tekst.strip(),

            flags=re.IGNORECASE
        ).strip()

        return len(kjerne) < 4

    if (
        svar_mangler_innhold(svar)
        and fakta_linje
    ):

        ja_nei_match = re.match(

            r"^(ja|nei)\b",

            svar.strip(),

            flags=re.IGNORECASE
        )

        prefiks = (

            f"{ja_nei_match.group(1).capitalize()}, "

            if ja_nei_match
            else ""
        )

        svar = (
            f"{prefiks}"
            f"{fakta_linje}"
        )

        print(
            "[Advarsel] Modellen ga et "
            "ufullstendig SVAR (kun Ja/Nei) "
            "— falt tilbake til FAKTA-linjen."
        )

    # ========================================================
    # FINN KILDE FOR FAKTA
    # ========================================================
    # Hensikt:
    # Forsøker å finne hvilken chunk og hvilken fil
    # som best støtter faktaene i det genererte svaret.
    #
    # Systemet sammenligner ordene i FAKTA-linjen med
    # innholdet i de chunkene som ble hentet fra
    # ChromaDB.
    #
    # Chunken med størst tekstlig overlapp blir valgt
    # som mest sannsynlige kilde.
    #
    # Dette gjør det mulig å vise:
    # - Hvilken fil informasjonen kom fra
    # - Hvilken chunk som inneholdt faktaene
    #
    # Resultat:
    # Svaret kan knyttes til en konkret dokumentkilde,
    # noe som gjør løsningen mer sporbar og lettere
    # å kontrollere.
    #
    # ========================================================
    def finn_kilde_for_fakta(
        fakta,
        kontekst_biter,
        metadata
    ):

        if (
            not fakta
            or fakta.strip().lower()
            in ("(ingen)", "ingen", "")
        ):

            return (
                metadata[0]["kilde"],
                metadata[0]["chunk_nr"]
            )

        fakta_ord = set(

            w.lower().strip(".,?!")

            for w in fakta.split()

            if len(w) > 3
        )

        beste_indeks = 0

        beste_overlapp = -1

        for i, bit in enumerate(
            kontekst_biter
        ):

            bit_ord = set(

                w.lower().strip(".,?!")

                for w in bit.split()
            )

            overlapp = len(
                fakta_ord & bit_ord
            )

            if overlapp > beste_overlapp:

                beste_overlapp = overlapp

                beste_indeks = i

        return (
            metadata[beste_indeks]["kilde"],
            metadata[beste_indeks]["chunk_nr"]
        )

    # ========================================================
    # FINN KILDE
    # ========================================================
    # Hensikt:
    # Kaller funksjonen som finner den mest
    # sannsynlige kilden til faktaene i svaret.
    #
    # Basert på innholdet i FAKTA-linjen forsøker
    # systemet å identifisere:
    # - Hvilken fil informasjonen kom fra
    # - Hvilken chunk som inneholdt informasjonen
    #
    # Denne informasjonen brukes til å dokumentere
    # hvor svaret kommer fra og gjør det mulig å
    # kontrollere og verifisere kildene.
    #
    # Resultat:
    # Svaret knyttes til en konkret fil og et
    # konkret chunknummer før resultatet sendes
    # tilbake til brukeren.
    #
    # ========================================================
    kilde_fil, chunk_nr = (
        finn_kilde_for_fakta(
            fakta_linje,
            dokument_kontekster,
            metadata_liste
        )
    )

    # ========================================================
    # RETURNER RESULTAT
    # ========================================================
    # Hensikt:
    # Samler alle resultatene fra spørsmålsbehandlingen
    # og returnerer dem til Flask og brukergrensesnittet.
    #
    # Informasjonen som returneres inkluderer:
    # - Svaret fra Qwen
    # - Fakta-linjen
    # - Sammenligning-linjen
    # - Kildefil
    # - Chunknummer
    # - Relevante chunks
    # - AI-statistikk
    #
    # Disse dataene brukes senere av API-et og
    # nettsiden for å vise både svar, kilder og
    # teknisk informasjon til brukeren.
    #
    # Resultat:
    # Hele resultatpakken sendes tilbake som et
    # strukturert Python-objekt som kan konverteres
    # til JSON og vises i nettleseren.
    #
    # ========================================================
    return {

        "svar":
            svar,

        "fakta":
            fakta_linje,

        "sammenligning":
            sammenligning_linje,

        "kilde":
            kilde_fil,

        "chunk":
            chunk_nr,

        "advarsel":
            None,

        "kandidater":
            undersokte_biter,

        "statistikk":
            {
                "brukt_tid_sekund":
                    round(
                        brukt_tid,
                        2
                    ),

                "antall_tokens":
                    genererte_tokens,

                "tokens_per_sekund":
                    round(
                        tokens_per_sekund,
                        2
                    )
            }
    }


# ============================================================
# FLASK
# ============================================================
# Hensikt:
# Oppretter webserveren som kobler sammen
# brukergrensesnittet og RAG-motoren.
#
# Flask håndterer:
# - Nettsiden som vises i nettleseren
# - API-kall fra brukergrensesnittet
# - Mottak av spørsmål
# - Retur av svar som JSON
#
# Flask fungerer som bindeleddet mellom:
#
# Nettleser
# ↓
# Flask
# ↓
# ChromaDB + Qwen
# ↓
# Flask
# ↓
# Nettleser
#
# Resultat:
# Brukeren kan stille spørsmål via en nettside
# og motta svar fra RAG-løsningen i sanntid.
#
# ============================================================
STATIC_MAPPE = (
    r"C:\working\python\Project html\static"
)

TEMPLATE_MAPPE = (
    r"C:\working\python\Project html\templates"
)


app = Flask(

    __name__,

    static_folder=STATIC_MAPPE,

    static_url_path="/static",

    template_folder=TEMPLATE_MAPPE
)


app.config[
    "TEMPLATES_AUTO_RELOAD"
] = True


# ============================================================
# HOVEDSIDE
# ============================================================
# Hensikt:
# Viser startsiden til RAG-applikasjonen når
# brukeren åpner systemet i nettleseren.
#
# Flask laster HTML-filen (index.html) og sender
# samtidig informasjon om hvilke moduser som er
# tilgjengelige for AI-modellen.
#
# Informasjon som sendes til nettsiden:
# - Tilgjengelige moduser
# - Standardmodus
#
# Resultat:
# Brukeren får opp webgrensesnittet og kan
# begynne å stille spørsmål til systemet.
#
# Arbeidsflyt:
#
# Nettleser
# ↓
# http://127.0.0.1:5000
# ↓
# Flask
# ↓
# index()
# ↓
# index.html
# ↓
# Brukergrensesnitt vises
#
# ============================================================
@app.route("/")
def index():

    return render_template(

        "index.html",

        moduser=list(
            MODUS_INSTRUKSER.keys()
        ),

        standard_modus=
            STANDARD_MODUS
    )


# ============================================================
# API - SPØRSMÅL
# ============================================================
# Hensikt:
# Mottar spørsmål fra webgrensesnittet og sender
# dem videre til RAG-motoren for behandling.
#
# Denne API-ruten fungerer som bindeleddet mellom
# nettleseren og funksjonen besvar_sporsmal().
#
# Arbeidsflyt:
# 1. Motta spørsmål fra nettsiden.
# 2. Les valgt modus og temperatur.
# 3. Kjør Smart Router ved behov.
# 4. Utfør dokumentsøk i ChromaDB.
# 5. Generer svar med Qwen.
# 6. Returner resultatet som JSON.
#
# Resultatet sendes tilbake til nettleseren og
# vises i brukergrensesnittet.
#
# Arbeidsflyt:
#
# Nettleser
# ↓
# /api/sporsmal
# ↓
# Flask
# ↓
# besvar_sporsmal()
# ↓
# ChromaDB + Qwen
# ↓
# JSON-svar
# ↓
# Nettleser
#
# ============================================================
@app.route(
    "/api/sporsmal",
    methods=["POST"]
)
def api_sporsmal():

    data = (
        request.get_json(
            force=True,
            silent=True
        )
        or {}
    )

    bruker_sporsmal = (
        data.get("sporsmal")
        or ""
    ).strip()

    gjeldende_modus = (
        data.get("modus")
        or STANDARD_MODUS
    )

    temperatur = (
        data.get("temperatur")
    )

    if not bruker_sporsmal:

        return jsonify(
            {
                "feil":
                    "Tomt spørsmål."
            }
        ), 400

    try:

        smart_melding = (
            smart_router_handling(
                bruker_sporsmal
            )
        )

        resultat = (
            besvar_sporsmal(
                bruker_sporsmal,
                gjeldende_modus,
                temperatur
            )
        )

        if smart_melding:

            resultat["svar"] = (
                f"{smart_melding}\n\n"
                f"{resultat['svar']}"
            )

        return jsonify(resultat)

    except Exception as feil:

        print(
            f"[Feil] Uventet feil: "
            f"{feil}"
        )

        return jsonify(
            {
                "feil":
                    f"Noe gikk galt "
                    f"på serveren: {feil}"
            }
        ), 500


# ============================================================
# API - NULLSTILL
# ============================================================
# Hensikt:
# Mottar forespørsler om å nullstille
# brukergrensesnittet.
#
# Denne ruten utfører ingen behandling av
# dokumenter eller AI-svar. Den returnerer
# kun en bekreftelse på at nullstillingen
# er mottatt.
#
# Brukes typisk når brukeren ønsker å:
# - Tømme samtalen
# - Starte et nytt søk
# - Nullstille visningen i nettleseren
#
# Arbeidsflyt:
#
# Nettleser
# ↓
# /api/nullstill
# ↓
# Flask
# ↓
# Status = OK
# ↓
# Nettleser
#
# Resultat:
# Brukergrensesnittet kan starte med en
# tom samtale uten å måtte laste siden på nytt.
#
# ============================================================

@app.route(
    "/api/nullstill",
    methods=["POST"]
)
def api_nullstill():

    return jsonify(
        {
            "status":
                "ok"
        }
    )


# ============================================================
# START SERVER
# ============================================================
# Hensikt:
# Dette er startpunktet for hele programmet.
#
# Når scriptet startes:
# 1. Mappeovervåkingen startes i en egen tråd.
# 2. Første indeksering fullføres.
# 3. Flask-serveren startes.
# 4. Systemet blir tilgjengelig i nettleseren.
#
# Arbeidsflyt:
#
# Start script
# ↓
# Start Watchdog
# ↓
# Første indeksering
# ↓
# Flask startes
# ↓
# http://127.0.0.1:5000
# ↓
# Brukeren kan stille spørsmål
#
# Resultat:
# Hele RAG-løsningen blir aktiv og klar til
# å motta spørsmål fra brukergrensesnittet.
#
# ============================================================
if __name__ == "__main__":

    bakgrunns_lytter = Thread(

        target=
            start_mappe_overvaking,

        daemon=True
    )

    bakgrunns_lytter.start()

    if not forste_indeksering_ferdig.wait(
        timeout=60
    ):

        print(
            "[Advarsel] Indeksering tok "
            "lengre tid enn forventet."
        )

    print(
        "[System] Serveren kjører på "
        "http://127.0.0.1:5000"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )