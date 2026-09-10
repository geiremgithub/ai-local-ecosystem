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

SKRIPT_VERSJON = "v54-web - viser relevante chunks ved hvert søk"

print(f"[System] Kjører skriptversjon: {SKRIPT_VERSJON}")


# ============================================================
# MASKINVARE - 
# ============================================================

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
# IMPORTER -
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
    # --------------------------------------------------------

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
        # ----------------------------------------------------

        for rot, _, filer in os.walk(MAPPESTI):

            for filnavn in filer:

                oppdater_fil_i_database(
                    os.path.join(
                        rot,
                        filnavn
                    )
                )

        # ----------------------------------------------------
        # START WATCHDOG
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
# ============================================================

def smart_router_handling(bruker_sporsmal):

    sporsmal_lower = bruker_sporsmal.lower()

    # --------------------------------------------------------
    # OUTLOOK
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

def besvar_sporsmal(
    bruker_sporsmal,
    gjeldende_modus,
    temperatur=None
):

    # --------------------------------------------------------
    # MODUS
    # --------------------------------------------------------

    if gjeldende_modus not in MODUS_INSTRUKSER:

        gjeldende_modus = STANDARD_MODUS

    # --------------------------------------------------------
    # TEMPERATUR
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

    meldinger.append(
        {
            "role": "user",
            "content": bruker_sporsmal
        }
    )

    # ========================================================
    # LAG PROMPT
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