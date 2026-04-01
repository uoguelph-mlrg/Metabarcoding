#!/usr/bin/env python3
"""
fetch_bin_sequences_from_bold.py

Reads taxonomic_data.csv with columns:
bin_uri,phylum,class,order,family,subfamily,genus,species

For each unique BIN URI this script:
 - fetches sequences from BOLD v3 sequence API (FASTA)
 - parses FASTA entries
 - chooses the most frequent identical sequence (tie -> longest)
 - performs sanity checks
 - writes BIN_data.csv containing original columns + sequence, n_sequences, seq_length, notes

It also writes a small cache file bin_sequences_cache.csv to allow resuming without re-downloading.
"""

import re
import time
import csv
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

# -----------------------
# User-editable settings
# -----------------------
INPUT_CSV = "Metabarcoding/data/taxonomic_data.csv"          # your input (must contain bin_uri)
OUTPUT_CSV = "Metabarcoding/data/BIN_data.csv"               # output with sequences added
CACHE_CSV = "bin_sequences_cache.csv"     # incremental cache to resume
BOLD_SEQUENCE_URL = "https://v3.boldsystems.org/index.php/API_Public/sequence"
REQUEST_TIMEOUT = 30                      # seconds
DELAY_BETWEEN_REQUESTS = 0.10             # polite delay; increase if BOLD complains
MAX_RETRIES = 5
COI_MIN_LEN = 400                         # typical COI barcode lower bound (flag if below)
COI_MAX_LEN = 800                         # typical COI barcode upper bound (flag if above)
# -----------------------

DNA_CHARS_RE = re.compile(r'^[ACGTN\-]+$')

def clean_sequence(seq: str) -> str:
    """Normalize a sequence: uppercase, remove whitespace, remove any characters not ACGTN-"""
    if seq is None:
        return ""
    s = re.sub(r'\s+', '', seq).upper()
    # keep hyphens (gaps) and N for ambiguous base
    s = re.sub(r'[^ACGTN\-]', '', s)
    return s

def parse_fasta(fasta_text: str):
    """Parse FASTA text -> list of (header, sequence) tuples"""
    entries = []
    header = None
    seq_lines = []
    for line in fasta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(seq_lines)))
            header = line[1:].strip()
            seq_lines = []
        else:
            seq_lines.append(line)
    if header is not None:
        entries.append((header, "".join(seq_lines)))
    return entries

def fetch_fasta_for_bin(bin_uri: str):
    """Fetch FASTA from BOLD sequence endpoint with simple retry/backoff. Returns raw text or None."""
    params = {"bin": bin_uri, "format": "fasta"}
    backoff = 1.0
    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(BOLD_SEQUENCE_URL, params=params, timeout=REQUEST_TIMEOUT,
                             headers={"User-Agent": "fetch_bin_sequences_script/1.0"})
            # if server returns 200 but empty body or a short 'No records found', return text for parsing
            return r.text
        except requests.RequestException as exc:
            last_exception = exc
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
            else:
                raise
    # unreachable normally
    if last_exception:
        raise last_exception
    return None

def choose_most_frequent_sequence(parsed_entries):
    """
    parsed_entries: list of (header, seq)
    Returns (representative_sequence_or_None, n_sequences, note)
    note is empty string if everything OK; otherwise describes issues
    """
    cleaned = []
    for hdr, seq in parsed_entries:
        c = clean_sequence(seq)
        if c:
            cleaned.append(c)

    if not cleaned:
        return None, 0, "no_valid_sequences"

    counter = Counter(cleaned)
    n_total = sum(counter.values())

    # choose most frequent; if tie choose the longest sequence among the tied ones
    max_count = max(counter.values())
    candidates = [s for s, cnt in counter.items() if cnt == max_count]
    best = max(candidates, key=len)

    # sanity notes
    notes = []
    # check for multiple distinct sequences
    if len(counter) > 1:
        notes.append("multiple_distinct_sequences")

    # check allowed characters
    if not DNA_CHARS_RE.match(best):
        notes.append("nonstandard_characters_in_sequence")

    seq_len = len(best)
    if seq_len < COI_MIN_LEN or seq_len > COI_MAX_LEN:
        notes.append(f"length_out_of_expected_range({seq_len})")

    note_str = ";".join(notes)
    return best, n_total, note_str

def load_cache(cache_path: Path):
    """Load cache csv into dict bin_uri->record"""
    if not cache_path.exists():
        return {}
    df = pd.read_csv(cache_path, dtype=str)
    cache = {}
    for _, row in df.iterrows():
        cache[row["bin_uri"]] = {
            "sequence": row.get("sequence", None),
            "n_sequences": int(row.get("n_sequences", 0)) if pd.notna(row.get("n_sequences", None)) else 0,
            "seq_length": int(row.get("seq_length")) if pd.notna(row.get("seq_length", None)) else None,
            "notes": row.get("notes", "")
        }
    return cache

def append_to_cache(cache_path: Path, record: dict):
    """Append a single record dict to cache CSV (creates file with header if needed)."""
    file_exists = cache_path.exists()
    with cache_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["bin_uri", "sequence", "n_sequences", "seq_length", "notes"])
        writer.writerow([
            record["bin_uri"],
            record["sequence"] if record["sequence"] is not None else "",
            record["n_sequences"],
            record["seq_length"] if record["seq_length"] is not None else "",
            record["notes"] or ""
        ])

def main():
    # read input CSV
    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    if "bin_uri" not in df.columns:
        raise SystemExit("Input CSV must contain 'bin_uri' column")

    cache_path = Path(CACHE_CSV)
    cache = load_cache(cache_path)
    # we'll produce final dataframe by merging later
    unique_bins = df["bin_uri"].astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
    print(f"Found {len(unique_bins)} unique non-empty BIN URIs in {INPUT_CSV}")

    results = {}  # bin_uri -> record

    # pre-populate with cache
    for b, rec in cache.items():
        results[b] = rec

    # iterate and fetch missing
    for idx, bin_uri in enumerate(unique_bins, start=1):
        bin_uri = bin_uri.strip()
        if not bin_uri:
            continue
        if bin_uri in results:
            if idx % 50 == 0:
                print(f"[{idx}/{len(unique_bins)}] {bin_uri} (cached)")
            continue

        print(f"[{idx}/{len(unique_bins)}] Fetching {bin_uri} ... ", end="", flush=True)
        try:
            raw = fetch_fasta_for_bin(bin_uri)
        except Exception as exc:
            print(f"ERROR_FETCH ({exc})")
            rec = {"bin_uri": bin_uri, "sequence": None, "n_sequences": 0, "seq_length": None, "notes": f"fetch_exception:{exc}"}
            results[bin_uri] = rec
            append_to_cache(cache_path, rec)
            time.sleep(DELAY_BETWEEN_REQUESTS)
            continue

        txt = (raw or "").strip()
        # handle obvious no-record messages
        if not txt:
            print("NO_RECORDS")
            rec = {"bin_uri": bin_uri, "sequence": None, "n_sequences": 0, "seq_length": None, "notes": "no_records"}
            results[bin_uri] = rec
            append_to_cache(cache_path, rec)
            time.sleep(DELAY_BETWEEN_REQUESTS)
            continue

        # if FASTA present -> parse
        parsed = []
        if txt.startswith(">"):
            parsed = parse_fasta(txt)
        else:
            # defensive parsing: try to locate any long DNA-like lines inside the response
            dna_candidates = []
            for line in txt.splitlines():
                line = line.strip()
                # consider a line DNA-like if it contains many A/C/G/T/N characters and long enough
                if len(line) >= 100 and re.search(r'[ACGTNacgtn]', line):
                    # keep only DNA-like characters
                    cand = re.sub(r'[^ACGTNacgtn\-]', '', line).upper()
                    if len(cand) >= 50:
                        dna_candidates.append(("extracted_line", cand))
            parsed = dna_candidates

        if not parsed:
            print("NO_VALID_SEQ")
            rec = {"bin_uri": bin_uri, "sequence": None, "n_sequences": 0, "seq_length": None, "notes": "no_valid_seq_in_response"}
            results[bin_uri] = rec
            append_to_cache(cache_path, rec)
            time.sleep(DELAY_BETWEEN_REQUESTS)
            continue

        best_seq, nseq, note = choose_most_frequent_sequence(parsed)
        if best_seq is None:
            print("NO_VALID_SEQ_AFTER_CLEAN")
            rec = {"bin_uri": bin_uri, "sequence": None, "n_sequences": nseq, "seq_length": None, "notes": note or "no_valid_seq_after_clean"}
            results[bin_uri] = rec
            append_to_cache(cache_path, rec)
            time.sleep(DELAY_BETWEEN_REQUESTS)
            continue

        seq_len = len(best_seq)
        rec = {"bin_uri": bin_uri, "sequence": best_seq, "n_sequences": nseq, "seq_length": seq_len, "notes": note or ""}
        results[bin_uri] = rec
        append_to_cache(cache_path, rec)
        print(f"OK (n={nseq}, len={seq_len})")
        time.sleep(DELAY_BETWEEN_REQUESTS)

    # now merge results back into original dataframe
    # prepare results DataFrame
    res_rows = []
    for bin_uri, rec in results.items():
        res_rows.append({
            "bin_uri": bin_uri,
            "sequence": rec.get("sequence", None) or "",
            "n_sequences": rec.get("n_sequences", 0),
            "seq_length": rec.get("seq_length", ""),
            "notes": rec.get("notes", "")
        })
    res_df = pd.DataFrame(res_rows).drop_duplicates(subset=["bin_uri"]).set_index("bin_uri")

    # merge on bin_uri preserving original rows and columns
    out_df = df.copy()
    # ensure all original columns preserved; add new columns
    out_df["sequence"] = out_df["bin_uri"].map(lambda b: res_df.at[b, "sequence"] if (b in res_df.index) else "")
    out_df["n_sequences"] = out_df["bin_uri"].map(lambda b: int(res_df.at[b, "n_sequences"]) if (b in res_df.index and pd.notna(res_df.at[b,"n_sequences"])) else 0)
    out_df["seq_length"] = out_df["bin_uri"].map(lambda b: int(res_df.at[b, "seq_length"]) if (b in res_df.index and res_df.at[b,"seq_length"] != "") else "")
    out_df["notes"] = out_df["bin_uri"].map(lambda b: res_df.at[b, "notes"] if (b in res_df.index) else "")

    # write final output
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {OUTPUT_CSV} ({len(out_df)} rows). Cache saved to {CACHE_CSV}.")

    # final quick sanity summary
    total_rows = len(out_df)
    with_seq = out_df["sequence"].astype(bool).sum()
    flagged = out_df[out_df["notes"].astype(bool)]
    print(f"{with_seq} / {total_rows} rows have a sequence. {len(flagged)} rows have notes (check 'notes' column).")

if __name__ == "__main__":
    main()
