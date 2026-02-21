#!/usr/bin/env python3
"""
PDF to CSV transaction extractor.
Converts bank/credit card statement PDFs to structured CSV via Marker + LLM.

Pipeline: PDF → Marker → .md → Substitutions → LLM → .csv → Pandas cleanup

Usage:
    uv run pdf_to_csv.py <input_pdf_or_dir> [--model MODEL] [--password PASS] [--overwrite] [-v]
"""

import argparse
import csv
import fcntl
import io
import json
import os
import pty
import re
import select
import shutil
import struct
import subprocess
import sys
import termios
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


# ===================================================================
# Marker configuration
# ===================================================================

def load_marker_config(config_path: Path) -> list:
    """Load Marker flags from config file, skipping comments."""
    flags = []
    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}", file=sys.stderr)
        return flags
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("--"):
                parts = line.split(None, 1)
                flag = parts[0]
                flags.append(flag)
                if len(parts) > 1:
                    flags.append(parts[1])
    return flags


# ===================================================================
# Progress bar alignment + PTY runner
# ===================================================================

LABEL_WIDTH = 28
COUNTER_WIDTH = 7
STATS_WIDTH = 28

PROGRESS_PATTERN = re.compile(
    r'([\r\n]|^)([A-Za-z][A-Za-z ]+):\s*(\d+%\|)([█▏▎▍▌▋▊▉ ]+\|)\s*(\d+/\d+)\s*(\[.+\])'
)
EMPTY_PROGRESS_PATTERN = re.compile(
    r'\r?\n?[A-Za-z][A-Za-z ]+:\s*0it \[\d+:\d+, \?it/s\]\r?\n?'
)


def align_progress_labels(text):
    """Normalize progress bar to fixed width format for alignment."""
    text = EMPTY_PROGRESS_PATTERN.sub('', text)
    text = re.sub(r'\n\n+', '\n', text)

    def replace_label(match):
        prefix = match.group(1)
        label = match.group(2).ljust(LABEL_WIDTH)
        percent = match.group(3).rjust(5)
        bar = match.group(4)
        counter = match.group(5).rjust(COUNTER_WIDTH)
        stats = match.group(6).ljust(STATS_WIDTH)
        used = LABEL_WIDTH + 2 + 5 + COUNTER_WIDTH + 1 + STATS_WIDTH
        bar_width = 100 - used
        bar_content = bar.strip('|')
        filled = bar_content.count('█') + sum(0.5 for c in bar_content if c in '▏▎▍▌▋▊▉')
        fill_ratio = filled / max(len(bar_content), 1)
        new_bar_inner = '█' * int(fill_ratio * (bar_width - 2)) + ' ' * (bar_width - 2 - int(fill_ratio * (bar_width - 2)))
        new_bar = f"{new_bar_inner}|"
        return f"{prefix}{label}: {percent}{new_bar} {counter} {stats}"

    return PROGRESS_PATTERN.sub(replace_label, text)


def run_with_pty(cmd, env):
    """Run command with PTY to get unbuffered output from tqdm progress bars."""
    master_fd, slave_fd = pty.openpty()
    cols, rows = 100, 24
    winsize = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
    env = env.copy()
    if "TERM" not in env:
        env["TERM"] = "xterm-256color"

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=slave_fd, env=env,
    )
    os.close(slave_fd)

    stdout_data = b""
    try:
        while True:
            ready, _, _ = select.select([master_fd, process.stdout], [], [], 0.1)
            if master_fd in ready:
                try:
                    data = os.read(master_fd, 1024)
                    if data:
                        text = data.decode("utf-8", errors="replace")
                        text = align_progress_labels(text)
                        sys.stderr.write(text)
                        sys.stderr.flush()
                except OSError:
                    pass
            if process.stdout in ready:
                chunk = process.stdout.read(1024)
                if chunk:
                    stdout_data += chunk
            if process.poll() is not None:
                while True:
                    try:
                        data = os.read(master_fd, 1024)
                        if not data:
                            break
                        text = data.decode("utf-8", errors="replace")
                        text = align_progress_labels(text)
                        sys.stderr.write(text)
                        sys.stderr.flush()
                    except OSError:
                        break
                break
    finally:
        os.close(master_fd)
        remaining = process.stdout.read()
        if remaining:
            stdout_data += remaining

    return process.returncode, stdout_data.decode("utf-8", errors="replace")


# ===================================================================
# PDF → Markdown (via Marker subprocess)
# ===================================================================

def process_single_pdf(input_path: Path, args, config_flags) -> Path:
    """Process a single PDF file via Marker and return the .md output path."""
    output_path = input_path.with_suffix(".md")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command — use patched wrapper for table row splitting fix
    script_dir = Path(__file__).parent.resolve()
    patched_script = script_dir / "marker_patched.py"
    cmd = [sys.executable, str(patched_script), str(input_path)]

    # Add config flags (skip output_dir, we'll add it manually)
    for flag in config_flags:
        if flag == "--output_dir":
            continue
        cmd.append(flag)

    cmd.extend(["--output_dir", str(output_dir)])

    # CLI overrides for Marker's LLM
    use_llm = args.use_llm
    if not args.use_llm and not args.no_llm:
        use_llm = "--use_llm" in config_flags

    if use_llm:
        if "--use_llm" not in cmd:
            cmd.append("--use_llm")
        if "--llm_service" not in cmd:
            cmd.extend(["--llm_service", "marker.services.ollama.OllamaService"])
        if "--ollama_base_url" not in cmd:
            cmd.extend(["--ollama_base_url", args.ollama_url])
    elif args.no_llm and "--use_llm" in cmd:
        idx = cmd.index("--use_llm")
        cmd.pop(idx)

    # CLI override for HTML tables
    html_tables = args.html_tables
    if not args.html_tables and not args.no_html_tables:
        html_tables = "--html_tables_in_markdown" in config_flags

    if html_tables:
        if "--html_tables_in_markdown" not in cmd:
            cmd.append("--html_tables_in_markdown")
    elif args.no_html_tables and "--html_tables_in_markdown" in cmd:
        idx = cmd.index("--html_tables_in_markdown")
        cmd.pop(idx)

    # CLI override for Marker's Ollama model
    if args.marker_model:
        if "--ollama_model" in cmd:
            idx = cmd.index("--ollama_model")
            cmd[idx + 1] = args.marker_model
        else:
            cmd.extend(["--ollama_model", args.marker_model])

    # CLI override for workers (batch sizes)
    if args.workers and args.workers > 1:
        for batch_type in ["layout", "detection", "recognition"]:
            flag = f"--{batch_type}_batch_size"
            if flag in cmd:
                idx = cmd.index(flag)
                cmd[idx + 1] = str(args.workers)
            else:
                cmd.extend([flag, str(args.workers)])

    if args.verbose:
        print(f"Running: {' '.join(cmd)}", file=sys.stderr)

    llm_status = " with LLM" if use_llm else ""
    table_status = " (HTML tables)" if html_tables else " (Markdown tables)"
    print(f"Converting: {input_path.name}{llm_status}{table_status}...", file=sys.stderr)
    print(f"Using Apple Silicon GPU (MPS) where supported", file=sys.stderr)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    if args.password:
        env["PDF_PASSWORD"] = args.password

    try:
        returncode, stdout = run_with_pty(cmd, env)

        if returncode != 0:
            error_msg = f"Marker failed with exit code {returncode}"
            if stdout:
                error_msg += f"\n{stdout}"
            raise subprocess.CalledProcessError(returncode, ' '.join(cmd), error_msg)

        expected_dir = output_dir / input_path.stem
        possible_outputs = [
            expected_dir / f"{input_path.stem}.md",
            expected_dir.with_suffix(".md"),
            expected_dir,
        ]

        final_output = None
        for out_path in possible_outputs:
            if out_path.is_file() and out_path.stat().st_size > 0:
                final_output = out_path
                break
            elif out_path.is_dir():
                md_files = list(out_path.glob("*.md"))
                for md_file in md_files:
                    if md_file.stat().st_size > 0:
                        final_output = md_file
                        break
                if final_output:
                    break

        if final_output and final_output != output_path:
            if output_path.exists():
                if output_path.is_dir():
                    shutil.rmtree(output_path)
                else:
                    output_path.unlink()
            shutil.copy2(final_output, output_path)

        if expected_dir.exists() and expected_dir.is_dir():
            shutil.rmtree(expected_dir)

        if output_path.exists():
            print(f"\nOutput: {output_path}", file=sys.stderr)
            return output_path
        else:
            raise FileNotFoundError("Marker did not produce output file")

    except FileNotFoundError as e:
        if "marker_single" in str(e):
            raise FileNotFoundError(
                "marker_single command not found. Make sure marker-pdf is installed."
            )
        raise


# ===================================================================
# Substitutions
# ===================================================================

def load_substitutions(config_path: Path) -> list:
    """Load find→replace pairs from SUBSTITUTIONS.md."""
    pairs = []
    if not config_path.exists():
        return pairs
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r'"([^"]*?)"\s*→\s*"([^"]*?)"', line)
            if match:
                pairs.append((match.group(1), match.group(2)))
    return pairs


def apply_substitutions(text: str, substitutions: list) -> str:
    """Apply all find→replace substitutions to text."""
    for find, replace in substitutions:
        text = text.replace(find, replace)
    return text


# ===================================================================
# LLM extraction (Ollama)
# ===================================================================

EXTRACTION_PROMPT = """You are a financial data extraction assistant. Extract ONLY the transaction rows from this bank/credit card statement markdown and output them as CSV.

RULES:
1. Output ONLY CSV lines — no headers, no explanations, no markdown fences.
2. Each line: DateTime,Description,Amount,Type
3. DateTime: keep exactly as shown (DD/MM/YYYY or DD/MM/YYYY HH:MM:SS).
4. Description: the merchant/transaction description. Keep it as-is from the table.
5. Amount: numeric value only — remove commas (e.g. 10413.00 not 10,413.00). No currency symbols.
6. Type: Look at the Amount column in the table. If the amount ends with "Cr" it is "Credit". Otherwise it is "Debit".
7. SKIP rows that are empty, headers, summaries, reward points, GST entries, or non-transaction data.
8. SKIP rows where the date or amount is missing or blank.
9. Include EVERY transaction row — do not skip any.

STATEMENT:
{md_content}

CSV OUTPUT:"""


def call_ollama(prompt: str, ollama_url: str, model: str, timeout: int = 180) -> str:
    """Call Ollama API and return the response text."""
    url = f"{ollama_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 8192,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "")
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot reach Ollama at {ollama_url}: {e}")
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}")


# ===================================================================
# LLM response parsing + deterministic post-processing
# ===================================================================

_DATE_RE = re.compile(r"\d{2}/\d{2}/\d{4}")
_AMOUNT_RE = re.compile(r"^\d+(?:\.\d+)?$")
_MD_TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")


def parse_llm_response(response: str) -> list:
    """Parse LLM CSV response into list of dicts."""
    rows = []
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("datetime") or line.startswith("```") or line.startswith("|"):
            continue
        if line.startswith("#") or line.startswith("---"):
            continue

        try:
            reader = csv.reader(io.StringIO(line))
            fields = next(reader)
        except (csv.Error, StopIteration):
            continue

        if len(fields) < 3:
            continue

        dt = fields[0].strip()
        desc = fields[1].strip()
        amount_raw = fields[2].strip()
        txn_type = fields[3].strip() if len(fields) > 3 else ""

        if not _DATE_RE.search(dt):
            continue

        amount_clean = re.sub(r"[₹,\s]", "", amount_raw)
        if amount_clean.lower().endswith("cr"):
            amount_clean = amount_clean[:-2].strip()
            if not txn_type:
                txn_type = "Credit"

        if not _AMOUNT_RE.match(amount_clean):
            continue

        if not desc:
            continue

        txn_type_lower = txn_type.lower()
        if txn_type_lower in ("credit", "cr"):
            txn_type = "Credit"
        elif txn_type_lower in ("debit", "dr", ""):
            txn_type = "Debit"

        rows.append({
            "DateTime": dt,
            "Description": desc,
            "Amount": amount_clean,
            "Type": txn_type,
        })

    return rows


def _build_credit_lookup(md_content: str) -> set:
    """
    Parse .md to build a set of (date, amount) pairs where amount had "Cr" suffix.
    Used to deterministically fix Credit/Debit classification.
    """
    credits = set()
    for line in md_content.splitlines():
        m = _MD_TABLE_ROW.match(line.strip())
        if not m:
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if len(cells) < 2:
            continue
        for cell in cells:
            if "Cr" in cell and re.search(r"[\d,.]+\s*Cr", cell):
                amt_match = re.search(r"([\d,.]+)\s*Cr", cell)
                if not amt_match:
                    continue
                amount_str = amt_match.group(1).replace(",", "")
                date_cell = cells[0].strip()
                date_match = _DATE_RE.search(date_cell)
                if date_match:
                    credits.add((date_match.group(), amount_str))
    return credits


def _fix_credit_debit(rows: list, credit_lookup: set) -> list:
    """Deterministically fix Type using Cr markers from original .md."""
    for row in rows:
        date_match = _DATE_RE.search(row["DateTime"])
        if not date_match:
            continue
        if (date_match.group(), row["Amount"]) in credit_lookup:
            row["Type"] = "Credit"
        else:
            row["Type"] = "Debit"
    return rows


def _deduplicate(rows: list) -> list:
    """Remove duplicate rows, keeping first occurrence."""
    seen = set()
    result = []
    for row in rows:
        key = (row["DateTime"], row["Amount"], row["Type"])
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


# ===================================================================
# Pandas cleanup
# ===================================================================

def clean_csv_with_pandas(csv_path: Path, verbose: bool = False) -> Path:
    """
    Final data cleaning pass using pandas:
    - Split DateTime → Date + Time
    - Shorten Type: Credit→C, Debit→D
    - Sort by Date, Time
    - Column order: Date, Time, Description, Amount, Type
    """
    df = pd.read_csv(csv_path, dtype=str)

    if df.empty:
        return csv_path

    # Split DateTime into Date and Time
    dt_parts = df["DateTime"].str.split(r"\s+", n=1, expand=True)
    df["Date"] = dt_parts[0]
    df["Time"] = dt_parts[1].fillna("") if dt_parts.shape[1] > 1 else ""

    # Shorten Type
    df["Type"] = df["Type"].map({"Credit": "C", "Debit": "D"}).fillna("D")

    # Ensure Amount is numeric for sorting consistency
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)

    # Sort by date then time
    df["_sort_dt"] = pd.to_datetime(
        df["Date"] + " " + df["Time"].replace("", "00:00:00"),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.sort_values("_sort_dt").drop(columns=["_sort_dt", "DateTime"])

    # Format Amount back to 2 decimal places
    df["Amount"] = df["Amount"].map(lambda x: f"{x:.2f}")

    # Final column order
    df = df[["Date", "Time", "Description", "Amount", "Type"]]

    df.to_csv(csv_path, index=False)

    credits = (df["Type"] == "C").sum()
    debits = (df["Type"] == "D").sum()
    if verbose:
        print(f"  Cleaned CSV: {len(df)} rows ({debits}D, {credits}C)", file=sys.stderr)

    return csv_path


# ===================================================================
# Transaction extraction: .md → LLM → .csv
# ===================================================================

def extract_transactions(
    md_content_raw: str,
    csv_path: Path,
    substitutions: list,
    ollama_url: str,
    model: str,
    verbose: bool = False,
) -> Path:
    """
    Core extraction pipeline: raw .md content → LLM → cleaned .csv.
    Accepts raw (pre-substitution) .md content so credit lookup is accurate.
    Returns csv_path on success, None on failure.
    """
    # Build credit lookup from raw .md BEFORE substitutions
    credit_lookup = _build_credit_lookup(md_content_raw)
    if verbose:
        print(f"  Found {len(credit_lookup)} credit markers in .md", file=sys.stderr)

    # Apply substitutions for cleaner LLM input
    md_content = apply_substitutions(md_content_raw, substitutions)

    # Build prompt and call LLM
    prompt = EXTRACTION_PROMPT.format(md_content=md_content)

    if verbose:
        print(f"  Calling LLM ({model}) for transaction extraction...", file=sys.stderr)

    response = call_ollama(prompt, ollama_url, model)

    if verbose:
        print(f"  LLM response: {len(response)} chars", file=sys.stderr)

    # Parse LLM output
    rows = parse_llm_response(response)

    if not rows:
        print(f"  ⚠ LLM returned no valid transaction rows", file=sys.stderr)
        if verbose:
            print(f"  Raw LLM response:\n{response[:500]}", file=sys.stderr)
        return None

    # Deterministic post-processing
    rows = _fix_credit_debit(rows, credit_lookup)
    rows = _deduplicate(rows)

    # Write intermediate CSV (DateTime format, before pandas cleanup)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["DateTime", "Description", "Amount", "Type"])
        writer.writeheader()
        writer.writerows(rows)

    credits = sum(1 for r in rows if r["Type"] == "Credit")
    debits = sum(1 for r in rows if r["Type"] == "Debit")
    print(
        f"  Extracted {len(rows)} transactions ({debits} debits, {credits} credits)",
        file=sys.stderr,
    )

    # Pandas cleanup: split Date/Time, shorten Type, sort
    clean_csv_with_pandas(csv_path, verbose=verbose)

    return csv_path


# ===================================================================
# Single file mode
# ===================================================================

def run_single(input_path: Path, args, config_flags, substitutions):
    """Process one PDF → .md + .csv"""
    print(f"Processing single file: {input_path}", file=sys.stderr)

    # Step 1: PDF → .md
    md_path = process_single_pdf(input_path, args, config_flags)

    # Step 2: .md → .csv via LLM
    csv_path = input_path.with_suffix(".csv")
    md_raw = md_path.read_text(encoding="utf-8")

    print(f"\nExtracting transactions → {csv_path.name}", file=sys.stderr)
    result = extract_transactions(
        md_raw, csv_path, substitutions,
        ollama_url=args.ollama_url, model=args.model, verbose=args.verbose,
    )
    if result:
        print(f"CSV output: {result}", file=sys.stderr)
    else:
        print("⚠ CSV extraction failed — .md file still available", file=sys.stderr)


# ===================================================================
# Directory mode
# ===================================================================

def _merge_csvs_pandas(csv_paths: list, output_path: Path, verbose: bool = False) -> Path:
    """Merge per-file CSVs into a single master CSV, deduplicate and sort."""
    frames = []
    for p in csv_paths:
        if p and p.exists():
            frames.append(pd.read_csv(p, dtype=str))
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["Date", "Time", "Description", "Amount", "Type"])

    # Sort by date then time
    df["_sort_dt"] = pd.to_datetime(
        df["Date"] + " " + df["Time"].replace("", "00:00:00"),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.sort_values("_sort_dt").drop(columns=["_sort_dt"])
    df.to_csv(output_path, index=False)

    credits = (df["Type"] == "C").sum()
    debits = (df["Type"] == "D").sum()
    print(
        f"  Merged {len(df)} unique transactions ({debits}D, {credits}C) → {output_path.name}",
        file=sys.stderr,
    )
    return output_path


def run_directory(input_path: Path, args, config_flags, substitutions):
    """
    Process all PDFs in a directory:
    1. Each PDF → individual .md file
    2. Each .md → per-file LLM extraction → per-file .csv
    3. Merge all per-file CSVs → master all_transactions.csv
    """
    print(f"Processing directory: {input_path}", file=sys.stderr)
    print(f"Searching for PDF files...", file=sys.stderr)

    pdf_files = sorted(
        [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"],
        key=lambda x: x.name,
    )
    if not pdf_files:
        print(f"Error: No PDF files found in: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF files:", file=sys.stderr)
    for pdf in pdf_files:
        print(f"  - {pdf.name}", file=sys.stderr)

    successful = []
    skipped = []
    failed = []
    csv_paths = []

    start_time = time.time()

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[{i}/{len(pdf_files)}] {pdf_file.name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        md_file = input_path / f"{pdf_file.stem}.md"

        # --- Step 1: PDF → .md ---
        try:
            if md_file.exists() and not args.overwrite:
                print(f"Skipping Marker (already exists): {md_file.name}", file=sys.stderr)
                skipped.append(pdf_file.name)
            else:
                process_single_pdf(pdf_file, args, config_flags)
                successful.append(pdf_file.name)
        except Exception as e:
            print(f"Error converting {pdf_file.name}: {e}", file=sys.stderr)
            failed.append(pdf_file.name)
            continue

        # --- Step 2: .md → per-file .csv via LLM ---
        if md_file.exists():
            per_file_csv = md_file.with_suffix(".csv")
            md_raw = md_file.read_text(encoding="utf-8")
            print(f"  Extracting transactions → {per_file_csv.name}", file=sys.stderr)
            try:
                result = extract_transactions(
                    md_raw, per_file_csv, substitutions,
                    ollama_url=args.ollama_url, model=args.model, verbose=args.verbose,
                )
                if result:
                    csv_paths.append(result)
            except Exception as e:
                print(f"  ⚠ LLM extraction failed: {e}", file=sys.stderr)

    marker_time = time.time() - start_time

    # --- Step 3: Merge per-file CSVs → master CSV ---
    if csv_paths:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Merging {len(csv_paths)} CSVs → all_transactions.csv", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        master_csv = input_path / "all_transactions.csv"
        _merge_csvs_pandas(csv_paths, master_csv, verbose=args.verbose)

        # Clean up per-file CSVs (intermediate artifacts)
        for p in csv_paths:
            if p and p.exists() and p != master_csv:
                p.unlink()
                if args.verbose:
                    print(f"  Removed intermediate: {p.name}", file=sys.stderr)

    total_time = time.time() - start_time

    # --- Summary ---
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Summary", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    if successful:
        print(f"  Converted: {len(successful)} files", file=sys.stderr)
    if skipped:
        print(f"  Skipped:   {len(skipped)} files (already had .md)", file=sys.stderr)
    if failed:
        print(f"  Failed:    {len(failed)} files", file=sys.stderr)
        for f in failed:
            print(f"    - {f}", file=sys.stderr)

    master = input_path / "all_transactions.csv"
    if master.exists():
        print(f"  CSV:       {master}", file=sys.stderr)
    else:
        print(f"  ⚠ No CSV produced", file=sys.stderr)

    print(f"  Total:     {total_time:.1f}s", file=sys.stderr)


# ===================================================================
# CLI
# ===================================================================

def main():
    script_dir = Path(__file__).parent.resolve()
    config_path = script_dir / "MARKER.md"
    subs_path = script_dir / "SUBSTITUTIONS.md"

    parser = argparse.ArgumentParser(
        description="Convert bank/credit card statement PDFs to structured CSV"
    )
    parser.add_argument("input", type=Path, help="PDF file or directory of PDFs")
    parser.add_argument("--password", "-p", default=None, help="PDF password")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # LLM model for transaction extraction (plug-and-play)
    parser.add_argument(
        "--model", "-m", default="llama3.1:8b",
        help="Ollama model for CSV extraction (default: llama3.1:8b)",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )

    # Marker-specific overrides (rarely needed — configured in MARKER.md)
    parser.add_argument("--config", type=Path, default=config_path, help="MARKER.md config path")
    parser.add_argument("--substitutions", type=Path, default=subs_path, help="SUBSTITUTIONS.md path")
    parser.add_argument("--use-llm", action="store_true", help="Enable Marker's LLM mode")
    parser.add_argument("--no-llm", action="store_true", help="Disable Marker's LLM mode")
    parser.add_argument("--html-tables", action="store_true", help="HTML tables in markdown")
    parser.add_argument("--no-html-tables", action="store_true", help="Markdown tables")
    parser.add_argument("--marker-model", default=None, help="Ollama model for Marker's LLM")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Batch size for workers")

    args = parser.parse_args()

    # Set password env var for the Marker subprocess
    if args.password:
        os.environ["PDF_PASSWORD"] = args.password

    # Load configs
    config_flags = load_marker_config(args.config)
    substitutions = load_substitutions(args.substitutions)

    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            print(f"Error: Not a PDF file: {input_path}", file=sys.stderr)
            sys.exit(1)
        try:
            run_single(input_path, args, config_flags, substitutions)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif input_path.is_dir():
        try:
            run_directory(input_path, args, config_flags, substitutions)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: Input must be a file or directory: {input_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
