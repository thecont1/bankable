# Bankable

End-to-end CLI tool for extracting structured transaction data from bank and credit card statement PDFs into clean CSV files.

**Pipeline:** PDF → Marker → `.md` → Substitutions → LLM → `.csv` → Pandas cleanup

## Features

- **PDF to CSV in one command** — single `pdf_to_csv.py` handles the entire pipeline
- **Plug-and-play LLM** — switch Ollama models via `--model` flag without code changes
- **Batch processing** — processes entire directories of PDFs, merges into one master CSV
- **Smart table extraction** — custom patches fix Marker's handling of collapsed financial tables
- **Password support** — handles encrypted PDFs via `--password` flag
- **Text substitutions** — user-configurable find/replace rules (`SUBSTITUTIONS.md`) clean up OCR artifacts before LLM processing
- **Deterministic post-processing** — Credit/Debit classification uses original Markdown markers, not LLM guesses
- **Apple Silicon optimized** — MPS GPU acceleration where supported

## Installation

```bash
uv sync
```

### Ollama Setup

```bash
brew install ollama
ollama serve
ollama pull llama3.1:8b
```

## Quick Start

```bash
# Single file → produces statement.md + statement.csv
uv run pdf_to_csv.py statement.pdf

# Batch directory → produces per-file .md files + all_transactions.csv
uv run pdf_to_csv.py /path/to/statements/

# Encrypted PDFs
uv run pdf_to_csv.py /path/to/statements/ --password "your_password"

# Use a different model
uv run pdf_to_csv.py statement.pdf --model llama3.2:3b

# Overwrite existing outputs
uv run pdf_to_csv.py /path/to/statements/ --overwrite --password "your_password"
```

## Output

### Single File
- `statement.md` — Markdown with extracted tables
- `statement.csv` — cleaned transaction data

### Directory
- `<name>.md` for each PDF — individual Markdown files
- `all_transactions.csv` — merged, deduplicated, sorted master CSV

### CSV Format

```
Date,Time,Description,Amount,Type
13/03/2025,21:13:01,SWIGGY,937.00,D
02/04/2025,12:29:45,NETBANKING TRANSFER,42545.00,C
```

- **Type**: `C` (Credit) or `D` (Debit)
- Sorted by date and time
- Deduplicated across statements

## Command Line Options

| Option | Description |
|--------|-------------|
| `input` | PDF file or directory of PDFs (required) |
| `--model`, `-m` | Ollama model for extraction (default: `llama3.1:8b`) |
| `--password`, `-p` | Password for encrypted PDFs |
| `--overwrite` | Overwrite existing output files |
| `--verbose`, `-v` | Verbose output |
| `--ollama-url` | Ollama base URL (default: `http://localhost:11434`) |
| `--config` | Marker config file (default: `MARKER.md`) |
| `--substitutions` | Substitutions config (default: `SUBSTITUTIONS.md`) |
| `--use-llm` | Enable Marker's built-in LLM mode |
| `--no-llm` | Disable Marker's built-in LLM mode |
| `--html-tables` | HTML tables in Markdown output |
| `--workers`, `-w` | Batch size for parallel processing |

## Configuration

### `MARKER.md`
Marker conversion settings (output format, table processing, DPI, etc.). CLI flags override these.

### `SUBSTITUTIONS.md`
User-maintained text replacements applied before LLM processing:

```
"<br>" → " "
"SMARTBUYBANGALORE" → "SMARTBUY BANGALORE"
"PayU*Swiggy Limited" → "SWIGGY"
```

## Project Structure

```
bankable/
├── pdf_to_csv.py          # Main program — end-to-end PDF → CSV
├── marker_patched.py      # Marker subprocess wrapper (applies patches)
├── table_split_patch.py   # Fixes collapsed transaction rows in tables
├── pdfprovider_patch.py   # Adds password support to Marker's PDF provider
├── MARKER.md              # Marker configuration
├── SUBSTITUTIONS.md       # Text substitution rules
└── pyproject.toml         # Dependencies
```

## Requirements

- Python 3.13+
- macOS with Apple Silicon recommended
- [Ollama](https://ollama.com) with a model pulled (e.g. `llama3.1:8b`)

## Notes

- **Existing files** are skipped by default. Use `--overwrite` to reprocess.
- **GPU**: Apple Silicon MPS where supported; some models fall back to CPU.
- **LLM accuracy**: `llama3.1:8b` is recommended over `llama3.2:3b` for fewer digit transposition errors.
- **Credit detection** uses `Cr` markers from the original Markdown tables, not LLM output.

## License

MIT License

## Acknowledgments

- [Marker](https://github.com/datalab-to/marker) — PDF to Markdown conversion
- [Ollama](https://ollama.com) — local LLM runtime
