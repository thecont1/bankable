# Bankable

A fast CLI tool for extracting data tables from computer-generated financial documents (bank statements, credit card statements, etc.) and preparing them for further processing.

## Features

- **Table-focused extraction** - Optimized for financial documents with complex tables
- **Multiple output formats** - Markdown tables or HTML tables in Markdown
- **LLM enhancement** - Optional Ollama integration for improved table parsing
- **Apple Silicon optimized** - Uses MPS GPU acceleration where supported
- **Real-time progress** - Streaming progress bars with consistent formatting
- **Configurable** - External config file (`MARKER.md`) for persistent settings

## Installation

```bash
uv sync
```

## Quick Start

```bash
# Basic conversion (fastest for computer-generated PDFs)
uv run pdf_to_md.py statement.pdf

# Overwrite existing output
uv run pdf_to_md.py statement.pdf --overwrite
```

## Performance

Processing time on Apple Silicon (M1 Max):

| Configuration | Time | Use Case |
|--------------|------|----------|
| Default (OCR disabled) | ~16s | Computer-generated PDFs |
| With OCR enabled | ~40-45s | Scanned documents |
| With LLM enhancement | ~2-3 min | Complex/ambiguous tables |

## Command Line Options

| Option | Description |
|--------|-------------|
| `input_pdf` | Path to input PDF file (required) |
| `--overwrite` | Overwrite existing output file |
| `--verbose`, `-v` | Show verbose output including full command |
| `--use-llm` | Enable LLM for improved table extraction |
| `--no-llm` | Disable LLM (overrides config) |
| `--html-tables` | Format tables as HTML in markdown |
| `--no-html-tables` | Use Markdown format for tables |
| `--ollama-model MODEL` | Ollama model (default: minimax-m2.5:cloud) |
| `--workers N`, `-w N` | Parallel workers for batch processing |
| `--config PATH` | Path to config file (default: MARKER.md) |

## Configuration

Settings are stored in `MARKER.md`. Edit this file to change defaults:

```markdown
# Active flags
--output_format markdown
--disable_image_extraction
--converter_cls marker.converters.table.TableConverter
--disable_ocr

# Inactive (commented)
# --use_llm
# --html_tables_in_markdown
```

CLI options override config file settings.

## Output

The converted Markdown file is saved alongside the input PDF with a `.md` extension.

## Requirements

- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3/M4) recommended
- marker-pdf
- Ollama (optional, for LLM enhancement)

### Ollama Setup (Optional)

```bash
brew install ollama
ollama serve
ollama pull minimax-m2.5:cloud
```

## Notes

- **OCR** is disabled by default for faster processing of computer-generated PDFs. Enable in `MARKER.md` for scanned documents.
- **GPU**: Uses Apple Silicon MPS where supported. Some models fall back to CPU.
- **Progress bars**: Empty progress bars (0 items) are automatically filtered.

## License

MIT License

## Acknowledgments

- [Marker](https://github.com/datalab-to/marker) - PDF to Markdown conversion
- [Ollama](https://ollama.com) - Local LLM runtime
