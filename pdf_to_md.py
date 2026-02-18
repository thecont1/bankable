#!/usr/bin/env python3
"""
PDF to Markdown converter using Marker.
Converts bank/credit card statement PDFs to Markdown format.
Reads configuration from MARKER.md config file.
"""

import argparse
import fcntl
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
from pathlib import Path

# Apply patch for PdfProvider password support
from pdfprovider_patch import *

# Width for progress bar labels (longest is "Running OCR Error Detection")
LABEL_WIDTH = 28
# Fixed width for the counter part (e.g., "54/54")
COUNTER_WIDTH = 7
# Fixed width for the rate/time part
STATS_WIDTH = 28

# Pattern to match tqdm progress bar output
# Groups: 1=prefix, 2=label, 3=percent, 4=bar, 5=counter, 6=stats
PROGRESS_PATTERN = re.compile(
    r'([\r\n]|^)([A-Za-z][A-Za-z ]+):\s*(\d+%\|)([█▏▎▍▌▋▊▉ ]+\|)\s*(\d+/\d+)\s*(\[.+\])'
)

# Pattern to match empty progress bars (0 items) - includes surrounding whitespace/newlines
EMPTY_PROGRESS_PATTERN = re.compile(
    r'\r?\n?[A-Za-z][A-Za-z ]+:\s*0it \[\d+:\d+, \?it/s\]\r?\n?'
)


def load_marker_config(config_path: Path) -> list:
    """Load Marker flags from config file, skipping comments."""
    flags = []

    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}", file=sys.stderr)
        return flags

    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and pure comment lines
            if not line or line.startswith("#"):
                continue

            # Check if line starts with -- (active flag)
            if line.startswith("--"):
                # Handle flag with value on same line
                parts = line.split(None, 1)  # Split on whitespace, max 2 parts
                flag = parts[0]
                flags.append(flag)
                if len(parts) > 1:
                    flags.append(parts[1])

    return flags


def align_progress_labels(text):
    """Normalize progress bar to fixed width format for alignment."""
    # Filter out empty progress bars (0 items)
    text = EMPTY_PROGRESS_PATTERN.sub('', text)
    # Clean up any resulting double newlines
    text = re.sub(r'\n\n+', '\n', text)
    
    def replace_label(match):
        prefix = match.group(1)
        label = match.group(2).ljust(LABEL_WIDTH)
        percent = match.group(3).rjust(5)  # "100%|" is 5 chars
        bar = match.group(4)
        counter = match.group(5).rjust(COUNTER_WIDTH)
        stats = match.group(6).ljust(STATS_WIDTH)
        
        # Calculate bar width to fill remaining space (target 100 total width)
        # Format: label(28) + ": " + percent(5) + bar + counter(7) + " " + stats(28)
        used = LABEL_WIDTH + 2 + 5 + COUNTER_WIDTH + 1 + STATS_WIDTH
        bar_width = 100 - used  # Should be ~29 chars for the bar
        
        # Normalize bar to fixed width
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
    
    # Set terminal size so tqdm knows the width (fixed at 100 for consistent output)
    cols = 100
    rows = 24
    winsize = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
    
    # Ensure TERM is set for Unicode support
    env = env.copy()
    if "TERM" not in env:
        env["TERM"] = "xterm-256color"
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=slave_fd,
        env=env,
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
                # Process ended, drain remaining output
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


def process_single_pdf(input_path: Path, args, config_flags):
    """Process a single PDF file and return the output path."""
    output_path = input_path.with_suffix(".md")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command from config
    cmd = ["marker_single", str(input_path)]

    # Add password if provided in .env file
    pdf_password = os.getenv("PDF_PASSWORD")
    if pdf_password:
        cmd.extend(["--password", pdf_password])

    # Add config flags (skip output_dir, we'll add it manually)
    for flag in config_flags:
        if flag == "--output_dir":
            continue
        cmd.append(flag)

    # Add output_dir
    cmd.extend(["--output_dir", str(output_dir)])

    # CLI overrides for LLM
    use_llm = args.use_llm
    if not args.use_llm and not args.no_llm:
        use_llm = "--use_llm" in config_flags

    if use_llm:
        if "--use_llm" not in cmd:
            cmd.append("--use_llm")
        # Ensure LLM service is set
        if "--llm_service" not in cmd:
            cmd.extend(["--llm_service", "marker.services.ollama.OllamaService"])
        if "--ollama_base_url" not in cmd:
            cmd.extend(["--ollama_base_url", "http://localhost:11434"])
    elif args.no_llm and "--use_llm" in cmd:
        # Remove --use_llm if explicitly disabled
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

    # CLI override for Ollama model
    if args.ollama_model:
        # Find and replace or add
        if "--ollama_model" in cmd:
            idx = cmd.index("--ollama_model")
            cmd[idx + 1] = args.ollama_model
        else:
            cmd.extend(["--ollama_model", args.ollama_model])

    # CLI override for workers (batch sizes)
    if args.workers and args.workers > 1:
        # Adjust batch sizes for parallel processing
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
    workers_status = (
        f" ({args.workers} workers)" if args.workers and args.workers > 1 else ""
    )

    print(
        f"Converting: {input_path.name}{llm_status}{table_status}{workers_status}...",
        file=sys.stderr,
    )
    print(f"Using Apple Silicon GPU (MPS) where supported", file=sys.stderr)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Suppress unnecessary warnings
    env["PYTHONWARNINGS"] = "ignore"
    
    # Set password if provided
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
            print("Done.", file=sys.stderr)
            return output_path
        else:
            raise FileNotFoundError("Marker did not produce output file")

    except FileNotFoundError as e:
        if "marker_single" in str(e):
            raise FileNotFoundError("marker_single command not found. Make sure marker-pdf is installed and in PATH.")
        raise
    except Exception as e:
        raise

def process_directory(input_path: Path, args, config_flags):
    """Process all PDFs in a directory and merge them into a single markdown file."""
    print(f"Processing directory: {input_path}", file=sys.stderr)
    print(f"Searching for PDF files...", file=sys.stderr)

    # Case-insensitive PDF file search
    pdf_files = []
    for file in input_path.iterdir():
        if file.is_file() and file.suffix.lower() == ".pdf":
            pdf_files.append(file)
    
    if not pdf_files:
        print(f"Error: No PDF files found in directory: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Sort files by filename
    pdf_files = sorted(pdf_files, key=lambda x: x.name)

    print(f"Found {len(pdf_files)} PDF files:", file=sys.stderr)
    for pdf in pdf_files:
        print(f"  - {pdf.name}", file=sys.stderr)

    merged_content = []
    successful_files = []
    skipped_files = []
    failed_files = []

    start_time = time.time()

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\nProcessing {i}/{len(pdf_files)}: {pdf_file.name}", file=sys.stderr)
        try:
            # Check if markdown file already exists
            md_file = input_path / f"{pdf_file.stem}.md"
            if md_file.exists() and not args.overwrite:
                print(f"Skipping {pdf_file.name}: Output file already exists", file=sys.stderr)
                skipped_files.append(pdf_file.name)
                # Still include existing file in merged content
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                merged_content.append(f"\n---\n")
                merged_content.append(f"# {pdf_file.stem}\n")
                merged_content.append(f"**Source File:** {pdf_file.name}\n")
                merged_content.append(f"---\n")
                merged_content.append(content)
                merged_content.append(f"\n")
            else:
                # Process new file
                output_path = process_single_pdf(pdf_file, args, config_flags)
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                merged_content.append(f"\n---\n")
                merged_content.append(f"# {pdf_file.stem}\n")
                merged_content.append(f"**Source File:** {pdf_file.name}\n")
                merged_content.append(f"---\n")
                merged_content.append(content)
                merged_content.append(f"\n")
                successful_files.append(pdf_file.name)
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}", file=sys.stderr)
            failed_files.append(pdf_file.name)

    total_time = time.time() - start_time

    # Create merged output with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    merged_output_path = input_path / f"merged_statements_{timestamp}.md"
    with open(merged_output_path, 'w', encoding='utf-8') as f:
        f.write("".join(merged_content))

    # Also create a symlink to the latest version
    latest_symlink = input_path / "merged_statements_latest.md"
    if latest_symlink.exists():
        if latest_symlink.is_symlink():
            latest_symlink.unlink()
        elif latest_symlink.is_file():
            latest_symlink.unlink()
        elif latest_symlink.is_dir():
            shutil.rmtree(latest_symlink)
    
    try:
        # Create symlink relative to the target
        latest_symlink.symlink_to(merged_output_path.name)
    except Exception as e:
        print(f"Warning: Could not create symlink to latest version: {e}", file=sys.stderr)
        # Fallback to absolute path if relative fails
        try:
            latest_symlink.symlink_to(merged_output_path)
        except Exception as e2:
            print(f"Warning: Could not create absolute symlink: {e2}", file=sys.stderr)

    print(f"\nSuccessfully processed {len(successful_files)} files:", file=sys.stderr)
    for file in successful_files:
        print(f"  - {file}", file=sys.stderr)

    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files (already processed):", file=sys.stderr)
        for file in skipped_files:
            print(f"  - {file}", file=sys.stderr)

    if failed_files:
        print(f"\nFailed to process {len(failed_files)} files:", file=sys.stderr)
        for file in failed_files:
            print(f"  - {file}", file=sys.stderr)

    print(f"\nMerged output: {merged_output_path}", file=sys.stderr)
    print(f"Total time: {total_time:.1f} seconds", file=sys.stderr)
    print("Done.", file=sys.stderr)
    return merged_output_path

def main():
    # Find config file - look in same directory as script
    script_dir = Path(__file__).parent.resolve()
    config_path = script_dir / "MARKER.md"

    # Parse arguments (only CLI-specific args, not Marker flags)
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown using Marker")
    parser.add_argument("input_pdf", type=Path, help="Path to input PDF file or directory")
    parser.add_argument(
        "--password", "-p", 
        default=None, 
        help="Password for password-protected PDF files"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )
    parser.add_argument(
        "--use-llm", action="store_true", help="Enable LLM (overrides config)"
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="Disable LLM (overrides config)"
    )
    parser.add_argument(
        "--html-tables",
        action="store_true",
        help="Enable HTML tables (overrides config)",
    )
    parser.add_argument(
        "--no-html-tables",
        action="store_true",
        help="Disable HTML tables (overrides config)",
    )
    parser.add_argument(
        "--ollama-model", default=None, help="Ollama model (overrides config)"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of workers (overrides config)",
    )
    parser.add_argument(
        "--config", type=Path, default=config_path, help="Path to config file"
    )

    args = parser.parse_args()

    # Load config from MARKER.md
    config_flags = load_marker_config(args.config)

    # Determine effective flags (CLI overrides config)
    use_llm = args.use_llm
    html_tables = args.html_tables

    # Check config for defaults if not set via CLI
    if not args.use_llm and not args.no_llm:
        use_llm = "--use_llm" in config_flags

    if not args.html_tables and not args.no_html_tables:
        html_tables = "--html_tables_in_markdown" in config_flags

    input_path = args.input_pdf.resolve()

    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            print(f"Error: Not a PDF file: {input_path}", file=sys.stderr)
            sys.exit(1)
        
        # Process single PDF file
        print(f"Processing single file: {input_path}", file=sys.stderr)
        try:
            output_path = process_single_pdf(input_path, args, config_flags)
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif input_path.is_dir():
        # Process directory of PDF files
        try:
            output_path = process_directory(input_path, args, config_flags)
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: Input must be a file or directory: {input_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
