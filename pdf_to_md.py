#!/usr/bin/env python3
"""
PDF to Markdown converter using Marker.
Converts bank/credit card statement PDFs to Markdown format.
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
from pathlib import Path

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


def align_progress_labels(text):
    """Normalize progress bar to fixed width format for alignment."""
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


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown using Marker")
    parser.add_argument("input_pdf", type=Path, help="Path to input PDF file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for improved table extraction (requires Ollama, slower but better results)",
    )
    parser.add_argument(
        "--html-tables",
        action="store_true",
        default=False,
        help="Format tables as HTML in markdown",
    )
    parser.add_argument(
        "--no-html-tables",
        action="store_false",
        dest="html_tables",
        help="Use Markdown format for tables instead of HTML (default)",
    )
    parser.add_argument(
        "--ollama-model",
        default="minimax-m2.5:cloud",
        help="Ollama model to use (default: minimax-m2.5:cloud)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )

    args = parser.parse_args()

    use_llm = args.use_llm
    html_tables = args.html_tables

    input_path = args.input_pdf.resolve()

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not input_path.is_file():
        print(f"Error: Not a file: {input_path}", file=sys.stderr)
        sys.exit(1)

    if input_path.suffix.lower() != ".pdf":
        print(f"Error: Not a PDF file: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = input_path.with_suffix(".md")

    if output_path.exists() and not args.overwrite:
        print(f"Error: Output file already exists: {output_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "marker_single",
        str(input_path),
        "--output_format",
        "markdown",
        "--disable_image_extraction",
        "--converter_cls",
        "marker.converters.table.TableConverter",
        "--output_dir",
        str(output_dir),
    ]

    if html_tables:
        cmd.append("--html_tables_in_markdown")

    if args.workers > 1:
        cmd.extend(["--detection_batch_size", str(args.workers)])
        cmd.extend(["--recognition_batch_size", str(args.workers)])
        cmd.extend(["--layout_batch_size", str(args.workers)])

    if use_llm:
        cmd.extend(
            [
                "--use_llm",
                "--llm_service",
                "marker.services.ollama.OllamaService",
                "--ollama_base_url",
                "http://localhost:11434",
                "--ollama_model",
                args.ollama_model,
            ]
        )

    if args.verbose:
        print(f"Running: {' '.join(cmd)}", file=sys.stderr)

    llm_status = " with LLM" if use_llm else ""
    table_status = " (HTML tables)" if html_tables else " (Markdown tables)"
    workers_status = f" ({args.workers} workers)" if args.workers > 1 else ""

    print(
        f"Converting: {input_path.name}{llm_status}{table_status}{workers_status}...",
        file=sys.stderr,
    )
    print(
        f"Using Apple Silicon GPU (MPS) where supported",
        file=sys.stderr,
    )

    # Set environment for unbuffered output and MPS optimization
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    try:
        returncode, stdout = run_with_pty(cmd, env)

        if returncode != 0:
            print(
                f"\nError: Marker failed with exit code {returncode}",
                file=sys.stderr,
            )
            if stdout:
                print(stdout, file=sys.stderr)
            sys.exit(returncode)

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

        # Clean up the directory Marker created
        if expected_dir.exists() and expected_dir.is_dir():
            shutil.rmtree(expected_dir)

        if output_path.exists():
            print(f"\nOutput: {output_path}", file=sys.stderr)
            print("Done.", file=sys.stderr)
            sys.exit(0)
        else:
            print("Error: Marker did not produce output file", file=sys.stderr)
            sys.exit(1)

    except FileNotFoundError:
        print("Error: marker_single command not found.", file=sys.stderr)
        print("Make sure marker-pdf is installed and in PATH.", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
