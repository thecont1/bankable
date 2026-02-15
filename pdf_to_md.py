#!/usr/bin/env python3
"""
PDF to Markdown converter using Marker.
Converts bank/credit card statement PDFs to Markdown format.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


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
        "--no-llm",
        action="store_true",
        help="Disable LLM for faster processing (LLM enabled by default)",
    )
    parser.add_argument(
        "--ollama-model",
        default="minimax-m2.5:cloud",
        help="Ollama model to use (default: minimax-m2.5:cloud)",
    )

    args = parser.parse_args()

    use_llm = not args.no_llm

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
        "--html_tables_in_markdown",
        "--converter_cls",
        "marker.converters.table.TableConverter",
        "--output_dir",
        str(output_dir),
    ]

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
    print(f"Converting: {input_path.name}{llm_status}...")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        while True:
            retcode = process.poll()

            if process.stderr:
                chunk = process.stderr.read(1024)
                if chunk:
                    if args.verbose:
                        print(chunk, file=sys.stderr, end="")
                    elif any(x in chunk for x in ["it/s]", "%", "Downloading"]):
                        print(chunk, file=sys.stderr, end="")

            if retcode is not None:
                break

            time.sleep(0.1)

        stdout, stderr = process.communicate()

        if retcode != 0 and retcode is not None:
            print(f"Error: Marker failed with exit code {retcode}", file=sys.stderr)
            if stdout:
                print(f"stdout: {stdout}", file=sys.stderr)
            if stderr:
                print(f"stderr: {stderr}", file=sys.stderr)
            sys.exit(retcode)

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

        if output_path.exists():
            print(f"Output: {output_path}")
            print("Done.")
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
