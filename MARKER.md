# Marker Master Control File
# This file documents all available marker_single flags.
# Edit this file to customize Marker behavior, then update pdf_to_md.py to read from it.
#
# Usage: marker_single INPUT_PDF [flags]

# ==============================================================================
# CORE OPTIONS (Most commonly used)
# ==============================================================================

# --output_format [markdown|json|html|chunks]
#   Output format: markdown, json, html, or chunks
--output_format markdown

# --output_dir PATH
#   Directory to save output files
# (set dynamically by pdf_to_md.py based on input PDF location)

# --page_range TEXT
#   Page range to convert (e.g., "0,5-10,20"). Comment out to convert all pages.
# --page_range

# --disable_image_extraction
#   Skip extracting images from the PDF
--disable_image_extraction

# --disable_ocr
#   Disable OCR processing (use only if PDF has selectable text)
# --disable_ocr

# --force_ocr
#   Force OCR on all pages (for scanned documents)
# --force_ocr

# ==============================================================================
# CONVERTER OPTIONS
# ==============================================================================

# --converter_cls TEXT
#   Converter class: marker.converters.pdf.PdfConverter (default)
#   or marker.converters.table.TableConverter (better for tables)
--converter_cls marker.converters.table.TableConverter

# ==============================================================================
# LLM OPTIONS (Enhanced accuracy)
# ==============================================================================

# --use_llm
#   Use LLM to improve accuracy (table merging, layout correction, etc.)
# (enabled by default in pdf_to_md.py, use --no-llm to disable)
# --use_llm

# --llm_service TEXT
#   LLM service to use:
#   - marker.services.gemini.GoogleGeminiService (Google Gemini)
#   - marker.services.ollama.OllamaService (Ollama local/cloud)
#   - marker.services.openai.OpenAIService (OpenAI)
#   - marker.services.claude.ClaudeService (Anthropic Claude)
#   - marker.services.azure_openai.AzureOpenAIService (Azure OpenAI)
#   - marker.services.vertex.GoogleVertexService (Google Vertex AI)
--llm_service marker.services.ollama.OllamaService

# --ollama_base_url TEXT
#   Base URL for Ollama service (default: http://localhost:11434)
--ollama_base_url http://localhost:11434

# --ollama_model TEXT
#   Ollama model to use (default: llama3.2-vision)
--ollama_model minimax-m2.5:cloud

# --gemini_api_key TEXT
#   API key for Google Gemini (if using Gemini service)
# --gemini_api_key YOUR_API_KEY

# --gemini_model_name TEXT
#   Gemini model name (default: gemini-2.0-flash)
# --gemini_model_name gemini-2.0-flash

# --openai_api_key TEXT
#   API key for OpenAI (if using OpenAI service)
# --openai_api_key YOUR_API_KEY

# --openai_model TEXT
#   OpenAI model name (default: gpt-4o-mini)
# --openai_model gpt-4o-mini

# --claude_api_key TEXT
#   API key for Claude (if using Claude service)
# --claude_api_key YOUR_API_KEY

# --claude_model_name TEXT
#   Claude model name (default: claude-3-7-sonnet-20250219)
# --claude_model_name claude-3-7-sonnet-20250219

# --max_concurrency INTEGER
#   Max concurrent requests to LLM (default: 3)
# --max_concurrency 3

# ==============================================================================
# TABLE PROCESSING OPTIONS
# ==============================================================================

# --html_tables_in_markdown
#   Format tables as HTML instead of Markdown
# --html_tables_in_markdown

# --max_table_rows INTEGER
#   Max rows to process with LLM for tables (default: 175)
# --max_table_rows 175

# --max_rows_per_batch INTEGER
#   Chunk large tables into batches (default: 60)
# --max_rows_per_batch 60

# --no_merge_tables_across_pages
#   Don't merge tables that span multiple pages
# --no_merge_tables_across_pages

# ==============================================================================
# RENDERING OPTIONS
# ==============================================================================

# --extract_images BOOLEAN
#   Extract images from document (default: True)
# --extract_images True

# --paginate_output
#   Add page separators in output
# --paginate_output

# --page_separator TEXT
#   Separator between pages (default: 48 dashes)
# --page_separator "------------------------------------------------"

# --keep_pageheader_in_output
#   Keep page headers in output
# --keep_pageheader_in_output

# --keep_pagefooter_in_output
#   Keep page footers in output
# --keep_pagefooter_in_output

# ==============================================================================
# OCR OPTIONS
# ==============================================================================

# --ocr_task_name TEXT
#   OCR mode: 'ocr_with_boxes' (default, better formatting)
#   or 'ocr_without_boxes' (faster, may lose formatting)
# --ocr_task_name ocr_with_boxes

# --disable_ocr_math
#   Disable inline math recognition in OCR
# --disable_ocr_math

# --drop_repeated_text
#   Drop repeated text in OCR results
# --drop_repeated_text

# --keep_chars
#   Keep individual characters in output
# --keep_chars

# ==============================================================================
# LAYOUT OPTIONS
# ==============================================================================

# --force_layout_block TEXT
#   Force all pages to be treated as a specific block type:
#   Caption, Code, Figure, Heading, List, Paragraph, SectionHeader, Table
# --force_layout_block

# --lowres_image_dpi INTEGER
#   DPI for low-res layout detection images (default: 96)
# --lowres_image_dpi 96

# --highres_image_dpi INTEGER
#   DPI for high-res OCR images (default: 192)
# --highres_image_dpi 192

# ==============================================================================
# PERFORMANCE OPTIONS
# ==============================================================================

# --disable_multiprocessing
#   Disable multiprocessing (use single thread)
# --disable_multiprocessing

# --layout_batch_size INTEGER
#   Batch size for layout model
# --layout_batch_size

# --detection_batch_size INTEGER
#   Batch size for detection model
# --detection_batch_size

# --recognition_batch_size INTEGER
#   Batch size for OCR recognition
# --recognition_batch_size

# --disable_tqdm
#   Disable progress bars
# --disable_tqdm

# ==============================================================================
# DEBUG OPTIONS
# ==============================================================================

# -d, --debug
#   Enable debug mode
# -d

# --debug_data_folder TEXT
#   Folder for debug output (default: debug_data)
# --debug_data_folder debug_data

# --debug_layout_images
#   Save layout debug images
# --debug_layout_images

# --debug_pdf_images
#   Save PDF debug images
# --debug_pdf_images

# --debug_json
#   Save block debug data as JSON
# --debug_json

# ==============================================================================
# PROCESSORS (Advanced)
# ==============================================================================

# --processors TEXT
#   Comma-separated list of processors (full module path required)
# --processors

# --config_json TEXT
#   Path to JSON config file with additional settings
# --config_json

# ==============================================================================
# EXTRACTION OPTIONS (for JSON/HTML output)
# ==============================================================================

# --page_schema TEXT
#   JSON schema for extraction (for extraction converters)
# --page_schema

# --existing_markdown TEXT
#   Pre-converted markdown for extraction (for extraction converters)
# --existing_markdown

# ==============================================================================
# AZURE OPENAI OPTIONS
# ==============================================================================

# --azure_endpoint TEXT
#   Azure OpenAI endpoint URL
# --azure_endpoint

# --azure_api_key TEXT
#   Azure OpenAI API key
# --azure_api_key

# --azure_api_version TEXT
#   Azure OpenAI API version
# --azure_api_version

# --deployment_name TEXT
#   Azure deployment name
# --deployment_name

# ==============================================================================
# VERTEX AI OPTIONS
# ==============================================================================

# --vertex_project_id TEXT
#   Google Cloud Project ID for Vertex AI
# --vertex_project_id

# --vertex_location TEXT
#   Google Cloud Location (default: us-central1)
# --vertex_location

# --vertex_dedicated
#   Use dedicated Vertex AI instance
# --vertex_dedicated

# ==============================================================================
# TIMEOUT & RETRY OPTIONS
# ==============================================================================

# --timeout INTEGER
#   Request timeout in seconds (default: 30)
# --timeout 30

# --max_retries INTEGER
#   Max retry attempts (default: 2)
# --max_retries 2

# --retry_wait_time INTEGER
#   Wait time between retries in seconds (default: 3)
# --retry_wait_time 3

# --max_output_tokens INTEGER
#   Max tokens in LLM response
# --max_output_tokens
