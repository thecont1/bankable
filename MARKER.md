# Marker Configuration File

#### Edit flags below to customize Marker behavior.
#### Lines starting with -- are active flags. Comment with # to disable.

# CORE OPTIONS

### Output format: markdown, json, html, or chunks
--output_format markdown

### Skip extracting images from the PDF (faster)
--disable_image_extraction

# CONVERTER OPTIONS

### Converter class: marker.converters.pdf.PdfConverter or marker.converters.table.TableConverter
--converter_cls marker.converters.table.TableConverter

# TABLE PROCESSING

### Row split threshold: fraction of rows that must need splitting before Marker
### activates row-splitting. Default 0.5 is too high for financial statements
### where header/summary rows are fine but transaction rows get collapsed.
### Lower values ensure collapsed transaction rows always get split out.
--row_split_threshold 0.15

### Format tables as HTML in markdown (better for complex financial tables)
### Comment out to use Markdown table format
#### --html_tables_in_markdown

# IMAGE RESOLUTION

### Higher DPI gives the table recognition model sharper cell boundaries.
### Default highres=192 is often too low for dense financial tables.
--highres_image_dpi 300

# LLM OPTIONS (Optional - requires Ollama or other LLM service)

### Use LLM for improved table extraction (slower but more accurate)
#### --use_llm

### LLM service: marker.services.ollama.OllamaService
#### --llm_service marker.services.ollama.OllamaService

### Ollama base URL
#### --ollama_base_url http://localhost:11434

### Ollama model
#### --ollama_model minimax-m2.5:cloud

# PERFORMANCE OPTIONS

### Batch sizes for parallel processing (adjust for your Mac)
#### --layout_batch_size 1
#### --detection_batch_size 1
#### --recognition_batch_size 1

### Disable multiprocessing
#### --disable_multiprocessing

# OTHER OPTIONS

### Page range: "0,5-10,20" (default: all pages)
#### --page_range

### Disable OCR: only safe when PDF has reliably selectable text.
### Disabled for now — OCR fallback is needed when pdftext fails to
### segment text into individual table cells.
#### --disable_ocr

### Force OCR on all pages (for scanned documents)
#### --force_ocr

### Enable debug mode
#### -d

### Disable progress bars
#### --disable_tqdm
