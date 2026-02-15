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

### Format tables as HTML in markdown (better for complex financial tables)
### Comment out to use Markdown table format
#### --html_tables_in_markdown

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

### Disable OCR (use only if PDF has selectable text)
#### --disable_ocr

### Force OCR on all pages (for scanned documents)
#### --force_ocr

### Enable debug mode
#### -d

### Disable progress bars
#### --disable_tqdm
