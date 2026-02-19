#!/usr/bin/env python3
"""
Test script to extract transaction data using Llama3.2:3b
"""
import json
import requests
import sys
import os
import re

def query_llama(prompt: str) -> str:
    """Query Llama3.2:3b using Ollama's API"""
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        print(f"Error querying Llama: {e}", file=sys.stderr)
        return ""

def clean_markdown(text: str) -> str:
    """Clean markdown by replacing HTML line breaks with single spaces"""
    # Replace all <br> tags with single spaces
    text = re.sub(r'<br\s*/?>', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Trim leading/trailing whitespace from lines
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            # For table lines, preserve proper formatting
            if stripped.startswith('|') and stripped.endswith('|'):
                # Split into cells, trim each cell, preserve separators
                cells = stripped.split('|')
                cleaned_cells = []
                for i, cell in enumerate(cells):
                    if i == 0 or i == len(cells) - 1:
                        continue
                    cleaned_cells.append(cell.strip())
                cleaned_lines.append(f"| {' | '.join(cleaned_cells)} |")
            else:
                cleaned_lines.append(stripped)
    return '\n'.join(cleaned_lines)


def extract_transactions(text: str) -> str:
    """Extract transactions from markdown using Llama3.2:3b"""
    # Clean markdown before processing with LLM
    cleaned_text = clean_markdown(text)
    
    prompt = f"""
You are an expert financial data extractor. Your task is to extract transaction data from the following bank/credit card statement text in Markdown format.

The extracted data must be a single Markdown table with ONLY these columns:
Date | Transaction Description | Amount (in Rs.)

Only include actual transaction data. Exclude any header information, account details, summary tables, or other non-transaction content.

Ensure the table is properly formatted and all transaction amounts are in Indian Rupees (Rs.).

IMPORTANT:
- Return ONLY the Markdown table with no additional text, explanations, or comments.
- Do NOT include any SQL examples, use cases, or additional analysis.
- Do NOT include negative amounts that don't make sense as transactions.
- Do NOT include duplicate transactions or reversed transactions.

Transaction text:
```
{cleaned_text}
```
"""
    
    response = query_llama(prompt)
    cleaned_response = clean_markdown(response)
    return cleaned_response.strip()

def test_with_sample():
    """Test with a sample transaction text from one of the files"""
    if len(sys.argv) > 1:
        md_file = sys.argv[1]
        if os.path.exists(md_file):
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                
            print("Testing LLM extraction...")
            print("=" * 50)
            
            extracted = extract_transactions(content)
            
            print("\nLLM Output:")
            print("=" * 50)
            print(extracted)
            
            return extracted
        else:
            print(f"File not found: {md_file}", file=sys.stderr)
            return None
    else:
        print("Usage: python test_llm_extraction.py <md_file>", file=sys.stderr)
        return None

if __name__ == "__main__":
    result = test_with_sample()
    if result and "| Date |" not in result:
        print("\nWarning: No transaction table extracted")
    sys.exit(0 if "| Date |" in (result or "") else 1)
