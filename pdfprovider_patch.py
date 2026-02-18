"""Temporary patch for Marker's PdfProvider to add password support"""
import os
import sys
from pathlib import Path
import contextlib

# Add venv to path
sys.path.insert(0, str(Path(__file__).parent / ".venv" / "lib" / "python3.13" / "site-packages"))

from marker.providers.pdf import PdfProvider
import pypdfium2 as pdfium
from pypdfium2 import PdfiumError


# Save original get_doc method
original_get_doc = PdfProvider.get_doc

# Add password support
@contextlib.contextmanager
def patched_get_doc(self):
    """Patch get_doc to accept password"""
    doc = None
    try:
        # Try to get password from config
        password = getattr(self, "password", None)
        # If password not set, try environment variable
        if not password:
            password = os.environ.get("PDF_PASSWORD")
        
        doc = pdfium.PdfDocument(self.filepath, password=password)
        
        # Must be called on the parent pdf, before retrieving pages to render correctly
        if self.flatten_pdf:
            doc.init_forms()
            
        yield doc
    except PdfiumError as e:
        # If password is wrong or document is encrypted and no password provided
        if "password" in str(e).lower() or "encrypted" in str(e).lower():
            raise Exception(f"PDF is encrypted - please provide a password: {e}")
        raise
    finally:
        if doc:
            doc.close()

# Apply the patch
PdfProvider.get_doc = patched_get_doc
PdfProvider.password = None

print("✅ PdfProvider patched successfully - password support added")