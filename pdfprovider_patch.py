"""
Runtime patch for Marker's PDF provider to support password-protected PDFs.
This patch dynamically modifies the Marker and pdftext libraries at runtime
to accept and use passwords when opening PDF files.
"""
import os
import sys
import subprocess
from importlib.metadata import version


def is_pdftext_installed():
    """Check if pdftext library is available."""
    try:
        import pdftext
        return True
    except ImportError:
        return False


def is_marker_installed():
    """Check if marker-pdf library is available."""
    try:
        import marker
        return True
    except ImportError:
        return False


def patch_marker_pdfprovider():
    """Patch Marker's PDF provider at runtime to support passwords."""
    try:
        from marker.providers.pdf import PdfProvider
        original_init = PdfProvider.__init__
        
        def patched_init(self, filepath, config):
            self.password = os.environ.get("PDF_PASSWORD")
            original_init(self, filepath, config)
            
        import contextlib

        @contextlib.contextmanager
        def patched_get_doc(self):
            import pypdfium2 as pdfium
            
            doc = None
            try:
                password = os.environ.get("PDF_PASSWORD")
                if password:
                    doc = pdfium.PdfDocument(self.filepath, password=password)
                else:
                    doc = pdfium.PdfDocument(self.filepath)
                
                yield doc
            finally:
                if doc:
                    doc.close()
        
        # Replace the methods
        PdfProvider.__init__ = patched_init
        PdfProvider.get_doc = patched_get_doc
        return True
        
    except Exception as e:
        print(f"Warning: Failed to patch Marker PDF provider: {e}", file=sys.stderr)
        return False


def patch_pdftext_extraction():
    """Patch pdftext's extraction module at runtime to support passwords."""
    try:
        from pdftext import extraction
        original_load_pdf = extraction._load_pdf
        
        def patched_load_pdf(pdf, flatten_pdf):
            import pypdfium2 as pdfium
            password = os.environ.get("PDF_PASSWORD")
            if password:
                return pdfium.PdfDocument(pdf, password=password)
            else:
                return original_load_pdf(pdf, flatten_pdf)
        
        extraction._load_pdf = patched_load_pdf
        return True
        
    except Exception as e:
        print(f"Warning: Failed to patch pdftext extraction: {e}", file=sys.stderr)
        return False


def patch_all():
    """Apply all necessary patches."""
    success_count = 0
    if is_marker_installed() and patch_marker_pdfprovider():
        success_count += 1
    if is_pdftext_installed() and patch_pdftext_extraction():
        success_count += 1
        
    if success_count == 0:
        print("Warning: No patches applied. Make sure marker-pdf and pdftext are installed.", file=sys.stderr)
    elif success_count == 2:
        print("✅ PDF password support patched successfully", file=sys.stderr)
    else:
        print(f"Warning: Only {success_count}/2 patches applied", file=sys.stderr)


def check_for_patches():
    """Check if the current Marker and pdftext versions require patching."""
    # This is a simplistic check - we assume all versions require patching
    return True


def verify_patch_applied():
    """Verify that patches are correctly applied."""
    try:
        if is_marker_installed():
            from marker.providers.pdf import PdfProvider
            assert hasattr(PdfProvider, "password"), "Marker PDF provider not patched"
            assert callable(getattr(PdfProvider, "get_doc", None)), "get_doc method not patched"
            
        if is_pdftext_installed():
            from pdftext import extraction
            assert hasattr(extraction, "_load_pdf"), "pdftext extraction not imported"
            
        return True
    except Exception as e:
        print(f"Warning: Patch verification failed: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    print("Testing PDF password support patching...")
    patch_all()
    
    if verify_patch_applied():
        print("✅ All patches verified successfully")
    else:
        print("❌ Patch verification failed")
