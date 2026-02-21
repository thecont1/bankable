#!/usr/bin/env python3
"""
Patched marker_single wrapper that applies the financial statement
table row-splitting fix before running Marker's converter.
"""

import os
import sys
import time

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Apply table split patch BEFORE importing Marker's converter
from table_split_patch import patch_table_processor
patch_table_processor()

# Apply PDF password patch only if a password is set
from pdfprovider_patch import patch_all
if os.environ.get("PDF_PASSWORD"):
    patch_all()
else:
    print("✅ Table split patch applied (no PDF password needed)", file=sys.stderr)

from marker.config.parser import ConfigParser
from marker.config.printer import CustomClickPrinter
from marker.logger import configure_logging, get_logger
from marker.models import create_model_dict
from marker.output import save_output

import click

configure_logging()
logger = get_logger()


@click.command(cls=CustomClickPrinter, help="Convert a single PDF to markdown (patched).")
@click.argument("fpath", type=str)
@ConfigParser.common_options
def convert_single_patched(fpath: str, **kwargs):
    models = create_model_dict()
    start = time.time()
    config_parser = ConfigParser(kwargs)

    converter_cls = config_parser.get_converter_cls()
    converter = converter_cls(
        config=config_parser.generate_config_dict(),
        artifact_dict=models,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    rendered = converter(fpath)
    out_folder = config_parser.get_output_folder(fpath)
    save_output(rendered, out_folder, config_parser.get_base_filename(fpath))

    logger.info(f"Saved markdown to {out_folder}")
    logger.info(f"Total time: {time.time() - start}")


if __name__ == "__main__":
    convert_single_patched()
