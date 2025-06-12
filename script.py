#!/usr/bin/env python3
"""
PDF Document Parser using Docling
Extracts text and tables from PDF documents (including scanned PDFs with Arabic text)
and exports each page as markdown format optimized for LLM consumption.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    TableFormerMode,
    TableStructureOptions,
    OcrOptions,
    TesseractCliOcrOptions,
    EasyOcrOptions
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DoclingPDFParser:
    """Main class for parsing PDF documents using Docling."""
    
    def __init__(self, pdf_path: str, ocr_enabled: bool = True, ocr_engine: str = "tesseract"):
        """
        Initialize the parser.
        
        Args:
            pdf_path: Path to the PDF file
            ocr_enabled: Whether to enable OCR for scanned documents
            ocr_engine: OCR engine to use ('tesseract' or 'easyocr')
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.ocr_enabled = ocr_enabled
        self.ocr_engine = ocr_engine
        self.extracted_pages = []
        
        # Initialize document converter with proper configuration
        self.converter = self._create_converter()
    
    def _create_converter(self) -> DocumentConverter:
        """
        Create and configure the document converter.
        
        Returns:
            Configured DocumentConverter instance
        """
        # Create pipeline options
        pipeline_options = PdfPipelineOptions()
        
        # Configure OCR if enabled
        if self.ocr_enabled:
            pipeline_options.do_ocr = True
            
            # Configure OCR options based on engine
            if self.ocr_engine == "tesseract":
                # Create Tesseract CLI-specific options
                ocr_options = TesseractCliOcrOptions(
                    lang=["ara", "eng"],  # Arabic and English
                    force_full_page_ocr=False,
                )
            else:  # easyocr
                # Create EasyOCR-specific options
                ocr_options = EasyOcrOptions(
                    lang=["ar", "en"],     # Arabic and English for EasyOCR
                    use_gpu=False,            # Set to True if you have GPU
                    force_full_page_ocr=False
                )
            
            pipeline_options.ocr_options = ocr_options
        else:
            pipeline_options.do_ocr = False
        
        # Configure table extraction
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True
        )
        
        # Create and return converter
        return DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )    
    def parse_document(self) -> List[str]:
        """
        Parse the entire document and extract content page by page.
        
        Returns:
            List of markdown strings, one for each page
        """
        logger.info(f"Starting to parse document: {self.pdf_path}")
        
        try:
            # Convert document
            result = self.converter.convert(str(self.pdf_path))
            
            # Get the document object
            document = result.document
            
            # Export to markdown (Docling has built-in markdown export)
            full_markdown = document.export_to_markdown()
            
            # Split by pages if page markers are present
            # Otherwise, process as a single document
            page_markdowns = self._split_markdown_by_pages(full_markdown)
            
            # If no page splitting was possible, process elements directly
            if len(page_markdowns) == 1 and not page_markdowns[0].strip():
                page_markdowns = self._process_elements_by_page(document)
            
            return page_markdowns
            
        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            raise
    
    def _split_markdown_by_pages(self, markdown: str) -> List[str]:
        """
        Split markdown content by page markers if present.
        
        Args:
            markdown: Full markdown content
            
        Returns:
            List of markdown strings split by page
        """
        # Look for page markers in the markdown
        # Docling might use different formats, so we'll try multiple patterns
        import re
        
        # Try to split by common page patterns
        page_patterns = [
            r'(?=# Page \d+)',
            r'(?=## Page \d+)',
            r'(?=\n---+\n)',  # Horizontal rules might indicate page breaks
            r'(?=<!-- Page \d+ -->)'
        ]
        
        pages = []
        for pattern in page_patterns:
            parts = re.split(pattern, markdown)
            if len(parts) > 1:
                # Found page breaks
                pages = [part.strip() for part in parts if part.strip()]
                break
        
        # If no page breaks found, return as single page
        if not pages:
            pages = [markdown]
        
        return pages
    
    def _process_elements_by_page(self, document) -> List[str]:
        """
        Process document elements and group by page.
        
        Args:
            document: Parsed document object
            
        Returns:
            List of markdown strings for each page
        """
        pages_content = {}
        
        # Get all elements with their page information
        for item in document.iterate_items():
            # Try to get page number from various possible attributes
            page_num = 1  # Default page
            
            # Check for page_number attribute
            if hasattr(item, 'page_number'):
                page_num = item.page_number
            # Check for prov (provenance) information
            elif hasattr(item, 'prov') and item.prov:
                for prov_item in item.prov:
                    if hasattr(prov_item, 'page'):
                        page_num = prov_item.page
                        break
                    elif hasattr(prov_item, 'page_number'):
                        page_num = prov_item.page_number
                        break
            
            if page_num not in pages_content:
                pages_content[page_num] = []
            
            pages_content[page_num].append(item)
        
        # Convert each page's content to markdown
        page_markdowns = []
        max_page = max(pages_content.keys()) if pages_content else 1
        
        for page_num in range(1, max_page + 1):
            page_items = pages_content.get(page_num, [])
            page_markdown = self._items_to_markdown(page_items, page_num)
            page_markdowns.append(page_markdown)
        
        return page_markdowns
    
    def _items_to_markdown(self, items: List, page_num: int) -> str:
        """
        Convert a list of document items to markdown.
        
        Args:
            items: List of document items
            page_num: Page number
            
        Returns:
            Markdown string
        """
        markdown_parts = [f"# Page {page_num}\n"]
        
        for item in items:
            # Get the markdown representation of the item
            if hasattr(item, 'export_to_markdown'):
                item_markdown = item.export_to_markdown()
                markdown_parts.append(item_markdown)
            elif hasattr(item, 'text'):
                # Handle text items
                text = str(item.text).strip()
                if text:
                    # Check item type for formatting
                    item_type = item.__class__.__name__.lower()
                    
                    if 'heading' in item_type or 'title' in item_type:
                        level = getattr(item, 'level', 2)
                        markdown_parts.append(f"{'#' * level} {text}\n")
                    elif 'list' in item_type:
                        markdown_parts.append(f"- {text}")
                    else:
                        markdown_parts.append(f"{text}\n")
            elif hasattr(item, 'data'):
                # Handle table items
                table_md = self._table_to_markdown(item)
                if table_md:
                    markdown_parts.append(table_md)
        
        return "\n".join(markdown_parts)
    
    def _table_to_markdown(self, table_item) -> str:
        """
        Convert a table item to markdown format.
        
        Args:
            table_item: Table item
            
        Returns:
            Markdown table string
        """
        try:
            # If the table has an export_to_markdown method, use it
            if hasattr(table_item, 'export_to_markdown'):
                return table_item.export_to_markdown()
            
            # Otherwise, manually convert
            if hasattr(table_item, 'data') and table_item.data:
                rows = table_item.data
                
                if not rows:
                    return ""
                
                markdown_lines = []
                
                for idx, row in enumerate(rows):
                    # Process cells
                    if isinstance(row, list):
                        cells = [str(cell).strip() for cell in row]
                    else:
                        cells = [str(row).strip()]
                    
                    # Create markdown row
                    row_text = "| " + " | ".join(cells) + " |"
                    markdown_lines.append(row_text)
                    
                    # Add separator after first row
                    if idx == 0:
                        separator = "|" + "|".join(["---"] * len(cells)) + "|"
                        markdown_lines.append(separator)
                
                return "\n".join(markdown_lines) + "\n"
            
            return ""
            
        except Exception as e:
            logger.warning(f"Error converting table to markdown: {e}")
            return ""
    
    def export_as_json(self, page_markdowns: List[str]) -> str:
        """
        Export the parsed content as JSON for LLM consumption.
        
        Args:
            page_markdowns: List of markdown strings for each page
            
        Returns:
            JSON string
        """
        # Create a flat list of content elements
        content_list = []
        
        for page_markdown in page_markdowns:
            # Split markdown into logical sections
            lines = page_markdown.split('\n')
            
            current_section = []
            for line in lines:
                line = line.strip()
                
                # Skip page headers
                if line.startswith('# Page'):
                    continue
                
                # Check if this is a new section (heading or table)
                if line and (line.startswith('#') or line.startswith('|')):
                    # Save previous section if exists
                    if current_section:
                        section_text = '\n'.join(current_section).strip()
                        if section_text:
                            content_list.append(section_text)
                        current_section = []
                
                # Add line to current section
                if line:
                    current_section.append(line)
                elif current_section:
                    # Empty line marks end of section
                    section_text = '\n'.join(current_section).strip()
                    if section_text:
                        content_list.append(section_text)
                    current_section = []
            
            # Don't forget the last section
            if current_section:
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    content_list.append(section_text)
        
        return json.dumps(content_list, ensure_ascii=False, indent=2)


def main():
    """Main function to handle command-line execution."""
    parser = argparse.ArgumentParser(
        description="Parse PDF documents using Docling and export as markdown"
    )
    parser.add_argument(
        "--pdf_path",
        type=str,
        required=True,
        help="Path to the PDF file to process"
    )
    parser.add_argument(
        "--disable_ocr",
        action="store_true",
        help="Disable OCR (use for text-based PDFs only)"
    )
    parser.add_argument(
        "--ocr_engine",
        type=str,
        choices=["tesseract", "easyocr"],
        default="tesseract",
        help="OCR engine to use (default: tesseract)"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["markdown", "json"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file path (if not specified, prints to stdout)"
    )
    
    from argparse import Namespace

    args = Namespace(**{
        "pdf_path":      "normalized.pdf",
        "disable_ocr":   False,                # True to turn OCR off
        "ocr_engine": "tesseract",
        "output_format": "markdown",           # or "json"
        "output_file":   "out.md"                  # or "out.md" to write to file
    })
    
    try:
        # Create parser instance
        parser_instance = DoclingPDFParser(
            args.pdf_path, 
            ocr_enabled=not args.disable_ocr,
            ocr_engine=args.ocr_engine
        )
        
        # Parse document
        logger.info("Starting document parsing...")
        page_markdowns = parser_instance.parse_document()
        
        # Prepare output
        if args.output_format == "markdown":
            # Combine all pages into one markdown document
            output = "\n\n---\n\n".join(page_markdowns)
        else:  # json
            output = parser_instance.export_as_json(page_markdowns)
        
        # Write output
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(output)
            logger.info(f"Output written to {args.output_file}")
        else:
            print(output)
        
        logger.info(f"Parsing complete. Processed {len(page_markdowns)} pages.")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()