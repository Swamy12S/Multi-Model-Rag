"""
Multi-modal document processor for handling text, tables, images, and OCR.
"""

import io
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import base64

import fitz  # PyMuPDF for PDF processing
import pytesseract
from PIL import Image
import numpy as np
from langchain_core.documents import Document

import pytesseract
import os

# Tell pytesseract where tesseract is installed
pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@dataclass
class MultiModalChunk:
    """Represents a chunk with metadata about its source and modality"""
    content: str
    modality: str  # "text", "table", "image_ocr", "image_metadata"
    page_num: int
    section_id: str  # e.g., "text_1", "table_2", "image_3"
    source_file: str
    metadata: Dict = None
    image_base64: Optional[str] = None  # For image chunks
    
    def to_document(self) -> Document:
        """Convert to LangChain Document"""
        doc_metadata = {
            "modality": self.modality,
            "page_num": self.page_num,
            "section_id": self.section_id,
            "source_file": self.source_file,
        }
        if self.metadata:
            doc_metadata.update(self.metadata)
        if self.image_base64:
            doc_metadata["image_base64"] = self.image_base64
            
        return Document(
            page_content=self.content,
            metadata=doc_metadata
        )


class MultiModalDocumentProcessor:
    """Process PDFs to extract text, tables, and images with metadata"""
    
    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr
        
    def process_pdf(self, pdf_path: str) -> List[MultiModalChunk]:
        """
        Extract text, tables, and images from PDF with OCR support
        Returns list of MultiModalChunk objects
        """
        chunks = []
        pdf_path = Path(pdf_path)
        
        try:
            pdf_doc = fitz.open(pdf_path)
            total_pages = pdf_doc.page_count
            
            for page_num in range(total_pages):
                page = pdf_doc[page_num]
                
                # Extract text
                text_chunks = self._extract_text(page, page_num, pdf_path.name)
                chunks.extend(text_chunks)
                
                # Extract tables
                table_chunks = self._extract_tables(page, page_num, pdf_path.name)
                chunks.extend(table_chunks)
                
                # Extract images with OCR
                if self.enable_ocr:
                    image_chunks = self._extract_images(page, page_num, pdf_path.name)
                    chunks.extend(image_chunks)
            
            pdf_doc.close()
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            
        return chunks
    
    def _extract_text(self, page: fitz.Page, page_num: int, source_file: str) -> List[MultiModalChunk]:
        """Extract plain text from page"""
        chunks = []
        
        try:
            text = page.get_text()
            if text.strip():
                chunks.append(
                    MultiModalChunk(
                        content=text.strip(),
                        modality="text",
                        page_num=page_num + 1,  # 1-indexed
                        section_id=f"text_{page_num}",
                        source_file=source_file,
                        metadata={"text_length": len(text)}
                    )
                )
        except Exception as e:
            print(f"Error extracting text from page {page_num}: {str(e)}")
            
        return chunks
    
    def _extract_tables(self, page: fitz.Page, page_num: int, source_file: str) -> List[MultiModalChunk]:
        """Extract tables from page using pdfplumber"""
        chunks = []
        
        try:
            import pdfplumber
            pdf_path = page.parent.name if hasattr(page.parent, 'name') else ""
            
            if pdf_path and Path(pdf_path).exists():
                with pdfplumber.open(pdf_path) as pdf:
                    if page_num < len(pdf.pages):
                        page_table = pdf.pages[page_num]
                        tables = page_table.extract_tables()
                        
                        for table_idx, table in enumerate(tables or []):
                            # Convert table to markdown format
                            table_md = self._table_to_markdown(table)
                            chunks.append(
                                MultiModalChunk(
                                    content=table_md,
                                    modality="table",
                                    page_num=page_num + 1,
                                    section_id=f"table_{page_num}_{table_idx}",
                                    source_file=source_file,
                                    metadata={
                                        "table_index": table_idx,
                                        "rows": len(table),
                                        "cols": len(table[0]) if table else 0
                                    }
                                )
                            )
        except ImportError:
            print("pdfplumber not installed. Skipping table extraction.")
        except Exception as e:
            print(f"Error extracting tables from page {page_num}: {str(e)}")
            
        return chunks
    
    def _extract_images(self, page: fitz.Page, page_num: int, source_file: str) -> List[MultiModalChunk]:
        """Extract images from page with OCR"""
        chunks = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img_ref in enumerate(image_list):
                try:
                    # Extract image
                    xref = img_ref[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    # Convert to PIL Image
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("ppm")
                        img = Image.open(io.BytesIO(img_data))
                    else:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix.tobytes("ppm")
                        img = Image.open(io.BytesIO(img_data))
                    
                    # Perform OCR
                    ocr_text = pytesseract.image_to_string(img)
                    
                    # Convert image to base64
                    img_base64 = self._image_to_base64(img)
                    
                    # Create metadata about the image
                    img_metadata = f"Image on page {page_num + 1}"
                    if ocr_text.strip():
                        img_metadata += f". Extracted text: {ocr_text[:500]}"
                    
                    chunks.append(
                        MultiModalChunk(
                            content=img_metadata,
                            modality="image_ocr",
                            page_num=page_num + 1,
                            section_id=f"image_{page_num}_{img_index}",
                            source_file=source_file,
                            metadata={
                                "image_index": img_index,
                                "image_size": img.size,
                                "ocr_text": ocr_text[:1000] if ocr_text else ""
                            },
                            image_base64=img_base64
                        )
                    )
                    
                except Exception as e:
                    print(f"Error processing image {img_index} on page {page_num}: {str(e)}")
                    
        except Exception as e:
            print(f"Error extracting images from page {page_num}: {str(e)}")
            
        return chunks
    
    @staticmethod
    def _table_to_markdown(table: List[List[str]]) -> str:
        """Convert table to markdown format"""
        if not table:
            return ""
        
        # Get headers (first row)
        headers = table[0]
        rows = table[1:]
        
        # Create markdown table
        md = "| " + " | ".join(str(h) for h in headers) + " |\n"
        md += "|" + "|".join(["---"] * len(headers)) + "|\n"
        
        for row in rows:
            md += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        
        return md
    
    @staticmethod
    def _image_to_base64(img: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
