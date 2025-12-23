# Multi-Modal RAG System - Implementation Guide

## Overview
This is a comprehensive Multi-Modal Retrieval-Augmented Generation (RAG) system that processes text, tables, images (with OCR), and metadata from complex PDF documents like IMF reports.

## Features Implemented

### 1. **Multi-Modal Document Ingestion** ✅
- **Text Extraction**: Extracts plain text from PDFs
- **Table Extraction**: Uses pdfplumber to extract tables and converts them to Markdown format
- **Image Processing**: Extracts images from PDFs
- **OCR (Optical Character Recognition)**: Uses Tesseract to extract text from images
- **Metadata Preservation**: Each chunk retains information about:
  - Source document
  - Page number
  - Section ID (for tracking)
  - Document modality type
  - Content preview

**Location**: `ragbase/multimodal_processor.py`

### 2. **Unified Multi-Modal Embedding Space** ✅
- All modalities (text, tables, OCR'd text) are embedded in the same vector space
- Uses `sentence-transformers/all-MiniLM-L6-v2` for consistent embeddings
- Enables semantic search across all modalities

**Location**: `ragbase/ingestor.py` (enhanced)

### 3. **Smart Chunking Strategy** ✅
Two-level chunking approach:
- **Semantic Chunking**: Uses semantic boundaries to split documents intelligently
- **Recursive Character Splitting**: Fallback for structured chunking with overlap (2048 chars, 128 overlap)
- **Modality-Aware**: Preserves structure for tables and images

**Location**: `ragbase/ingestor.py`

### 4. **Multi-Modal Retrieval** ✅
- **ModaModalRetriever**: Custom retriever that groups documents by modality
- **Prioritization**: Retrieves text first, then tables, then images for optimal context
- **Reranking**: Compatible with FlashRank for relevance scoring
- **Filtering**: Optional LLM-based chain filtering

**Location**: `ragbase/retriever.py` (enhanced)

### 5. **Citation-Aware QA Chatbot** ✅
- **Source Attribution**: Every answer includes citations with:
  - Page number
  - Section ID
  - Document modality (TEXT, TABLE, IMAGE_OCR)
  - Source file name
- **Interactive Interface**: Streamlit-based chatbot with conversation history
- **Citation Format**: `[Source: filename.pdf, Page 5, Section table_4, Modality: TABLE]`

**Location**: `ragbase/chain.py` (enhanced)

### 6. **Multi-Model LLM Support** ✅
- **Single Model Mode**: Uses one LLM (Llama 3.1 or Gemma2)
- **Multi-Model Mode**: Ensemble approach using multiple LLMs simultaneously
- **Model Routing**: Can switch between local (Ollama) and remote (Groq) models
- **Comparison**: Shows responses from different models for verification

**Location**: `ragbase/model.py` (enhanced)

### 7. **Comprehensive Evaluation Suite** ✅
Metrics for benchmark evaluation:
- **Modality Diversity**: Measures document type coverage (0-1)
- **Citation Completeness**: Checks metadata presence (0-1)
- **Answer Relevance**: LLM-based grounding assessment (0-1)
- **Modality Coverage**: Tracks which document types were retrieved
- **Citation Accuracy**: Validates citation formatting and completeness

**Location**: `ragbase/evaluator.py`

## Architecture

```
PDF Documents
    ↓
MultiModalDocumentProcessor
    ├── Text Extraction
    ├── Table Extraction → Markdown
    ├── Image Extraction
    └── OCR Processing
    ↓
MultiModalChunk Objects (with metadata)
    ↓
Semantic + Recursive Chunking
    ↓
HuggingFace Embeddings (all modalities)
    ↓
Qdrant Vector Store
    ↓
MultiModalRetriever (modality-prioritized)
    ↓
LLM Chain (with citations)
    ↓
Streamlit Interface
```

## Configuration

### `ragbase/config.py`
```python
class Model:
    USE_MULTI_MODEL = True          # Enable multi-model ensemble
    REMOTE_LLM = "llama-3.1-70b"   # Primary LLM
    REMOTE_LLM_2 = "mixtral-8x7b"  # Secondary LLM
    TEMPERATURE = 0.0
    MAX_TOKENS = 8000

class Retriever:
    USE_MULTIMODAL = True           # Enable multi-modal processing
    USE_CITATIONS = True            # Include source citations
    USE_RERANKER = True
    USE_CHAIN_FILTER = False
```

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# Also install system dependency:
sudo apt-get install tesseract-ocr  # On Linux
# Or download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Configuration Options (in Streamlit UI)
- **Enable Multi-Model LLM**: Toggle for single vs. multiple models
- **Enable Multi-Modal Processing**: Process text, tables, and images
- **Show Citations**: Display source attribution for answers

### 4. Upload Documents
Upload PDF files containing text, tables, charts, and images. The system will:
1. Extract all content modalities
2. Generate embeddings
3. Store in vector database
4. Enable citation-based QA

### 5. Ask Questions
Questions are answered with:
- Relevant context from all modalities
- Citations showing exact sources
- Conversation history maintained

## Evaluation Usage

```python
from ragbase.evaluator import MultiModalRAGEvaluator, create_benchmark_queries

# Initialize evaluator
evaluator = MultiModalRAGEvaluator(llm)

# Run evaluation suite
queries = create_benchmark_queries()
results = evaluator.run_evaluation_suite(
    queries=queries,
    retriever=retriever,
    llm_chain=chain,
    expected_modalities=["text", "table", "image_ocr"]
)

# Access results
for eval_result in results:
    print(f"Query: {eval_result.query}")
    for metric in eval_result.metrics:
        print(f"  {metric.name}: {metric.value:.2f}")
    print(f"  Modality Coverage: {eval_result.modality_coverage}")
```

## Key Files

| File | Purpose |
|------|---------|
| `multimodal_processor.py` | Multi-modal document extraction |
| `ingestor.py` | Enhanced document ingestion pipeline |
| `retriever.py` | Multi-modal aware retrieval |
| `chain.py` | Citation-aware QA chains |
| `evaluator.py` | Evaluation metrics and benchmarks |
| `model.py` | Multi-model LLM management |
| `app.py` | Streamlit UI |
| `config.py` | Configuration management |

## Supported Document Types

- **Text**: Plain paragraphs, headings, lists
- **Tables**: Multi-column data with headers
- **Images**: Diagrams, charts, scanned content (with OCR)
- **Metadata**: Footnotes, page references, sections

## Performance Characteristics

- **Embedding Model**: All-MiniLM-L6-v2 (384 dimensions)
- **Vector Store**: Qdrant (vector database)
- **Reranking**: FlashRank (MS MARCO model)
- **LLM Options**:
  - Local: Gemma2 9B via Ollama
  - Remote: Llama 3.1 70B or Mixtral 8x7B via Groq

## Future Enhancements

- [ ] Layout-aware PDF parsing (preserve document structure)
- [ ] Chart interpretation with vision models
- [ ] Named entity extraction from tables
- [ ] Document summarization per modality
- [ ] Confidence scores for retrieved chunks
- [ ] Multi-language support

## Dependencies

Core dependencies:
- `langchain`: LLM orchestration
- `langchain-qdrant`: Vector store integration
- `PyMuPDF`: PDF processing
- `pytesseract`: OCR
- `pdfplumber`: Table extraction
- `sentence-transformers`: Embeddings
- `streamlit`: Web UI
- `groq`: LLM API

See `requirements.txt` for complete list with versions.

## Troubleshooting

### OCR Not Working
- Ensure Tesseract is installed: `which tesseract`
- On Windows, download from: https://github.com/UB-Mannheim/tesseract/wiki
- Update path in code if needed: `pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`

### Missing API Keys
- Set `GROQ_API_KEY` in `.env` file for remote LLM access
- For local models, ensure Ollama is running

### Vector Store Issues
- Vector store persists in `docs-db/` directory
- Clear old data: `rm -rf docs-db/` then re-ingest documents

## Evaluation Results Example

```
Query: What are the key financial metrics?
Retrieved Docs: 5
  - Text chunks: 3
  - Table chunks: 1
  - Image chunks: 1

Metrics:
  Modality Diversity: 0.75
  Citation Completeness: 1.00
  Answer Relevance: 0.92
  Modality Coverage: 0.67

Answer:
"According to the IMF Article IV report, key metrics include:
1. GDP growth rate of 3.2% (from Table 2, page 15)
2. Inflation at 4.1% (from main text, page 8)
3. Trade surplus of $2.3B (from Chart 3, page 22)

[Source: IMF_Article_IV.pdf, Page 15, Section table_2, Modality: TABLE]
[Source: IMF_Article_IV.pdf, Page 8, Section text_0, Modality: TEXT]
[Source: IMF_Article_IV.pdf, Page 22, Section image_1, Modality: IMAGE_OCR]"
```

---

**Created**: December 2025  
**Status**: Fully Implemented ✅
