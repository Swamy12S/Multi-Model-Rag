# Multi-Modal RAG System - Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# System dependency for OCR (IMPORTANT!)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows:
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Or use: choco install tesseract
```

### 2. Set Up API Keys
Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from: https://console.groq.com

### 3. Verify Installation
```bash
python test_implementation.py
```

You should see:
```
✅ PASS - Imports
✅ PASS - Configuration
✅ PASS - MultiModalChunk
✅ PASS - Processor Init
✅ PASS - Evaluator Init
✅ PASS - Citation Format

🎉 All tests passed! The Multi-Modal RAG system is ready to use.
```

### 4. Run the Application
```bash
streamlit run app.py
```

The Streamlit app will open at `http://localhost:8501`

---

## 📝 Usage

### In the Streamlit Interface:

1. **Upload PDFs**
   - Click "Upload PDF files" button
   - Select one or more PDF documents

2. **Configure Options**
   - ☑️ Enable Multi-Model LLM (compare responses from multiple LLMs)
   - ☑️ Enable Multi-Modal Processing (extract text, tables, images, OCR)
   - ☑️ Show Citations (display sources for answers)

3. **Ask Questions**
   - Type your question in the chat input
   - The system will search across all document modalities
   - Answers include citations showing:
     - Source document
     - Page number
     - Section ID
     - Modality type (TEXT, TABLE, IMAGE_OCR)

### Example Queries:
```
"What are the main financial metrics?"
"What does Table 3 show?"
"Summarize the key findings from images/charts"
"What recommendations are made?"
```

---

## 🎯 Features Explained

### Multi-Modal Processing
The system extracts and processes:
- **TEXT**: Paragraphs, headings, lists
- **TABLES**: Converts to Markdown format with headers and rows
- **IMAGES**: Uses OCR to extract text from diagrams and charts
- **METADATA**: Maintains page numbers, sections, file references

### Citation System
Every answer includes sources:
```
[Source: IMF_Article_IV.pdf, Page 15, Section table_2, Modality: TABLE]
```

### Multi-Model LLM
When enabled, queries multiple LLMs simultaneously:
- **Model 1**: Llama 3.1 (70B)
- **Model 2**: Mixtral (8x7B)
- Compare responses for accuracy and completeness

---

## 📊 Evaluation

Run the evaluation suite to benchmark your system:

```python
from ragbase.evaluator import MultiModalRAGEvaluator, create_benchmark_queries
from ragbase.model import create_llm
from ragbase.ingestor import Ingestor
from ragbase.retriever import create_retriever
from ragbase.chain import create_citation_aware_chain

# Setup
llm = create_llm()
ingestor = Ingestor(use_multimodal=True)
vector_store = ingestor.ingest(["your_document.pdf"])
retriever = create_retriever(llm, vector_store, use_multimodal=True)
chain = create_citation_aware_chain(llm, retriever)

# Evaluate
evaluator = MultiModalRAGEvaluator(llm)
queries = create_benchmark_queries()
results = evaluator.run_evaluation_suite(queries, retriever, chain)

# View results
for result in results:
    print(f"Query: {result.query}")
    print(f"Modality Coverage: {result.modality_coverage}")
    for metric in result.metrics:
        print(f"  {metric.name}: {metric.value:.2f}")
```

---

## 🔧 Configuration

Edit `ragbase/config.py` to customize:

```python
class Model:
    LOCAL_LLM = "gemma2:9b"              # Local model via Ollama
    REMOTE_LLM = "llama-3.1-70b-versatile"  # Primary remote
    REMOTE_LLM_2 = "mixtral-8x7b-32768"    # Secondary remote
    USE_LOCAL = False                   # Use local or remote?
    USE_MULTI_MODEL = True              # Enable ensemble?
    TEMPERATURE = 0.0
    MAX_TOKENS = 8000

class Retriever:
    USE_RERANKER = True                 # Rerank results?
    USE_MULTIMODAL = True               # Enable multi-modal?
    USE_CITATIONS = True                # Show citations?
```

---

## 📂 Project Structure

```
Multi_model-RAG/
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── test_implementation.py           # Verification tests
├── .env                           # API keys (create this)
├── ragbase/
│   ├── __init__.py
│   ├── config.py                  # Configuration
│   ├── model.py                   # LLM management
│   ├── ingestor.py                # Document ingestion
│   ├── multimodal_processor.py    # Multi-modal extraction
│   ├── retriever.py               # Vector retrieval
│   ├── chain.py                   # QA chains
│   ├── evaluator.py               # Evaluation metrics
│   ├── session_history.py         # Chat history
│   └── uploader.py                # File upload handling
├── models/                        # Pre-downloaded models
├── docs-db/                       # Vector database (auto-created)
└── images/                        # UI assets
```

---

## 🆘 Troubleshooting

### "No module named 'pytesseract'"
```bash
pip install pytesseract
# Also install Tesseract system package (see Setup section)
```

### "OCR not working"
Verify Tesseract installation:
```bash
which tesseract          # Linux/Mac
where tesseract         # Windows
```

If not found, reinstall from: https://github.com/UB-Mannheim/tesseract/wiki

### "GROQ_API_KEY not found"
1. Create `.env` file in project root
2. Add: `GROQ_API_KEY=your_key_here`
3. Get key from: https://console.groq.com

### Vector database issues
Clear old data and reingest:
```bash
rm -rf docs-db/
# Then upload new documents in the UI
```

### Out of memory
Reduce chunk size in `config.py`:
```python
# In Ingestor.__init__
self.recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,  # Reduce from 2048
    chunk_overlap=64,
)
```

---

## 📚 Documentation

- **Full Details**: See `MULTIMODAL_IMPLEMENTATION.md`
- **Summary of Changes**: See `IMPLEMENTATION_SUMMARY.md`
- **Code Examples**: See docstrings in each module

---

## 🎓 Example: Processing an IMF Report

1. **Upload** an IMF Article IV report PDF
2. **System extracts**:
   - Economic text (Modality: TEXT)
   - Financial tables (Modality: TABLE)
   - Graphs/charts with OCR (Modality: IMAGE_OCR)

3. **Ask questions**:
   - "What's the GDP growth rate?" → Finds in text
   - "Show me Table 1" → Extracts table data
   - "What do the charts show?" → OCR'd image analysis

4. **Get answers with citations**:
   ```
   GDP growth is 3.2% [Source: IMF_2024.pdf, Page 8, 
   Section text_0, Modality: TEXT]
   
   Key metrics:
   - Inflation: 4.1% [Source: IMF_2024.pdf, Page 15, 
     Section table_2, Modality: TABLE]
   - Trade surplus: $2.3B [Source: IMF_2024.pdf, Page 22,
     Section image_1, Modality: IMAGE_OCR]
   ```

---

## ✅ Verification Checklist

Before running in production:

- [ ] `test_implementation.py` passes all tests
- [ ] API keys configured in `.env`
- [ ] Tesseract installed and accessible
- [ ] Test document uploaded successfully
- [ ] Multi-modal processing working (tables and images extracted)
- [ ] Citations appearing in answers
- [ ] Conversation history maintained across queries

---

## 🚀 Next Steps

1. **Deploy**: Run `streamlit run app.py` for production use
2. **Customize**: Modify prompts in `ragbase/chain.py`
3. **Extend**: Add more LLMs in `ragbase/model.py`
4. **Optimize**: Adjust chunk sizes and retrieval parameters
5. **Monitor**: Track metrics with evaluation suite

---

**Ready to use! 🎉**

For questions or issues, check the full documentation in `MULTIMODAL_IMPLEMENTATION.md`
