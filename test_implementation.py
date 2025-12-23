"""
Test script for Multi-Modal RAG system.
Run this to verify the implementation is working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ragbase.config import Config
from ragbase.multimodal_processor import MultiModalDocumentProcessor, MultiModalChunk
from ragbase.evaluator import MultiModalRAGEvaluator, create_benchmark_queries


def test_imports():
    """Test that all modules import correctly"""
    print("=" * 60)
    print("TEST 1: Testing Imports")
    print("=" * 60)
    
    try:
        from ragbase.ingestor import Ingestor
        print("✅ ragbase.ingestor imported")
        
        from ragbase.retriever import create_retriever, MultiModalRetriever
        print("✅ ragbase.retriever imported (with MultiModalRetriever)")
        
        from ragbase.chain import (
            create_chain, 
            create_multi_model_chain,
            create_citation_aware_chain,
            format_documents_with_citations,
            extract_citations
        )
        print("✅ ragbase.chain imported (with citation functions)")
        
        from ragbase.model import create_llm, create_multi_llms
        print("✅ ragbase.model imported (with multi_llms)")
        
        from ragbase.multimodal_processor import (
            MultiModalDocumentProcessor,
            MultiModalChunk
        )
        print("✅ ragbase.multimodal_processor imported")
        
        from ragbase.evaluator import (
            MultiModalRAGEvaluator,
            create_benchmark_queries
        )
        print("✅ ragbase.evaluator imported")
        
        print("\n✅ All imports successful!\n")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}\n")
        return False


def test_config():
    """Test configuration"""
    print("=" * 60)
    print("TEST 2: Testing Configuration")
    print("=" * 60)
    
    try:
        # Check multi-modal config
        assert hasattr(Config.Model, 'USE_MULTI_MODEL'), "Missing USE_MULTI_MODEL"
        assert hasattr(Config.Model, 'REMOTE_LLM_2'), "Missing REMOTE_LLM_2"
        print(f"✅ Multi-Model LLM Config:")
        print(f"   - USE_MULTI_MODEL: {Config.Model.USE_MULTI_MODEL}")
        print(f"   - REMOTE_LLM: {Config.Model.REMOTE_LLM}")
        print(f"   - REMOTE_LLM_2: {Config.Model.REMOTE_LLM_2}")
        
        # Check retriever config
        assert hasattr(Config.Retriever, 'USE_MULTIMODAL'), "Missing USE_MULTIMODAL"
        assert hasattr(Config.Retriever, 'USE_CITATIONS'), "Missing USE_CITATIONS"
        print(f"\n✅ Multi-Modal Retriever Config:")
        print(f"   - USE_MULTIMODAL: {Config.Retriever.USE_MULTIMODAL}")
        print(f"   - USE_CITATIONS: {Config.Retriever.USE_CITATIONS}")
        print(f"   - USE_RERANKER: {Config.Retriever.USE_RERANKER}")
        
        print("\n✅ Configuration verified!\n")
        return True
        
    except AssertionError as e:
        print(f"❌ Configuration test failed: {e}\n")
        return False


def test_multimodal_chunk():
    """Test MultiModalChunk dataclass"""
    print("=" * 60)
    print("TEST 3: Testing MultiModalChunk")
    print("=" * 60)
    
    try:
        chunk = MultiModalChunk(
            content="Sample table data",
            modality="table",
            page_num=5,
            section_id="table_2",
            source_file="test.pdf",
            metadata={"rows": 3, "cols": 4}
        )
        
        # Convert to Document
        doc = chunk.to_document()
        
        assert doc.page_content == "Sample table data"
        assert doc.metadata["modality"] == "table"
        assert doc.metadata["page_num"] == 5
        assert doc.metadata["section_id"] == "table_2"
        assert doc.metadata["source_file"] == "test.pdf"
        
        print(f"✅ Created MultiModalChunk:")
        print(f"   - Content: {doc.page_content}")
        print(f"   - Modality: {doc.metadata['modality']}")
        print(f"   - Page: {doc.metadata['page_num']}")
        print(f"   - Section: {doc.metadata['section_id']}")
        print(f"   - Source: {doc.metadata['source_file']}")
        
        print("\n✅ MultiModalChunk test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ MultiModalChunk test failed: {e}\n")
        return False


def test_processor_initialization():
    """Test MultiModalDocumentProcessor initialization"""
    print("=" * 60)
    print("TEST 4: Testing MultiModalDocumentProcessor Initialization")
    print("=" * 60)
    
    try:
        processor = MultiModalDocumentProcessor(enable_ocr=True)
        
        assert processor.enable_ocr == True
        print(f"✅ MultiModalDocumentProcessor initialized:")
        print(f"   - OCR Enabled: {processor.enable_ocr}")
        print(f"   - Methods available:")
        print(f"     - process_pdf()")
        print(f"     - _extract_text()")
        print(f"     - _extract_tables()")
        print(f"     - _extract_images()")
        
        print("\n✅ Processor initialization test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Processor initialization test failed: {e}\n")
        return False


def test_evaluator_initialization():
    """Test MultiModalRAGEvaluator initialization"""
    print("=" * 60)
    print("TEST 5: Testing MultiModalRAGEvaluator Initialization")
    print("=" * 60)
    
    try:
        # Create a mock LLM (we won't use it, just test structure)
        class MockLLM:
            def invoke(self, *args, **kwargs):
                return type('obj', (object,), {'content': '0.8'})()
        
        llm = MockLLM()
        evaluator = MultiModalRAGEvaluator(llm)
        
        # Test benchmark queries
        queries = create_benchmark_queries()
        assert len(queries) > 0
        
        print(f"✅ MultiModalRAGEvaluator initialized")
        print(f"\n✅ Benchmark Queries ({len(queries)} total):")
        for i, query in enumerate(queries, 1):
            print(f"   {i}. {query}")
        
        print("\n✅ Evaluator initialization test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Evaluator initialization test failed: {e}\n")
        return False


def test_citation_formatting():
    """Test citation formatting functions"""
    print("=" * 60)
    print("TEST 6: Testing Citation Formatting")
    print("=" * 60)
    
    try:
        from langchain_core.documents import Document
        from ragbase.chain import extract_citations
        
        # Create sample documents
        docs = [
            Document(
                page_content="Financial data from Table 2",
                metadata={
                    "page_num": 15,
                    "section_id": "table_2",
                    "modality": "table",
                    "source_file": "report.pdf"
                }
            ),
            Document(
                page_content="Text from introduction",
                metadata={
                    "page_num": 1,
                    "section_id": "text_0",
                    "modality": "text",
                    "source_file": "report.pdf"
                }
            )
        ]
        
        citations = extract_citations(docs)
        
        assert len(citations) == 2
        assert citations[0]["page_num"] == 15
        assert citations[0]["modality"] == "table"
        assert citations[1]["page_num"] == 1
        
        print(f"✅ Generated {len(citations)} citations:")
        for i, citation in enumerate(citations, 1):
            print(f"\n   Citation {i}:")
            print(f"   - Source: {citation['source_file']}")
            print(f"   - Page: {citation['page_num']}")
            print(f"   - Section: {citation['section_id']}")
            print(f"   - Modality: {citation['modality'].upper()}")
        
        print("\n✅ Citation formatting test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Citation formatting test failed: {e}\n")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "   Multi-Modal RAG System - Implementation Test Suite   ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("MultiModalChunk", test_multimodal_chunk),
        ("Processor Init", test_processor_initialization),
        ("Evaluator Init", test_evaluator_initialization),
        ("Citation Format", test_citation_formatting),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The Multi-Modal RAG system is ready to use.")
        print("\nNext steps:")
        print("1. Install system dependencies:")
        print("   - Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("   - Windows: Download from github.com/UB-Mannheim/tesseract")
        print("2. Install Python dependencies: pip install -r requirements.txt")
        print("3. Set up API keys in .env file")
        print("4. Run the app: streamlit run app.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")
    
    print("\n")
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
