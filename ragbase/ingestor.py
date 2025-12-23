from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFium2Loader
from sentence_transformers import SentenceTransformer
from typing import Iterable
from langchain_core.embeddings import Embeddings


class SentenceTransformersEmbeddings(Embeddings):
    """Lightweight wrapper around `sentence_transformers.SentenceTransformer`.

    Provides the `embed_documents` and `embed_query` methods expected by
    vectorstore constructors like `Qdrant.from_documents`.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        embs = self.model.encode(list(texts), show_progress_bar=False)
        # Ensure a list of lists
        return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embs]

    def embed_query(self, text: str) -> list[float]:
        emb = self.model.encode([text], show_progress_bar=False)[0]
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def __call__(self, texts):
        """Callable interface for backwards compatibility with older vectorstore APIs.

        Accepts a single string or an iterable of strings.
        """
        if isinstance(texts, str):
            return self.embed_query(texts)
        try:
            return self.embed_documents(texts)
        except TypeError:
            return self.embed_query(str(texts))
from langchain_core.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbase.config import Config
from ragbase.multimodal_processor import MultiModalDocumentProcessor

class Ingestor:
    def __init__(self, use_multimodal: bool = True):
        # Initialize embeddings with error handling
        try:
            # Use sentence-transformers locally to avoid fastembed HF download issues
            self.embeddings = SentenceTransformersEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Initialize text splitters
            self.semantic_splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="interquartile"
            )
            
            self.recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2048,
                chunk_overlap=128,
                add_start_index=True,
            )
            
            # Initialize multi-modal processor
            self.use_multimodal = use_multimodal
            if use_multimodal:
                self.mm_processor = MultiModalDocumentProcessor(enable_ocr=True)
            else:
                self.mm_processor = None
            
        except Exception as e:
            print(f"Error initializing embeddings: {str(e)}")
            raise

    def ingest(self, doc_paths: List[Path]) -> VectorStore:
        try:
            documents = []
            
            if self.use_multimodal:
                # Use multi-modal processing
                for doc_path in doc_paths:
                    mm_chunks = self.mm_processor.process_pdf(str(doc_path))
                    # Convert multi-modal chunks to documents
                    for chunk in mm_chunks:
                        documents.append(chunk.to_document())
            else:
                # Fallback to original text-only processing
                for doc_path in doc_paths:
                    loaded_documents = PyPDFium2Loader(str(doc_path)).load()
                    document_text = "\n".join(
                        [doc.page_content for doc in loaded_documents]
                    )
                    semantic_chunks = self.semantic_splitter.create_documents([document_text])
                    final_chunks = self.recursive_splitter.split_documents(semantic_chunks)
                    documents.extend(final_chunks)
            
            # Create and return Qdrant vector store
            return Qdrant.from_documents(
                documents=documents,
                embedding=self.embeddings,
                path=Config.Path.DATABASE_DIR,
                collection_name=Config.Database.DOCUMENTS_COLLECTION,
            )
            
        except Exception as e:
            print(f"Error in document ingestion: {str(e)}")
            raise