from typing import Optional, List

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_filter import LLMChainFilter
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_core.documents import Document
from langchain_qdrant import Qdrant

from ragbase.config import Config
from ragbase.model import create_embeddings, create_reranker


class MultiModalRetriever(VectorStoreRetriever):
    """Enhanced retriever that handles multi-modal documents"""
    
    def invoke(self, input, config=None):
        """Override to filter results by modality if needed"""
        results = super().invoke(input, config)
        
        # Group results by modality for better context
        modality_groups = {}
        for doc in results:
            modality = doc.metadata.get("modality", "text")
            if modality not in modality_groups:
                modality_groups[modality] = []
            modality_groups[modality].append(doc)
        
        # Reorder: prioritize text, then tables, then images
        modality_priority = ["text", "table", "image_ocr"]
        reordered = []
        for modality in modality_priority:
            if modality in modality_groups:
                reordered.extend(modality_groups[modality])
        
        # Add any remaining modalities
        for modality in modality_groups:
            if modality not in modality_priority:
                reordered.extend(modality_groups[modality])
        
        return reordered


def create_retriever(
    llm: BaseLanguageModel, 
    vector_store: Optional[VectorStore] = None,
    use_multimodal: bool = True
) -> VectorStoreRetriever:
    """Create a retriever with optional multi-modal support"""
    if not vector_store:
        vector_store = Qdrant.from_existing_collection(
            embedding=create_embeddings(),
            collection_name=Config.Database.DOCUMENTS_COLLECTION,
            path=Config.Path.DATABASE_DIR,
        )

    if use_multimodal:
        retriever = MultiModalRetriever(
            vectorstore=vector_store,
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    else:
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

    if Config.Retriever.USE_RERANKER:
        retriever = ContextualCompressionRetriever(
            base_compressor=create_reranker(), base_retriever=retriever
        )

    if Config.Retriever.USE_CHAIN_FILTER:
        retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainFilter.from_llm(llm), base_retriever=retriever
        )

    return retriever
