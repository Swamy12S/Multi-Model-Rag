"""
Evaluation suite for multi-modal RAG system.
Tests retrieval quality, citation accuracy, and answer relevance.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import ChatPromptTemplate


class ModalityType(Enum):
    """Types of document modalities"""
    TEXT = "text"
    TABLE = "table"
    IMAGE_OCR = "image_ocr"


@dataclass
class EvaluationMetric:
    """Container for evaluation metric"""
    name: str
    value: float
    description: str


@dataclass
class QueryEvaluation:
    """Results for a single query evaluation"""
    query: str
    retrieved_docs: List[Document]
    answer: str
    metrics: List[EvaluationMetric]
    modality_coverage: Dict[str, int]


class MultiModalRAGEvaluator:
    """Evaluate multi-modal RAG system performance"""
    
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
    
    def evaluate_retrieval_quality(
        self, 
        query: str, 
        retrieved_docs: List[Document],
        expected_modalities: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate quality of retrieved documents
        
        Metrics:
        - Document relevance (0-1)
        - Modality diversity (0-1)
        - Citation completeness (0-1)
        """
        metrics = {}
        
        # 1. Check modality diversity
        modality_count = self._count_modalities(retrieved_docs)
        modality_score = len(modality_count) / 4  # Max 4 modalities
        metrics["modality_diversity"] = min(modality_score, 1.0)
        
        # 2. Check citation completeness
        citation_score = self._check_citation_completeness(retrieved_docs)
        metrics["citation_completeness"] = citation_score
        
        # 3. Check if expected modalities are present
        if expected_modalities:
            found_modalities = set(modality_count.keys())
            expected_set = set(expected_modalities)
            coverage = len(found_modalities & expected_set) / len(expected_set)
            metrics["modality_coverage"] = coverage
        
        return metrics
    
    def evaluate_answer_relevance(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Document]
    ) -> float:
        """
        Evaluate if answer is grounded in retrieved context
        Returns score 0-1
        """
        if not retrieved_docs:
            return 0.0
        
        # Build evaluation prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an evaluation expert. Rate how well the answer 
            is grounded in the provided context.
            
            Rate on scale 0-1:
            - 0: Answer contradicts or is unrelated to context
            - 0.5: Answer partially matches context
            - 1: Answer is well-supported by context
            
            Respond with just the score as a decimal."""),
            ("human", """Query: {query}
            
            Context:
            {context}
            
            Answer:
            {answer}
            
            Relevance score:""")
        ])
        
        context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        
        chain = prompt | self.llm
        try:
            response = chain.invoke({
                "query": query,
                "context": context,
                "answer": answer
            })
            score = float(response.content.strip())
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5
    
    def evaluate_citation_accuracy(
        self,
        retrieved_docs: List[Document],
        answer: str
    ) -> Dict[str, any]:
        """
        Evaluate if citations match retrieved documents
        """
        results = {
            "has_citations": self._has_citations(answer),
            "citation_count": self._count_citations(answer),
            "doc_count": len(retrieved_docs),
            "citation_to_doc_ratio": 0.0
        }
        
        if results["doc_count"] > 0:
            results["citation_to_doc_ratio"] = results["citation_count"] / results["doc_count"]
        
        return results
    
    def evaluate_multimodal_coverage(
        self,
        retrieved_docs: List[Document]
    ) -> Dict[str, int]:
        """Count document coverage by modality"""
        return self._count_modalities(retrieved_docs)
    
    def run_evaluation_suite(
        self,
        queries: List[str],
        retriever,
        llm_chain,
        expected_modalities: List[str] = None
    ) -> List[QueryEvaluation]:
        """
        Run complete evaluation suite on multiple queries
        """
        results = []
        
        for query in queries:
            # Retrieve documents
            retrieved_docs = retriever.invoke(query)
            
            # Get answer
            answer_result = llm_chain.invoke({
                "question": query,
                "chat_history": []
            })
            answer = answer_result.content if hasattr(answer_result, 'content') else str(answer_result)
            
            # Evaluate
            retrieval_metrics = self.evaluate_retrieval_quality(
                query, retrieved_docs, expected_modalities
            )
            relevance_score = self.evaluate_answer_relevance(query, answer, retrieved_docs)
            citation_info = self.evaluate_citation_accuracy(retrieved_docs, answer)
            modality_coverage = self.evaluate_multimodal_coverage(retrieved_docs)
            
            # Build metric objects
            metrics = [
                EvaluationMetric("Modality Diversity", retrieval_metrics.get("modality_diversity", 0), 
                               "Diversity of document types retrieved"),
                EvaluationMetric("Citation Completeness", retrieval_metrics.get("citation_completeness", 0),
                               "Completeness of source citations"),
                EvaluationMetric("Answer Relevance", relevance_score,
                               "How well answer is grounded in context"),
            ]
            
            if "modality_coverage" in retrieval_metrics:
                metrics.append(
                    EvaluationMetric("Modality Coverage", retrieval_metrics["modality_coverage"],
                                   "Coverage of expected document types")
                )
            
            results.append(QueryEvaluation(
                query=query,
                retrieved_docs=retrieved_docs,
                answer=answer,
                metrics=metrics,
                modality_coverage=modality_coverage
            ))
        
        return results
    
    # Helper methods
    @staticmethod
    def _count_modalities(docs: List[Document]) -> Dict[str, int]:
        """Count documents by modality"""
        counts = {}
        for doc in docs:
            modality = doc.metadata.get("modality", "unknown")
            counts[modality] = counts.get(modality, 0) + 1
        return counts
    
    @staticmethod
    def _check_citation_completeness(docs: List[Document]) -> float:
        """Check if all documents have required metadata"""
        if not docs:
            return 0.0
        
        required_fields = ["page_num", "section_id", "modality"]
        complete_count = 0
        
        for doc in docs:
            if all(field in doc.metadata for field in required_fields):
                complete_count += 1
        
        return complete_count / len(docs)
    
    @staticmethod
    def _has_citations(text: str) -> bool:
        """Check if answer contains citations"""
        return "[Source:" in text or "[source:" in text
    
    @staticmethod
    def _count_citations(text: str) -> int:
        """Count citations in answer"""
        return text.count("[Source:") + text.count("[source:")


def create_benchmark_queries() -> List[str]:
    """Create benchmark queries for evaluation"""
    return [
        "What are the key financial metrics mentioned in the document?",
        "Are there any tables or charts? What do they show?",
        "What are the main policy recommendations?",
        "Summarize the document's findings in one sentence.",
        "What images or diagrams are included and what do they illustrate?",
    ]
