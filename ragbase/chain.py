import re
from operator import itemgetter
from typing import List, Dict

from langchain.schema.runnable import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableBranch
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.output_parsers import StrOutputParser

from ragbase.config import Config
from ragbase.session_history import get_session_history

SYSTEM_PROMPT = """
Utilize the provided contextual information to respond to the user question.
If the answer is not found within the context, state that the answer cannot be found.
Prioritize concise responses (maximum of 3 sentences) and use a list where applicable.
The contextual information is organized with the most relevant source appearing first.
Each source is separated by a horizontal rule (---).

Context:
{context}

Use markdown formatting where appropriate.
"""

SYSTEM_PROMPT_WITH_CITATIONS = """
You are a helpful document analyst. Answer user questions based on the provided context.
Always provide citations using [Source: Page X, Section Y, Modality: Z] format.
If information comes from multiple modalities (text, table, image), mention all sources.

Context:
{context}

Guidelines:
- Be accurate and concise (max 3 sentences)
- Always cite your sources with page and section info
- If answer not in context, clearly state so
- Use lists/tables for structured data
"""


def remove_links(text: str) -> str:
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, "", text)


def format_documents(documents: List[Document]) -> str:
    texts = []
    for doc in documents:
        texts.append(doc.page_content)
        texts.append("---")

    return remove_links("\n".join(texts))


def format_documents_with_citations(documents: List[Document]) -> str:
    """Format documents with multi-modal citations"""
    texts = []
    
    for doc in documents:
        page_num = doc.metadata.get("page_num", "?")
        section_id = doc.metadata.get("section_id", "")
        modality = doc.metadata.get("modality", "text")
        source_file = doc.metadata.get("source_file", "unknown")
        
        # Build citation
        citation = f"[Source: {source_file}, Page {page_num}, Section {section_id}, Modality: {modality.upper()}]"
        
        # Add content with citation
        texts.append(f"{doc.page_content}\n{citation}")
        texts.append("---")
    
    return remove_links("\n".join(texts))


def extract_citations(documents: List[Document]) -> List[Dict]:
    """Extract structured citation information from documents"""
    citations = []
    
    for i, doc in enumerate(documents):
        citation_info = {
            "index": i + 1,
            "source_file": doc.metadata.get("source_file", "unknown"),
            "page_num": doc.metadata.get("page_num", "?"),
            "section_id": doc.metadata.get("section_id", ""),
            "modality": doc.metadata.get("modality", "text"),
            "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        }
        citations.append(citation_info)
    
    return citations


def create_multi_model_response(llms: List[BaseLanguageModel], formatted_context: str, question: str) -> str:
    """Generate responses from multiple LLMs and combine them"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    
    responses = []
    for i, llm in enumerate(llms, 1):
        chain = prompt | llm
        try:
            response = chain.invoke({
                "context": formatted_context,
                "question": question
            })
            responses.append(f"**Model {i} Response:**\n{response.content}")
        except Exception as e:
            responses.append(f"**Model {i} Response:** Error - {str(e)}")
    
    return "\n\n---\n\n".join(responses)


def create_chain(llm: BaseLanguageModel, retriever: VectorStoreRetriever) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question")
            | retriever.with_config({"run_name": "context_retriever"})
            | format_documents
        )
        | prompt
        | llm
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    ).with_config({"run_name": "chain_answer"})


def create_citation_aware_chain(llm: BaseLanguageModel, retriever: VectorStoreRetriever) -> Runnable:
    """Create a chain that includes multi-modal citations"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_WITH_CITATIONS),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    def format_with_citations(question_dict):
        """Retrieve documents and format with citations"""
        question = question_dict.get("question", "")
        documents = retriever.invoke(question)
        formatted_context = format_documents_with_citations(documents)
        question_dict["context"] = formatted_context
        question_dict["citations"] = extract_citations(documents)
        return question_dict

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_documents_with_citations(
                retriever.invoke(x.get("question", ""))
            ),
            citations=lambda x: extract_citations(
                retriever.invoke(x.get("question", ""))
            )
        )
        | prompt
        | llm
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    ).with_config({"run_name": "citation_aware_chain"})


def create_multi_model_chain(llms: List[BaseLanguageModel], retriever: VectorStoreRetriever) -> Runnable:
    """Create a chain that uses multiple LLMs in ensemble"""
    def multi_model_llm(input_dict):
        """Wrapper to handle multiple LLMs"""
        from langchain_core.messages import AIMessage
        responses = create_multi_model_response(
            llms, 
            input_dict.get("context", ""), 
            input_dict.get("question", "")
        )
        return AIMessage(content=responses)
    
    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question")
            | retriever.with_config({"run_name": "context_retriever"})
            | format_documents
        )
        | RunnablePassthrough(input_dict=lambda x: x)
        | multi_model_llm
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    ).with_config({"run_name": "multi_model_chain_answer"})


async def ask_question(chain: Runnable, question: str, session_id: str):
    async for event in chain.astream_events(
        {"question": question},
        config={
            "callbacks": [ConsoleCallbackHandler()] if Config.DEBUG else [],
            "configurable": {"session_id": session_id},
        },
        version="v2",
        include_names=["context_retriever", "chain_answer"],
    ):
        event_type = event["event"]
        if event_type == "on_retriever_end":
            yield event["data"]["output"]
        if event_type == "on_chain_stream":
            yield event["data"]["chunk"].content
