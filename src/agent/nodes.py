"""Individual nodes for the RAG workflow."""
from typing import Dict, Any
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from agent.components.query_transformer import QueryTransformer
from agent.components.router import IntelligentRouter
from agent.components.retriever import EnhancedRetriever
from agent.components.reranker import DocumentReranker
from agent.components.evaluator import AnswerEvaluator
from agent.config import Configuration
from agent.embeddings import DocumentProcessor
from .state import RAGState


class RAGNodes:
    """Collection of workflow nodes for RAG system."""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=config.llm_model,
            temperature=config.temperature,
            google_api_key=config.google_api_key
        )
        
        # Initialize components
        self.query_transformer = QueryTransformer(config)
        self.router = IntelligentRouter(config)
        self.retriever = EnhancedRetriever(config)
        self.reranker = DocumentReranker(config)
        self.evaluator = AnswerEvaluator(config)
        self.document_processor = DocumentProcessor(config)
        
        # Answer generation prompt
        self.answer_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""Bạn là một trợ lý AI hữu ích. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp.

Câu hỏi: {question}

Thông tin tham khảo:
{context}

Hướng dẫn:
1. Trả lời chính xác dựa trên thông tin đã cho
2. Nếu không có đủ thông tin, hãy nói rõ
3. Trích dẫn nguồn khi cần thiết
4. Trả lời bằng tiếng Việt rõ ràng, dễ hiểu

Câu trả lời:"""
        )
    
    def detect_and_load_urls_node(self, state: RAGState) -> RAGState:
        """Detect URLs in question and load documents from them."""
        try:
            question = state["messages"]
            
            # Extract URLs from question using regex
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, question)
            
            if not urls:
                # No URLs found, continue with normal flow
                return {
                    **state,
                    "metadata": {**state.get("metadata", {}), "step": "no_urls_detected"}
                }
            
            print(f"🔗 Phát hiện {len(urls)} URL trong câu hỏi")
            for url in urls:
                print(f"  - {url}")
            
            # Load documents from detected URLs
            loaded_docs = self.document_processor.load_documents_from_urls(urls)
            
            if loaded_docs:
                # Add documents to retriever
                succ = self.retriever.add_documents(loaded_docs)
                print(f"✅ Đã tải và lưu trữ {len(loaded_docs)} đoạn văn từ URL")
                
                # Clean question by removing URLs for better processing
                clean_question = question
                for url in urls:
                    clean_question = clean_question.replace(url, "").strip()
                
                # If question becomes empty after URL removal, create a general question
                if not clean_question or len(clean_question.split()) < 3:
                    clean_question = f"Hãy tóm tắt nội dung chính từ các tài liệu đã được tải."
                
                return {
                    **state,
                    "messages": clean_question,  # Update question without URLs
                    "metadata": {
                        **state.get("metadata", {}), 
                        "step": "urls_loaded_and_indexed",
                        "loaded_urls": urls,
                        "loaded_docs": loaded_docs,
                        "original_question": question,
                        "retriever_update_success": succ
                    }
                }
            else:
                return {
                    **state,
                    "error": f"Không thể tải tài liệu từ URL: {urls}",
                    "metadata": {
                        **state.get("metadata", {}), 
                        "step": "url_loading_failed",
                        "failed_urls": urls
                    }
                }
                
        except Exception as e:
            return {
                **state,
                "error": f"URL detection/loading failed: {str(e)}",
                "metadata": {**state.get("metadata", {}), "step": "url_detection_error"}
            }
    
    def transform_query_node(self, state: RAGState) -> RAGState:
        """Transform user query into multiple variants."""
        try:
            transformed_queries = self.query_transformer.transform_query(state["messages"])
            return {
                **state,
                "transformed_queries": transformed_queries,
                "metadata": {**state.get("metadata", {}), "step": "query_transformed"}
            }
        except Exception as e:
            return {
                **state,
                "transformed_queries": [state["messages"]],
                "error": f"Query transformation failed: {str(e)}",
                "metadata": {**state.get("metadata", {}), "step": "query_transform_error"}
            }
    
    def route_query_node(self, state: RAGState) -> RAGState:
        """Route query to appropriate processing path."""
        try:
            route = self.router.route_query(state["messages"])
            return {
                **state,
                "route": route,
                "metadata": {**state.get("metadata", {}), "step": "query_routed", "route": route}
            }
        except Exception as e:
            return {
                **state,
                "route": "vectorstore",  # Default fallback
                "error": f"Routing failed: {str(e)}",
                "metadata": {**state.get("metadata", {}), "step": "routing_error"}
            }
    
    def retrieve_documents_node(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents from vector store."""
        try:
            queries = state.get("transformed_queries", [state["messages"]])
            documents = self.retriever.retrieve_documents(queries)
            
            return {
                **state,
                "documents": documents,
                "metadata": {
                    **state.get("metadata", {}), 
                    "step": "documents_retrieved",
                    "num_documents": len(documents)
                }
            }
        except Exception as e:
            return {
                **state,
                "documents": [],
                "error": f"Document retrieval failed: {str(e)}",
                "metadata": {**state.get("metadata", {}), "step": "retrieval_error"}
            }
    
    def rerank_documents_node(self, state: RAGState) -> RAGState:
        """Re-rank retrieved documents for better relevance."""
        try:
            documents = state.get("documents", [])
            if not documents:
                return {
                    **state,
                    "ranked_documents": [],
                    "metadata": {**state.get("metadata", {}), "step": "no_documents_to_rerank"}
                }
            
            ranked_documents = self.reranker.rerank_documents(state["messages"], documents)
            
            return {
                **state,
                "ranked_documents": ranked_documents,
                "metadata": {
                    **state.get("metadata", {}), 
                    "step": "documents_reranked",
                    "num_ranked": len(ranked_documents)
                }
            }
        except Exception as e:
            return {
                **state,
                "ranked_documents": state.get("documents", [])[:self.config.rerank_top_k],
                "error": f"Document re-ranking failed: {str(e)}",
                "metadata": {**state.get("metadata", {}), "step": "reranking_error"}
            }
    
    def generate_answer_node(self, state: RAGState) -> RAGState:
        """Generate answer based on retrieved documents."""
        try:
            documents = state.get("ranked_documents", [])
            
            # Handle different routes
            if state.get("route") == "direct_response":
                return self._generate_direct_response(state)
            elif state.get("route") == "websearch":
                return self._generate_websearch_response(state)
            else:  # vectorstore route
                return self._generate_rag_response(state, documents)
                
        except Exception as e:
            return {
                **state,
                "answer": f"Xin lỗi, tôi không thể trả lời câu hỏi này do lỗi hệ thống: {str(e)}",
                "error": f"Answer generation failed: {str(e)}",
                "metadata": {**state.get("metadata", {}), "step": "generation_error"}
            }
    
    def evaluate_answer_node(self, state: RAGState) -> RAGState:
        """Evaluate answer quality and determine if retry is needed."""
        try:
            answer = state.get("answer", "")
            documents = state.get("ranked_documents", [])
            
            quality_score, needs_retry, feedback = self.evaluator.evaluate_answer(
                state["messages"], answer, documents
            )
            
            # Check retry count to prevent infinite loops
            retry_count = state.get("metadata", {}).get("retry_count", 0)
            should_retry = needs_retry and retry_count < self.config.max_retries
            
            return {
                **state,
                "quality_score": quality_score,
                "needs_retry": should_retry,
                "metadata": {
                    **state.get("metadata", {}),
                    "step": "answer_evaluated",
                    "retry_count": retry_count + (1 if should_retry else 0),
                    "evaluation_feedback": feedback
                }
            }
        except Exception as e:
            return {
                **state,
                "quality_score": 0.5,  # Neutral score
                "needs_retry": False,   # Don't retry on evaluation error
                "error": f"Answer evaluation failed: {str(e)}",
                "metadata": {**state.get("metadata", {}), "step": "evaluation_error"}
            }
    
    def improve_answer_node(self, state: RAGState) -> RAGState:
        """Improve answer based on evaluation feedback."""
        try:
            feedback = state.get("metadata", {}).get("evaluation_feedback", "")
            improvement_prompt = self.evaluator.generate_improvement_prompt(
                feedback, state["messages"]
            )
            
            # Use context from ranked documents
            documents = state.get("ranked_documents", [])
            context = self._format_context(documents)
            
            full_prompt = f"{improvement_prompt}\n\nThông tin tham khảo:\n{context}"
            
            response = self.llm.invoke(full_prompt)
            improved_answer = response.content.strip()
            
            return {
                **state,
                "answer": improved_answer,
                "metadata": {
                    **state.get("metadata", {}),
                    "step": "answer_improved",
                    "improvement_attempt": True
                }
            }
        except Exception as e:
            return {
                **state,
                "error": f"Answer improvement failed: {str(e)}",
                "needs_retry": False,  # Stop retrying on improvement error
                "metadata": {**state.get("metadata", {}), "step": "improvement_error"}
            }
    
    def _generate_direct_response(self, state: RAGState) -> RAGState:
        """Generate direct response without document retrieval."""
        response = self.llm.invoke(f"Trả lời câu hỏi sau một cách ngắn gọn: {state['messages']}")
        
        return {
            **state,
            "answer": response.content.strip(),
            "metadata": {**state.get("metadata", {}), "step": "direct_response_generated"}
        }
    
    def _generate_websearch_response(self, state: RAGState) -> RAGState:
        """Generate response indicating web search is needed."""
        return {
            **state,
            "answer": "Câu hỏi này cần thông tin cập nhật từ web. Tính năng tìm kiếm web sẽ được triển khai trong phiên bản tiếp theo.",
            "metadata": {**state.get("metadata", {}), "step": "websearch_placeholder"}
        }
    
    def _generate_rag_response(self, state: RAGState, documents: list) -> RAGState:
        """Generate RAG-based response using retrieved documents."""
        if not documents:
            return {
                **state,
                "answer": "Tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu để trả lời câu hỏi này.",
                "metadata": {**state.get("metadata", {}), "step": "no_relevant_documents"}
            }
        
        context = self._format_context(documents)
        
        response = self.llm.invoke(
            self.answer_prompt.format(
                question=state["messages"],
                context=context
            )
        )
        
        return {
            **state,
            "answer": response.content.strip(),
            "metadata": {
                **state.get("metadata", {}), 
                "step": "rag_answer_generated",
                "context_length": len(context)
            }
        }
    
    def _format_context(self, documents: list) -> str:
        """Format documents into context string."""
        if not documents:
            return "Không có thông tin tham khảo."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "Không rõ nguồn")
            context_parts.append(f"{i}. {content}\n   (Nguồn: {source})")
        
        return "\n\n".join(context_parts)