"""Re-ranking component for better relevance."""
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json


class DocumentReranker:
    """Re-rank retrieved documents for better relevance."""
    
    def __init__(self, config):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=config.llm_model,
            temperature=0,
            google_api_key=config.google_api_key
        )
        
        self.rerank_prompt = PromptTemplate(
            input_variables=["question", "documents"],
            template="""Bạn là chuyên gia đánh giá độ liên quan của tài liệu. Hãy xếp hạng các đoạn văn theo mức độ liên quan đến câu hỏi.

Câu hỏi: {question}

Các đoạn văn:
{documents}

Hãy đánh giá từng đoạn văn (0-10 điểm) và sắp xếp theo thứ tự giảm dần. Chỉ trả lời JSON theo format:
{{"rankings": [{{"index": số_thứ_tự, "score": điểm_số, "reasoning": "lý do ngắn gọn"}}]}}"""
        )
    
    def rerank_documents(self, question: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank documents based on relevance to the question."""
        if not documents:
            return []
        
        # Take only top documents for re-ranking to save tokens
        docs_to_rerank = documents[:min(len(documents), 10)]
        
        try:
            # Format documents for prompt
            doc_text = ""
            for i, doc in enumerate(docs_to_rerank):
                doc_text += f"\n{i+1}. {doc['content'][:500]}...\n"
            
            response = self.llm.invoke(
                self.rerank_prompt.format(
                    question=question,
                    documents=doc_text
                )
            )
            
            # Parse rankings
            result = json.loads(response.content.strip())
            rankings = result.get("rankings", [])
            
            # Apply rankings
            reranked_docs = []
            for ranking in rankings:
                idx = ranking["index"] - 1  # Convert to 0-based index
                if 0 <= idx < len(docs_to_rerank):
                    doc = docs_to_rerank[idx].copy()
                    doc["rerank_score"] = ranking["score"]
                    doc["rerank_reasoning"] = ranking["reasoning"]
                    reranked_docs.append(doc)
            
            # Sort by rerank score
            reranked_docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            # Return top K after re-ranking
            return reranked_docs[:self.config.rerank_top_k]
            
        except Exception as e:
            print(f"Re-ranking error: {e}")
            # Fallback: return documents sorted by original similarity
            return sorted(
                docs_to_rerank,
                key=lambda x: x.get("max_similarity", 0),
                reverse=True
            )[:self.config.rerank_top_k]