"""Self-correction and evaluation component."""
from typing import Dict, Any, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json


class AnswerEvaluator:
    """Evaluate answer quality and determine if retry is needed."""
    
    def __init__(self, config):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=config.llm_model,
            temperature=0,
            google_api_key=config.google_api_key
        )
        
        self.evaluation_prompt = PromptTemplate(
            input_variables=["question", "answer", "documents"],
            template="""Bạn là chuyên gia đánh giá chất lượng câu trả lời. Hãy đánh giá câu trả lời dựa trên các tiêu chí sau:

Câu hỏi: {question}

Câu trả lời: {answer}

Tài liệu tham khảo:
{documents}

Tiêu chí đánh giá:
1. Độ chính xác (0-10): Câu trả lời có chính xác không?
2. Độ liên quan (0-10): Câu trả lời có trả lời đúng câu hỏi không?
3. Tính đầy đủ (0-10): Câu trả lời có đủ thông tin không?
4. Sử dụng nguồn (0-10): Có sử dụng tốt thông tin từ tài liệu không?
5. Độ rõ ràng (0-10): Câu trả lời có dễ hiểu không?

Trả lời theo format JSON:
{{
    "accuracy": điểm_số,
    "relevance": điểm_số,
    "completeness": điểm_số,
    "source_usage": điểm_số,
    "clarity": điểm_số,
    "overall_score": điểm_trung_bình,
    "needs_improvement": true/false,
    "feedback": "nhận xét chi tiết"
}}"""
        )
    
    def evaluate_answer(
        self, 
        question: str, 
        answer: str, 
        documents: list
    ) -> Tuple[float, bool, str]:
        """
        Evaluate answer quality.
        Returns: (quality_score, needs_retry, feedback)
        """
        try:
            # Format documents for evaluation
            doc_text = ""
            for i, doc in enumerate(documents[:3]):  # Use top 3 docs
                doc_text += f"\n{i+1}. {doc.get('content', '')[:300]}...\n"
            
            response = self.llm.invoke(
                self.evaluation_prompt.format(
                    question=question,
                    answer=answer,
                    documents=doc_text
                )
            )
            
            # Parse evaluation result
            result = json.loads(response.content.strip())
            
            overall_score = result.get("overall_score", 5.0) / 10.0  # Normalize to 0-1
            needs_retry = (
                overall_score < self.config.min_quality_score or 
                result.get("needs_improvement", False)
            )
            feedback = result.get("feedback", "No feedback available")
            
            return overall_score, needs_retry, feedback
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            # Default: assume good quality if can't evaluate
            return 0.8, False, f"Evaluation failed: {str(e)}"
    
    def generate_improvement_prompt(self, feedback: str, original_question: str) -> str:
        """Generate a prompt for improving the answer."""
        return f"""
Câu hỏi gốc: {original_question}

Phản hồi từ đánh giá: {feedback}

Hãy cải thiện câu trả lời dựa trên phản hồi này. Tập trung vào:
1. Khắc phục các vấn đề được chỉ ra
2. Cung cấp thông tin chính xác hơn
3. Làm rõ các điểm mơ hồ
4. Sử dụng tốt hơn thông tin từ tài liệu

Câu trả lời được cải thiện:"""