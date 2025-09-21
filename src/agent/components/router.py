"""Intelligent routing component."""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json


class IntelligentRouter:
    """Route queries to appropriate processing paths."""
    
    def __init__(self, config):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=config.llm_model,
            temperature=0,
            google_api_key=config.google_api_key
        )
        
        self.router_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Bạn là một hệ thống định tuyến thông minh. Hãy xác định cách xử lý tốt nhất cho câu hỏi.

Câu hỏi: {question}

Các lựa chọn xử lý:
1. "vectorstore" - Tìm kiếm trong cơ sở dữ liệu nội bộ (cho câu hỏi về tài liệu, thông tin cụ thể)
2. "websearch" - Tìm kiếm web (cho thông tin thời sự, cập nhật mới)
3. "direct_response" - Trả lời trực tiếp (cho câu hỏi đơn giản, chào hỏi, toán học cơ bản)

Hãy phản hồi CHÍNH XÁC theo định dạng JSON:
{{"route": "tên_route", "reasoning": "lý do lựa chọn"}}"""
        )
    
    def route_query(self, question: str) -> str:
        """Determine the best route for processing the query."""
        try:
            response = self.llm.invoke(
                self.router_prompt.format(question=question)
            )
            
            # Parse JSON response
            result = json.loads(response.content.strip())
            route = result.get("route", "vectorstore")
            
            # Validate route
            valid_routes = ["vectorstore", "websearch", "direct_response"]
            if route not in valid_routes:
                return "vectorstore"  # Default fallback
                
            return route
            
        except Exception as e:
            print(f"Routing error: {e}")
            return "vectorstore"  # Safe default