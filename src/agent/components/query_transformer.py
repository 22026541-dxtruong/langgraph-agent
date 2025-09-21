"""Query transformation component for better retrieval."""
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


class QueryTransformer:
    """Transform user queries for better retrieval."""
    
    def __init__(self, config):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=config.llm_model,
            temperature=0,
            google_api_key=config.google_api_key
        )
        
        self.transform_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Bạn là chuyên gia tối ưu hóa truy vấn. Hãy tạo {max_queries} phiên bản khác nhau của câu hỏi để cải thiện việc tìm kiếm thông tin.

Câu hỏi gốc: {question}

Yêu cầu:
1. Giữ nguyên ý nghĩa chính
2. Sử dụng từ đồng nghĩa và cách diễn đạt khác
3. Thêm context hoặc làm rõ chi tiết
4. Mỗi phiên bản trên 1 dòng riêng

Các phiên bản câu hỏi:"""
        )
    
    def transform_query(self, question: str) -> List[str]:
        """Transform a single query into multiple variants."""
        try:
            response = self.llm.invoke(
                self.transform_prompt.format(
                    question=question,
                    max_queries=self.config.max_transformed_queries
                )
            )
            
            # Parse response into list of queries
            queries = [question]  # Always include original
            lines = response.content.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and line not in queries:
                    # Remove numbering if present
                    if '. ' in line[:5]:
                        line = line.split('. ', 1)[1]
                    queries.append(line)
            
            return queries[:self.config.max_transformed_queries + 1]
            
        except Exception as e:
            print(f"Query transformation error: {e}")
            return [question]  # Fallback to original