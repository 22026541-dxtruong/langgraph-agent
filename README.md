# LangGraph Agent với Memory, Research, Retrieval và ReAct

Tính năng:
- **Memory**: Lưu trữ và truy xuất thông tin dài hạn
- **Research**: Tìm kiếm thông tin trên web bằng Tavily
- **Retrieval**: Tìm kiếm trong cơ sở kiến thức local bằng FAISS
- **ReAct**: Agent reasoning với Think-Act-Observe pattern

## Cài đặt

1. Clone project và cài đặt dependencies:
```bash
python -m pip install -e .
```

2. Cấu hình environment variables trong file `.env`:
```bash
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=langgraph-agent
```

3. Chạy trên LangSmith
```bash
pip install langgraph-cli
langgraph dev
```

## Tính năng

### 1. Memory
- `save_to_memory(key, value)`: Lưu thông tin
- `recall_from_memory(key)`: Truy xuất thông tin
- Tự động tóm tắt cuộc hội thoại dài

### 2. Research  
- `web_search(query)`: Tìm kiếm web với Tavily
- Tìm kiếm nâng cao với nhiều kết quả
- Tự động xử lý lỗi

### 3. Retrieval
- `add_documents_to_retrieval(documents)`: Thêm tài liệu
- `retrieve_documents(query)`: Tìm kiếm tài liệu liên quan
- Sử dụng FAISS vector store

### 4. ReAct Pattern
- **Think**: Agent suy nghĩ về nhiệm vụ
- **Act**: Sử dụng tools phù hợp  
- **Observe**: Quan sát kết quả
- **Respond**: Trả lời người dùng

### 5. Graph Flow
<div align="center">
  <img src="./static/studio_ui.png" alt="Graph view in LangGraph studio UI" width="75%" />
</div>