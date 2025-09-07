import os
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
import json
import requests
from bs4 import BeautifulSoup
import time

document_store = None
persistent_memory = {}
conversation_history = {}

# Setup data directories
DATA_DIR = Path("./data")
MEMORY_FILE = DATA_DIR / "memory.json"
DOCUMENTS_DIR = DATA_DIR / "documents"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True)


def load_persistent_memory():
    """Load memory from file."""
    global persistent_memory
    try:
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                persistent_memory = json.load(f)
    except Exception as e:
        print(f"Failed to load memory: {e}")
        persistent_memory = {}


def save_persistent_memory():
    """Save memory to file."""
    global persistent_memory
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(persistent_memory, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save memory: {e}")


# Load memory on startup
load_persistent_memory()

# Initialize DuckDuckGo search
search_tool = TavilySearchResults(max_results=4)

# Initialize embeddings for retrieval
import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", async_client=False)

@tool
def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web for information using Tavily.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
    
    Returns:
        List of search results with title, content, and URL
    """
    try:
        response = search_tool.invoke({
            "query": query,
        })
        
        # results = []
        # for result in response.get("results", []):
        #     results.append({
        #         "title": result.get("title", ""),
        #         "content": result.get("content", ""),
        #         "url": result.get("url", ""),
        #         "score": result.get("score", 0),
        #         "timestamp": datetime.now().isoformat()
        #     })
        
        return response
    except Exception as e:
        return [{"error": f"Web search failed: {str(e)}"}]


@tool
def web_search_news(query: str, max_results: int = 5, days: int = 7) -> List[Dict[str, Any]]:
    """
    Search for recent news using Tavily.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        days: Search within last N days
    
    Returns:
        List of news results with title, content, URL, and published date
    """
    try:
        # Add news-specific keywords and time constraint
        news_query = f"{query} news recent {days} days"
        
        response = search_tool.invoke({
            "query": news_query,
        })    
        # results = []
        # for result in response.get("results", []):
        #     results.append({
        #         "title": result.get("title", ""),
        #         "content": result.get("content", ""),
        #         "url": result.get("url", ""),
        #         "score": result.get("score", 0),
        #         "published_date": result.get("published_date", ""),
        #         "timestamp": datetime.now().isoformat(),
        #         "type": "news"
        #     })
        
        return response
    except Exception as e:
        return [{"error": f"News search failed: {str(e)}"}]
    
@tool
def scrape_webpage(url: str, max_length: int = 2000) -> Dict[str, Any]:
    """
    Lấy nội dung từ một trang web.
    
    Args:
        url: URL của trang web cần lấy nội dung.
        max_length: Độ dài tối đa của văn bản cần lấy. Mặc định là 2000 ký tự.
    
    Returns:
        Từ điển chứa nội dung đã được lấy từ trang web.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Cơ chế thử lại với thời gian chờ tăng dần (exponential backoff)
        response = None
        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()  # Kiểm tra mã lỗi HTTP
                break  # Thoát khỏi vòng lặp nếu thành công
            except requests.exceptions.RequestException as e:
                if attempt < 2:
                    print(f"Lỗi khi lấy {url}: {e}. Đang thử lại...")
                    time.sleep(2 ** attempt)  # Thời gian chờ tăng dần
                else:
                    return {"error": f"Không thể lấy nội dung từ {url}: {e}"}
        
        if response is None:
            return {"error": f"Không thể lấy nội dung từ {url}: Không nhận được phản hồi."}
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Xóa các thẻ không cần thiết (script, style, footer, header, aside, v.v.)
        for tag in soup(["script", "style", "footer", "header", "aside"]):
            tag.decompose()
        
        # Lấy nội dung văn bản từ trang web
        text = soup.get_text()
        
        # Làm sạch văn bản
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Giới hạn độ dài văn bản
        if len(clean_text) > max_length:
            clean_text = clean_text[:max_length]
        
        return {
            "url": url,
            "title": soup.title.string if soup.title else "Không có tiêu đề",
            "content": clean_text,
            "timestamp": datetime.now().isoformat(),
            "word_count": len(clean_text.split())
        }
    
    except Exception as e:
        return {"error": f"Không thể lấy nội dung từ {url}: {str(e)}"}

@tool
def add_documents_to_retrieval(documents: List[str], metadata: Optional[List[Dict]] = None) -> str:
    """
    Add documents to the retrieval system with enhanced metadata.
    
    Args:
        documents: List of document texts to add
        metadata: Optional metadata for each document
    
    Returns:
        Status message
    """
    global document_store
    
    try:
        if metadata is None:
            metadata = [{"source": "user_input", "timestamp": datetime.now().isoformat()} 
                       for _ in documents]
        
        docs = []
        for i, doc in enumerate(documents):
            doc_metadata = metadata[i] if i < len(metadata) else {}
            doc_metadata.update({
                "doc_id": f"doc_{datetime.now().timestamp()}_{i}",
                "word_count": len(doc.split()),
                "char_count": len(doc)
            })
            docs.append(Document(page_content=doc, metadata=doc_metadata))
        
        if document_store is None:
            document_store = FAISS.from_documents(docs, embeddings)
        else:
            document_store.add_documents(docs)
        
        # Save documents to file
        for i, doc in enumerate(docs):
            doc_file = DOCUMENTS_DIR / f"doc_{doc.metadata['doc_id']}.txt"
            with open(doc_file, 'w', encoding='utf-8') as f:
                f.write(f"Metadata: {json.dumps(doc.metadata)}\n\n")
                f.write(doc.page_content)
        
        return f"Successfully added {len(documents)} documents to retrieval system"
    except Exception as e:
        return f"Failed to add documents: {str(e)}"


@tool
def retrieve_documents(query: str, k: int = 3, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents with enhanced filtering.
    
    Args:
        query: Search query
        k: Number of documents to retrieve
        filter_metadata: Optional metadata filters
    
    Returns:
        List of relevant documents with content and metadata
    """
    global document_store
    
    if document_store is None:
        return [{"error": "No documents in retrieval system"}]
    
    try:
        docs_and_scores = document_store.similarity_search_with_score(query, k=k*2)  # Get more to filter
        
        filtered_docs = []
        for doc, score in docs_and_scores:
            if filter_metadata:
                # Apply metadata filters
                if not all(doc.metadata.get(key) == value for key, value in filter_metadata.items()):
                    continue
            
            filtered_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score),
                "rank": len(filtered_docs) + 1
            })
            
            if len(filtered_docs) >= k:
                break
        
        return filtered_docs
    except Exception as e:
        return [{"error": f"Retrieval failed: {str(e)}"}]


@tool
def save_to_memory(key: str, value: str, category: str = "general") -> str:
    """
    Enhanced memory system with categories and persistence.
    
    Args:
        key: Memory key
        value: Value to store
        category: Memory category (general, user_info, preferences, facts)
    
    Returns:
        Status message
    """
    global persistent_memory
    
    if category not in persistent_memory:
        persistent_memory[category] = {}
    
    persistent_memory[category][key] = {
        "value": value,
        "timestamp": datetime.now().isoformat(),
        "access_count": 0
    }
    
    save_persistent_memory()
    return f"Saved to memory [{category}]: {key} = {value[:50]}..."


@tool
def recall_from_memory(key: str = None, category: str = None) -> str:
    """
    Enhanced memory recall with search capabilities.
    
    Args:
        key: Specific memory key to retrieve
        category: Memory category to search in
    
    Returns:
        Stored value(s) or search results
    """
    global persistent_memory
    
    if not persistent_memory:
        return "No memories stored yet"
    
    # If specific key requested
    if key:
        for cat, memories in persistent_memory.items():
            if category and cat != category:
                continue
            if key in memories:
                memories[key]["access_count"] += 1
                save_persistent_memory()
                return memories[key]["value"]
        return f"No memory found for key: {key}"
    
    # If category requested, return all keys in category
    if category:
        if category in persistent_memory:
            keys = list(persistent_memory[category].keys())
            return f"Keys in {category}: {', '.join(keys)}"
        return f"No memories in category: {category}"
    
    # Return overview of all memories
    overview = []
    for cat, memories in persistent_memory.items():
        overview.append(f"{cat}: {len(memories)} items")
    
    return f"Memory overview: {'; '.join(overview)}"


@tool
def search_memory(query: str) -> List[Dict[str, Any]]:
    """
    Search through all memories for relevant content.
    
    Args:
        query: Search query
    
    Returns:
        List of matching memories
    """
    global persistent_memory
    query_lower = query.lower()
    results = []
    
    for category, memories in persistent_memory.items():
        for key, memory_data in memories.items():
            value = memory_data["value"]
            
            # Simple text matching
            if (query_lower in key.lower() or 
                query_lower in value.lower()):
                results.append({
                    "category": category,
                    "key": key,
                    "value": value,
                    "timestamp": memory_data["timestamp"],
                    "access_count": memory_data["access_count"]
                })
    
    return results

@tool
def calculate(expression: str) -> str:
    """
    Enhanced calculator with more functions.
    
    Args:
        expression: Mathematical expression to evaluate
    
    Returns:
        Result of calculation or error message
    """
    import math
    
    try:
        # Add math functions to allowed operations
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "pi": math.pi, "e": math.e, "log": math.log,
            "ceil": math.ceil, "floor": math.floor
        }
        
        # Only allow safe operations and math functions
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                return f"Error: '{name}' is not allowed in calculations"
        
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_current_time() -> str:
    """Get current date and time."""
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')})"


@tool
def set_reminder(message: str, minutes_from_now: int) -> str:
    """
    Set a reminder (stored in memory).
    
    Args:
        message: Reminder message
        minutes_from_now: Minutes from now to remind
    
    Returns:
        Status message
    """
    reminder_time = datetime.now() + timedelta(minutes=minutes_from_now)
    reminder_key = f"reminder_{reminder_time.timestamp()}"
    
    reminder_data = {
        "message": message,
        "reminder_time": reminder_time.isoformat(),
        "created_at": datetime.now().isoformat()
    }
    
    # Call the original function, not the tool wrapper
    globals()["save_to_memory"].__wrapped__(reminder_key, json.dumps(reminder_data), "reminders")
    return f"Reminder set for {reminder_time.strftime('%Y-%m-%d %H:%M:%S')}: {message}"


@tool
def check_reminders() -> List[Dict[str, Any]]:
    """Check for active reminders."""
    current_time = datetime.now()
    active_reminders = []
    
    if "reminders" in persistent_memory:
        for key, reminder_data in persistent_memory["reminders"].items():
            try:
                reminder_info = json.loads(reminder_data["value"])
                reminder_time = datetime.fromisoformat(reminder_info["reminder_time"])
                
                if reminder_time <= current_time:
                    active_reminders.append({
                        "message": reminder_info["message"],
                        "reminder_time": reminder_info["reminder_time"],
                        "overdue_minutes": int((current_time - reminder_time).total_seconds() / 60)
                    })
            except:
                continue
    
    return active_reminders


@tool
def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze text for various metrics.
    
    Args:
        text: Text to analyze
    
    Returns:
        Analysis results
    """
    words = text.split()
    sentences = text.split('.')
    
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "char_count": len(text),
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
        "most_common_words": {word: words.count(word) for word in set(words) if words.count(word) > 1},
        "estimated_reading_time_minutes": len(words) / 200  # 200 words per minute average
    }


# Export all tools
all_tools = [
    web_search,
    web_search_news,
    scrape_webpage,
    add_documents_to_retrieval,
    retrieve_documents,
    save_to_memory,
    recall_from_memory,
    search_memory,
    calculate,
    get_current_time,
    set_reminder,
    check_reminders,
    analyze_text
]
