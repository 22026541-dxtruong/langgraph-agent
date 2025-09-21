"""Embedding utilities and document processing."""
from typing import List, Dict, Any
import os
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader, DirectoryLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
from pathlib import Path
import requests
from bs4.filter import SoupStrainer
from bs4 import BeautifulSoup


class DocumentProcessor:
    """Process and prepare documents for embedding."""
    
    def __init__(self, config):
        self.config = config
        
        # Use SentenceTransformersTokenTextSplitter for better token management
        self.text_splitter = SentenceTransformersTokenTextSplitter(
            model_name=config.embedding_model,
            tokens_per_chunk=256,  # Optimal for most sentence transformer models
            chunk_overlap=50,
            # separators=["\n\n", "\n", ". ", " "]
        )
    
    def load_documents_from_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Load and process documents from web URLs."""
        if not urls:
            print("No URLs provided.")
            return []
        
        processed_docs = []
        
        for url in urls:
            try:
                print(f"Loading from URL: {url}")
                
                # Create WebBaseLoader for the URL
                loader = WebBaseLoader(
                    web_paths=[url],
                    # bs_kwargs={
                    #     "parse_only": SoupStrainer(
                    #         ["article", "main", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6"]
                    #     )
                    # }
                )
                
                # Load documents
                docs = loader.load()
                
                if not docs:
                    print(f"No content found at {url}")
                    continue
                
                # Process each document
                for doc in docs:
                    # Clean and extract text content
                    content = doc.page_content.strip()
                    if len(content) < 100:  # Skip very short content
                        continue
                    
                    # Split into chunks
                    chunks = self.text_splitter.split_text(content)
                    
                    # Create document entries for each chunk
                    for i, chunk in enumerate(chunks):
                        if len(chunk.strip()) > 50:  # Only keep meaningful chunks
                            processed_docs.append({
                                "content": chunk.strip(),
                                "source": url,
                                "title": self._extract_title_from_url(url),
                                "chunk_id": i,
                                "total_chunks": len(chunks),
                                "url": url
                            })
                
                print(f"✅ Processed {len([d for d in processed_docs if d['source'] == url])} chunks from {url}")
                
            except Exception as e:
                print(f"❌ Error loading {url}: {str(e)}")
                continue
        
        print(f"📚 Total processed: {len(processed_docs)} document chunks from {len(urls)} URLs")
        return processed_docs
    
    def load_documents_from_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Load and process documents from a directory (fallback method)."""
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist.")
            return []
        
        # Load documents
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        
        try:
            raw_docs = loader.load()
            print(f"Loaded {len(raw_docs)} raw documents")
        except Exception as e:
            print(f"Error loading documents: {e}")
            return []
        
        # Split documents into chunks using SentenceTransformers splitter
        processed_docs = []
        for doc in raw_docs:
            chunks = self.text_splitter.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                processed_docs.append({
                    "content": chunk,
                    "source": doc.metadata.get("source", "unknown"),
                    "title": Path(doc.metadata.get("source", "")).stem,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
        
        print(f"Created {len(processed_docs)} document chunks")
        return processed_docs

    def process_text_input(self, text: str, source: str = "user_input") -> List[Dict[str, Any]]:
        """Process a single text input into chunks."""
        chunks = self.text_splitter.split_text(text)
        
        processed_docs = []
        for i, chunk in enumerate(chunks):
            processed_docs.append({
                "content": chunk,
                "source": source,
                "title": "User Input",
                "chunk_id": i,
                "total_chunks": len(chunks)
            })
        
        return processed_docs
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract a meaningful title from URL."""
        try:
            # Try to get title from the page
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title_tag = soup.find('title')
            if title_tag and title_tag.text.strip():
                return title_tag.text.strip()[:100]  # Limit title length
            
            # Fallback: use URL path
            from urllib.parse import urlparse
            parsed = urlparse(url)
            path_parts = [part for part in parsed.path.split('/') if part]
            if path_parts:
                return path_parts[-1].replace('-', ' ').replace('_', ' ').title()
            
            return parsed.netloc
            
        except Exception:
            # Final fallback: use domain name
            from urllib.parse import urlparse
            return urlparse(url).netloc


def setup_web_documents() -> List[str]:
    """Return a list of sample web URLs for testing."""
    sample_urls = [
        # Vietnamese content
        "https://vi.wikipedia.org/wiki/H%E1%BB%93_Ch%C3%AD_Minh",
        "https://vi.wikipedia.org/wiki/L%E1%BB%8Bch_s%E1%BB%AD_Vi%E1%BB%87t_Nam",
        
        # Technology content  
        "https://python.langchain.com/docs/introduction/",
        "https://python.langchain.com/docs/tutorials/rag/",
        
        # AI/ML content
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Large_language_model"
    ]
    
    print(f"📋 Sample URLs for document loading:")
    for i, url in enumerate(sample_urls, 1):
        print(f"  {i}. {url}")
    
    return sample_urls


def setup_sample_documents(data_dir: str = "./data/documents") -> List[str]:
    """Create sample documents for testing (fallback if web loading fails)."""
    os.makedirs(data_dir, exist_ok=True)
    
    sample_docs = {
        "vietnam_history.txt": """
Lịch sử Việt Nam là một chặng đường dài với nhiều biến động. 
Việt Nam có lịch sử hàng ngàn năm với nhiều triều đại phong kiến.
Trong thế kỷ 20, Việt Nam đã trải qua các cuộc kháng chiến chống thực dân Pháp và đế quốc Mỹ.
Hồ Chí Minh là lãnh tụ vĩ đại của dân tộc Việt Nam, người đã dẫn dắt cuộc đấu tranh giành độc lập.
Ngày 2/9/1945 là ngày Việt Nam tuyên bố độc lập, thành lập nước Việt Nam Dân chủ Cộng hòa.
Chiến tranh Việt Nam kéo dài từ 1955 đến 1975, kết thúc với việc thống nhất đất nước.
Sau 1975, Việt Nam bước vào thời kỳ đổi mới từ năm 1986, mở cửa kinh tế.
""",
        
        "technology.txt": """
Trí tuệ nhân tạo (AI) đang phát triển mạnh mẽ và thay đổi nhiều lĩnh vực.
Machine Learning là một nhánh quan trọng của AI, cho phép máy tính học từ dữ liệu.
Deep Learning sử dụng mạng neural nhiều lớp để học các pattern phức tạp.
Natural Language Processing (NLP) giúp máy tính hiểu và xử lý ngôn ngữ tự nhiên.
Large Language Models như GPT, Claude, Gemini đang cách mạng hóa cách chúng ta tương tác với AI.
RAG (Retrieval Augmented Generation) kết hợp việc tìm kiếm thông tin và tạo sinh câu trả lời.
Vector databases như ChromaDB, Pinecone, Weaviate lưu trữ embeddings để tìm kiếm semantic.
""",
        
        "programming.txt": """
Python là một ngôn ngữ lập trình phổ biến và dễ học, được sử dụng rộng rãi trong AI/ML.
LangChain là một framework mạnh mẽ để xây dựng ứng dụng AI với LLMs.
LangGraph cho phép xây dựng các workflow phức tạp với AI agents và state management.
Vector databases như ChromaDB, Pinecone giúp lưu trữ và tìm kiếm embeddings hiệu quả.
Embedding models chuyển đổi text thành vector số để so sánh độ tương đồng semantic.
Sentence Transformers là thư viện phổ biến để tạo embeddings chất lượng cao.
Hugging Face cung cấp hàng ngàn pre-trained models cho NLP và computer vision.
"""
    }
    
    for filename, content in sample_docs.items():
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    print(f"Created {len(sample_docs)} sample documents in {data_dir}")
    
    # Return file paths for loading
    return [os.path.join(data_dir, filename) for filename in sample_docs.keys()]


if __name__ == "__main__":
    # Test web document loading
    urls = setup_web_documents()
    processor = DocumentProcessor(type('Config', (), {'embedding_model': 'all-MiniLM-L6-v2'})())
    
    # Try loading from web
    web_docs = processor.load_documents_from_urls(urls[:2])  # Test with first 2 URLs
    
    if web_docs:
        print(f"✅ Successfully loaded {len(web_docs)} chunks from web")
        print(f"Sample content: {web_docs[0]['content'][:200]}...")
    else:
        print("❌ Web loading failed, falling back to sample documents")
        setup_sample_documents()