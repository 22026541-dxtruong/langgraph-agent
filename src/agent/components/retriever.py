"""Enhanced retrieval component with multiple strategies."""
from typing import List, Dict, Any, Union, Mapping
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import uuid


class EnhancedRetriever:
    """Advanced retrieval with multiple strategies and fusion."""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize embedding model using LangChain
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=config.chroma_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(config.collection_name)
        except:
            # Create collection if it doesn't exist
            self.collection = self.client.create_collection(
                name=config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Add documents to the vector store."""
        if not documents:
            return "No documents to add."
    
        try:
            ids = [str(uuid.uuid4()) for _ in documents]
            texts = [doc["content"] for doc in documents]
        
            # Kiểm tra và đảm bảo rằng metadata không rỗng
            metadatas = [
                doc.get("metadata", {"source": "unknown", "title": "", "chunk_id": i}) 
                for i, doc in enumerate(documents)
            ]
        
            embeddings = self.embedding_model.embed_documents(texts)
        
            # Kiểm tra dữ liệu trước khi thêm vào collection
            if not all(metadatas):
                raise ValueError("One or more documents have empty metadata.")
        
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            return "Documents added successfully."
        except Exception as e:
            return f"Error adding documents: {str(e)}"

    
    def retrieve_documents(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Retrieve documents using multiple queries with fusion."""
        all_results = []
        query_weights = [1.0] + [0.7] * (len(queries) - 1)  # Original query gets higher weight
    
        if not queries:
            return all_results  # Nếu không có query, trả về danh sách rỗng

        for query, weight in zip(queries, query_weights):
            try:
                # Tạo embedding cho query
                query_embedding = self.embedding_model.embed_query(query)
                
                # Gửi query đến ChromaDB và nhận kết quả
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=self.config.top_k,
                    include=["documents", "metadatas", "distances"]
                )
                # Process results for this query
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0], 
                    results["distances"][0]
                )):
                    # Convert distance to similarity score
                    similarity = 1 - distance
                    weighted_score = similarity * weight
                
                    all_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity": similarity,
                        "weighted_score": weighted_score,
                        "query": query,
                        "rank": i + 1
                    })
            except Exception as e:
                print(f"Error processing query '{query}': {str(e)}")
                continue  # Nếu có lỗi, tiếp tục với query tiếp theo
    
        if not all_results:
            print("No results found after fusion.")

        # Fusion: Combine results from all queries
        return self._reciprocal_rank_fusion(all_results)
    
    def _reciprocal_rank_fusion(self, results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        """Apply Reciprocal Rank Fusion to combine multiple result sets."""
        doc_scores = {}

        for result in results:
            # Kiểm tra nếu có đầy đủ dữ liệu
            if "content" not in result or "rank" not in result or "weighted_score" not in result or "similarity" not in result:
                continue  # Bỏ qua kết quả không hợp lệ
        
            doc_id = result.get("content", "")
            if not doc_id:  # Nếu không có content
                continue
        
            rank = result["rank"]
            rrf_score = 1 / (k + rank)
            weighted_score = result.get("weighted_score", 0)

            # Đảm bảo weighted_score là kiểu số hợp lệ
            if not isinstance(weighted_score, (int, float)):
                weighted_score = 0

            print(f"Processing doc {doc_id} with rank {rank}, RRF score: {rrf_score}, Weighted score: {weighted_score}")

            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "total_score": 0,
                    "max_similarity": 0,
                    "query_count": 0
                }

            doc_scores[doc_id]["total_score"] += rrf_score + weighted_score
            doc_scores[doc_id]["max_similarity"] = max(
                doc_scores[doc_id]["max_similarity"], result.get("similarity", 0)
            )
            doc_scores[doc_id]["query_count"] += 1

        # Sắp xếp theo total_score
        ranked_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["total_score"],
            reverse=True
        )

        # Lọc theo similarity threshold
        filtered_docs = [
            doc for doc in ranked_docs
            if doc["max_similarity"] >= self.config.similarity_threshold
        ]

        print(f"Documents after applying similarity threshold: {len(filtered_docs)}")

        # In ra các tài liệu có similarity
        for doc in filtered_docs:
            print(f"Filtered doc ID: {doc['content']} - Max Similarity: {doc['max_similarity']}")

        # Trả về kết quả đã lọc và giới hạn theo top_k
        return filtered_docs[:self.config.top_k]
