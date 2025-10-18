"""
Vector database integration using LanceDB for fast similarity search
"""

from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import lancedb
from loguru import logger


class FrameWiseVectorStore:
    """Vector database for storing and searching frame embeddings"""
    
    def __init__(
        self,
        db_path: str | Path = "framewise.db",
        table_name: str = "frames"
    ):
        """
        Initialize the vector store
        
        Args:
            db_path: Path to the LanceDB database
            table_name: Name of the table to store frames
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._db = None
        self._table = None
        
        logger.info(f"Initializing vector store at: {self.db_path}")
    
    def _connect(self):
        """Connect to the database"""
        if self._db is None:
            self._db = lancedb.connect(str(self.db_path))
            logger.debug(f"Connected to database: {self.db_path}")
    
    def create_table(
        self,
        embeddings: List[Dict],
        mode: str = "overwrite"
    ):
        """
        Create or update the frames table with embeddings
        
        Args:
            embeddings: List of embedding dictionaries from FrameWiseEmbedder
            mode: 'overwrite' to replace, 'append' to add to existing
        """
        self._connect()
        
        if not embeddings:
            logger.warning("No embeddings provided")
            return
        
        logger.info(f"Creating table '{self.table_name}' with {len(embeddings)} entries")
        
        # Prepare data for LanceDB
        # LanceDB expects a list of dictionaries with consistent schema
        data = []
        for emb in embeddings:
            # Combine image and text embeddings into a single vector
            # This enables hybrid search
            combined_embedding = np.concatenate([
                emb["image_embedding"],
                emb["text_embedding"]
            ])
            
            data.append({
                "frame_id": emb["frame_id"],
                "timestamp": emb["timestamp"],
                "text": emb["text"],
                "frame_path": emb["frame_path"],
                "extraction_reason": emb["extraction_reason"],
                "quality_score": emb["quality_score"],
                "vector": combined_embedding.tolist(),  # Combined embedding
                "image_vector": emb["image_embedding"].tolist(),  # Separate for image-only search
                "text_vector": emb["text_embedding"].tolist(),  # Separate for text-only search
            })
        
        # Create or overwrite table
        if mode == "overwrite":
            self._table = self._db.create_table(
                self.table_name,
                data=data,
                mode="overwrite"
            )
        else:
            # Append to existing table
            if self.table_name in self._db.table_names():
                self._table = self._db.open_table(self.table_name)
                self._table.add(data)
            else:
                self._table = self._db.create_table(self.table_name, data=data)
        
        logger.success(f"Table '{self.table_name}' created with {len(data)} entries")
    
    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        search_type: str = "hybrid"
    ) -> List[Dict]:
        """
        Search for similar frames using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            search_type: 'hybrid', 'image', or 'text'
            
        Returns:
            List of matching frames with similarity scores
        """
        self._connect()
        
        if self._table is None:
            if self.table_name not in self._db.table_names():
                raise ValueError(f"Table '{self.table_name}' does not exist")
            self._table = self._db.open_table(self.table_name)
        
        # Choose which vector to search
        vector_column = {
            "hybrid": "vector",
            "image": "image_vector",
            "text": "text_vector"
        }.get(search_type, "vector")
        
        # Perform search
        results = (
            self._table
            .search(query_embedding.tolist(), vector_column_name=vector_column)
            .limit(limit)
            .to_list()
        )
        
        logger.debug(f"Found {len(results)} results for {search_type} search")
        return results
    
    def search_by_text(
        self,
        query_text: str,
        embedder,
        limit: int = 5,
        search_type: str = "hybrid"
    ) -> List[Dict]:
        """
        Search using a text query
        
        Args:
            query_text: Text query (e.g., "How do I export?")
            embedder: FrameWiseEmbedder instance to generate query embedding
            limit: Number of results to return
            search_type: 'hybrid', 'image', or 'text'
            
        Returns:
            List of matching frames with similarity scores
        """
        logger.info(f"Searching for: '{query_text}'")
        
        # Generate query embedding based on search type
        if search_type == "text":
            # Text-only search
            query_embedding = embedder.embed_text(query_text)
        elif search_type == "image":
            # For image search with text query, we need CLIP's text encoder
            # For now, use text embedding as approximation
            query_embedding = embedder.embed_text(query_text)
        else:  # hybrid
            # For hybrid search, we need to match the combined vector dimensions
            # Get both text and image embeddings
            text_emb = embedder.embed_text(query_text)
            
            # For text query on hybrid search, duplicate text embedding
            # to match the combined (image + text) vector size
            # This gives equal weight to both modalities
            image_dim = 512  # CLIP dimension
            text_dim = len(text_emb)
            
            # Create a combined vector: [text_emb, text_emb_padded]
            # Pad or truncate to match image dimension
            if text_dim < image_dim:
                # Pad with zeros
                text_as_image = np.pad(text_emb, (0, image_dim - text_dim))
            else:
                # Truncate
                text_as_image = text_emb[:image_dim]
            
            query_embedding = np.concatenate([text_as_image, text_emb])
        
        return self.search(query_embedding, limit, search_type)
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary with database statistics
        """
        self._connect()
        
        if self.table_name not in self._db.table_names():
            return {
                "exists": False,
                "total_frames": 0
            }
        
        table = self._db.open_table(self.table_name)
        count = table.count_rows()
        
        return {
            "exists": True,
            "total_frames": count,
            "table_name": self.table_name,
            "db_path": str(self.db_path)
        }
    
    def delete_table(self):
        """Delete the frames table"""
        self._connect()
        
        if self.table_name in self._db.table_names():
            self._db.drop_table(self.table_name)
            logger.info(f"Deleted table: {self.table_name}")
        else:
            logger.warning(f"Table '{self.table_name}' does not exist")
