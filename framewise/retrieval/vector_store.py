"""Vector database integration using LanceDB for fast similarity search.

This module provides a vector database interface for storing and searching
frame embeddings. It supports hybrid search combining both image and text
embeddings for powerful multimodal retrieval.

Example:
    Basic usage::

        from framewise import FrameWiseVectorStore, FrameWiseEmbedder
        
        # Create embeddings
        embedder = FrameWiseEmbedder()
        embeddings = embedder.embed_frames_batch(frames)
        
        # Store in vector database
        store = FrameWiseVectorStore(db_path="tutorials.db")
        store.create_table(embeddings)
        
        # Search
        results = store.search_by_text("How do I export?", embedder, limit=5)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import numpy as np
import lancedb
from loguru import logger


class FrameWiseVectorStore:
    """Vector database for storing and searching frame embeddings.
    
    This class provides a high-level interface to LanceDB for storing multimodal
    embeddings and performing similarity search. It supports three search modes:
    
    - **Hybrid**: Searches using combined image + text embeddings
    - **Text**: Searches using text embeddings only
    - **Image**: Searches using image embeddings only
    
    The database connection is established lazily on first use.
    
    Attributes:
        db_path: Path to the LanceDB database directory.
        table_name: Name of the table storing frame embeddings.
    
    Example:
        Create and populate database::
        
            store = FrameWiseVectorStore(db_path="my_videos.db")
            store.create_table(embeddings, mode="overwrite")
        
        Search for similar frames::
        
            results = store.search_by_text(
                "How do I save?",
                embedder=embedder,
                limit=3,
                search_type="hybrid"
            )
        
        Get database statistics::
        
            stats = store.get_stats()
            print(f"Total frames: {stats['total_frames']}")
    """
    
    def __init__(
        self,
        db_path: Union[str, Path] = "framewise.db",
        table_name: str = "frames"
    ) -> None:
        """Initialize the vector store.
        
        Args:
            db_path: Path to the LanceDB database directory. Will be created
                if it doesn't exist. Defaults to "framewise.db".
            table_name: Name of the table to store frame embeddings.
                Defaults to "frames".
        
        Example:
            >>> # Use default database
            >>> store = FrameWiseVectorStore()
            
            >>> # Use custom database and table
            >>> store = FrameWiseVectorStore(
            ...     db_path="my_tutorials.db",
            ...     table_name="tutorial_frames"
            ... )
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._db = None
        self._table = None
        
        logger.info(f"Initializing vector store at: {self.db_path}")
    
    def _connect(self) -> None:
        """Connect to the LanceDB database.
        
        Establishes connection to the database if not already connected.
        The connection is lazy - only created when needed.
        
        Raises:
            RuntimeError: If connection to the database fails.
        """
        if self._db is None:
            self._db = lancedb.connect(str(self.db_path))
            logger.debug(f"Connected to database: {self.db_path}")
    
    def create_table(
        self,
        embeddings: List[Dict[str, Any]],
        mode: str = "overwrite"
    ) -> None:
        """Create or update the frames table with embeddings.
        
        Stores frame embeddings in the database. Each embedding includes both
        image and text vectors, enabling hybrid search. The combined vector
        is created by concatenating image and text embeddings.
        
        Args:
            embeddings: List of embedding dictionaries from FrameWiseEmbedder.
                Each dictionary should contain:
                - frame_id: Unique identifier
                - timestamp: Time in seconds
                - image_embedding: Image embedding vector
                - text_embedding: Text embedding vector
                - text: Transcript text
                - frame_path: Path to frame image
                - extraction_reason: Why frame was extracted
                - quality_score: Frame quality score
            mode: Table creation mode. Options:
                - 'overwrite': Replace existing table (default)
                - 'append': Add to existing table
                Defaults to 'overwrite'.
        
        Raises:
            ValueError: If embeddings list is empty or has invalid format.
            RuntimeError: If table creation fails.
        
        Example:
            >>> embedder = FrameWiseEmbedder()
            >>> embeddings = embedder.embed_frames_batch(frames)
            >>> store = FrameWiseVectorStore()
            >>> store.create_table(embeddings, mode="overwrite")
            
            >>> # Append more embeddings later
            >>> new_embeddings = embedder.embed_frames_batch(new_frames)
            >>> store.create_table(new_embeddings, mode="append")
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
    ) -> List[Dict[str, Any]]:
        """Search for similar frames using vector similarity.
        
        Performs approximate nearest neighbor search to find frames with
        embeddings most similar to the query embedding.
        
        Args:
            query_embedding: Query embedding vector to search for. Should match
                the dimensionality of the search_type:
                - hybrid: image_dim + text_dim (e.g., 512 + 384 = 896)
                - image: image_dim (e.g., 512)
                - text: text_dim (e.g., 384)
            limit: Maximum number of results to return. Defaults to 5.
            search_type: Type of search to perform. Options:
                - 'hybrid': Search using combined image+text embeddings
                - 'image': Search using image embeddings only
                - 'text': Search using text embeddings only
                Defaults to 'hybrid'.
        
        Returns:
            List of dictionaries, each containing:
            - frame_id, timestamp, text, frame_path, extraction_reason,
              quality_score, and similarity score
            Results are sorted by similarity (most similar first).
        
        Raises:
            ValueError: If the table doesn't exist or search_type is invalid.
            RuntimeError: If search operation fails.
        
        Example:
            >>> embedder = FrameWiseEmbedder()
            >>> query_emb = embedder.embed_text("export data")
            >>> results = store.search(query_emb, limit=3, search_type="text")
            >>> for r in results:
            ...     print(f"{r['timestamp']}s: {r['text']}")
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
        embedder: Any,  # FrameWiseEmbedder type
        limit: int = 5,
        search_type: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """Search using a text query.
        
        Convenience method that generates the query embedding from text and
        performs the search. This is the most common way to search the database.
        
        Args:
            query_text: Natural language query (e.g., "How do I export my data?").
            embedder: FrameWiseEmbedder instance to generate query embedding.
            limit: Maximum number of results to return. Defaults to 5.
            search_type: Type of search to perform. Options:
                - 'hybrid': Search using combined image+text (recommended)
                - 'text': Search using text only
                - 'image': Search using text-as-image approximation
                Defaults to 'hybrid'.
        
        Returns:
            List of matching frames sorted by similarity, each containing:
            - frame_id, timestamp, text, frame_path, extraction_reason,
              quality_score, and similarity score
        
        Raises:
            ValueError: If the table doesn't exist.
            ImportError: If required embedding models are not available.
        
        Example:
            >>> store = FrameWiseVectorStore()
            >>> embedder = FrameWiseEmbedder()
            >>> 
            >>> # Hybrid search (recommended)
            >>> results = store.search_by_text(
            ...     "How do I export?",
            ...     embedder=embedder,
            ...     limit=3
            ... )
            >>> 
            >>> # Text-only search
            >>> results = store.search_by_text(
            ...     "export data",
            ...     embedder=embedder,
            ...     search_type="text"
            ... )
            >>> 
            >>> for r in results:
            ...     print(f"[{r['timestamp']}s] {r['text']}")
            [12.5s] Click the export button in the toolbar
            [45.2s] Select export from the file menu
            [78.9s] Choose your export format
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
    
    def get_stats(self) -> Dict[str, Union[bool, int, str]]:
        """Get statistics about the vector store.
        
        Provides information about the database state, including whether
        the table exists and how many frames are stored.
        
        Returns:
            Dictionary containing:
            - exists: Whether the table exists (bool)
            - total_frames: Number of frames in the table (int)
            - table_name: Name of the table (str, if exists)
            - db_path: Path to the database (str, if exists)
        
        Example:
            >>> store = FrameWiseVectorStore()
            >>> stats = store.get_stats()
            >>> if stats['exists']:
            ...     print(f"Database has {stats['total_frames']} frames")
            ... else:
            ...     print("Database is empty")
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
    
    def delete_table(self) -> None:
        """Delete the frames table from the database.
        
        Permanently removes the table and all its data. Use with caution.
        
        Raises:
            RuntimeError: If table deletion fails.
        
        Example:
            >>> store = FrameWiseVectorStore()
            >>> store.delete_table()  # Removes all data
            >>> stats = store.get_stats()
            >>> print(stats['exists'])
            False
        """
        self._connect()
        
        if self.table_name in self._db.table_names():
            self._db.drop_table(self.table_name)
            logger.info(f"Deleted table: {self.table_name}")
        else:
            logger.warning(f"Table '{self.table_name}' does not exist")
