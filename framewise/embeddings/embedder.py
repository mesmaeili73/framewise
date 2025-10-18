"""Multimodal embedding generation for frames and text.

This module provides functionality to generate embeddings for both images and text
using state-of-the-art models (CLIP for images, Sentence Transformers for text).
These embeddings enable semantic search across video content.

Example:
    Basic usage::

        from framewise import FrameWiseEmbedder, FrameExtractor
        
        # Extract frames first
        frames = FrameExtractor().extract("video.mp4", transcript)
        
        # Generate embeddings
        embedder = FrameWiseEmbedder(device="cuda")
        embeddings = embedder.embed_frames_batch(frames)
        
        print(f"Generated {len(embeddings)} embeddings")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Union
import numpy as np
from PIL import Image
import torch
from loguru import logger

from framewise.core.frame_extractor import ExtractedFrame
from framewise.core.transcript_extractor import TranscriptSegment


class FrameWiseEmbedder:
    """Generate multimodal embeddings for text and images.
    
    This class provides a unified interface for generating embeddings using:
    - **CLIP** (Contrastive Language-Image Pre-training) for images
    - **Sentence Transformers** for text
    
    Both models are loaded lazily on first use to minimize initialization overhead.
    Supports batch processing for efficient embedding generation.
    
    Attributes:
        text_model_name: Name of the sentence transformer model.
        vision_model_name: Name of the CLIP vision model.
        device: Device being used ('cuda' or 'cpu').
    
    Example:
        Single embeddings::
        
            embedder = FrameWiseEmbedder()
            text_emb = embedder.embed_text("Click the export button")
            image_emb = embedder.embed_image("frame.jpg")
        
        Batch processing (recommended)::
        
            texts = ["First sentence", "Second sentence"]
            text_embs = embedder.embed_text_batch(texts)
            
            images = ["frame1.jpg", "frame2.jpg"]
            image_embs = embedder.embed_image_batch(images)
        
        Full frame embedding::
        
            frames = extractor.extract("video.mp4", transcript)
            embeddings = embedder.embed_frames_batch(frames)
    """
    
    def __init__(
        self,
        text_model: str = "all-MiniLM-L6-v2",
        vision_model: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ) -> None:
        """Initialize the embedder with text and vision models.
        
        Args:
            text_model: Name of the sentence transformer model to use for text
                embeddings. Popular options:
                - 'all-MiniLM-L6-v2': Fast, good quality (default)
                - 'all-mpnet-base-v2': Better quality, slower
                - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual
                Defaults to 'all-MiniLM-L6-v2'.
            vision_model: Name of the CLIP model to use for image embeddings.
                Popular options:
                - 'openai/clip-vit-base-patch32': Balanced (default)
                - 'openai/clip-vit-large-patch14': Better quality, slower
                Defaults to 'openai/clip-vit-base-patch32'.
            device: Device to run models on. Options:
                - 'cuda': Use GPU (requires CUDA)
                - 'cpu': Use CPU only
                - None: Auto-detect (use GPU if available)
                Defaults to None (auto-detect).
        
        Example:
            >>> # Use GPU with larger models
            >>> embedder = FrameWiseEmbedder(
            ...     text_model="all-mpnet-base-v2",
            ...     vision_model="openai/clip-vit-large-patch14",
            ...     device="cuda"
            ... )
            
            >>> # CPU-only with fast models
            >>> embedder = FrameWiseEmbedder(
            ...     text_model="all-MiniLM-L6-v2",
            ...     device="cpu"
            ... )
        """
        self.text_model_name = text_model
        self.vision_model_name = vision_model
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing FrameWiseEmbedder on {self.device}")
        
        # Lazy load models
        self._text_model = None
        self._vision_model = None
        self._vision_processor = None
    
    def _load_text_model(self) -> None:
        """Lazy load the text embedding model.
        
        Loads the Sentence Transformer model on first use to avoid initialization
        overhead when the embedder is created but not immediately used.
        
        Raises:
            ImportError: If sentence-transformers package is not installed.
            RuntimeError: If the model fails to load.
        """
        if self._text_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading text model: {self.text_model_name}")
                self._text_model = SentenceTransformer(
                    self.text_model_name,
                    device=self.device
                )
                logger.success("Text model loaded")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Install it with: pip install sentence-transformers"
                )
    
    def _load_vision_model(self) -> None:
        """Lazy load the vision embedding model.
        
        Loads the CLIP model and processor on first use to avoid initialization
        overhead when the embedder is created but not immediately used.
        
        Raises:
            ImportError: If transformers package is not installed.
            RuntimeError: If the model fails to load.
        """
        if self._vision_model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
                logger.info(f"Loading vision model: {self.vision_model_name}")
                self._vision_model = CLIPModel.from_pretrained(self.vision_model_name)
                self._vision_processor = CLIPProcessor.from_pretrained(self.vision_model_name)
                self._vision_model = self._vision_model.to(self.device)
                logger.success("Vision model loaded")
            except ImportError:
                raise ImportError(
                    "transformers is not installed. "
                    "Install it with: pip install transformers"
                )
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text.
        
        Converts text into a dense vector representation using Sentence Transformers.
        The embedding captures semantic meaning, enabling similarity search.
        
        Args:
            text: Text string to embed. Can be a word, sentence, or paragraph.
        
        Returns:
            Embedding vector as numpy array. Dimension depends on the model
            (typically 384 for MiniLM, 768 for MPNet).
        
        Raises:
            ImportError: If sentence-transformers is not installed.
        
        Example:
            >>> embedder = FrameWiseEmbedder()
            >>> emb = embedder.embed_text("Click the export button")
            >>> print(emb.shape)
            (384,)
            >>> # Embeddings can be compared for similarity
            >>> emb2 = embedder.embed_text("Press the save icon")
            >>> similarity = np.dot(emb, emb2)
        """
        self._load_text_model()
        embedding = self._text_model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_text_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently.
        
        Processes multiple texts in batches for better performance compared to
        embedding one-by-one. Shows progress bar for large batches.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process in each batch. Larger batches
                are faster but use more memory. Defaults to 32.
        
        Returns:
            Array of embeddings with shape (n_texts, embedding_dim).
        
        Raises:
            ImportError: If sentence-transformers is not installed.
        
        Example:
            >>> embedder = FrameWiseEmbedder()
            >>> texts = [
            ...     "Click the button",
            ...     "Select the menu",
            ...     "Open the dialog"
            ... ]
            >>> embeddings = embedder.embed_text_batch(texts)
            >>> print(embeddings.shape)
            (3, 384)
        """
        self._load_text_model()
        embeddings = self._text_model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings
    
    def embed_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Generate embedding for an image using CLIP.
        
        Converts an image into a dense vector representation using CLIP's vision
        encoder. The embedding captures visual features and can be compared with
        text embeddings for multimodal search.
        
        Args:
            image_path: Path to the image file. Supports common formats
                (jpg, png, etc.).
        
        Returns:
            Embedding vector as numpy array. Dimension is 512 for CLIP base models.
        
        Raises:
            ImportError: If transformers package is not installed.
            FileNotFoundError: If the image file doesn't exist.
            PIL.UnidentifiedImageError: If the file is not a valid image.
        
        Example:
            >>> embedder = FrameWiseEmbedder()
            >>> emb = embedder.embed_image("frame_0001.jpg")
            >>> print(emb.shape)
            (512,)
        """
        self._load_vision_model()
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Process and embed
        inputs = self._vision_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self._vision_model.get_image_features(**inputs)
        
        # Convert to numpy
        embedding = image_features.cpu().numpy().flatten()
        return embedding
    
    def embed_image_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 8
    ) -> np.ndarray:
        """Generate embeddings for multiple images efficiently.
        
        Processes multiple images in batches for better performance compared to
        embedding one-by-one. Particularly beneficial when using GPU.
        
        Args:
            image_paths: List of paths to image files.
            batch_size: Number of images to process in each batch. Larger batches
                are faster but use more GPU memory. Defaults to 8.
        
        Returns:
            Array of embeddings with shape (n_images, embedding_dim).
        
        Raises:
            ImportError: If transformers package is not installed.
            FileNotFoundError: If any image file doesn't exist.
        
        Example:
            >>> embedder = FrameWiseEmbedder(device="cuda")
            >>> images = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]
            >>> embeddings = embedder.embed_image_batch(images, batch_size=8)
            >>> print(embeddings.shape)
            (3, 512)
        """
        self._load_vision_model()
        
        all_embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load images
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            
            # Process batch
            inputs = self._vision_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self._vision_model.get_image_features(**inputs)
            
            # Convert to numpy
            embeddings = image_features.cpu().numpy()
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def embed_frame(self, frame: ExtractedFrame) -> Dict[str, Union[str, float, np.ndarray, None]]:
        """Generate embeddings for both the frame image and its transcript.
        
        Creates multimodal embeddings by processing both the visual content
        (image) and textual content (transcript) of a frame.
        
        Args:
            frame: ExtractedFrame object containing image path and optional
                transcript segment.
        
        Returns:
            Dictionary containing:
            - frame_id: Unique frame identifier
            - timestamp: Frame timestamp in seconds
            - image_embedding: Image embedding vector
            - text_embedding: Text embedding vector (or None if no transcript)
            - text: Transcript text (or None if no transcript)
            - frame_path: Path to the frame image
            - extraction_reason: Why this frame was extracted
            - quality_score: Frame quality score
        
        Raises:
            FileNotFoundError: If the frame image file doesn't exist.
        
        Example:
            >>> embedder = FrameWiseEmbedder()
            >>> frame = ExtractedFrame(
            ...     frame_id="frame_0001",
            ...     path=Path("frame.jpg"),
            ...     timestamp=12.5,
            ...     transcript_segment=segment
            ... )
            >>> result = embedder.embed_frame(frame)
            >>> print(result['image_embedding'].shape)
            (512,)
            >>> print(result['text_embedding'].shape)
            (384,)
        """
        # Embed the image
        image_embedding = self.embed_image(frame.path)
        
        # Embed the transcript text if available
        text_embedding = None
        text = None
        if frame.transcript_segment:
            text = frame.transcript_segment.text
            text_embedding = self.embed_text(text)
        
        return {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "image_embedding": image_embedding,
            "text_embedding": text_embedding,
            "text": text,
            "frame_path": str(frame.path),
            "extraction_reason": frame.extraction_reason,
            "quality_score": frame.quality_score,
        }
    
    def embed_frames_batch(
        self,
        frames: List[ExtractedFrame],
        batch_size: int = 8
    ) -> List[Dict[str, Union[str, float, np.ndarray, None]]]:
        """Generate embeddings for multiple frames efficiently.
        
        Processes multiple frames in batches, generating both image and text
        embeddings. This is significantly faster than processing frames one-by-one,
        especially when using GPU.
        
        Args:
            frames: List of ExtractedFrame objects to embed.
            batch_size: Number of images to process in each batch. Larger batches
                are faster but use more GPU memory. Defaults to 8.
        
        Returns:
            List of dictionaries, one per frame, each containing:
            - frame_id, timestamp, image_embedding, text_embedding, text,
              frame_path, extraction_reason, quality_score
        
        Raises:
            ImportError: If required packages are not installed.
            FileNotFoundError: If any frame image file doesn't exist.
        
        Example:
            >>> embedder = FrameWiseEmbedder(device="cuda")
            >>> frames = extractor.extract("video.mp4", transcript)
            >>> embeddings = embedder.embed_frames_batch(frames, batch_size=16)
            >>> print(f"Generated {len(embeddings)} multimodal embeddings")
            Generated 15 multimodal embeddings
            >>> # Each embedding has both image and text components
            >>> print(embeddings[0]['image_embedding'].shape)
            (512,)
            >>> print(embeddings[0]['text_embedding'].shape)
            (384,)
        """
        logger.info(f"Embedding {len(frames)} frames...")
        
        # Extract image paths and texts
        image_paths = [frame.path for frame in frames]
        texts = [
            frame.transcript_segment.text if frame.transcript_segment else ""
            for frame in frames
        ]
        
        # Batch embed images
        logger.info("Generating image embeddings...")
        image_embeddings = self.embed_image_batch(image_paths, batch_size)
        
        # Batch embed texts
        logger.info("Generating text embeddings...")
        text_embeddings = self.embed_text_batch(texts, batch_size=32)
        
        # Combine into result dictionaries
        results = []
        for i, frame in enumerate(frames):
            result = {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "image_embedding": image_embeddings[i],
                "text_embedding": text_embeddings[i],
                "text": texts[i],
                "frame_path": str(frame.path),
                "extraction_reason": frame.extraction_reason,
                "quality_score": frame.quality_score,
            }
            results.append(result)
        
        logger.success(f"Generated embeddings for {len(frames)} frames")
        return results
    
    def get_embedding_dimensions(self) -> Dict[str, int]:
        """Get the dimensions of the text and image embeddings.
        
        Loads both models (if not already loaded) and returns their embedding
        dimensions. Useful for setting up vector databases or understanding
        memory requirements.
        
        Returns:
            Dictionary with 'text_embedding_dim' and 'image_embedding_dim' keys.
        
        Raises:
            ImportError: If required packages are not installed.
        
        Example:
            >>> embedder = FrameWiseEmbedder()
            >>> dims = embedder.get_embedding_dimensions()
            >>> print(dims)
            {'text_embedding_dim': 384, 'image_embedding_dim': 512}
            >>> # Combined embedding dimension for hybrid search
            >>> total_dim = dims['text_embedding_dim'] + dims['image_embedding_dim']
            >>> print(f"Combined dimension: {total_dim}")
            Combined dimension: 896
        """
        self._load_text_model()
        self._load_vision_model()
        
        # Get dimensions
        text_dim = self._text_model.get_sentence_embedding_dimension()
        
        # For CLIP, we need to check the model config
        image_dim = self._vision_model.config.projection_dim
        
        return {
            "text_embedding_dim": text_dim,
            "image_embedding_dim": image_dim
        }
