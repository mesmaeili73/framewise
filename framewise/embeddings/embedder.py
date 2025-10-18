"""
Multimodal embedding generation for frames and text
"""

from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
from PIL import Image
import torch
from loguru import logger

from framewise.core.frame_extractor import ExtractedFrame
from framewise.core.transcript_extractor import TranscriptSegment


class FrameWiseEmbedder:
    """Generate embeddings for text and images using state-of-the-art models"""
    
    def __init__(
        self,
        text_model: str = "all-MiniLM-L6-v2",
        vision_model: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        """
        Initialize the embedder with text and vision models
        
        Args:
            text_model: Sentence transformer model for text embeddings
            vision_model: CLIP model for image embeddings
            device: Device to run on ('cuda', 'cpu', or None for auto)
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
    
    def _load_text_model(self):
        """Lazy load the text embedding model"""
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
    
    def _load_vision_model(self):
        """Lazy load the vision embedding model"""
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
        """
        Generate embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        self._load_text_model()
        embedding = self._text_model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_text_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts (faster than one-by-one)
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        self._load_text_model()
        embeddings = self._text_model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings
    
    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """
        Generate embedding for an image using CLIP
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Embedding vector as numpy array
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
        image_paths: List[str | Path],
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Generate embeddings for multiple images (faster than one-by-one)
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings (n_images, embedding_dim)
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
    
    def embed_frame(self, frame: ExtractedFrame) -> Dict:
        """
        Generate embeddings for both the frame image and its transcript
        
        Args:
            frame: ExtractedFrame object with image and transcript
            
        Returns:
            Dictionary with both embeddings and metadata
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
    ) -> List[Dict]:
        """
        Generate embeddings for multiple frames efficiently
        
        Args:
            frames: List of ExtractedFrame objects
            batch_size: Batch size for processing
            
        Returns:
            List of dictionaries with embeddings and metadata
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
        """
        Get the dimensions of the embeddings
        
        Returns:
            Dictionary with text and image embedding dimensions
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
