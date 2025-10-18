"""Transcript correction utilities for fixing common transcription errors.

This module provides tools to correct common speech recognition errors in
video transcripts, particularly useful for product names, technical terms,
and domain-specific vocabulary that Whisper might misrecognize.

Example:
    Basic usage::

        from framewise.utils.transcript_corrections import TranscriptCorrector
        
        # Define corrections
        corrector = TranscriptCorrector({
            "Defali": "Definely",
            "expot": "export"
        })
        
        # Correct a transcript
        corrected = corrector.correct_transcript(transcript)
"""

from __future__ import annotations

from typing import Dict, List
import re
from loguru import logger

from framewise.core.transcript_extractor import Transcript, TranscriptSegment


class TranscriptCorrector:
    """Correct common transcription errors in video transcripts.
    
    This class applies find-and-replace corrections to transcript text,
    with case-insensitive matching and case-preserving replacement.
    Useful for fixing product names, technical terms, and other
    domain-specific vocabulary.
    
    Attributes:
        corrections: Dictionary mapping incorrect terms to correct terms.
    
    Example:
        Create and use corrector::
        
            corrector = TranscriptCorrector({
                "Defali": "Definely",
                "expot": "export",
                "clique": "click"
            })
            
            text = "Defali helps you expot documents"
            corrected = corrector.correct_text(text)
            print(corrected)  # "Definely helps you export documents"
        
        Correct full transcript::
        
            transcript = extractor.extract("video.mp4")
            corrected_transcript = corrector.correct_transcript(transcript)
    """
    
    def __init__(self, corrections: Optional[Dict[str, str]] = None) -> None:
        """Initialize the corrector with correction rules.
        
        Args:
            corrections: Dictionary mapping incorrect terms to correct terms.
                Keys are the misrecognized terms, values are the correct terms.
                Matching is case-insensitive, but replacement preserves case.
                Defaults to None (empty corrections).
        
        Example:
            >>> corrector = TranscriptCorrector({
            ...     "Defali": "Definely",
            ...     "expot": "export"
            ... })
            >>> corrector.corrections
            {'Defali': 'Definely', 'expot': 'export'}
        """
        self.corrections = corrections or {}
    
    def add_correction(self, incorrect: str, correct: str) -> None:
        """Add a single correction rule.
        
        Args:
            incorrect: The misrecognized term to find.
            correct: The correct term to replace it with.
        
        Example:
            >>> corrector = TranscriptCorrector()
            >>> corrector.add_correction("Defali", "Definely")
            >>> corrector.add_correction("expot", "export")
        """
        self.corrections[incorrect] = correct
        logger.debug(f"Added correction: '{incorrect}' → '{correct}'")
    
    def add_corrections(self, corrections: Dict[str, str]) -> None:
        """Add multiple correction rules at once.
        
        Args:
            corrections: Dictionary of incorrect -> correct term mappings.
        
        Example:
            >>> corrector = TranscriptCorrector()
            >>> corrector.add_corrections({
            ...     "Defali": "Definely",
            ...     "expot": "export",
            ...     "clique": "click"
            ... })
        """
        self.corrections.update(corrections)
        logger.debug(f"Added {len(corrections)} corrections")
    
    def correct_text(self, text: str) -> str:
        """Apply corrections to text with case preservation.
        
        Performs case-insensitive matching but preserves the case pattern
        of the original text in the replacement:
        - ALL CAPS → ALL CAPS
        - Title Case → Title Case
        - lowercase → lowercase
        
        Args:
            text: Text to correct.
        
        Returns:
            Corrected text with case preserved.
        
        Example:
            >>> corrector = TranscriptCorrector({"defali": "definely"})
            >>> corrector.correct_text("DEFALI is great")
            'DEFINELY is great'
            >>> corrector.correct_text("Defali is great")
            'Definely is great'
            >>> corrector.correct_text("defali is great")
            'definely is great'
        """
        corrected = text
        
        for incorrect, correct in self.corrections.items():
            # Case-insensitive replacement with case preservation
            def replace_preserve_case(match):
                original = match.group(0)
                if original.isupper():
                    return correct.upper()
                elif original[0].isupper():
                    return correct.capitalize()
                else:
                    return correct.lower()
            
            pattern = re.compile(re.escape(incorrect), re.IGNORECASE)
            corrected = pattern.sub(replace_preserve_case, corrected)
        
        return corrected
    
    def correct_segment(self, segment: TranscriptSegment) -> TranscriptSegment:
        """Correct a single transcript segment.
        
        Creates a new segment with corrected text while preserving
        the original timing information.
        
        Args:
            segment: TranscriptSegment to correct.
        
        Returns:
            New TranscriptSegment with corrected text and original timing.
        
        Example:
            >>> corrector = TranscriptCorrector({"expot": "export"})
            >>> segment = TranscriptSegment(0.0, 2.5, "Click expot button")
            >>> corrected = corrector.correct_segment(segment)
            >>> print(corrected.text)
            Click export button
        """
        corrected_text = self.correct_text(segment.text)
        
        return TranscriptSegment(
            start=segment.start,
            end=segment.end,
            text=corrected_text
        )
    
    def correct_transcript(self, transcript: Transcript) -> Transcript:
        """Correct all segments in a transcript.
        
        Applies corrections to all segments and the full text, creating
        a new Transcript object with corrected content.
        
        Args:
            transcript: Transcript to correct.
        
        Returns:
            New Transcript with corrected segments and full text.
            Original video path and language are preserved.
        
        Example:
            >>> corrector = TranscriptCorrector({
            ...     "Defali": "Definely",
            ...     "expot": "export"
            ... })
            >>> transcript = extractor.extract("video.mp4")
            >>> corrected = corrector.correct_transcript(transcript)
            >>> # Save corrected version
            >>> corrected.save("corrected_transcript.json")
        """
        logger.info(f"Correcting transcript with {len(self.corrections)} rules")
        
        corrected_segments = [
            self.correct_segment(seg) for seg in transcript.segments
        ]
        
        corrected_full_text = self.correct_text(transcript.full_text)
        
        # Count corrections made
        corrections_made = sum(
            1 for orig, corr in zip(transcript.segments, corrected_segments)
            if orig.text != corr.text
        )
        
        if corrections_made > 0:
            logger.success(f"Applied corrections to {corrections_made} segments")
        else:
            logger.info("No corrections needed")
        
        return Transcript(
            video_path=transcript.video_path,
            language=transcript.language,
            segments=corrected_segments,
            full_text=corrected_full_text
        )


# Common product/brand name corrections
COMMON_CORRECTIONS = {
    # Add your product-specific corrections here
    "Defali": "Definely",
    "DefaliDraft": "DefinelyDraft",
    # Add more as needed
}


def create_product_corrector(product_terms: Optional[Dict[str, str]] = None) -> TranscriptCorrector:
    """Create a corrector with common product name corrections.
    
    Convenience function that creates a TranscriptCorrector pre-loaded with
    common product name corrections, plus any additional custom terms.
    
    Args:
        product_terms: Additional product-specific terms to correct.
            These will be merged with COMMON_CORRECTIONS. Defaults to None.
    
    Returns:
        TranscriptCorrector instance with combined corrections.
    
    Example:
        >>> # Use common corrections only
        >>> corrector = create_product_corrector()
        
        >>> # Add custom product terms
        >>> corrector = create_product_corrector({
        ...     "MyProduct": "MyProduct Pro",
        ...     "expot": "export"
        ... })
        >>> 
        >>> # Apply to transcript
        >>> corrected = corrector.correct_transcript(transcript)
    """
    corrections = COMMON_CORRECTIONS.copy()
    if product_terms:
        corrections.update(product_terms)
    
    return TranscriptCorrector(corrections)
