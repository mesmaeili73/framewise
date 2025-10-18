"""
Transcript correction utilities for fixing common transcription errors
"""

from typing import Dict, List
from loguru import logger

from framewise.core.transcript_extractor import Transcript, TranscriptSegment


class TranscriptCorrector:
    """Correct common transcription errors in video transcripts"""
    
    def __init__(self, corrections: Dict[str, str] = None):
        """
        Initialize the corrector with a dictionary of corrections
        
        Args:
            corrections: Dictionary mapping incorrect -> correct terms
                        e.g., {"Defali": "Definely", "expot": "export"}
        """
        self.corrections = corrections or {}
    
    def add_correction(self, incorrect: str, correct: str):
        """Add a correction rule"""
        self.corrections[incorrect] = correct
        logger.debug(f"Added correction: '{incorrect}' → '{correct}'")
    
    def add_corrections(self, corrections: Dict[str, str]):
        """Add multiple correction rules"""
        self.corrections.update(corrections)
        logger.debug(f"Added {len(corrections)} corrections")
    
    def correct_text(self, text: str) -> str:
        """
        Apply corrections to text
        
        Args:
            text: Text to correct
            
        Returns:
            Corrected text
        """
        corrected = text
        
        for incorrect, correct in self.corrections.items():
            # Case-insensitive replacement
            # But preserve original case in output
            import re
            
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
        """
        Correct a transcript segment
        
        Args:
            segment: TranscriptSegment to correct
            
        Returns:
            New TranscriptSegment with corrected text
        """
        corrected_text = self.correct_text(segment.text)
        
        return TranscriptSegment(
            start=segment.start,
            end=segment.end,
            text=corrected_text
        )
    
    def correct_transcript(self, transcript: Transcript) -> Transcript:
        """
        Correct all segments in a transcript
        
        Args:
            transcript: Transcript to correct
            
        Returns:
            New Transcript with corrected segments
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


def create_product_corrector(product_terms: Dict[str, str] = None) -> TranscriptCorrector:
    """
    Create a corrector with common product name corrections
    
    Args:
        product_terms: Additional product-specific terms to correct
        
    Returns:
        TranscriptCorrector instance
    """
    corrections = COMMON_CORRECTIONS.copy()
    if product_terms:
        corrections.update(product_terms)
    
    return TranscriptCorrector(corrections)
