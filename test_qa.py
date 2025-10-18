"""
Test the complete FrameWise Q&A system with Claude
"""

from pathlib import Path
from loguru import logger
from framewise import (
    TranscriptExtractor,
    FrameExtractor,
    FrameWiseEmbedder,
    FrameWiseVectorStore,
    FrameWiseQA,
)
from framewise.utils.transcript_corrections import create_product_corrector


def main():
    """Test Q&A system with Claude"""
    
    VIDEO_PATH = "/Users/mes/Desktop/Screen Recording 2025-10-17 at 15.45.45.mov"
    
    logger.info("🎬 FrameWise Q&A System Test")
    logger.info("=" * 60)
    
    # Check if we already have processed data
    db_path = Path("test_outputs/search_test.db")
    frames_dir = Path("test_outputs/frames")
    
    if not db_path.exists() or not frames_dir.exists():
        logger.info("\n📦 Processing video (first time setup)...")
        logger.info("-" * 60)
        
        # Extract transcript
        logger.info("Extracting transcript...")
        transcript_extractor = TranscriptExtractor(model_size="base")
        transcript = transcript_extractor.extract(VIDEO_PATH)
        logger.success(f"✓ {len(transcript.segments)} segments")
        
        # Correct common transcription errors
        logger.info("Correcting transcription errors...")
        corrector = create_product_corrector()
        transcript = corrector.correct_transcript(transcript)
        logger.success("✓ Transcript corrected")
        
        # Extract frames
        logger.info("Extracting frames...")
        frame_extractor = FrameExtractor(
            strategy="hybrid",
            max_frames_per_video=15,
            scene_threshold=0.3,
            quality_threshold=0.1
        )
        frames = frame_extractor.extract(
            video_path=VIDEO_PATH,
            transcript=transcript,
            output_dir="test_outputs/frames"
        )
        logger.success(f"✓ {len(frames)} frames")
        
        if len(frames) == 0:
            logger.error("No frames extracted - cannot proceed")
            return
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embedder = FrameWiseEmbedder()
        embeddings = embedder.embed_frames_batch(frames, batch_size=4)
        logger.success(f"✓ {len(embeddings)} embeddings")
        
        # Create database
        logger.info("Creating vector database...")
        vector_store = FrameWiseVectorStore(
            db_path="test_outputs/search_test.db",
            table_name="frames"
        )
        vector_store.create_table(embeddings, mode="overwrite")
        logger.success("✓ Database created")
    else:
        logger.info("\n✓ Using existing processed data")
        embedder = FrameWiseEmbedder()
        vector_store = FrameWiseVectorStore(
            db_path="test_outputs/search_test.db",
            table_name="frames"
        )
    
    # Initialize Q&A system
    logger.info("\n🤖 Initializing Q&A System with Claude...")
    logger.info("-" * 60)
    
    try:
        qa = FrameWiseQA(
            vector_store=vector_store,
            embedder=embedder,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=0.7
        )
        logger.success("✓ Q&A system ready")
    except ValueError as e:
        logger.error(f"❌ {e}")
        logger.info("\n💡 To fix:")
        logger.info("1. Copy .env.example to .env")
        logger.info("2. Add your Claude API key to .env")
        logger.info("3. Run this script again")
        return
    
    # Ask questions
    logger.info("\n🔍 Asking Questions...")
    logger.info("=" * 60)
    
    questions = [
        "How do I open the Definely add-in?",
        "What happens after I click Open Defali?",
        "How do I get started with DefaliDraft?",
    ]
    
    for i, question in enumerate(questions, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Question {i}: {question}")
        logger.info("-" * 60)
        
        try:
            response = qa.ask(question, num_results=3)
            
            logger.success("\n✅ Answer:")
            logger.info(f"{response['answer']}")
            
            logger.info(f"\n📊 Supporting Evidence ({response['num_frames_used']} frames):")
            for j, frame in enumerate(response['relevant_frames'], 1):
                logger.info(f"\n  Frame {j}:")
                logger.info(f"    Time: {frame['timestamp']:.1f}s")
                logger.info(f"    Text: '{frame['text']}'")
                if frame['frame_path']:
                    logger.info(f"    Image: {Path(frame['frame_path']).name}")
        
        except Exception as e:
            logger.error(f"❌ Error: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.success("✅ Q&A Test Complete!")
    logger.info("\n🎯 What just happened:")
    logger.info("   1. Retrieved relevant frames from vector database")
    logger.info("   2. Built context with transcripts + timestamps")
    logger.info("   3. Claude generated natural language answers")
    logger.info("   4. Provided visual evidence (frame references)")
    
    logger.info("\n🚀 FrameWise is now a complete AI assistant!")


if __name__ == "__main__":
    main()
