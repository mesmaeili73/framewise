"""
Test FrameWise with Tableau tutorial video
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
from framewise.utils.transcript_corrections import TranscriptCorrector


def main():
    """Process Tableau video and test Q&A"""
    
    VIDEO_PATH = "/Users/mes/Downloads/videoplayback.mp4"
    
    if not Path(VIDEO_PATH).exists():
        logger.error(f"❌ Video not found: {VIDEO_PATH}")
        return
    
    logger.info("🎬 FrameWise - Tableau Tutorial Test")
    logger.info("=" * 60)
    
    # Step 1: Extract transcript
    logger.info("\n📝 Step 1: Extracting transcript...")
    transcript_extractor = TranscriptExtractor(model_size="base", language="en")
    transcript = transcript_extractor.extract(
        VIDEO_PATH,
        output_path="test_outputs/tableau_transcript.json"
    )
    logger.success(f"✓ Extracted {len(transcript.segments)} segments")
    logger.info(f"   Duration: ~{transcript.segments[-1].end:.1f}s")
    
    # Show first few segments
    logger.info("\n📄 First segments:")
    for seg in transcript.segments[:3]:
        logger.info(f"   [{seg.start:.1f}s]: {seg.text}")
    
    # Step 2: Apply corrections (if needed)
    logger.info("\n🔧 Step 2: Checking for corrections...")
    corrector = TranscriptCorrector({
        # Add Tableau-specific corrections if needed
        # "Tablo": "Tableau",
    })
    transcript = corrector.correct_transcript(transcript)
    logger.success("✓ Transcript ready")
    
    # Step 3: Extract frames
    logger.info("\n🖼️  Step 3: Extracting keyframes...")
    frame_extractor = FrameExtractor(
        strategy="hybrid",
        max_frames_per_video=20,  # More frames for longer video
        scene_threshold=0.3,
        quality_threshold=0.2  # Lower threshold for more frames
    )
    
    frames = frame_extractor.extract(
        video_path=VIDEO_PATH,
        transcript=transcript,
        output_dir="test_outputs/tableau_frames"
    )
    logger.success(f"✓ Extracted {len(frames)} keyframes")
    
    if len(frames) == 0:
        logger.warning("No frames extracted - try lowering quality_threshold")
        return
    
    # Show some frames
    logger.info("\n📸 Sample frames:")
    for frame in frames[:3]:
        logger.info(f"   {frame.timestamp:.1f}s: {frame.extraction_reason}")
        if frame.transcript_segment:
            logger.info(f"      '{frame.transcript_segment.text[:60]}...'")
    
    # Step 4: Generate embeddings
    logger.info("\n🧠 Step 4: Generating embeddings...")
    embedder = FrameWiseEmbedder()
    embeddings = embedder.embed_frames_batch(frames, batch_size=8)
    logger.success(f"✓ Generated {len(embeddings)} embeddings")
    
    # Step 5: Create vector database
    logger.info("\n💾 Step 5: Creating vector database...")
    vector_store = FrameWiseVectorStore(
        db_path="test_outputs/tableau.db",
        table_name="frames"
    )
    vector_store.create_table(embeddings, mode="overwrite")
    logger.success("✓ Database created")
    
    # Step 6: Initialize Q&A
    logger.info("\n🤖 Step 6: Initializing Q&A with Claude...")
    qa = FrameWiseQA(
        vector_store=vector_store,
        embedder=embedder,
        model="claude-3-5-sonnet-20241022"
    )
    logger.success("✓ Q&A system ready")
    
    # Step 7: Ask questions about Tableau
    logger.info("\n🔍 Step 7: Asking Questions about Tableau...")
    logger.info("=" * 60)
    
    questions = [
        "How do I create a visualization in Tableau?",
        "How do I connect to data?",
        "What are the first steps in Tableau?",
    ]
    
    for i, question in enumerate(questions, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"❓ Question {i}: {question}")
        logger.info("-" * 60)
        
        try:
            response = qa.ask(question, num_results=3)
            
            logger.success("\n💬 Claude's Answer:")
            logger.info(f"{response['answer']}")
            
            logger.info(f"\n📊 Evidence ({response['num_frames_used']} frames):")
            for j, frame in enumerate(response['relevant_frames'], 1):
                logger.info(f"   Frame {j}: {frame['timestamp']:.1f}s")
                logger.info(f"   Text: '{frame['text'][:80]}...'")
        
        except Exception as e:
            logger.error(f"❌ Error: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.success("✅ Tableau Video Test Complete!")
    logger.info("\n📁 Output:")
    logger.info("   - test_outputs/tableau_transcript.json")
    logger.info("   - test_outputs/tableau_frames/")
    logger.info("   - test_outputs/tableau.db/")
    logger.info("\n🎯 FrameWise successfully processed a Tableau tutorial!")


if __name__ == "__main__":
    main()
