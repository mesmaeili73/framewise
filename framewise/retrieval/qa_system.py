"""
Question-Answering system using LLM with retrieved frames
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional
    pass

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from framewise.embeddings.embedder import FrameWiseEmbedder
from framewise.retrieval.vector_store import FrameWiseVectorStore


class FrameWiseQA:
    """Question-Answering system for video tutorials using LLM"""
    
    SYSTEM_PROMPT = """You are FrameWise, an AI assistant that helps users understand tutorial videos.

You have access to:
1. Video transcripts with timestamps
2. Screenshots from key moments in the video
3. Context about when things were said and shown

Your role:
- Answer user questions about the tutorial clearly and concisely
- Reference specific timestamps when relevant
- Describe what's shown in screenshots when helpful
- Be precise and actionable

Guidelines:
- Keep answers focused and practical
- Mention timestamps in format: "at 12.5 seconds"
- If showing a screenshot, describe what's visible
- If you're not sure, say so - don't make up information
"""
    
    def __init__(
        self,
        vector_store: FrameWiseVectorStore,
        embedder: FrameWiseEmbedder,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Q&A system
        
        Args:
            vector_store: FrameWiseVectorStore instance with indexed frames
            embedder: FrameWiseEmbedder instance for query embedding
            model: Claude model to use
            max_tokens: Maximum tokens in response
            temperature: LLM temperature (0-1)
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter"
            )
        
        # Initialize Claude with LangChain
        self.llm = ChatAnthropic(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
        )
        
        logger.info(f"Initialized FrameWiseQA with {model}")
    
    def ask(
        self,
        question: str,
        num_results: int = 3,
        search_type: str = "hybrid",
        include_frames: bool = True,
    ) -> Dict:
        """
        Ask a question about the tutorial video
        
        Args:
            question: User's question
            num_results: Number of relevant frames to retrieve
            search_type: 'hybrid', 'text', or 'image'
            include_frames: Whether to include frame paths in response
            
        Returns:
            Dictionary with answer, relevant frames, and metadata
        """
        logger.info(f"Question: '{question}'")
        
        # Step 1: Retrieve relevant frames
        logger.info(f"Retrieving {num_results} relevant frames...")
        results = self.vector_store.search_by_text(
            query_text=question,
            embedder=self.embedder,
            limit=num_results,
            search_type=search_type
        )
        
        if not results:
            logger.warning("No relevant frames found")
            return {
                "answer": "I couldn't find any relevant information in the tutorial video for that question.",
                "relevant_frames": [],
                "confidence": 0.0
            }
        
        logger.success(f"Found {len(results)} relevant frames")
        
        # Step 2: Build context from retrieved frames
        context = self._build_context(results)
        
        # Step 3: Generate answer with LLM
        logger.info("Generating answer with Claude...")
        answer = self._generate_answer(question, context)
        
        # Step 4: Format response
        response = {
            "answer": answer,
            "relevant_frames": [
                {
                    "timestamp": r["timestamp"],
                    "text": r["text"],
                    "frame_path": r["frame_path"] if include_frames else None,
                    "frame_id": r["frame_id"],
                }
                for r in results
            ],
            "num_frames_used": len(results),
            "search_type": search_type,
        }
        
        logger.success("Answer generated")
        return response
    
    def _build_context(self, results: List[Dict]) -> str:
        """Build context string from retrieved frames"""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            timestamp = result["timestamp"]
            text = result["text"]
            
            context_parts.append(
                f"[Frame {i} at {timestamp:.1f}s]\n"
                f"Transcript: \"{text}\"\n"
            )
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Claude"""
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", """Based on the following information from a tutorial video, please answer the user's question.

Retrieved Context:
{context}

User Question: {question}

Please provide a clear, helpful answer. Reference specific timestamps when relevant.""")
        ])
        
        # Create chain
        chain = prompt | self.llm
        
        # Invoke
        response = chain.invoke({
            "context": context,
            "question": question
        })
        
        return response.content
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        num_results: int = 3,
    ) -> Dict:
        """
        Multi-turn conversation with context
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            num_results: Number of frames to retrieve per question
            
        Returns:
            Dictionary with answer and relevant frames
        """
        # Get the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        if not last_user_message:
            raise ValueError("No user message found in conversation")
        
        # For now, treat as single question
        # TODO: Implement conversation memory
        return self.ask(last_user_message, num_results=num_results)
    
    def batch_ask(
        self,
        questions: List[str],
        num_results: int = 3,
    ) -> List[Dict]:
        """
        Answer multiple questions
        
        Args:
            questions: List of questions
            num_results: Number of frames per question
            
        Returns:
            List of answer dictionaries
        """
        logger.info(f"Answering {len(questions)} questions...")
        
        answers = []
        for i, question in enumerate(questions, 1):
            logger.info(f"\nQuestion {i}/{len(questions)}")
            answer = self.ask(question, num_results=num_results)
            answers.append(answer)
        
        logger.success(f"Answered {len(questions)} questions")
        return answers
