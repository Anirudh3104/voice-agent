"""
Main entry point for the Voice-Based Government Scheme Agent (Telugu).

Flow:
Mic (STT) → Telugu Text → Agent (Eligibility + RAG) → Telugu Response → TTS
"""

# -----------------------------
# Imports
# -----------------------------
from data.scheme import schemes

from agent.agent_loop import agent_loop
from rag.knowledge import build_scheme_knowledge
from rag.vector_store import build_faiss_index

from voice.stt import listen_telugu   # your STT function
from voice.tts import speak_telugu    # your TTS function


# -----------------------------
# Initialization
# -----------------------------
def initialize_system():
    """
    Initialize RAG knowledge and vector store.
    This runs once at startup.
    """
    print("🔧 Initializing scheme knowledge and vector store...")

    scheme_knowledge = build_scheme_knowledge(schemes)
    index, embedding_model = build_faiss_index(scheme_knowledge)

    print("✅ System initialized successfully.\n")

    return scheme_knowledge, index, embedding_model


# -----------------------------
# Main Voice Agent Loop
# -----------------------------
def main():
    # Initialize tools
    scheme_knowledge, index, embedding_model = initialize_system()

    print("🎙️ Voice-Based Government Scheme Agent is READY\n")

    while True:
        # -----------------------------
        # 1. Listen to user (STT)
        # -----------------------------
        print("🎧 వినిపించండి... (Speak in Telugu)")
        user_text = listen_telugu()

        if not user_text or user_text.strip() == "":
            speak_telugu("మీ మాటలు వినిపించలేదు. దయచేసి మళ్లీ ప్రయత్నించండి.")
            continue

        print(f"📝 మీరు చెప్పింది: {user_text}")

        # Exit condition
        if user_text.lower() in ["exit", "quit", "బయటకు"]:
            speak_telugu("ధన్యవాదాలు! మళ్లీ కలుద్దాం.")
            break

        # -----------------------------
        # 2. Run Agent (Text-based)
        # -----------------------------
        response_text = agent_loop(
            user_input=user_text,
            scheme_knowledge=scheme_knowledge
        )

        if not response_text:
            response_text = "క్షమించండి, మళ్లీ చెప్పగలరా?"

        print(f"🤖 Agent Response:\n{response_text}")

        # -----------------------------
        # 3. Speak response (TTS)
        # -----------------------------
        speak_telugu(response_text)


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
