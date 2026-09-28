"""
chat_engine.py
==============
Builds the LangChain LCEL retrieval chain for StudyBuddy using:
  - ChatGroq as the LLM (fast, free inference)
  - ChromaDB as the retriever (top-k semantic search)
  - RunnableWithMessageHistory for multi-turn conversation memory
  - Two operating modes controlled by the `use_outside_knowledge` flag:
      * Strict mode  — answers ONLY from uploaded materials
      * Extended mode — may use general knowledge, clearly labelling its source
"""

import os
import logging
from typing import List

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of chunks to retrieve per query
TOP_K = 4

# Available Groq models for this account — shown in the sidebar dropdown
AVAILABLE_MODELS = {
    "GPT-OSS 20B (Fast ⚡)": "openai/gpt-oss-20b",
    "GPT-OSS 120B (Best Quality 🏆)": "openai/gpt-oss-120b",
    "Qwen 3.8 27B (Multilingual 🌐)": "qwen/qwen3.8-27b",
}

# Default model — fast and capable for academic Q&A
DEFAULT_MODEL = "openai/gpt-oss-20b"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

# Used when the user has NOT enabled outside knowledge.
# The LLM must answer STRICTLY from the retrieved context.
STRICT_SYSTEM_PROMPT = """You are StudyBuddy, a warm and knowledgeable AI study assistant for university students.

Tone: Always respond in a natural, professional, and kind manner. Be encouraging and supportive.

---
ABOUT STUDYBUDDY (you may always use this knowledge to help users, regardless of mode):

StudyBuddy is an AI-powered study assistant that helps students understand their course materials.
Here is how it works:

1. Upload your materials — Use the sidebar on the left to upload one or more lecture PDFs
   (slides, syllabi, textbook chapters, etc.).
2. Ask questions — Once uploaded, type any question about your materials in the chat box below.
   StudyBuddy will find the most relevant passages and explain them clearly.
3. View sources — Every answer includes a "View source passages" expander so you can see
   exactly which part of your PDF the answer came from, with the page number.
4. Manage your knowledge base — In the sidebar you can tick/untick individual files to choose
   which materials the bot searches, or delete files you no longer need.
5. Choose a model — The sidebar lets you switch between fast and more powerful AI models.
6. Answer modes:
   - Strict mode (default, lock icon): Answers come ONLY from your uploaded PDFs.
   - Extended mode (toggle in sidebar): The bot can also draw on its general knowledge,
     clearly labelling anything that doesn't come from your materials with a ⚠️ tag.
7. Clear chat — Use the "Clear Chat History" button in the sidebar to start a fresh conversation.

Tips:
- Use text-based PDFs (not scanned image PDFs) for best results.
- You can upload multiple files and switch between them at any time.
- The knowledge base persists between sessions — you don't need to re-upload files each time.
- If you haven't uploaded any files yet, start by clicking the upload area in the sidebar.
---

STRICT RULE FOR ACADEMIC QUESTIONS: For any questions about the student's course content,
answer ONLY from the information provided in the <context> section below. Do NOT use your
training data to answer academic questions.

If the academic answer is not in the context, respond with something like:
"I wasn't able to find that in your uploaded materials. Could you double-check the file was
uploaded, or try rephrasing your question?"

When answering academic questions:
- Be clear, concise, and educational.
- Reference the source document and page number when relevant
  (e.g. "According to lecture3.pdf, page 5...").
- Use bullet points or numbered lists for multi-part answers.
- If the context partially answers the question, share what you found and note what is missing.

<context>
{context}
</context>"""


# Used when the user HAS enabled outside knowledge.
# The LLM MUST clearly distinguish between context-grounded and general-knowledge answers.
EXTENDED_SYSTEM_PROMPT = """You are StudyBuddy, a warm and knowledgeable AI study assistant for university students.

Tone: Always respond in a natural, professional, and kind manner. Be encouraging and supportive.

---
ABOUT STUDYBUDDY (you may always use this knowledge to help users, regardless of mode):

StudyBuddy is an AI-powered study assistant that helps students understand their course materials.
Here is how it works:

1. Upload your materials — Use the sidebar on the left to upload one or more lecture PDFs
   (slides, syllabi, textbook chapters, etc.).
2. Ask questions — Once uploaded, type any question about your materials in the chat box below.
   StudyBuddy will find the most relevant passages and explain them clearly.
3. View sources — Every answer includes a "View source passages" expander so you can see
   exactly which part of your PDF the answer came from, with the page number.
4. Manage your knowledge base — In the sidebar you can tick/untick individual files to choose
   which materials the bot searches, or delete files you no longer need.
5. Choose a model — The sidebar lets you switch between fast and more powerful AI models.
6. Answer modes:
   - Strict mode (default, lock icon): Answers come ONLY from your uploaded PDFs.
   - Extended mode (toggle in sidebar): The bot can also draw on its general knowledge,
     clearly labelling anything that doesn't come from your materials with a ⚠️ tag.
7. Clear chat — Use the "Clear Chat History" button in the sidebar to start a fresh conversation.

Tips:
- Use text-based PDFs (not scanned image PDFs) for best results.
- You can upload multiple files and switch between them at any time.
- The knowledge base persists between sessions — you don't need to re-upload files each time.
- If you haven't uploaded any files yet, start by clicking the upload area in the sidebar.
---

For academic questions, you have access to two sources of information:
1. The student's uploaded course materials (provided in <context> below).
2. Your general training knowledge.

IMPORTANT RULES:
- ALWAYS check the uploaded materials FIRST.
- For any information found in the uploaded materials, cite the source
  (e.g. "According to lecture3.pdf, page 5...").
- For any information from your general knowledge (not from the uploaded materials),
  you MUST prefix it with: "⚠️ [General Knowledge — not from your materials]:"
- Clearly separate context-based and general-knowledge sections in your answer.
- If the uploaded materials fully cover the question, do NOT add general knowledge unless asked.

When answering:
- Be clear, concise, and educational.
- Use bullet points or numbered lists for multi-part answers.

<context>
{context}
</context>"""


# ---------------------------------------------------------------------------
# Context formatting helper
# ---------------------------------------------------------------------------

def format_docs(docs: list) -> str:
    """
    Formats a list of retrieved Document objects into a single context string
    that is injected into the system prompt.

    Each chunk is prefixed with its source and page number so the LLM can
    reference them accurately.

    Args:
        docs (list): List of LangChain Document objects from ChromaDB retrieval.

    Returns:
        str: Formatted context string, or a message if no docs were retrieved.
    """
    if not docs:
        return "No relevant passages were found in the uploaded materials."

    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        formatted_chunks.append(
            f"[Passage {i} | Source: {source}, Page {page}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted_chunks)


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def build_chain(
    vector_store: Chroma,
    groq_api_key: str,
    use_outside_knowledge: bool = False,
    model_name: str = DEFAULT_MODEL,
    selected_sources: list = None,
):
    """
    Builds and returns a RunnableWithMessageHistory chain that:
      1. Retrieves the top-k most relevant chunks from ChromaDB
         (optionally filtered to only the user-selected source files)
      2. Formats them into a context string
      3. Injects the context + conversation history into the chosen system prompt
      4. Calls the Groq LLM and streams the response
      5. Maintains full multi-turn conversation memory per session

    Args:
        vector_store (Chroma): The populated ChromaDB vector store.
        groq_api_key (str): Groq API key for authenticating requests.
        use_outside_knowledge (bool): If True, uses the extended prompt that
            allows general knowledge with clear labelling. Defaults to False.
        model_name (str): Groq model to use. Defaults to DEFAULT_MODEL.
        selected_sources (list | None): List of filenames to restrict retrieval
            to. None or empty list means search all uploaded files.

    Returns:
        RunnableWithMessageHistory: The fully assembled conversational chain.
    """
    # 1. Initialise the Groq LLM
    llm = ChatGroq(
        api_key=groq_api_key,
        model=model_name,
        temperature=0.3,
        max_tokens=1024,
        streaming=True,
    )

    # 2. Build search_kwargs — apply source filter when specific files are selected
    search_kwargs: dict = {"k": TOP_K}
    if selected_sources:
        if len(selected_sources) == 1:
            # Exact match filter for a single file
            search_kwargs["filter"] = {"source": selected_sources[0]}
        else:
            # $in operator for multiple files
            search_kwargs["filter"] = {"source": {"$in": list(selected_sources)}}

    # 3. Configure the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    # 3. Select the appropriate system prompt based on mode
    system_prompt = EXTENDED_SYSTEM_PROMPT if use_outside_knowledge else STRICT_SYSTEM_PROMPT

    # 4. Build the full prompt template
    #    MessagesPlaceholder injects the full conversation history here
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # 5. Assemble the LCEL chain using the pipe operator (|)
    #    Flow: question → retrieve docs → format context → fill prompt → LLM → parse output
    chain = (
        {
            # Retrieve relevant context for the current question
            "context": RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
            # Pass the question through unchanged
            "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
            # Pass chat history through unchanged
            "chat_history": RunnablePassthrough() | RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        | prompt
        | llm
        | StrOutputParser()   # Converts AIMessage → plain string
    )

    # 6. Wrap with RunnableWithMessageHistory for automatic history management
    #    This handles injecting and updating the conversation history each turn.
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=_get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    logger.info(
        f"Chain built — model={model_name}, "
        f"k={TOP_K}, "
        f"mode={'extended' if use_outside_knowledge else 'strict'}"
    )

    return chain_with_history


# ---------------------------------------------------------------------------
# Session history store
# ---------------------------------------------------------------------------

# In-memory store mapping session_id → InMemoryChatMessageHistory.
# Since StudyBuddy is a single-user Streamlit app, we use one fixed session ID.
# For a multi-user deployment, this would be replaced by a database-backed store.
_session_store: dict[str, InMemoryChatMessageHistory] = {}


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Returns (or creates) the InMemoryChatMessageHistory object for a given session ID.

    RunnableWithMessageHistory calls this function automatically on each invocation
    to load and then update the conversation history.

    Args:
        session_id (str): Unique identifier for the chat session.

    Returns:
        InMemoryChatMessageHistory: The message history for this session.
    """
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
        logger.info(f"Created new chat history for session: {session_id}")
    return _session_store[session_id]


def clear_session_history(session_id: str) -> None:
    """
    Clears all conversation history for the given session.
    Called when the user clicks "Clear Chat" in the UI.

    Args:
        session_id (str): The session whose history should be cleared.
    """
    if session_id in _session_store:
        _session_store[session_id].clear()
        logger.info(f"Cleared chat history for session: {session_id}")


def get_session_messages(session_id: str) -> List[BaseMessage]:
    """
    Returns the current list of messages for a session.
    Useful for displaying history in the Streamlit UI without re-querying.

    Args:
        session_id (str): The session ID.

    Returns:
        List[BaseMessage]: Ordered list of human/AI messages.
    """
    history = _get_session_history(session_id)
    return history.messages  # type: ignore[attr-defined]
