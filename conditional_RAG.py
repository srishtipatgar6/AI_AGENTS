
import os
from pathlib import Path
from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --------------------------------------------------
# STEP 1: PDF PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

ACADEMIC_PDF = BASE_DIR / "academics_handbook.pdf"
FEE_PDF = BASE_DIR / "fee_structure.pdf"

# --------------------------------------------------
# STEP 2: INITIALIZE EMBEDDINGS
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# STEP 3: BUILD RAG RETRIEVERS
# --------------------------------------------------

def build_retriever(pdf_path: Path):

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Please check whether the PDF exists."
        )

    print(f"\nLoading PDF: {pdf_path.name}")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    if not documents:
        raise ValueError(f"No content extracted from {pdf_path.name}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(f"No text chunks generated from {pdf_path.name}")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    print(f"Successfully indexed {len(chunks)} chunks.")

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )


# Build retrievers
# Both PDFs must exist in the same folder as this Python file.

fee_retriever = build_retriever(FEE_PDF)

# Only build the academic retriever if the PDF exists.
# This allows the fee assistant to run without the academic PDF.

academic_retriever = None

if ACADEMIC_PDF.exists():
    academic_retriever = build_retriever(ACADEMIC_PDF)
else:
    print(
        "\nWARNING: academics_handbook.pdf not found. "
        "Academic queries will not have document retrieval."
    )

# --------------------------------------------------
# STEP 4: INITIALIZE GROQ LLM
# --------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please configure it in your .env file."
    )

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.4,
    api_key=GROQ_API_KEY
)

# --------------------------------------------------
# STEP 5: DEFINE STATE
# --------------------------------------------------

class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# --------------------------------------------------
# STEP 6: CLASSIFIER NODE
# --------------------------------------------------

def classifier_node(state: State) -> dict:

    last_message = state["messages"][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"

        "Use 'academic' for questions about attendance, exams, grading, "
        "credits, promotion, course structure, summer training, "
        "or degree requirements.\n"

        "Use 'fee' for questions about tuition, payment, refund, "
        "late charges, scholarships, or money-related topics.\n"

        "Use 'general' for greetings, casual talk, or unrelated questions.\n\n"

        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)

    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"

    elif "fee" in category:
        category = "fee"

    else:
        category = "general"

    return {"query_type": category}


# --------------------------------------------------
# STEP 7: ACADEMIC RAG NODE
# --------------------------------------------------

def academic_rag_node(state: State) -> dict:

    query = state["messages"][-1].content

    if academic_retriever is None:
        return {
            "retrieved_context": "ACADEMIC_DOCUMENT_NOT_AVAILABLE"
        }

    docs = academic_retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    if not context.strip():
        context = "NO_RELEVANT_CONTEXT_FOUND"

    return {"retrieved_context": context}


# --------------------------------------------------
# STEP 8: FEE RAG NODE
# --------------------------------------------------

def fee_rag_node(state: State) -> dict:

    query = state["messages"][-1].content

    docs = fee_retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    if not context.strip():
        context = "NO_RELEVANT_CONTEXT_FOUND"

    return {"retrieved_context": context}


# --------------------------------------------------
# STEP 9: GENERAL NODE
# --------------------------------------------------

def general_node(state: State) -> dict:

    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED"
    }


# --------------------------------------------------
# STEP 10: RESPONSE NODE
# --------------------------------------------------

def response_node(state: State) -> dict:

    query = state["messages"][-1].content

    programme = state.get("programme", "Unknown")

    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":

        prompt = (
            f"You are a friendly college assistant talking to a "
            f"{programme} student.\n\n"
            f"Answer this general question:\n{query}"
        )

    elif context == "ACADEMIC_DOCUMENT_NOT_AVAILABLE":

        prompt = (
            "The academic handbook PDF is unavailable. "
            "Do not invent college-specific rules, attendance "
            "requirements, or academic policies. "
            "Politely inform the student that the academic document "
            "is not currently available.\n\n"
            f"Question: {query}"
        )

    elif context == "NO_RELEVANT_CONTEXT_FOUND":

        prompt = (
            "The provided college documents do not contain relevant "
            "information to answer this question. "
            "Do not invent college-specific policies or fee amounts. "
            "Politely tell the student that the information could "
            "not be found in the available documents.\n\n"
            f"Question: {query}"
        )

    else:

        prompt = (
            f"You are a college assistant helping a "
            f"{programme} student.\n\n"

            "Answer using ONLY the following official document context. "
            "Do not invent fees, deadlines, rules, or policies. "
            "If the answer is missing, clearly say so.\n\n"

            f"Context:\n{context}\n\n"

            f"Question: {query}\n\n"

            "Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)

    return {
        "messages": [
            ("ai", response.content.strip())
        ]
    }


# --------------------------------------------------
# STEP 11: ROUTER FUNCTION
# --------------------------------------------------

def route_query(state: State):

    if state["query_type"] == "academic":
        return "academic_rag"

    elif state["query_type"] == "fee":
        return "fee_rag"

    else:
        return "general"


# --------------------------------------------------
# STEP 12: BUILD LANGGRAPH
# --------------------------------------------------

graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

graph.add_edge(START, "classifier")

graph.add_conditional_edges(
    "classifier",
    route_query
)

graph.add_edge("academic_rag", "response")
graph.add_edge("fee_rag", "response")
graph.add_edge("general", "response")

graph.add_edge("response", END)

app = graph.compile()


# --------------------------------------------------
# STEP 13: RUN COLLEGE ASSISTANT
# --------------------------------------------------

print("\nWelcome to the College Assistant!\n")

print("Which programme are you in?")
print("1. BCA")
print("2. BBA")
print("3. B.Com (H)")

choice = input("\nEnter 1, 2 or 3: ").strip()

programme_map = {
    "1": "BCA",
    "2": "BBA",
    "3": "B.Com (H)"
}

student_programme = programme_map.get(choice, "BCA")

print(
    f"\nGreat! You're set as a {student_programme} student."
)

while True:

    user_query = input("\nYou: ").strip()

    if user_query.lower() in ["exit", "quit"]:
        print("\nThank you for using the College Assistant!")
        break

    if not user_query:
        continue

    try:

        result = app.invoke({
            "programme": student_programme,
            "messages": [
                ("human", user_query)
            ]
        })

        print(
            f"\nAssistant: {result['messages'][-1].content}"
        )

    except Exception as e:

        print(f"\nError occurred: {e}")
