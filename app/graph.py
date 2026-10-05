from typing import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from rag import load_vectorstore


# ==========================================
# STATE
# ==========================================

class GraphState(TypedDict, total=False):
    question: str
    context: str
    documents: list
    answer: str


# ==========================================
# LOAD VECTOR STORE
# ==========================================

vectorstore = load_vectorstore()


# ==========================================
# OLLAMA LLM
# ==========================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# ==========================================
# RETRIEVE
# ==========================================

def retrieve(state: GraphState):

    question = state["question"]

    documents = vectorstore.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return {
        "question": question,
        "context": context,
        "documents": documents
    }


# ==========================================
# GENERATE
# ==========================================

def generate(state: GraphState):

    question = state["question"]
    context = state["context"]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are Ashok's A.I., a helpful Java study assistant.

Answer the user's question using ONLY the provided context.

Do not use outside knowledge.

If the answer is not available in the context, say:

I don't have enough information in the provided document.

Give a clear and simple explanation suitable for a Java student.
"""
            ),
            (
                "human",
                """
Context:

{context}

Question:

{question}
"""
            )
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    answer = response.content

    if isinstance(answer, list):

        text_parts = []

        for item in answer:

            if isinstance(item, dict):

                text_parts.append(
                    item.get("text", "")
                )

            elif isinstance(item, str):

                text_parts.append(item)

        answer = "".join(text_parts)

    return {
        "answer": answer
    }


# ==========================================
# LANGGRAPH
# ==========================================

graph_builder = StateGraph(GraphState)

graph_builder.add_node(
    "retrieve",
    retrieve
)

graph_builder.add_node(
    "generate",
    generate
)

graph_builder.set_entry_point(
    "retrieve"
)

graph_builder.add_edge(
    "retrieve",
    "generate"
)

graph_builder.add_edge(
    "generate",
    END
)

graph = graph_builder.compile()


# ==========================================
# TERMINAL CHAT
# ==========================================

if __name__ == "__main__":

    print("=" * 50)
    print("ASHOK'S A.I. - JAVA RAG CHATBOT")
    print("=" * 50)
    print("Model: llama3.2")
    print("Embedding: nomic-embed-text")
    print("Type 'exit' to quit.")
    print("=" * 50)

    while True:

        try:

            question = input("\nYou: ")

        except (KeyboardInterrupt, EOFError):

            print("\nGoodbye!")

            break

        question = question.strip()

        if question.lower() in ["exit", "quit"]:

            print("Goodbye!")

            break

        if not question:

            continue

        try:

            result = graph.invoke(
                {
                    "question": question
                }
            )

            print("\nAssistant:")
            print(result.get("answer", ""))

        except Exception as e:

            print("\nERROR TYPE:", type(e).__name__)
            print("ERROR:", repr(e))