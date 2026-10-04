from typing import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from rag import load_vectorstore


# -----------------------------
# State
# -----------------------------

class GraphState(TypedDict, total=False):
    question: str
    context: str
    documents: list
    answer: str


# -----------------------------
# Load models
# -----------------------------

vectorstore = load_vectorstore()

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# -----------------------------
# Retrieve documents
# -----------------------------

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
        "context": context,
        "documents": documents
    }


# -----------------------------
# Generate answer
# -----------------------------

def generate(state: GraphState):

    question = state["question"]
    context = state["context"]

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful Java study assistant.

Answer the question using ONLY the provided context.

If the answer is not present in the context,
say exactly:

"I don't have enough information in the provided document."

Do not invent information.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return {
        "answer": response.content
    }


# -----------------------------
# Build LangGraph
# -----------------------------

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


# -----------------------------
# Terminal chatbot
# -----------------------------

if __name__ == "__main__":

    print("=" * 50)
    print("JAVA RAG CHATBOT")
    print("=" * 50)

    while True:

        question = input("\nYou: ")

        if question.lower() in ["exit", "quit"]:

            print("Goodbye!")

            break

        result = graph.invoke(
            {
                "question": question
            }
        )

        print("\nAssistant:")
        print(result["answer"])