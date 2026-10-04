import streamlit as st
from graph import graph


# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Java RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ---------------------------------
# Header
# ---------------------------------

st.title("🤖 Java RAG Chatbot")

st.caption(
    "Ask questions from your Java notes using RAG + LangGraph + Ollama."
)


# ---------------------------------
# Clear Chat
# ---------------------------------

if st.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.rerun()


# ---------------------------------
# Initialize Chat History
# ---------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# ---------------------------------
# Display Previous Messages
# ---------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Show sources for assistant messages
        if (
            message["role"] == "assistant"
            and message.get("documents")
        ):

            with st.expander("📚 Retrieved Sources"):

                for i, document in enumerate(
                    message["documents"],
                    start=1
                ):

                    st.markdown(f"### Source {i}")

                    st.write(document)


# ---------------------------------
# Chat Input
# ---------------------------------

question = st.chat_input(
    "Ask something about Java..."
)


# ---------------------------------
# Process Question
# ---------------------------------

if question:

    # Save user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Display user question
    with st.chat_message("user"):

        st.markdown(question)


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching your Java notes..."
        ):

            result = graph.invoke(
                {
                    "question": question
                }
            )

        answer = result["answer"]

        documents = result.get(
            "documents",
            []
        )


        # Display answer
        st.markdown(answer)


        # Display retrieved sources
        if documents:

            with st.expander(
                "📚 Retrieved Sources"
            ):

                for i, document in enumerate(
                    documents,
                    start=1
                ):

                    st.markdown(
                        f"### Source {i}"
                    )

                    st.write(
                        document.page_content
                    )


    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "documents": [
                document.page_content
                for document in documents
            ]
        }
    )