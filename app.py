import streamlit as st

from src.retriever import load_embeddings
from src.rag_pipeline import answer_question

# 1. Streamlit Page Configuration

st.set_page_config(
    page_title="AI Teaching Assistant", page_icon="🎓", layout="centered"
)

# 2. Application Title

st.title("🎓 AI Teaching Assistant")

st.caption("Ask questions about the Sigma Web Development course.")

# 3. Load Embeddings


@st.cache_resource
def get_embeddings():
    return load_embeddings()


try:
    df = get_embeddings()

except Exception as e:
    st.error(f"Unable to load course embeddings: {e}")
    st.stop()

# 4. Sidebar

st.sidebar.title("Settings")

st.sidebar.metric("Available Course Chunks", len(df))

top_k = st.sidebar.slider(
    "Number of chunks to retrieve", min_value=1, max_value=10, value=3
)

# 5. Initialize Chat History

if "messages" not in st.session_state:
    st.session_state.messages = []

# 6. Format Timestamps


def format_timestamp(seconds):

    seconds = int(float(seconds))

    minutes = seconds // 60
    remaining_seconds = seconds % 60

    return f"{minutes:02d}:{remaining_seconds:02d}"


# 7. Display Previous Messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display retrieved sources
        if message.get("sources"):

            with st.expander("View Course Sources"):

                for source in message["sources"]:

                    start = format_timestamp(source["start"])
                    end = format_timestamp(source["end"])

                    st.markdown(f"""
                        **Video:** {source["title"]}

                        **Video Number:** {source["number"]}

                        **Timestamp:** {start} - {end}

                        **Content:** {source["text"]}
                        """)

                    st.divider()


# 8. User Input

user_query = st.chat_input("Ask something about the course...")


if user_query:

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Searching course content and generating answer..."):

            try:

                result = answer_question(query=user_query, df=df, top_k=top_k)

                answer = result["answer"]

                sources = result["sources"][
                    ["title", "number", "start", "end", "text", "similarity_score"]
                ].to_dict(orient="records")

                st.markdown(answer)

                # Display sources
                with st.expander("View Course Sources"):

                    for source in sources:

                        start = format_timestamp(source["start"])
                        end = format_timestamp(source["end"])

                        st.markdown(f"""
                            **Video:** {source["title"]}

                            **Video Number:** {source["number"]}

                            **Timestamp:** {start} - {end}

                            **Content:** {source["text"]}

                            **Similarity Score:** {source["similarity_score"]:.4f}
                            """)

                        st.divider()

            except Exception as e:

                answer = f"Something went wrong: {e}"
                sources = []

                st.error(answer)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
