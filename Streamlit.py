import streamlit as st
from CourtAssistant import chatbot, load_pdf_to_db
import uuid


# -------------------------
# Helper Functions
# -------------------------

def generate_thread_id():
    return str(uuid.uuid4())


def create_new_chat():

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    if thread_id not in st.session_state["conversations"]:
        st.session_state["conversations"][thread_id] = []


# -------------------------
# Session State Initialization
# -------------------------

if "conversations" not in st.session_state:
    st.session_state["conversations"] = {}

if "thread_id" not in st.session_state:
    create_new_chat()

if "pdf_loaded" not in st.session_state:
    st.session_state["pdf_loaded"] = False

if "pdf_path" not in st.session_state:
    st.session_state["pdf_path"] = None


# -------------------------
# Main UI
# -------------------------

st.title("⚖️ Smart Court Assistant")

st.write(
    "Ask legal questions based on Supreme Court judgments and uploaded case files."
)


# -------------------------
# Sidebar
# -------------------------

st.sidebar.title("My Conversations")

if st.sidebar.button("➕ New Conversation"):
    create_new_chat()


for thread_id in st.session_state["conversations"]:

    label = f"Chat {thread_id[:8]}"

    if st.sidebar.button(label, key=thread_id):
        st.session_state["thread_id"] = thread_id


# -------------------------
# PDF Upload
# -------------------------

uploaded_file = st.sidebar.file_uploader(
    "Select a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    pdf_path = f"temp_{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    load_pdf_to_db(pdf_path)

    st.session_state["pdf_loaded"] = True
    st.session_state["pdf_path"] = pdf_path

    st.sidebar.success("PDF processed successfully!")


# -------------------------
# Current Chat
# -------------------------

current_thread = st.session_state["thread_id"]

messages = st.session_state["conversations"][current_thread]


# -------------------------
# Display Previous Messages
# -------------------------

for msg in messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------
# User Input
# -------------------------

user_query = st.chat_input("Enter your legal question")


if user_query:

    # -------------------------
    # Show User Message
    # -------------------------

    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state["conversations"][current_thread].append(
        {
            "role": "user",
            "content": user_query
        }
    )

    # -------------------------
    # Generate Response
    # -------------------------

    with st.chat_message("assistant"):

        placeholder = st.empty()

        try:

            result = chatbot.invoke(
                {
                    "query": user_query,
                    "pdf_uploaded": st.session_state["pdf_loaded"]
                }
            )

            answer = result["response"]

            placeholder.markdown(answer)

        except Exception as e:

            answer = f"❌ Error: {str(e)}"

            placeholder.error(answer)

    # -------------------------
    # Save Assistant Message
    # -------------------------

    st.session_state["conversations"][current_thread].append(
        {
            "role": "assistant",
            "content": answer
        }
    )