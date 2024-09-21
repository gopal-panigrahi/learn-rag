import streamlit as st
from store import get_store
from bot import get_bot

st.set_page_config(layout="wide", page_title="Learn RAG", page_icon=":robot_face:")

store = get_store()
bot = get_bot()

st.header("Car Expert - RAG", divider="rainbow")

uploaded_file = st.sidebar.file_uploader(
    "Add File", type=["pdf"], key="add_file_uploader"
)

if uploaded_file is not None:
    with st.spinner("Adding file to repository"):
        success, message = store.add_file(uploaded_file)
        st.toast(message, icon="🟢" if success else "🔴")

st.sidebar.markdown("Files In Knowledge Repository")
for f in store.get_uploaded_files():
    st.sidebar.markdown(f)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(name=message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Enter your query...")
if query:
    with st.chat_message("human"):
        st.markdown(query)
        st.session_state.messages.append({"role": "human", "content": query})

    with st.chat_message(name="ai"):
        response = st.write_stream(bot.get_response(query))
        st.session_state.messages.append({"role": "assistant", "content": response})
