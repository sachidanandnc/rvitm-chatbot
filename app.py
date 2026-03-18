# app.py
import streamlit as st
from rag_backend import init_rag_chain

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RVITM Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 RVITM College Assistant")
st.caption("Ask me anything about courses, admissions, labs, or faculty!")

# ─── LOAD RAG CHAIN ONCE ──────────────────────────────────────────────────────

@st.cache_resource
def load_chain():
    with st.spinner("Loading college knowledge base... please wait ⏳"):
        return init_rag_chain()

chain = load_chain()

# ─── SESSION STATE ────────────────────────────────────────────────────────────

if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 1

if "chats" not in st.session_state:
    st.session_state.chats = [{
        "id": st.session_state.chat_counter,
        "title": "New Chat",
        "messages": []
    }]

if "active_chat" not in st.session_state:
    st.session_state.active_chat = st.session_state.chat_counter

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_active_chat():
    for chat in st.session_state.chats:
        if chat["id"] == st.session_state.active_chat:
            return chat
    return st.session_state.chats[0]

def start_new_chat():
    st.session_state.chat_counter += 1
    new_id = st.session_state.chat_counter
    st.session_state.chats.insert(0, {
        "id": new_id,
        "title": "New Chat",
        "messages": []
    })
    st.session_state.active_chat = new_id

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 💬 Chats")

    if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
        start_new_chat()
        st.rerun()

    st.markdown("---")

    for i, chat in enumerate(st.session_state.chats):
        is_active = chat["id"] == st.session_state.active_chat
        label = f"{'▶ ' if is_active else ''}{chat['title']}"
        if st.button(label, key=f"chat_btn_{i}", use_container_width=True):
            st.session_state.active_chat = chat["id"]
            st.rerun()

# ─── ACTIVE CHAT ──────────────────────────────────────────────────────────────

active_chat = get_active_chat()

for message in active_chat["messages"]:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

# ─── CHAT INPUT ───────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about RVITM courses, admissions, faculty..."):

    active_chat["messages"].append({
        "role": "user",
        "content": prompt,
        "avatar": "🧑‍💻"
    })
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Auto-title using first message
    if active_chat["title"] == "New Chat":
        active_chat["title"] = prompt[:40] + ("..." if len(prompt) > 40 else "")

    # Get answer from RAG chain
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Searching through college records..."):
            result = chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": str(active_chat["id"])}}
            )
            answer = result["answer"]

        st.markdown(answer)

        active_chat["messages"].append({
            "role": "assistant",
            "content": answer,
            "avatar": "🎓"
        })

    st.rerun()