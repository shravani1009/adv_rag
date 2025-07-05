import os
import asyncio
# Set USER_AGENT first to avoid warnings from any web requests during imports
os.environ["USER_AGENT"] = "Advanced-RAG-Agent/1.0"

import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
# Updated import for HuggingFace embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub
import time

# Load environment variables
load_dotenv()

# Fix for asyncio event loop issue
try:
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
except:
    pass

# Page configuration
st.set_page_config(
    page_title="🤖 Advanced RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .tool-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-container {
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        animation: fadeIn 0.3s ease-in;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,0,0,0.05);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
        border-left: 4px solid #4CAF50;
    }
    .agent-message {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
        margin-right: 2rem;
        border-left: 4px solid #FF9800;
    }
    .tool-usage {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        font-size: 0.9rem;
        font-style: italic;
        border-left: 4px solid #9C27B0;
    }
    .example-btn {
        width: 100%;
        margin: 0.2rem 0;
        padding: 0.5rem;
        background-color: #f0f2f6;
        border: 1px solid #ddd;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .example-btn:hover {
        background-color: #e0e2e6;
        transform: translateY(-2px);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_tools_and_agent():
    """Initialize all tools and create the agent"""
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Wikipedia Tool
        wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
        
        # ArXiv Tool
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        
        # RAG Tool - Load and process document
        loader = WebBaseLoader("https://arxiv.org/abs/2309.17074")
        docs = loader.load()
        documents = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        ).split_documents(docs)
        
        vectordb = FAISS.from_documents(documents, embeddings)
        retriever = vectordb.as_retriever()
        
        retriever_tool = create_retriever_tool(
            retriever,
            "langsmith_search",
            "Search LangSmith for relevant information. For any questions regarding the langsmith, you must use this tool to search for relevant information."
        )
        
        # Combine tools
        tools = [wiki_tool, arxiv_tool, retriever_tool]
        
        # Initialize LLM
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("⚠️ GOOGLE_API_KEY not found in environment variables!")
            return None, None
            
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=google_api_key
        )
        
        # Create agent
        prompt = hub.pull("hwchase17/openai-tools-agent")
        agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        return agent_executor, tools
        
    except Exception as e:
        st.error(f"❌ Error initializing agent: {str(e)}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Advanced RAG Agent</h1>', unsafe_allow_html=True)
    st.markdown("### Ask questions and get intelligent answers from multiple sources!")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_executor" not in st.session_state:
        with st.spinner("🔄 Initializing AI Agent... This may take a moment."):
            agent_executor, tools = initialize_tools_and_agent()
            st.session_state.agent_executor = agent_executor
            st.session_state.tools = tools
            if agent_executor:
                st.success("✅ Agent initialized successfully!")
            else:
                st.error("❌ Failed to initialize agent. Please check your API key.")
                return
    if "waiting_for_response" not in st.session_state:
        st.session_state.waiting_for_response = False
    
    # Create layout
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Sidebar content
        st.markdown("### 🛠️ Agent Tools")
        
        st.markdown("""
        <div class="tool-card">
            <h4>📚 Wikipedia Search</h4>
            <p>General knowledge queries</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tool-card">
            <h4>🔬 ArXiv Search</h4>
            <p>Academic papers & research</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tool-card">
            <h4>📄 LangSmith RAG</h4>
            <p>Specific document retrieval</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**🤖 Model:** Gemini-1.5-Flash")
        st.markdown("**🔍 Embeddings:** HuggingFace")
        st.markdown("**💾 Vector DB:** FAISS")
        
        st.markdown("---")
        st.markdown("### � Chat Statistics")
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")
        if st.session_state.messages:
            user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
            agent_msgs = len([m for m in st.session_state.messages if m["role"] == "agent"])
            st.markdown(f"**Questions:** {user_msgs}")
            st.markdown(f"**Responses:** {agent_msgs}")
    
    with col1:
        st.markdown("### 💬 Chat Interface")
        
        # Add example questions at the top of chat
        if not st.session_state.messages:
            st.markdown("#### 💡 Try asking about:")
            
            example_questions = [
                "What is LangSmith?",
                "Tell me about machine learning", 
                "Latest research in AI",
                "How does RAG work?",
                "What is quantum computing?"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(example_questions):
                with cols[i % 2]:
                    if st.button(question, key=f"example_{i}", use_container_width=True):
                        if not st.session_state.waiting_for_response:
                            process_user_input(question)
            
            st.markdown("---")
        
        # Chat container
        chat_container = st.container()
        with chat_container:
            
            # Display chat history
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                elif message["role"] == "agent":
                    st.markdown(f"""
                    <div class="chat-message agent-message">
                        <strong>🤖 Agent:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                elif message["role"] == "tool":
                    st.markdown(f"""
                    <div class="chat-message tool-usage">
                        <strong>🛠️ Tools Used:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("Ask me anything... 🚀", disabled=st.session_state.waiting_for_response)
        
        # Handle user input
        if user_input and not st.session_state.waiting_for_response:
            process_user_input(user_input)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🤖 Advanced RAG Agent | Powered by Gemini-1.5-Flash, LangChain & Streamlit<br>
        💡 Ask questions about general topics, research papers, or LangSmith specifically!
    </div>
    """, unsafe_allow_html=True)

def process_user_input(user_input):
    """Process user input and get agent response"""
    if not st.session_state.agent_executor:
        st.error("❌ Agent not initialized!")
        return
    
    # Set waiting state
    st.session_state.waiting_for_response = True
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get agent response
    try:
        with st.spinner("🤔 Agent is thinking..."):
            response = st.session_state.agent_executor.invoke({"input": user_input})
            agent_response = response.get("output", "Sorry, I couldn't generate a response.")
            
            # Add agent response
            st.session_state.messages.append({"role": "agent", "content": agent_response})
            
            # Show which tools were used (if available)
            if "intermediate_steps" in response and response["intermediate_steps"]:
                tools_used = [step[0].tool for step in response["intermediate_steps"]]
                if tools_used:
                    tools_str = ", ".join(set(tools_used))
                    st.session_state.messages.append({"role": "tool", "content": tools_str})
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.session_state.messages.append({"role": "agent", "content": f"Sorry, I encountered an error: {str(e)}"})
    
    finally:
        # Reset waiting state
        st.session_state.waiting_for_response = False
        # Force rerun to update UI
        st.rerun()

if __name__ == "__main__":
    main()
