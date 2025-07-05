# 🤖 Advanced RAG Agent - Web Application

A powerful AI agent with web interface that can answer questions using multiple information sources through Retrieval-Augmented Generation (RAG).

## 🚀 Features

- **🌐 Web Interface**: Beautiful Streamlit-based UI
- **🤖 Multi-Source Intelligence**: Wikipedia, ArXiv, and custom document search
- **🆓 Completely Free**: Uses Gemini-1.5-Flash and free tools
- **🔍 Smart RAG**: Retrieval-Augmented Generation for accurate answers
- **📱 Responsive Design**: Works on desktop and mobile

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Model**: Google Gemini-1.5-Flash
- **Embeddings**: HuggingFace all-MiniLM-L6-v2
- **Vector DB**: FAISS
- **Framework**: LangChain

## 📋 Prerequisites

1. **Google AI API Key**: Get it from [Google AI Studio](https://makersuite.google.com/)
2. **Python 3.8+**

## 🔧 Local Setup

1. **Clone/Download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables**:
   - Create `.env` file in the project root
   - Add your API key:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     ```
4. **Run the application**:
   ```bash
   streamlit run app.py
   ```
5. **Open your browser** to `http://localhost:8501`

## 🌐 Deployment Options

### Option 1: Streamlit Community Cloud (FREE)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add your `GOOGLE_API_KEY` in Streamlit secrets
5. Deploy!

### Option 2: Heroku
1. Create `Procfile`:
   ```
   web: sh setup.sh && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [general]\n\
   email = \"your-email@domain.com\"\n\
   " > ~/.streamlit/credentials.toml
   echo "\
   [server]\n\
   headless = true\n\
   enableCORS=false\n\
   port = $PORT\n\
   " > ~/.streamlit/config.toml
   ```
3. Deploy to Heroku

### Option 3: Docker
1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```
2. Build and run:
   ```bash
   docker build -t rag-agent .
   docker run -p 8501:8501 --env-file .env rag-agent
   ```

## 📱 Usage

1. **Open the web app**
2. **Ask questions** like:
   - "What is LangSmith?" (uses custom RAG)
   - "Tell me about machine learning" (uses Wikipedia)
   - "Latest AI research papers" (uses ArXiv)
3. **View real-time responses** with tool usage indicators
4. **Chat history** is maintained during the session

## 🎯 Example Questions

- **General Knowledge**: "What is quantum computing?"
- **Research Papers**: "Recent developments in transformer models"
- **LangSmith Specific**: "How does LangSmith help with LLM applications?"
- **Technical Topics**: "Explain RAG architecture"

## 🔒 Security

- API keys stored in `.env` file (never commit to git)
- Environment variables for deployment
- No sensitive data in source code

## 🚀 Performance

- **First load**: ~30 seconds (downloads models)
- **Subsequent queries**: 2-5 seconds
- **Caching**: Models cached for faster responses

## 🛟 Troubleshooting

1. **"API key not found"**: Check your `.env` file
2. **"Resource exhausted"**: Switch to Gemini-1.5-Flash (free tier)
3. **Slow loading**: Models download on first run
4. **Import errors**: Run `pip install -r requirements.txt`

## 📊 Project Structure

```
adv_rag/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
├── README.md          # This file
└── agents.ipynb       # Original notebook development
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**🚀 Ready to deploy your AI agent to the world!**
