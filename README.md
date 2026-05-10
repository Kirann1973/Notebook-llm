# NotebookLM RAG — Assignment 03

A full RAG (Retrieval-Augmented Generation) pipeline that lets you upload any PDF or TXT document and chat with its content.

---

## Live Demo

Link : notebook-llm-production.up.railway.app

> Example: https://notebook-llm-production.up.railway.app/

---

## Tech Stack

| Layer            | Technology                          |
| ---------------- | ----------------------------------- |
| LLM (Generation) | **Groq API** — llama3-70b-8192      |
| Embeddings       | **OpenAI** — text-embedding-3-large |
| Vector DB        | **Qdrant**                          |
| Backend          | **Node.js + Express**               |
| Frontend         | **Vanilla HTML/JS**                 |

---

## RAG Pipeline

```
File Upload
    ↓
1. INGESTION — PDFLoader / fs.readFile
    ↓
2. CHUNKING — RecursiveCharacterTextSplitter (size: 1200, overlap: 200)
    ↓
3. EMBEDDING — OpenAI text-embedding-3-large
    ↓
4. STORAGE — Qdrant vector store (unique collection per document)
    ↓
  [User asks question]
    ↓
5. RETRIEVAL — Cosine similarity search, top 4 chunks
    ↓
6. GENERATION — Groq llama3-70b-8192, context-grounded prompt
    ↓
  Answer (with source page citations)
```

---

## Chunking Strategy

Uses **RecursiveCharacterTextSplitter** from LangChain.

- **Chunk size**: 1200 characters (~300-400 tokens)
- **Overlap**: 200 characters
- **Separator priority**: paragraph breaks (`\n\n`) → newlines (`\n`) → sentences (`. `) → spaces (` `) → characters

This strategy keeps semantic units (paragraphs, sentences) together as much as possible, and the overlap prevents important context from being lost at chunk boundaries.

---

## Setup

### 1. Clone & install

```bash
git clone <your-repo-url>
cd notebooklm-rag
npm install
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

- `GROQ_API_KEY` — from https://console.groq.com/keys (free)
- `OPENAI_API_KEY` — from https://platform.openai.com/api-keys (needed for embeddings)
- `QDRANT_URL` — local or Qdrant Cloud URL

### 3. Start Qdrant

```bash
# Local via Docker
docker run -p 6333:6333 qdrant/qdrant

# OR use Qdrant Cloud: https://cloud.qdrant.io
```

### 4. Run the app

```bash
npm run dev
# Open http://localhost:3000
```

---

## Deployment (Render + Qdrant Cloud)

1. Create a free cluster on [Qdrant Cloud](https://cloud.qdrant.io)
2. Push code to GitHub (public repo)
3. Deploy on [Render](https://render.com):
   - New → Web Service → Connect repo
   - Build command: `npm install`
   - Start command: `npm start`
   - Add all env vars from `.env`

---

## Example Queries

- "Summarize the main topic of this document"
- "What does page 3 say about embeddings?"
- "List all key points mentioned in the introduction"
- "What is the conclusion of this document?"

---

## Limitations

- Session state is in-memory; refreshing the browser resets the session
- Max file size: 20 MB
- Qdrant collections persist until manually deleted
- Groq rate limits apply on the free tier
