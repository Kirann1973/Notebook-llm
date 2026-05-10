/**
 * rag.js — Full RAG Pipeline
 *
 * LLM (generation): Groq API         — free, fast
 * Embeddings:       @xenova/transformers (local, runs on your machine, NO API key needed)
 * Vector DB:        Qdrant
 */

import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { QdrantVectorStore } from "@langchain/qdrant";
import { Embeddings } from "@langchain/core/embeddings";
import { pipeline } from "@xenova/transformers";
import Groq from "groq-sdk";

// ----------------------------------------------------------------
// CONFIG
// ----------------------------------------------------------------

/**
 * Chunking strategy: Recursive Character Text Splitter
 * Splits by: paragraph → newline → sentence → word → character
 * - chunkSize:    max characters per chunk
 * - chunkOverlap: overlap to avoid losing context at boundaries
 */
const CHUNK_SIZE = 1200;
const CHUNK_OVERLAP = 200;

// Number of top chunks to retrieve per query
const RETRIEVAL_K = 4;

/**
 * Local embedding model (downloads ~25MB on first run, then cached)
 * "Xenova/all-MiniLM-L6-v2" — fast, lightweight, good quality
 * No API key needed. Runs entirely on your CPU.
 *
 * Other options if you want higher quality (larger download):
 *   "Xenova/all-mpnet-base-v2"       (~420MB, better quality)
 *   "Xenova/paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
 */
const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";

/**
 * Groq model for answer generation
 * Free options: "llama3-70b-8192" | "llama3-8b-8192" | "mixtral-8x7b-32768"
 * TODO (YOU): Change if needed — full list: https://console.groq.com/docs/models
 */
const GENERATION_MODEL = "llama-3.3-70b-versatile";

// ----------------------------------------------------------------
// LOCAL EMBEDDINGS — wraps @xenova/transformers into LangChain format
// ----------------------------------------------------------------

/**
 * XenovaEmbeddings
 * A custom LangChain-compatible embeddings class using @xenova/transformers.
 * Downloads the model on first run and caches it locally in ./node_modules/.cache
 * No internet needed after first download.
 */
class XenovaEmbeddings extends Embeddings {
  constructor() {
    super({});
    this.pipelinePromise = null; // lazy-loaded pipeline
  }

  // Lazy-load the pipeline once, reuse across calls
  async getPipeline() {
    if (!this.pipelinePromise) {
      console.log(
        `[EMBEDDINGS] Loading local model: ${EMBEDDING_MODEL} (first run downloads ~25MB)`
      );
      this.pipelinePromise = pipeline("feature-extraction", EMBEDDING_MODEL);
    }
    return this.pipelinePromise;
  }

  // Mean pooling: average all token embeddings → single vector
  meanPool(output, attentionMask) {
    const { data, dims } = output;
    const [batchSize, seqLen, hiddenSize] = dims;
    const result = [];

    for (let b = 0; b < batchSize; b++) {
      const vec = new Array(hiddenSize).fill(0);
      let count = 0;

      for (let s = 0; s < seqLen; s++) {
        const maskVal = attentionMask?.data
          ? attentionMask.data[b * seqLen + s]
          : 1;

        if (maskVal > 0) {
          for (let h = 0; h < hiddenSize; h++) {
            vec[h] += data[b * seqLen * hiddenSize + s * hiddenSize + h];
          }
          count++;
        }
      }

      // Normalize
      const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0)) || 1;
      result.push(vec.map((v) => v / (norm * count || 1)));
    }
    return result;
  }

  // Embed a batch of texts
  async embedDocuments(texts) {
    const extractor = await this.getPipeline();
    const results = [];

    // Process in small batches to avoid memory issues on large docs
    const BATCH_SIZE = 16;
    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batch = texts.slice(i, i + BATCH_SIZE);
      const output = await extractor(batch, {
        pooling: "mean",
        normalize: true,
      });
      const { data, dims } = output;
      const [batchSize, hiddenSize] = [batch.length, dims[dims.length - 1]];

      for (let b = 0; b < batchSize; b++) {
        results.push(
          Array.from(data.slice(b * hiddenSize, (b + 1) * hiddenSize))
        );
      }
    }
    return results;
  }

  // Embed a single query string
  async embedQuery(text) {
    const vecs = await this.embedDocuments([text]);
    return vecs[0];
  }
}

// ----------------------------------------------------------------
// STEP 1 — INGESTION
// ----------------------------------------------------------------
async function loadDocument(filePath, mimeType) {
  if (mimeType === "application/pdf" || filePath.endsWith(".pdf")) {
    const loader = new PDFLoader(filePath);
    return await loader.load();
  }
  const fs = await import("fs/promises");
  const text = await fs.readFile(filePath, "utf-8");
  return [{ pageContent: text, metadata: { source: filePath } }];
}

// ----------------------------------------------------------------
// STEP 2 — CHUNKING
// ----------------------------------------------------------------
async function chunkDocuments(docs) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
    separators: ["\n\n", "\n", ". ", " ", ""],
  });
  return await splitter.splitDocuments(docs);
}

// ----------------------------------------------------------------
// STEP 3 & 4 — EMBEDDING + STORAGE
// ----------------------------------------------------------------
async function embedAndStore(chunks, collectionName) {
  const embeddings = new XenovaEmbeddings();

  /**
   * TODO (YOU): Make sure Qdrant is accessible:
   *   Local:  docker run -p 6333:6333 qdrant/qdrant
   *   Cloud:  QDRANT_URL + QDRANT_API_KEY in .env
   */
  await QdrantVectorStore.fromDocuments(chunks, embeddings, {
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY || undefined,
    collectionName,
  });

  return { chunksCount: chunks.length, collectionName };
}

// ----------------------------------------------------------------
// STEP 5 — RETRIEVAL
// ----------------------------------------------------------------
async function retrieveChunks(query, collectionName) {
  const embeddings = new XenovaEmbeddings();

  const vectorStore = await QdrantVectorStore.fromExistingCollection(
    embeddings,
    {
      url: process.env.QDRANT_URL,
      apiKey: process.env.QDRANT_API_KEY || undefined,
      collectionName,
    }
  );

  const retriever = vectorStore.asRetriever({ k: RETRIEVAL_K });
  return await retriever.invoke(query);
}

// ----------------------------------------------------------------
// STEP 6 — GENERATION via Groq
// ----------------------------------------------------------------
async function generateAnswer(query, chunks) {
  /**
   * TODO (YOU): GROQ_API_KEY must be in your .env
   *   Free key: https://console.groq.com/keys
   */
  const client = new Groq({ apiKey: process.env.GROQ_API_KEY });

  const context = chunks
    .map((chunk, i) => {
      const page =
        chunk.metadata?.loc?.pageNumber ?? chunk.metadata?.page ?? "?";
      return `[Chunk ${i + 1} | Page ${page}]\n${chunk.pageContent}`;
    })
    .join("\n\n---\n\n");

  const systemPrompt = `You are a document assistant. Answer the user's question using ONLY the document context below.

Rules:
- Answer strictly from the provided context. Do NOT use your own training knowledge.
- If the answer is NOT found in the context, say: "I couldn't find this in the document."
- Cite page numbers when possible (e.g., "According to page 3, ...").
- Be concise but complete.

Document Context:
${context}`;

  const response = await client.chat.completions.create({
    model: GENERATION_MODEL,
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: query },
    ],
    temperature: 0.2,
    max_tokens: 1024,
  });

  return {
    answer: response.choices[0].message.content,
    sources: chunks.map((c) => ({
      page: c.metadata?.loc?.pageNumber ?? c.metadata?.page ?? "?",
      preview: c.pageContent.substring(0, 150) + "...",
    })),
  };
}

// ----------------------------------------------------------------
// PUBLIC API
// ----------------------------------------------------------------
export async function indexDocument(filePath, mimeType, collectionName) {
  const docs = await loadDocument(filePath, mimeType);
  const chunks = await chunkDocuments(docs);
  return await embedAndStore(chunks, collectionName);
}

export async function queryDocument(question, collectionName) {
  const chunks = await retrieveChunks(question, collectionName);
  if (!chunks.length) {
    return {
      answer: "No relevant content found in the document.",
      sources: [],
    };
  }
  return await generateAnswer(question, chunks);
}
