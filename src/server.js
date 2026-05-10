/**
 * server.js — Express HTTP Server
 * Exposes REST endpoints consumed by the frontend UI
 */

import "dotenv/config";
import express from "express";
import multer from "multer";
import path from "path";
import { fileURLToPath } from "url";
import { v4 as uuidv4 } from "uuid";
import { indexDocument, queryDocument } from "./rag.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();

// ----------------------------------------------------------------
// MIDDLEWARE
// ----------------------------------------------------------------
app.use(express.json());
app.use(express.static(path.join(__dirname, "../public"))); // serve the frontend

// Multer: handles multipart/form-data (file uploads)
// TODO (YOU): Files are saved to ./uploads/ on your server.
//   On a cloud platform (Render, Railway, Fly.io) you may need
//   to use ephemeral storage or an S3 bucket instead.
const upload = multer({
  dest: path.join(__dirname, "../uploads/"),
  limits: { fileSize: 20 * 1024 * 1024 }, // 20 MB max
  fileFilter: (req, file, cb) => {
    const allowed = ["application/pdf", "text/plain"];
    if (allowed.includes(file.mimetype)) cb(null, true);
    else cb(new Error("Only PDF and .txt files are supported."));
  },
});

// ----------------------------------------------------------------
// ROUTES
// ----------------------------------------------------------------

/**
 * POST /api/upload
 * Accepts a PDF or TXT file, runs the full indexing pipeline,
 * and returns a session collectionName to use for subsequent queries.
 */
app.post("/api/upload", upload.single("document"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded." });

    // Generate a unique collection name for this document session
    // e.g., "doc_3f2a1b" — stored in Qdrant as an isolated namespace
    const collectionName = `doc_${uuidv4().split("-")[0]}`;

    console.log(`[UPLOAD] Processing: ${req.file.originalname} → ${collectionName}`);

    const result = await indexDocument(
      req.file.path,
      req.file.mimetype,
      collectionName
    );

    console.log(`[UPLOAD] Done — ${result.chunksCount} chunks indexed.`);

    res.json({
      success: true,
      collectionName,
      fileName: req.file.originalname,
      chunksCount: result.chunksCount,
    });
  } catch (err) {
    console.error("[UPLOAD ERROR]", err);
    res.status(500).json({ error: err.message || "Upload failed." });
  }
});

/**
 * POST /api/query
 * Accepts { question, collectionName }, retrieves relevant chunks,
 * generates a grounded answer, and returns it.
 */
app.post("/api/query", async (req, res) => {
  try {
    const { question, collectionName } = req.body;

    if (!question || !collectionName) {
      return res.status(400).json({ error: "Missing question or collectionName." });
    }

    console.log(`[QUERY] "${question}" on ${collectionName}`);

    const result = await queryDocument(question, collectionName);

    res.json({ success: true, ...result });
  } catch (err) {
    console.error("[QUERY ERROR]", err);
    res.status(500).json({ error: err.message || "Query failed." });
  }
});

// ----------------------------------------------------------------
// START
// ----------------------------------------------------------------
// TODO (YOU): On deployment platforms (Render, Railway, etc.)
//   PORT is automatically set by the platform via environment variable.
//   Locally it defaults to 3000.
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`\n🚀 NotebookLM RAG running at http://localhost:${PORT}\n`);
  console.log("Make sure:");
  console.log("  ✓ OPENAI_API_KEY is set in .env");
  console.log("  ✓ Qdrant is running (docker run -p 6333:6333 qdrant/qdrant)");
  console.log("    OR QDRANT_URL points to your Qdrant Cloud cluster\n");
});
