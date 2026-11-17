# DocuMind Plan

1. **Scaffold:** Set up the repo, uv backend, Vite frontend, Docker Compose with Postgres and pgvector, and a health endpoint.
2. **Ingestion:** Upload PDF/DOCX/MD files, parse and chunk them with page numbers kept, embed the chunks and store them in pgvector.
3. **Retrieval + streaming answers:** Run vector search over chunks and stream Claude answers in which every claim cites its document and page.
4. **Frontend chat:** Build the React chat UI with file upload, streamed responses and clickable citations.
5. **Hybrid search + re-ranking:** Combine full-text and vector search, then re-rank the results to improve relevance.
6. **Evals:** Build a question/answer test set and measure retrieval and citation accuracy.
7. **Auth + deploy:** Add user accounts so each user sees only their own documents, then deploy to production.
