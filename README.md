**LinkedRAG** is a RAG app that allows users to upload their LinkedIn connections data and interact with it through natural language queries. The system processes CSV files exported from LinkedIn, creates vector embeddings of the connection information, and LLM models to answer questions about the user's professional network.

## TODO

- [x] Add local model option
- [x] Add robust CSV schema mapping for LinkedIn exports (auto-detect common column names)
- [x] Parse and normalize dates in Connected On; support recent/most-recent queries
- [ ] Add optional enrichment from LinkedIn profile URLs with rate limiting and caching
- [ ] Extract and store additional fields (location, industry, headline) when available
- [ ] Improve hiring intent detection with heuristic signals beyond embeddings
- [ ] Persist FAISS index and processed connections to disk; add load on startup
- [ ] Provide Windows-safe vector index fallback when FAISS is unavailable
- [x] Ensure embedding dimension consistency across mock and OpenAI modes
- [x] Deduplicate connections and harden stable unique ID generation
- [x] Expose retrieved citations in chat responses (names, companies, links)
- [x] Add filters in UI sidebar (company, role, location) and sorting
- [ ] Integrate dynamic suggested queries from ChatBot into UI
- [ ] Export query results to CSV (selected or all matches)
- [ ] Add unit tests for DataProcessor, VectorStore, and ChatBot search paths
- [ ] Add logging and user-facing error messages with remediation tips
- [ ] Support ingesting LinkedIn Data Archive ZIP (auto-locate Connections.csv)
- [x] Improve connections list search UX with fielded and highlighted matches
- [ ] Add ability to save/load multiple datasets and switch between them
- [ ] Pass chat history to LLM for multi-turn context when generating answers
- [ ] Ensure complete app functionality across ingest, retrieval, chat, and persistence
- [ ] Add automated LinkedIn data ingestion via API with auth and rate limits
- [ ] Enrich profiles from external sources (e.g., GitHub, Crunchbase, company sites)
- [ ] Add advanced filtering and multi-field sorting for search results
- [ ] Build connection relationship graph and basic network analysis metrics
- [ ] Export search results and insights to CSV and JSON formats