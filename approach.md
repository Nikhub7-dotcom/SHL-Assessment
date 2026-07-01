# SHL Assessment Recommender — Approach Document

## Problem & Design Choices

The task was to build a conversational agent that takes a hiring manager from a vague intent to a grounded shortlist of SHL assessments. The core challenge is that hiring managers often don't know the right assessment vocabulary, so keyword search fails them. A conversational agent that clarifies, recommends, refines, and compares solves this.

**Architecture:**
- FastAPI for the stateless REST API (lightweight, fast, auto-docs via Swagger)
- FAISS + sentence-transformers (all-MiniLM-L6-v2) for semantic retrieval over the catalog
- Groq (llama-3.3-70b-versatile) for generating grounded, structured replies
- The index is pre-built once at deploy time and loaded into memory at startup

The API is fully stateless — every POST /chat carries the full conversation history. The server stores nothing per conversation, which makes it horizontally scalable and easy to deploy on free-tier hosting.

## Retrieval Setup

Each catalog item is converted into a descriptive text string combining its name, description, category keys, and job levels. These are encoded into 384-dimensional vectors using all-MiniLM-L6-v2 and stored in a FAISS IndexFlatL2 index (exact search, appropriate for ~400 items).

At query time, we combine the last user message with the full conversation history into a single retrieval query. This ensures that context from earlier turns (e.g. "mid-level" mentioned two turns ago) still influences which assessments are retrieved. The top 15 results are injected into the LLM's system prompt as a CATALOG CONTEXT block.

This approach grounds the LLM strictly in real catalog data — it can only recommend what it has seen in the context block, which prevents hallucination of non-existent assessments.

## Prompt Design

The system prompt has three parts:

1. **Rules** — explicit behavioral instructions covering when to clarify, when to recommend, when to refuse, and how to handle refinement and comparison
2. **Few-shot examples** — six concrete examples derived from the provided sample conversations showing exactly what correct JSON output looks like for each scenario (vague query, role given, refinement, confirmation, comparison, off-topic)
3. **Catalog context** — the 15 retrieved assessments injected fresh on every call

The output is constrained to a strict JSON schema with three fields: `reply`, `recommendations`, and `end_of_conversation`. The few-shot examples teach the model the schema better than instructions alone.

## What Didn't Work

- **llama-3.1-8b-instant** was too small to reliably follow the JSON schema and behavioral rules — it kept giving generic welcome messages regardless of the user's input
- **Forced JSON mode (response_format)** with qwen/qwen3.6-27b caused empty responses on Groq — dropped it in favor of prompt-based JSON enforcement with fallback parsing
- **Single retrieval query from last message only** missed context from earlier turns — fixed by combining full conversation history into the retrieval query

## Evaluation Approach

The 10 provided sample conversation traces were used in two ways:

1. **Few-shot examples** — 6 representative patterns were extracted and embedded directly in the system prompt to teach the model correct behavior
2. **Local replay testing** — a test script (`test_conversations.py`) replays conversations turn by turn against the live endpoint and prints recommendations at each step, making it easy to spot when the agent clarifies too much, recommends wrong assessments, or fails to set end_of_conversation correctly

Key behaviors tested manually:
- Agent asks clarifying question for vague queries, recommends immediately when role is given
- Refinement updates the shortlist without restarting
- Comparison answers are drawn from catalog data only
- Off-topic requests are refused
- All returned URLs exist in the scraped catalog

## AI Tools Used

Claude was used for code assistance — generating boilerplate, debugging errors, and suggesting prompt improvements. All design decisions (retrieval strategy, prompt structure, schema design, model selection) were made and understood by the developer. The code reflects actual understanding of the system.