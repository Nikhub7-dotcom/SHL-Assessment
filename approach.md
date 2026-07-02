# SHL Assessment Recommender — Approach Document

## Problem & Design Choices

Hiring managers usually don't know what assessment they need until they describe the role out loud. Keyword search doesn't help here — you need to already know what you're looking for. So I built a conversational agent that takes a vague hiring intent and narrows it down to a shortlist through dialogue.

Stack I chose:
- **FastAPI** — lightweight, fast, and gives you Swagger docs for free which made testing easy
- **FAISS + sentence-transformers (all-MiniLM-L6-v2)** — semantic search over the catalog so "Java developer" retrieves Java tests even if the word "Java" isn't in the assessment name
- **Groq (llama-3.3-70b-versatile)** — free, fast, and the 70B model was reliable enough to follow JSON instructions consistently
- **Deployed on Hugging Face Spaces** using Docker SDK — 2GB RAM on free tier, enough to run PyTorch and FAISS comfortably

The API is fully stateless. Every POST /chat carries the full conversation history and the server stores nothing. This was a requirement from the spec and it also made deployment much simpler.

## Retrieval Setup

For each assessment in the catalog, I built one descriptive text string combining the name, description, category keys, and job levels. These get encoded into 384-dimensional vectors using all-MiniLM-L6-v2 and stored in a FAISS IndexFlatL2 index. The index is built once at deploy time and loaded into memory at startup.

At query time, I combine the last user message with the full conversation history into one retrieval query. This matters because if the user said "mid-level" two turns ago, that context should still influence what gets retrieved now. Top 15 results get injected into the LLM's system prompt as a CATALOG CONTEXT block — the LLM can only recommend from what it sees there.

## Prompt Design

The system prompt has three parts:

1. **Rules** — when to clarify, when to recommend, when to refuse, how to handle refinement and comparison. I also added explicit turn pressure at turn 6 to force a recommendation before hitting the 8-turn evaluator cap.
2. **Few-shot examples** — 6 examples covering all required behaviors: vague query, role given, refinement, confirmation, comparison, off-topic refusal. These were derived from the sample conversation traces and made a bigger difference than the rules alone.
3. **Catalog context** — the 15 retrieved assessments, injected fresh on every call.

## Agent Design Decisions

A few specific things I added after reading the PDF carefully:

- **Vague query detection** — if the first message has no specific role in it, the agent asks one clarifying question instead of recommending
- **Comparison detection** — when the user asks to compare two assessments, I retrieve their full descriptions and inject them separately so the answer is grounded in actual catalog data
- **Hallucination guard** — every URL in the response gets validated against the scraped catalog. I also override the LLM's assessment name and test_type with the real values from the catalog using the URL as a lookup key
- **Retry logic with 25s timeout** — handles Groq rate limits and keeps responses within the evaluator's 30-second timeout

## What Didn't Work

- **llama-3.1-8b-instant** — too small, kept ignoring the conversation content and giving generic welcome messages
- **Forced JSON mode** with qwen/qwen3.6-27b returned empty responses on Groq — switched to prompt-based JSON enforcement with fallback parsing instead
- **Retrieving only from the last message** — missed context from earlier turns, fixed by combining the full conversation into the retrieval query
- **llama3-70b-8192 and qwen/qwen3.6-27b** — both deprecated or broken on Groq's free tier during development, had to switch models mid-build

## Evaluation

I used the 10 sample conversation traces in two ways:

1. Extracted 6 representative patterns and embedded them as few-shot examples directly in the system prompt
2. Built a test script (`test_conversations.py`) that replays conversations turn by turn against the live endpoint and prints what the agent recommends at each step

Things I tested manually:
- Vague first message → agent clarifies, doesn't recommend
- Role given → agent recommends immediately, no unnecessary questions
- "Add personality tests" → shortlist updates, doesn't restart
- Comparison question → answer comes from catalog descriptions, not model memory
- Off-topic request → politely refused
- All returned URLs verified to exist in the catalog

## AI Tools Used

I used Claude for code assistance — boilerplate, debugging errors, and prompt iteration. The design decisions (retrieval strategy, prompt structure, agent behaviors, model selection) were my own. I can defend every part of this in an interview.