# 🚀 Autonomous Multi-Agent Research System

This project implements an **agentic AI system** with:

- Multi-agent orchestration (Planner, Router, Critic)
- Adaptive RAG (vector DB + web search)
- Self-correction loops
- Evaluation framework (custom benchmark + scoring)

---

## 📂 Dataset

### References
Stored in `data/references/`:
- Semiconductor policies (India, Taiwan)
- EV policies (FAME II)
- Global supply chain reports

### Benchmarks
Stored in `data/benchmarks/`:
- `test_set.json` → questions
- `ground_truth.json` → evaluation logic

---

## 🎯 Goal
Build a system that:
- Plans → retrieves → verifies → improves answers
- Minimizes hallucination
- Produces structured, reliable outputs

---

## 🛠️ Tech Stack
- LangGraph
- LangChain
- Ollama (Llama3)
- ChromaDB

---

## 🚧 Status
Phase 1: Planning ✅  
Phase 2: RAG (In Progress)