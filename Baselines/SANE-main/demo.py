#!/usr/bin/env python
"""Demo: SANE pipeline as a library — ingest multiple documents, then query."""

import sane

config = sane.load_config("config/default.yaml")
config.method.method = "sane"
config.method.k = 5
config.method.w = 20

agent = sane.create_agent(config)

# Ingest multiple documents (realistic multi-doc scenario)
docs = [
    ("doc1", "The capital of France is Paris. It is known for the Eiffel Tower."),
    ("doc2", "The capital of Germany is Berlin. The Brandenburg Gate is a famous landmark."),
    ("doc3", "Paris has over 2 million residents. The Louvre museum is located there."),
    ("doc4", "Berlin has around 3.6 million residents. It is the largest city in Germany."),
    ("doc5", "France is a country in Western Europe. Its official language is French."),
]
for doc_id, text in docs:
    agent.ingest_text(text=text, doc_id=doc_id)

response = agent.send("What is the capital of France and how many people live there?")
print("Response:", response)
print()
print("Timing:", agent.get_timing_summary())
