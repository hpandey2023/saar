# Retrieval evaluation

Do not judge retrieval from one impressive example. Build a small set of real questions with known source files, including paraphrases, acronyms, code questions, multilingual questions, similar-but-wrong documents, and facts that should not be found.

## Create a labeled set

```yaml
queries:
  - query: How quickly must a high-priority request be acknowledged?
    expected_sources:
      - request-handling.md
    scopes: [operations]

  - query: Where is production MFA required?
    expected_citations:
      - security/access.md#production
```

Expected values are case-insensitive substrings matched against `scope/source_file` and `citation`.

## Measure one configuration

```bash
saar --config configs/local.yaml evaluate evaluation.yaml \
  --limit 8 --output reports/local.json
```

The report includes:

- hit rate at K: proportion with an expected source in the first K results;
- mean reciprocal rank at K: rewards placing the correct source near the top;
- p50 and p95 retrieval latency;
- per-question rank, citations, and latency.

## Compare two providers

Use separate collections so one provider does not overwrite the other:

```bash
saar --config configs/local.yaml index
saar --config configs/local.yaml evaluate evaluation.yaml \
  --output reports/local.json

saar --config configs/voyage.yaml index
saar --config configs/voyage.yaml evaluate evaluation.yaml \
  --output reports/voyage.json

saar compare reports/local.json reports/voyage.json
```

Choose thresholds before looking at the result. For example: no reduction in hit rate, MRR improvement of at least 0.05, p95 below the agent's latency budget, and acceptable operating cost. Report confidence cautiously when the query set is small.

## Test with and without MCP

Retrieval metrics prove that search finds labeled context; they do not prove the agent uses it well. Run an agent-level A/B test with a fixed model, prompt, temperature, and question set:

1. Disable the MCP and record answer correctness, supported-claim rate, citations, and latency.
2. Enable the MCP with an instruction to search corpus-dependent questions.
3. Blind the answer labels and compare the same metrics.
4. Inspect tool traces to confirm the MCP was actually called.

Include answerable and unanswerable questions. A good retrieval system should improve grounded answers and make unsupported questions more likely to receive an honest “insufficient evidence” response.
