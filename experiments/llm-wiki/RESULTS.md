# Phandelver LLM Wiki and Cognee comparison

Completed November 23, 2026 (Pacific). **This LLM Wiki configuration was slower
and substantially less complete than the saved Cognee indexes.** Removing
Clef reranking helped slightly, but the main problem was lost detail in the
generated chapter notes. Cognee remains the stronger factual-retrieval choice
for this module under these tested settings.
[Audited metrics](results-20261123.json) · [Protocol and replay instructions](README.md).

## Scope and controls

This trial compares [CROW in LLM Wiki v2](https://github.com/crow-hq/llm-wiki-v2/tree/3333c71485bf271a89544090647a91abe25c70ca)
with the saved [stock Cognee](../inkling/stock-cognee/RESULTS.md) and
[adaptive Cognee](../inkling/fp8-cognee/RESULTS.md) Phandelver indexes. Each
covers the same thirteen full-module source documents. The benchmark retains
the original 48 factual questions, q37's predeclared ambiguous exclusion, and
the independent question about how Wave Echo Cave feeds into the second half.

All new inference uses the existing local services: native Inkling 2 Mini for
writing and answering, stock Clef for CROW classification, and Ollama's
`qwen3-embedding:0.6b` for vectors. Clef runs on Spark 2 alongside Inkling;
the experimental `clef-card-tuned` head is not used. Neither model service is
restarted. Inkling uses the selected 411,520-token context, FP8 KV and fixed
13 GiB cache, with thinking blocked, temperature zero and seed 604.

The factual reader policy matches the Cognee trials: answer only from supplied
evidence, preserve negation, state unknown when unsupported, and keep the answer
within 120 words. The output cap is 4,096 tokens. Upstream CROW selection,
conflict handling and answer postprocessing remain enabled; its supported
custom answer step supplies this controlled reader. Evidence packing differs
by backend. This measures that adapter rather than the native wiki answer prompt.

Two wiki arms use eight selected notes per question: hybrid vector/lexical
selection with Clef reranking, then the same selection with reranking disabled.
Conflict checks remain enabled in both arms. A separate BM25-only diagnostic
measures source-document representation without generating answers. Finding a
note derived from the right chapter does not establish that the requested fact
survived in it.

## Ingestion

The new wiki took **3,263.34 seconds (54 minutes 23 seconds)** to ingest the
thirteen documents. It created thirteen notes and performed no merges. Native
Inkling made 18 writer calls, using 220,186 input and 104,017 output tokens;
captured writer time was 3,218.78 seconds. Clef made 25 classification calls,
taking 43.65 seconds in total. The generic vector-cache preparation then took
2.73 seconds, before any benchmark question was issued.

Compression varied substantially. The Chapter 5 source contained 79,685
characters; its complete saved note contained 3,371. Chapter 1 retained
24,452 characters from a 34,104-character source. These are character counts,
including note metadata in the output, rather than semantic retention scores.

The 100,000-character read budget split Chapters 6 and 8 into two parts followed
by a combine call. One Chapter 8 part reached the 16,384-token output limit;
the upstream library retained its partial plaintext and applied the part budget.
Thus, successful ingestion does not mean every writer response ended naturally
or every source fact survived. The library also logged missing exact values
after a combine step; that check is not a count of missing semantic facts.
A larger read budget could avoid those split paths on this large-context model,
but was not tested in this run.

The saved Cognee graphs were originally extracted by **Qwen**, whereas this
wiki was written by **Inkling**. Consequently, these preparation times and
quality differences do not isolate indexing architecture. The answer model and
serving configuration match the historical Cognee comparisons.

## Factual answers

Both factual arms are complete.

| Measure | Stock Cognee | Adaptive Cognee | Wiki with Clef reranking | Wiki reranking off |
|---|---:|---:|---:|---:|
| Query p50 | 10.87 s | 6.03 s | 21.74 s | 19.33 s |
| Query p95 | 14.72 s | 8.41 s | 25.29 s | 22.97 s |
| Sum of 48 queries | 514.61 s | 300.32 s | 1,034.96 s | 938.23 s |
| Complete correct answers | 46/47 | 44/47 | 21/47 | 22/47 |
| Fully grounded complete answers | 44/47 | 44/47 | 21/47 | 22/47 |
| Reader input tokens | 844,840 | 204,946 | 1,669,743 | 1,801,276 |
| Reader output tokens | 3,355 | 3,069 | 3,852 | 3,790 |

The wiki arm made 96 classifier calls, taking 218.68 seconds, in addition to
811.92 seconds of reader time. It supplied 1.98 times the stock input-token
volume and 8.15 times adaptive's. That is consistent with reading eight large
chapter notes plus classifier overhead; this capture does not isolate GPU
prefill, decode or contention costs. Its median was 2.00 times stock's and
3.60 times adaptive's. This is also a worse-quality result, not an equal-quality
speed tradeoff.

Disabling reranking saved 9.3% of summed query time and recovered q12, the Grol
question. Classifier time fell to 71.26 seconds across 48 conflict checks, while
reader time rose to 864.75 seconds with a different, larger evidence selection.
It also avoided the q22 invented explanation, abstaining instead. This single
ablation does not establish that reranking is generally harmful.

With reranking, the 47 adjudicable answers comprised 21 complete, 12 partial,
12 abstained and two incorrect. Without reranking, they comprised 22 complete,
12 partial, 12 abstained and one incorrect. All six item questions passed in
both arms; the bestiary summary had discarded
the stat-block mechanics needed for q47 and q48. The review retains the frozen
rubric, including the Spider identity target in q13 and q38's trigger-and-round
scope. Complete answers must be supported by both the actual supplied notes and
the original book; faithful repetition of a wrong note does not suffice.
Accepting q13's correct fates and room without requiring Tharden's killer would
add one complete answer to each wiki arm; the overall conclusion is unchanged.

Both wiki arms represented the required original chapter in **47/48** selected
note sets. BM25 alone represented it in **48/48**, selecting eight of the thirteen
notes. These are broad source-document coverage counts, including excluded q37;
they do not measure retained facts, and no BM25-only answers were generated.
High routing coverage coexisted with much lower factual completeness.

The source checks distinguish several failure modes:

| Example | Observed result | What the source check establishes |
|---|---|---|
| Goblin flood timing, q04 | Exact round sequence recovered | A genuine gain over both historical Cognee answers |
| King Grol’s companions, q12 | Reader abstained | The relevant Chapter 3 note retained the answer but was not selected |
| Ruxithid’s awareness, q22 | Earlier goblin encounter offered as the cause | The true clairvoyance explanation was absent from the notes; the reader invented the connection |
| Halia’s original offer, q05 | Original offer unknown; later miners’ quest supplied | The notes omitted the requested bargain and compressed a later quest in a way that included an already-returned miner among the missing |
| Gundren’s original escort payment, q01 | Later reward treated as the original promise | Distinct transactions were conflated despite a relevant chapter being selected |

The correct chapter and the correct answer are separate measures. Retrieving
more generated notes cannot recover a fact that was removed while writing them.
Conversely, q12 shows that improving selection could still recover some facts
already present in the wiki. A relevance score alone conflates those cases.

The representation reaching the ranker matters too. Upstream CROW embeds the
title, summary and first 3,000 body characters, combines that with lexical
search over the full body, and reranks candidates using a question-selected
**600-character passage** plus title and summary. The reader then receives the
full eight selected notes. In q12, the captured Chapter 3 reranking passage
described Agatha and Old Owl Well; it did not contain Snarl or Vyerith, although
the full note contained both. The observed ranking miss therefore does not
isolate Clef's ability to recognize the correct passage when shown it.

## Independent module agent

The agent reads one result, identifies remaining gaps and chooses its next
question, with the original eight-search budget and action schema. It receives
generated wiki notes as evidence; Cognee supplied original module chunks.
The original question is its only initial query, without reviewer-suggested
follow-ups or prior answers. Review checks the final explanation against the
original book as well as the notes actually supplied.

| Measure | Stock Cognee | Adaptive Cognee | Wiki with Clef |
|---|---:|---:|---:|
| Research loop | 259.24 s | 175.72 s | **300.62 s** |
| Client elapsed | 263.71 s | 180.03 s | 300.64 s |
| Searches / model calls | 6 / 13 | 6 / 12 | 4 / 11 |
| Evidence units accumulated | 18 raw passages | 25 raw passages | 11 derived notes |
| Protocol corrections | 1 | 0 | 3 |
| Input / output tokens | 290,123 / 4,324 | 78,419 / 4,263 | 456,331 / 4,468 |
| Total model tokens | 294,447 | 82,682 | 460,799 |
| Broad return and goblin-investigation bridge explained | Yes | Yes | Yes |
| Concrete Nundro escort hook covered | Yes | Yes | No |
| First-half psionic-goblin foreshadowing explained | Yes | Yes | No |

The wiki research loop took **16.0% longer than stock and 71.1% longer than
adaptive**, despite using fewer searches. It consumed 1.56 times stock's total
model tokens and 5.57 times adaptive's. Its client setup occurred inside an
already-running worker; research-loop time is the closer comparison with the
historical standalone Cognee clients. These are different trajectories and
evidence representations, rather than a decoder-speed benchmark.

The first observation already supplied the notes containing the Chapter 4
ending and Chapter 5 opening. Subsequent searches concentrated on whether
Nezznar was connected to the mind flayer plot. Two attempted exact duplicate
queries were rejected. A third correction rejected a finish action with a
nonempty remaining-gaps field; the retry cleared that field while retaining
the uncertainty in the final answer. Four searches remained unused.

The final answer recovered the optional continuation, level progression,
mine-profit award, return to Phandalin, goblin crimes, Gwyn, Ruxithid and the
later Far Realm plot. It avoided inventing a Forge-to-Obelisk mechanism or a
physical note on the goblin corpse. However, it omitted the concrete Nundro
escort hook and did not explain the early psionic-goblin foreshadowing, even
though both were available in its accumulated notes. More searches did not
ensure that all useful connections already in evidence appeared in the answer.

There were also wording problems. Its Qunbraxel sentence broadens his role
from coordinating the fanatics' **obelisk-retrieval efforts** to coordinating
the fanatics generally; the sources call them his overlords. Another sentence
grammatically describes the goblins as kidnapped townsfolk, although the
intended kidnapping relationship is clear from the surrounding material.
These measured outputs are retained unchanged. The introductory note's
wording about avenging Gundren also appears in the original module and is
not treated as an invented death claim.

The outcome is a correct broad transition with omitted concrete links and
wording caveats. It does not establish a quality gain over the historical
Cognee agents; stock gave the more complete, consistently supported explanation.

## Verification and service state

The audit verified all 768 factual note-context appearances, actual agent
reader contexts, generation controls, source hashes and 11,868 protected
historical files. Querying left wiki text unchanged. All 96 factual and 11
agent LLM responses stopped naturally with zero reasoning tokens; the earlier
ingestion limit event is recorded separately above. Every captured HTTP request
succeeded. The embedding model and digest matched the Cognee baseline, with
no API truncation.

At **18:09 Pacific**, Inkling and Clef had the same container IDs, images,
start times and commands as before this run. Clef was ready, Inkling still
advertised 411,520 tokens, Qwen model discovery passed, and the existing
embedding-only Ollama inventory was present. The memory guard remained active
without tripping; final sampled headroom was **13.12 GiB**. Neither application
clients nor model launch settings were changed.

## Interpretation limits

This is one run per question and arm, and one module agent, on shared hardware.
Historical Cognee runs occurred at different times. Clef uses uncalibrated
custom-model thresholds, not the upstream paper's model configuration. Source
review is by Codex, rather than blinded human adjudication. All module text,
generated notes, prompts and answers remain in ignored local storage; public
exports contain metrics, hashes and review flags.
