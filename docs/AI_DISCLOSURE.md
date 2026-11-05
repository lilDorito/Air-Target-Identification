# Disclosure of AI Tool Use

## 1. Tools and what they were used for

The main assistant was Claude (Anthropic). Other general-purpose chat assistants were used occasionally, for the same kinds of tasks.

| Area | How AI was used |
|---|---|
| Code | Mainly as a refactoring tool: cleaning up and unifying code we wrote, with some help on implementation logic (e.g. temperature scaling) |
| Plots | Suggested ideas for figures; we implemented the ones we found useful, not without the refactoring assistant help of course |
| Documentation | README, `docs/` and the notebook's explanatory text were largely written by AI from our dictation and then edited by us |
| Data | Web research, AI-recalled reference figures and audit passes (see section 5) |
| Web app | primarily AI-generated design on top of our api, verified and corrected by us |
| Consulting | Discussing approaches and deployment strategy |

## 2. Influence on methodological decisions

Small. The research question, the dataset, the feature set and the three scenarios, the three targets, the choice of LightGBM and Optuna, and keeping operators separate from country of origin were all our own ideas.

Two decisions came partly from talking things through with AI, alongside public sources we read:

- **Simulated observations.** At first we did not want to sample from spec ranges. We later concluded that simulation is necessary for this problem.
- **Calibration.** We were not very familiar with temperature scaling. AI helped us understand it and decide how to apply it.

## 3. Written or substantially changed by us

- **Code:** written mostly by us. AI refactored it for style and consistency, and helped with parts of the calibration logic.
- **Text:** README, the documents in `docs/` and the notebook's text were largely written or rewritten by AI from our instructions. The content, structure and conclusions are ours, and we corrected the text repeatedly.

## 4. Verified and understood independently

- Every piece of code was reviewed by hand after each refactoring. We did not accept AI changes without checking them, and the unit tests exist for that reason.
- We reran the tests and the notebook, compared results before and after changes, and ran the experiments and the deployment ourselves.
- We understand what every file and document does. Placeholder files that turned out to be unnecessary (e.g. per-folder READMEs) were removed.

## 5. Data provenance

Summary of [`DATA_COLLECTION_NOTICE.md`](DATA_COLLECTION_NOTICE.md):

- **Sourced (133 rows):** researched online with AI-assisted web search. Every row lists its sources.
- **AI-recalled (402 rows):** reference figures from Claude's general knowledge, not fetched live. A random sample and a full audit were done; about 1 in 6 checked figures was off by more than 5%. The clearest errors were fixed, but some likely remain.
- **Rule-estimated (~36% of filled cells):** fixed rules based on physics and size, never on the label.
- **Operators:** from AI recall, with a few high-impact entries checked live.

All steps were reviewed and approved by the authors.

---

*In the spirit of full honesty: this disclosure was also written by AI, from our dictation. Yeah, we are that lazy about documentation, but we did read it and can confirm it is true.*