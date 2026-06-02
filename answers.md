# AI Validation Reflection

## Did the AI suggest an architecture that was too complex for the dataset?

Not exactly — but it could become one depending on how much data we actually collect.

The proposed CNN has three convolutional blocks, BatchNorm at every stage, GlobalAveragePooling, and a dense head with dropout. For a clean, well-curated dataset of 1,000+ clips that's a defensible choice. The GlobalAveragePooling in particular is smarter than a flat Flatten layer because it prevents the parameter explosion that would crush a small dataset.

That said, we flagged a real risk: if we end up with fewer than 150 samples per class, this architecture will overfit even with dropout and early stopping. We'd need to either drop to two conv blocks or lean much harder on data augmentation. The AI gave us the architecture for the optimistic scenario. We need to pressure-test it against the actual dataset size before treating it as settled.

---

## Did the AI confuse MFCCs with spectrogram images?

No. The distinction was handled clearly and correctly.

The design document included a comparison table that laid out the tradeoffs directly: MFCCs compress the spectral envelope into a small coefficient vector, which makes them fast and useful for classical ML baselines, but they lose the full frequency information a CNN needs. Mel-spectrograms keep that information as a 2D time-frequency image — which is what makes them CNN-compatible.

The AI also correctly placed MFCCs in the baseline context (the SVM can work with a 40-coefficient mean MFCC vector) and Mel-spectrograms in the CNN context. That's the right call. What we verified ourselves: the math behind the output shape. With a 4-second clip at 22050 Hz, hop length 512, we get ⌈(4×22050)/512⌉ = 173 frames. The stated shape of (128, 173) checks out.

---

## Did the AI suggest an evaluation metric that was insufficient?

No — it made the right choice for exactly the right reason.

Macro F1-score was designated as the primary metric with a clear explanation: if one class dominates the dataset, a model that always predicts the majority class can score high on accuracy while being completely useless. Macro F1 weights all five classes equally regardless of sample count, which is what you want for an imbalanced industrial monitoring scenario.

What we added to the checklist ourselves: per-class F1 broken out individually, not just the macro average. The macro number can hide a class that's at near-zero recall. We also added a confusion matrix as a required output — the AI included it in the evaluation suite, but we made it mandatory rather than optional.

---

## Did the AI produce code or explanations that required correction?

A few things needed attention.

The test file uses `sys.path.insert(0, ...)` to reach `src/`. That works locally but breaks the moment someone runs pytest from a different working directory or adds the project as a package. A proper `conftest.py` at the root or a `pyproject.toml` with `pythonpath = ["src"]` would be more robust. We caught this before running the test suite.

In `features.py`, the `import soundfile as sf` is placed inside the per-file loop rather than at the top of the module. It runs correctly — Python caches imports — but it's misleading and violates the project's own convention of keeping imports at module level.

The `dataset.py` calls `tf.keras.utils.to_categorical` on the full split at construction time. For a dataset this size that's fine, but if the dataset ever grows it pins everything in memory unnecessarily. Not a bug, but worth knowing if we scale.

None of these were wrong in a way that would break training. They were code quality issues, not logic errors.

---

## What did your team learn by validating the AI-generated information?

The main takeaway: AI output on well-documented ML pipelines is largely trustworthy at the architecture and metric level, but assumes a clean, well-sized dataset that may not exist yet.

We went in expecting to find conceptual mistakes. We didn't find many. What we found instead were implementation choices that were correct for the happy path but brittle at the edges — the test path handling, the in-loop import, the assumption that ~1,000 samples would be available. The AI optimized for the scenario described in the design document without flagging what happens if reality diverges from that scenario.

That's a useful failure mode to understand. The AI is confident. It doesn't caveat its architectural choices with "unless your dataset is smaller than 800 samples." It doesn't warn that the test file will break if you change your working directory. You have to bring that skepticism yourself.

The other thing we learned: verifying the math manually — the spectrogram frame count, the split ratios, the parameter counts in the conv layers — forced us to actually understand the pipeline rather than just run it. That's probably the most valuable thing about the validation exercise. We can now explain why every number in the architecture exists.
