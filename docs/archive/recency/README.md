# Recency draft

`check_recency_draft.py` preserves the original contents of
`tests/test_recency.py`. It expects transaction data and functions such as
`prepare_rows` and `fill_structural_missing` that are absent from this repository.
It therefore cannot run against the COMPAS or Adult configs.

The draft was moved out of test discovery without changing its contents.
The corresponding existing `recover_recency` helper in `src/preprocess.py`
was preserved. Neither is required by the Week 9 Adult task.
