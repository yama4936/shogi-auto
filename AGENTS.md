# Development instructions

- This is an early CPU shogi baseline with a local browser UI, not a trained or tournament-ready engine.
- Use Python 3.11+; keep CPU development and Kaggle GPU experiments separate.
- Run `bash scripts/setup.sh`, activate `.venv`, and run `python -m unittest discover -s tests -v` after changes.
- Do not require CUDA, network access, or credentials for the standard tests.
- Preserve Kaggle's installed PyTorch/CUDA combination. Do not automatically upgrade torch.
- Keep notebooks output-free in Git. Record real GPU results separately; never describe mocked tests as GPU validation.
- Never commit secrets, game-server credentials, datasets, or checkpoints. Review licenses before adding engines or training data.
- Do not submit to tournaments, start paid resources, or publish Kaggle artifacts without authorization.
- Basic USI and bounded CPU search exist. Full USI clock/ponder support, CSA, training and tournament hosting remain future work.
- Keep the web UI local by default. Test browser behavior with the optional `scripts/browser_smoke.py`; never expose arbitrary SFEN fixtures through the public API.
