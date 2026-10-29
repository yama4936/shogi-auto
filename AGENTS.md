# Development instructions

- This is an early shogi AI scaffold, not a trained or tournament-ready engine.
- Use Python 3.11+; keep CPU development and Kaggle GPU experiments separate.
- Run `bash scripts/setup.sh` and `python -m unittest discover -s tests -v` after changes.
- Do not require CUDA, network access, or credentials for the standard tests.
- Preserve Kaggle's installed PyTorch/CUDA combination. Do not automatically upgrade torch.
- Keep notebooks output-free in Git. Record real GPU results separately; never describe mocked tests as GPU validation.
- Never commit secrets, game-server credentials, datasets, or checkpoints. Review licenses before adding engines or training data.
- Do not submit to tournaments, start paid resources, or publish Kaggle artifacts without authorization.
- Protocol support (USI / CSA), search, training and tournament hosting remain future work.
