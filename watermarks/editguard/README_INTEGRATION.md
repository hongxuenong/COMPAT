# EditGuard integration

Source repo: https://github.com/xuanyuzhang21/EditGuard (upstream `code/` mirrored here).

- The `clean.pth` checkpoint is **not** auto-downloadable. Download it from the Google Drive
  link in the upstream EditGuard README and place it at `checkpoints/clean.pth`
  (or point `EDITGUARD_CKPT`, or pass `ckpt_path=` to the API).
- A CUDA GPU is strongly recommended; the model and several upstream ops assume CUDA.
- API: `from editguard import add_watermark, verify_watermark`.
