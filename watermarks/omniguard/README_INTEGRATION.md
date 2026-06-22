# OmniGuard integration

Source: https://github.com/xuanyuzhang21/OmniGuard (inference code vendored from repo root; CVPR 2025).

Checkpoints are NOT auto-downloadable. Get `checkpoint.zip` from PKU Disk
(https://disk.pku.edu.cn/link/AAB048898581E047DE9519CE140F991B3A , code `5bvw`) or Google Drive
(https://drive.google.com/file/d/1khdBDUDIRIhPIKlV0ictcbTdWLh-WFY_/view). Unzip it and place
`model_checkpoint_01500.pt`, `encoder_Q.ckpt`, `decoder_Q.ckpt`, and `checkpoint-175.pth` into
`watermarks/omniguard/checkpoint/` (or any dir, then set `OMNIGUARD_CKPT` / pass `ckpt_path`).

GPU required: the upstream IWT (`iwt_init`) and `init_model` call `.cuda()` directly, so a CUDA
device is needed for both embed and verify. Extra deps: torch, torchvision, kornia, timm, fvcore,
albumentations, opencv-python, Pillow, numpy.
