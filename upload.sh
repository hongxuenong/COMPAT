smc rsync -azvu \
    --exclude='*.pyc' \
    --exclude='*.DS_Store' \
    --exclude='*__pycache__' \
    --exclude='*.ipynb_checkpoints' \
    --exclude='venv' \
    --exclude='pretrained' \
    --exclude='.git' \
    . \
    10.60.2.172:/data/xuenong_hong/projects/COMPAT/
