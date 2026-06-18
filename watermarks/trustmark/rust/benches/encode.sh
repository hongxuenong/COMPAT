#! /usr/bin/env bash

# Copyright 2025 Adobe
# All Rights Reserved.
#
# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

set -euxo pipefail

python -m timeit \
        -s "from PIL import Image; from trustmark import TrustMark; tm=TrustMark(verbose=True, model_type='Q', encoding_type=TrustMark.Encoding.BCH_5); image = Image.open('../images/ufo_240.jpg').convert('RGB')" \
        "tm.encode(image, '0100100100100001000101001010', MODE='BINARY')"
