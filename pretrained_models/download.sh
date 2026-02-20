#!/bin/bash

rootdir=$(dirname "$0")

function check_sha256() {
    echo "$1 $2" | sha256sum -c - || { echo "SHA256 verification failed for $2"; exit 1; }
}

# autoenc
# necessary for checkerboard-L-2x and checkerboard-L-4x
mkdir -p "$rootdir/autoenc"
wget https://huggingface.co/deigen/checkerboardgen/resolve/main/pretrained_models/autoenc/autoencoder_config.yaml?download=true -O "$rootdir/autoenc/autoencoder_config.yaml"
wget https://huggingface.co/deigen/checkerboardgen/resolve/main/pretrained_models/autoenc/checkpoint.pth?download=true -O "$rootdir/autoenc/checkpoint.pth"
check_sha256 1d4c1cefa2a2edb55f1260645601e235c98ee41e4b51785c1dbcdabde4a60d14  $rootdir/autoenc/autoencoder_config.yaml
check_sha256 7503e2afd945546a0ba77cbb863227e8e5b1e1498190725c9358c6fed5720afa  $rootdir/autoenc/checkpoint.pth

# checkerboard-L-2x
mkdir -p "$rootdir/checkerboard-L-2x"
wget https://huggingface.co/deigen/checkerboardgen/resolve/main/pretrained_models/checkerboard-L-2x/config.yaml?download=true -O "$rootdir/checkerboard-L-2x/config.yaml"
wget https://huggingface.co/deigen/checkerboardgen/resolve/main/pretrained_models/checkerboard-L-2x/checkpoint-last.pth?download=true -O "$rootdir/checkerboard-L-2x/checkpoint-last.pth"
check_sha256 d0b1c226a3b6bc84827855d2228bc21642774d99145d5978636f2a1e3553f297  $rootdir/checkerboard-L-2x/checkpoint-last.pth
check_sha256 8719010494d7a909f36831a600cbda45c8223377600aa493abb28d3caba1f32b  $rootdir/checkerboard-L-2x/config.yaml

# checkerboard-L-4x
mkdir -p "$rootdir/checkerboard-L-4x"
wget https://huggingface.co/deigen/checkerboardgen/resolve/main/pretrained_models/checkerboard-L-4x/config.yaml?download=true -O "$rootdir/checkerboard-L-4x/config.yaml"
wget https://huggingface.co/deigen/checkerboardgen/resolve/main/pretrained_models/checkerboard-L-4x/checkpoint-last.pth?download=true -O "$rootdir/checkerboard-L-4x/checkpoint-last.pth"
check_sha256 e359552cb2b1d7ace912d0aaac2450e80680e2635b9cb537442aba3823692fa9  $rootdir/checkerboard-L-4x/checkpoint-last.pth
check_sha256 5cc319d36b881d9c6f492e4ded345e0c5abd1ff6e5648424e55ba86f4ba2f30a  $rootdir/checkerboard-L-4x/config.yaml
