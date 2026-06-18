# TrustMark CLI

The Rust implementation includes a CLI wrapper for the `trustmark` crate.

## Installation

To install the CLI, run this command from the `trustmark/crates/trustmark-cli` directory:

```
cargo install --locked --path .
```

### Downloading models 

To use the CLI, you must first create a `models` directory and download the models, [if you haven't already done so](../../README.md#download-models). Enter these commands from the `trustmark/rust` directory:

```
mkdir models
cargo xtask fetch-models
```

## Usage

View CLI help information by entering this command:

```
trustmark [encode | decode] help
```

The basic command syntax is:

```
trustmark --models <MODELS> [encode | decode]
```

Where `<MODELS>` is the relative path to the directory containing models.
Use the `encode` subcommand to encode a watermark into an image and the `decode` subcommand to decode a watermark from an image.

### Encoding watermarks

To encode a watermark into an image, use the `encode` subcommand:

```
trustmark --models <MODELS> encode [OPTIONS] -i <INPUT> -o <OUTPUT>
```

Options:

| Option |  Description | Allowed Values |
|--------|--------------|----------------|
| `-i <INPUT>` | Path to the image to encode. | Relative file path. | 
| `-o <OUTPUT>` | Path to file in which to save the watermarked image. | Relative file path. |
| `-w, --watermark <WATERMARK>` | The watermark (payload) to encode.  | Any a binary string such as  `0101010101`. Only 0 and 1 characters are allowed. Maximum length is governed by the version selected.  Default is a random binary string. |
| `--version <VERSION>`  |  The BCH version to encode with. | One of `BCH_SUPER` (default), `BCH_5`, `BCH_4`, or `BCH_3`. |
| `--variant <VARIANT>`  | The model variant to encode with. | `Q` (default), `B`, `C`, and `P`. |
| `--quality <QUALITY>`  | If the requested output format is JPEG, the output quality to encode. | A number between 0 and 100. The default is 90. |
| `-h, --help` | Display help information. | N/A |

### Decoding watermarks

To decode a watermark from an image, use the `decode` subcommand:

```
trustmark --models <MODELS> decode [OPTIONS] -i <INPUT>
```

| Option |  Description | Allowed Values |
|--------|--------------|----------------|
| `-i <INPUT>` | Path to the image to decode. | Relative file path. |
| `--variant <VARIANT>`  | The model variant to decode with.  Must match variant used to encode the watermark. | `Q` (default), `B`, `C`, and `P`.  |
| `-h, --help` | Display help information. | N/A |

## Examples

To encode a watermark into one of the sample images, run this command from the workspace root:

```sh
trustmark -m ./models encode -i ../images/ghost.png -o ../images/ghost_encoded.png
```

Then to decode the watermark from this image, run this command from the workspace root:

```sh
trustmark -m ./models decode -i ../images/ghost_encoded.png
```

You'll see something like this in your terminal:

```
Found watermark: 0101111001101001000000011011010100100010011101101101000001101
```
