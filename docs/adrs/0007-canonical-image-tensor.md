# ADR 0007 — Images Reach Backbones as Float RGB Tensors; Batches Are Lists

Status: Accepted

## Decision

1. Every accepted image form (PIL image, numpy array, torch tensor, file path) is converted by one function, `embedfit.data.to_rgb_tensor`, to a float32 RGB tensor shaped `[3, H, W]` with values in [0, 1] at native resolution. Backbone adapters apply checkpoint-specific resizing and normalisation to that tensor.
2. Floating-point input outside [0, 1] is an error, not a cue to rescale.
3. In `EmbedFitModel.encode`, a `list` or `tuple` is a batch and anything else is one image. 4-D arrays are rejected.
4. `ImageBackbone.freeze()` pins the backbone in eval mode until `unfreeze()`.

## Rationale

A tensor is the lossless common denominator: converting float tensors to PIL would quantise them to 8 bits, while every provider's preprocessing (torchvision v2 transforms, timm transforms, Hugging Face image processors) can consume tensors. Centralising the conversion gives one place for validation, privacy-safe error messages, and grayscale handling (ADR 0003 still leaves all checkpoint-specific work to the adapter).

Guessing the scale of float data silently produces wrong embeddings when the guess fails, and an image that happens to be dark would be indistinguishable from a [0, 1] image. An explicit error costs the caller one line.

Shape-based batch detection is ambiguous (`[3, H, W]` versus a batch of three grayscale images), and `AGENTS.md` forbids inferring ambiguous semantics.

"Frozen" must mean the representation cannot change. Parameters with `requires_grad=False` can still drift through batch-norm running statistics if a parent module calls `train()`, which the two-stage flow (ADR 0002) does routinely.

## Consequences

Errors from image loading omit file paths, because file names may embed sensitive identifiers. They are raised without a chained cause, since the underlying `FileNotFoundError` or decoder error names the path and would otherwise reach any traceback or `logger.exception` output. `encode` reports the failing image's position in the input, which is enough to find the file. Paths of model directories (`save_pretrained` / `from_pretrained`) do appear in messages: they locate artifacts the caller created, not input data.

High-bit-depth PIL modes are decoded explicitly (`I;16` scaled by 65535, `F` validated against [0, 1], `I` rejected) because Pillow's RGB conversion clips them to 8 bits. The EXIF orientation tag is applied so the tensor matches the image as displayed.

Adapters resize the float tensor, whereas provider reference pipelines usually resize 8-bit pixels. The results differ by roughly 0.001 of the intensity range on average. The float path is kept because it does not quantise float and 16-bit input.

`encode` always raises on a corrupt image. Dataset-level options such as skipping corrupt images belong in the data pipeline on top of `to_rgb_tensor`; see the amendment below.

## Amendment — encoded bytes and the `skip` policy

Status: Accepted (data adapters, v0.1)

1. Encoded image file bytes (`bytes`) are an accepted image form everywhere a path is: in `ImageSample.image`, `EmbedFitModel.encode`, `predict`, and the backbone `preprocess` methods. `to_rgb_tensor` decodes them exactly as it decodes a file (same EXIF, high-bit-depth, and multi-frame rules). Error messages for bytes describe them as "encoded image bytes" and never include their content; like file errors, they have no chained cause.
2. An image with more than twice Pillow's decompression-bomb limit raises `ImageInputError` (no chained cause) instead of Pillow's own `DecompressionBombError`; between one and two times the limit Pillow only warns and the image is decoded. Other errors Pillow plugins raise on malformed data (`ValueError`, `SyntaxError`, `EOFError`, `struct.error`, `IndexError`) are also reported as `ImageInputError`.
3. The `skip` policy lives in `ImageDataset` (`on_corrupt_image="skip"`), not in `encode`. It checks every file-path and bytes image once, when the dataset is built, and drops the records that are missing, undecodable, truncated, or over twice the decompression-bomb limit. A directory or a file without read permission still raises. `encode` still always raises.

Rationale: the Hugging Face `datasets.Image` feature stores encoded bytes. Decoding them to PIL images up front would hold every decoded image in memory, and a lazily opened PIL image keeps its decoded pixels after the first use; a third input form that decodes like a file keeps one decode path and decodes per use. Skipping is decided at construction because samplers index a dataset by position: dropping records later would change its length during training and misalign `predict` outputs with labels.
