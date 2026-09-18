# Q38PACK v1

Q38PACK is the model container used by qwen382743090. It is intentionally
narrow: Qwen3.8-27B on RTX 3090 / SM86 is the first target.

## Goals

- Constant-time mmap of model metadata and tensor directory.
- 4 KiB default tensor alignment for future Driver API / VMM mapping.
- No runtime dependency on GGUF parsing.
- Preserve the original GGUF metadata region losslessly.
- Preserve quantized tensor payloads byte-for-byte in v1.
- Allow future SM86-specific repacks without changing the public server API.

## File layout

All integer fields are little-endian.

    fixed header (256 bytes)
    tensor directory (tensor_count * 256 bytes)
    compact JSON manifest
    raw GGUF metadata region
    padding to pack alignment
    tensor payloads with per-tensor alignment

### Header fields

    char[8]  magic = "Q38PACK\0"
    u32      version = 1
    u32      header_bytes = 256
    u32      flags
    u32      tensor_count
    u32      alignment
    u32      reserved
    u64      directory_offset
    u64      directory_bytes
    u64      manifest_offset
    u64      manifest_bytes
    u64      raw_gguf_meta_offset
    u64      raw_gguf_meta_bytes
    u64      data_offset
    u64      data_bytes
    u64      source_file_size
    u64      source_data_offset

The remaining bytes in the 256-byte header are reserved.

### Tensor directory entry

Every entry is exactly 256 bytes.

    char[160] name
    u32       ndim
    u32       ggml_type
    u64[4]    dims
    u64       data_offset
    u64       stored_bytes
    u64       source_offset
    u32       role
    u32       flags
    ...       reserved to 256 bytes

stored_bytes is the source GGUF storage span for the tensor. In v1 this can
include alignment bytes at the end of a tensor span. This is deliberate: v1
copies the source storage exactly and does not reinterpret quantization blocks.

## Compatibility contract

The converter embeds the original GGUF header, key/value metadata, and tensor
directory in raw_gguf_meta. This lets future versions recover tokenizer data
and metadata that are intentionally summarized in the compact JSON manifest.

Q38PACK v1 is a container conversion only. It does not requantize weights and
therefore must not introduce model-quality changes by itself.
