# Q38PACK v1/v2

Q38PACK is the execution container used by Q38RT. The first optimization target is Qwen3.8-27B on RTX 3090 / GA102 / SM86.

## Compatibility

- v1: source-preserving container. Tensor payloads are copied from GGUF.
- v2: same 256-byte header and tensor-directory entry size, with per-tensor execution-layout metadata in the 24 bytes that were reserved by v1.
- The C++ and Python readers accept both v1 and v2.
- The original GGUF header, metadata and tensor directory remain embedded in raw_gguf_meta in both versions.

No Q38PACK v2 layout is allowed to change decoded model weights. Specialized layouts are byte/layout transforms, not requantization.

## Top-level layout

All integer fields are little-endian.

    fixed header (256 bytes)
    tensor directory (tensor_count * 256 bytes)
    compact JSON manifest
    raw GGUF metadata region
    padding to pack alignment
    tensor payloads / execution layouts

## Header

    char[8]  magic = "Q38PACK\0"
    u32      version = 1 or 2
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

## Tensor directory

Every entry remains exactly 256 bytes.

### v1

    char[160] name
    u32       ndim
    u32       ggml_type
    u64[4]    dims
    u64       data_offset
    u64       stored_bytes
    u64       source_offset
    u32       role
    u32       flags
    u8[24]    reserved

### v2

The first 232 bytes are unchanged. The former 24-byte reserved tail becomes:

    u32       layout
    u32       layout_flags
    u64       aux0_offset
    u64       aux1_offset

Current layout IDs:

    0 = GGUF_NATIVE
    1 = SM86_Q5K_SOA

data_offset points to the primary plane. stored_bytes is the physical span owned by the tensor, including internal plane-alignment padding. aux0_offset and aux1_offset are absolute file offsets.

## SM86_Q5K_SOA

Real RTX 3090 measurements on blk.0.attn_gate.weight [5120,6144] showed:

    GGUF-native Q5_K:
      0.0864 ms
      ~250.2 GB/s effective weight bandwidth

    SM86 Q5_K SoA prototype:
      0.0651 ms
      ~332.3 GB/s original-Q5_K-equivalent bandwidth
      ~339.9 GB/s physical repacked bandwidth

Decoded weights remained exactly equivalent in the CPU reference dequantizer.

Standard Q5_K uses 176 bytes per 256 values:

    fp16 d, dmin          4
    packed scale/min     12
    qh                   32
    qs                  128
                       ----
                        176

SM86_Q5K_SOA expands only the metadata:

    META logical record per block:
      fp16 d, dmin                        4
      8 x {u8 scale, u8 min}             16
                                         --
                                         20

    QH plane: 32 bytes/block
    QS plane: 128 bytes/block

Total logical storage is 180 bytes/block, a 2.27% increase before tiny 128-byte plane-alignment padding. The 5-bit quantized weights themselves remain compact.

A Q5_K tensor is stored as:

    data_offset -> META plane, 20-byte stride
    aux0_offset -> QH plane,   32-byte stride
    aux1_offset -> QS plane,  128-byte stride

This removes 6-bit scale/min unpacking from the token-time GEMV hot loop and gives QH/QS GPU-friendly power-of-two strides.

## Conversion

Q38PACK v2 is the default:

    python3 tools/gguf_to_q38pack.py model.gguf model.q38pack

Q5_K repacking is vectorized with NumPy:

    python3 -m pip install numpy

To create the old source-preserving format:

    python3 tools/gguf_to_q38pack.py model.gguf model-v1.q38pack --format-version 1

Non-Q5_K tensor types currently remain GGUF_NATIVE in v2. They will only get specialized layouts after independent RTX 3090 benchmarks demonstrate a useful performance/size tradeoff.
