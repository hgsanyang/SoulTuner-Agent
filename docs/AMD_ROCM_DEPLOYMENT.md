# AMD ROCm deployment

SoulTuner supports three independent runtime profiles:

- CPU: metadata, Graph retrieval and the API provider path;
- NVIDIA CUDA: the existing `docker-compose.gpu.yml` overlay;
- AMD ROCm: `docker-compose.amd.yml` with the official AMD PyTorch image.

The business code uses PyTorch's `torch.cuda` device namespace on both NVIDIA
and AMD. A ROCm build is identified by `torch.version.hip`; the namespace name
does not mean the host is NVIDIA.

## Prerequisites

1. A Linux host supported by the selected ROCm release;
2. `/dev/kfd` and `/dev/dri` available to Docker;
3. the current user allowed to use the `video` and `render` groups;
4. model licenses accepted and model caches mounted outside the image.

## Start

```bash
cp .env.example .env
docker compose \
  -f docker-compose.yml \
  -f docker-compose.amd.yml \
  --profile gpu \
  up -d --build
```

Verify the runtime before ingesting audio:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.amd.yml \
  exec backend python scripts/assert_cuda.py
```

The report must show `ROCm/HIP`, the visible device count and the AMD GPU name.
The service fails closed when `MUSIC_REQUIRE_ACCELERATOR=1` but no device is
visible, avoiding an accidental multi-hour CPU fallback.

The default base image is an official AMD `rocm/pytorch` release. Override
`ROCM_PYTORCH_IMAGE` only with a tag validated for the target GPU and driver.
