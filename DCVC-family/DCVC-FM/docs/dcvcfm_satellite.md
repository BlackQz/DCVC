# DCVC-FM Satellite Adapter

This extension keeps DCVC-FM as the main learned video codec and adds a
satellite-aware semantic transmission path for training and evaluation.

## Architecture

The original DCVC-FM modules remain responsible for:

- intra coding with `DMCI`;
- motion estimation and compensation with `DMC.optic_flow`, `mv_encoder`, and
  `motion_compensation`;
- contextual encoder/decoder;
- hyperprior and four-part spatial entropy priors;
- feature modulation and quantization scalers;
- reconstruction generation.

The new satellite path is implemented under `src/models/satellite/`:

- `capacity_controller.py`: continuous capacity control using
  `capacity = bandwidth_mbps * (1 - packet_loss_rate) * log2(1 + snr_linear)`.
  It outputs target bpp interval, q index, lambda values, base budget,
  enhancement budget, and keep ratios.
- `slot_adapter.py`: Slot Attention adapter over reconstructed/key-frame
  features. It returns slots, masks, and object importance maps resized to
  latent space.
- `official_slot_attention.py`: PyTorch compatibility layer for the official
  TensorFlow Slot Attention implementation in `../slot-attention/model.py`.
  It preserves the official object-discovery autoencoder structure while
  keeping gradients inside the DCVC-FM PyTorch graph.
- `token_selector.py`: base/enhancement latent selector. The keep score is
  `w_mag * residual + w_novelty * novelty + w_obj * object + w_temporal * temporal`
  with defaults `0.45 / 0.25 / 0.20 / 0.10`. It uses straight-through top-k:
  forward masks are hard and nested, while backward gradients flow through a
  soft sigmoid relaxation.
- `channel.py`: AWGN, Rayleigh, and time-varying satellite-style Rician channel
  simulation with row packet loss. Base latents use stronger protection than
  enhancement latents.
- `dcvc_fm_satellite.py`: wrapper that reuses the DCVC-FM forward internals and
  inserts selection/channel perturbation only in the differentiable latent path.

The original `test_video.py`, `src/models/video_model.py`, and bitstream
`compress/decompress` path are not modified.

When Stage A or `--disable_satellite` is used, the wrapper bypasses token
selection and channel perturbation completely. P-frame latents, DPB contents,
and bpp match the original `DMC.forward_one_frame` path.

`forward_sequence` pads non-16-aligned inputs with replicate padding before
calling DCVC-FM and crops reconstructed frames back to the original size.

## Training

For the recommended paper-training recipe, use the curriculum entry point in
[`dcvcfm_satellite_training_plan.md`](dcvcfm_satellite_training_plan.md).  The
older A/B/C commands below remain useful for quick compatibility checks and
small ablations.

Stage A, reproduce/evaluate original DCVC-FM behavior:

```bash
cd DCVC-family/DCVC-FM
python -m training.train_dcvcfm_satellite \
  --stage A \
  --data_dir /path/to/frame_dataset \
  --model_path_i checkpoints/cvpr2024_image.pth.tar \
  --model_path_p checkpoints/cvpr2024_video.pth.tar \
  --save_dir checkpoints/dcvcfm_satellite_stageA \
  --max_steps 0
```

Stage B, freeze DCVC-FM and train the satellite adapters:

```bash
python -m training.train_dcvcfm_satellite \
  --stage B \
  --data_dir /path/to/frame_dataset \
  --model_path_i checkpoints/cvpr2024_image.pth.tar \
  --model_path_p checkpoints/cvpr2024_video.pth.tar \
  --save_dir checkpoints/dcvcfm_satellite \
  --channel_type satellite \
  --slot_adapter_h 128 --slot_adapter_w 128 \
  --snr_min 1 --snr_max 25 \
  --bandwidth_min_mbps 1 --bandwidth_max_mbps 25 \
  --pkt_loss_max 0.5 \
  --val_conditions "12,10,0.0;5,10,0.0;12,10,0.2;10,2,0.0;20,18,0.0;5,2,0.1"
```

Stage C, conservative joint fine-tuning:

```bash
python -m training.train_dcvcfm_satellite \
  --stage C \
  --data_dir /path/to/frame_dataset \
  --resume checkpoints/dcvcfm_satellite/best.pt \
  --save_dir checkpoints/dcvcfm_satellite_stageC \
  --lr 1e-5 \
  --channel_type satellite
```

Before training updates, the script evaluates the initialized model and saves it
as `best.pt`. This preserves the starting baseline if early robust training
temporarily degrades reconstruction quality.

Stage B trainable modules include Slot Attention, slot-to-latent FiLM
modulation, token selection weights, and learnable capacity offsets. This avoids
the hard-mask gradient break that would otherwise make adapter training a no-op.

## Evaluation

Single condition:

```bash
python -m training.evaluate_dcvcfm_satellite \
  --data_dir /path/to/frame_dataset/val \
  --ckpt checkpoints/dcvcfm_satellite/best.pt \
  --model_path_i checkpoints/cvpr2024_image.pth.tar \
  --model_path_p checkpoints/cvpr2024_video.pth.tar \
  --channel_type satellite \
  --snr_db 20 \
  --bandwidth_mbps 25 \
  --packet_loss_rate 0 \
  --output_dir results/dcvcfm_satellite/good
```

Formal scan:

```bash
bash run_dcvcfm_satellite_formal_eval.sh \
  --data-root /path/to/frame_dataset/val \
  --checkpoint checkpoints/dcvcfm_satellite/best.pt
```

The JSON output contains PSNR, SSIM, MS-SSIM fallback, LPIPS, DISTS, VMAF
placeholder, bpp, kbps, keep ratio, base/enhancement ratios, processing time,
transmission time, input source, number of GoPs, and number of frames. LPIPS and
DISTS are reported as `null` if their optional packages are unavailable.

## Bandwidth Response Check

Run the formal bandwidth scan and compare:

- `results/dcvcfm_satellite/bandwidth/bw_1/eval_results.json`
- `results/dcvcfm_satellite/bandwidth/bw_2/eval_results.json`
- `results/dcvcfm_satellite/bandwidth/bw_5/eval_results.json`
- `results/dcvcfm_satellite/bandwidth/bw_10/eval_results.json`
- `results/dcvcfm_satellite/bandwidth/bw_20/eval_results.json`
- `results/dcvcfm_satellite/bandwidth/bw_25/eval_results.json`

Expected diagnostics:

- capacity, target bpp, actual bpp, and keep ratio should generally increase;
- low bandwidth should have enhancement ratio near zero;
- BW25/BW1 bpp ratio should be checked and should ideally exceed 2.5;
- PSNR should trend upward with bandwidth, allowing small content-level noise.

## Compatibility

The original DCVC-FM code remains independently runnable:

```bash
python test_video.py \
  --model_path_i checkpoints/cvpr2024_image.pth.tar \
  --model_path_p checkpoints/cvpr2024_video.pth.tar \
  --rate_num 4 \
  --test_config dataset_config_example_yuv420.json \
  --cuda 1 \
  --worker 1 \
  --write_stream 0 \
  --output_path output.json
```

The satellite wrapper is a training/evaluation model, not a replacement for the
standard entropy-coded bitstream path.
