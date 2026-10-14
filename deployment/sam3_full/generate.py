"""Disjoint, resumable SAM3 shards; retain original mask/semantic conventions."""
import argparse
import json
import os
from pathlib import Path
import struct
import time
from contextlib import contextmanager
from output_store import ImageWriter, LocalStore, OssStore


def names_from_colmap(path):
    with open(path, 'rb') as f:
        count = struct.unpack('<Q', f.read(8))[0]
        names = []
        for _ in range(count):
            f.read(64)
            name = bytearray()
            while True:
                char = f.read(1)
                if not char:
                    raise ValueError('Truncated COLMAP images.bin')
                if char == b'\0':
                    break
                name.extend(char)
            points = struct.unpack('<Q', f.read(8))[0]
            f.seek(points * 24, 1)
            names.append(name.decode())
    if len(set(names)) != count:
        raise ValueError('Duplicate image names')
    return sorted(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', help='Defaults to data; separate directory for smoke test')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--status-dir', required=True)
    parser.add_argument('--output-backend', choices=['filesystem', 'oss'], default='filesystem')
    parser.add_argument('--oss-credentials', default=os.environ.get('OSS_CREDS_FILE'))
    parser.add_argument('--upload-workers', type=int, default=8)
    parser.add_argument('--upload-queue', type=int, default=64)
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    data = Path(args.data)
    output = Path(args.output or args.data)
    status_dir = Path(args.status_dir)
    status_file = status_dir / f'shard_{args.shard}.json'
    store = OssStore(output, args.oss_credentials) if args.output_backend == 'oss' else LocalStore(output)
    names = names_from_colmap(data / 'sparse/0_full/images.bin')[args.shard::args.shards]
    # List once instead of issuing two OSS stat operations per image.
    existing = {}
    for folder in ['mask_sam3', 'semantic_weights']:
        existing[folder] = store.existing(folder)
    pending = [n for n in names if any(str(Path(n).with_suffix('.png')) not in existing[k]
                                      for k in existing)]
    if args.limit:
        pending = pending[:args.limit]
    started = time.monotonic()
    timings = {}
    upload_seconds = encode_seconds = 0
    inferred = submitted = 0
    writer = None
    def status(state, done, **extra):
        elapsed = time.monotonic() - started
        record = dict(state=state, shard=args.shard, shards=args.shards, batch_size=args.batch_size,
                      assigned=len(names), pending=len(pending), completed=done,
                      elapsed_seconds=elapsed, images_per_second=done / max(elapsed, 1),
                      eta_seconds=(len(pending)-done)*elapsed/done if done else None,
                      checkpoint=args.checkpoint,
                      output_backend=args.output_backend, inferred=inferred, submitted=submitted,
                      upload_queue=len(writer.futures) if writer else 0,
                      upload_worker_seconds_per_image=upload_seconds / max(done, 1),
                      png_encode_worker_seconds_per_image=encode_seconds / max(done, 1),
                      stage_seconds_per_image={k: round(v / max(done, 1), 4) for k, v in timings.items()}, **extra)
        store.status(status_file, json.dumps(record, indent=2).encode())
        print(json.dumps(record), flush=True)
    status('loading', 0)
    if not pending:
        status('complete', 0)
        return
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.data_misc import FindStage, interpolate
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'CUDA unavailable'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=0.2)

    @contextmanager
    def timed(stage, cuda=False):
        if cuda:
            torch.cuda.synchronize()
        start = time.monotonic()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize()
            timings[stage] = timings.get(stage, 0) + time.monotonic() - start

    text_cache = {}
    def predict(state, prompt, find, geometry):
        if prompt not in text_cache:
            text_cache[prompt] = model.backbone.forward_text([prompt], device=processor.device)
        state['backbone_out'].update(text_cache[prompt])
        return model.forward_grounding(backbone_out=state['backbone_out'],
                                       find_input=find, find_target=None, geometric_prompt=geometry)

    def merge(result, threshold, shape, dilation):
        masks = result['masks'][result['scores'] > threshold]
        combined = np.zeros(shape, dtype=np.uint8)
        if masks.shape[0]:
            array = masks.any(dim=0).cpu().numpy().squeeze(0)
            if array.shape != shape:
                array = cv2.resize(array.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            combined |= (array > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation+1, 2*dilation+1))
        return cv2.dilate(combined, kernel, iterations=1)

    done = 0
    last = time.monotonic()
    writer = ImageWriter(store, args.upload_workers, args.upload_queue)

    def collect(timeout=0):
        nonlocal done, upload_seconds, encode_seconds, last
        completed = writer.collect(timeout=timeout)
        for item in completed:
            done += 1
            upload_seconds += item['upload_seconds']
            encode_seconds += item['encode_seconds']
        if (completed and done == len(completed)) or time.monotonic() - last >= 30:
            status('running', done)
            last = time.monotonic()

    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for offset in range(0, len(pending), args.batch_size):
                collect()
                batch_names = pending[offset:offset + args.batch_size]
                keys = [str(Path(name).with_suffix('.png')) for name in batch_names]
                need_mask = [key not in existing['mask_sam3'] for key in keys]
                need_semantic = [key not in existing['semantic_weights'] for key in keys]
                with timed('read_decode'):
                    images = []
                    for name in batch_names:
                        with Image.open(data / 'images' / name) as source:
                            images.append(source.convert('RGB'))
                b = len(images)
                with timed('image_encoder', cuda=True):
                    state = processor.set_image_batch(images)
                find = FindStage(img_ids=torch.arange(b, device='cuda'),
                                 text_ids=torch.zeros(b, dtype=torch.long, device='cuda'),
                                 input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                                 input_points=None, input_points_mask=None)
                geometry = model._get_dummy_prompt(num_prompts=b)
                binary, excluded = [None] * b, [None] * b
                backgrounds = [np.zeros((im.height, im.width), dtype=np.uint8) for im in images]
                prompts = ['person', 'floor', 'wall', 'ceiling'] if any(need_semantic) else ['person']
                for prompt in prompts:
                    with timed('person_decoder' if prompt == 'person' else 'background_decoder', cuda=True):
                        prediction = predict(state, prompt, find, geometry)
                    with timed('postprocess', cuda=True):
                        scores = (prediction['pred_logits'].sigmoid() *
                                  prediction['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
                        for i, im in enumerate(images):
                            if prompt != 'person' and not need_semantic[i]:
                                continue
                            keep = scores[i] > 0.2
                            shape = (im.height, im.width)
                            masks = interpolate(prediction['pred_masks'][i][keep].unsqueeze(1),
                                                shape, mode='bilinear', align_corners=False).sigmoid() > 0.5
                            result = dict(masks=masks, scores=scores[i][keep])
                            if prompt == 'person':
                                if need_mask[i]:
                                    binary[i] = merge(result, 0.3, shape, 5) * 255
                                if need_semantic[i]:
                                    excluded[i] = merge(result, 0.2, shape, 3)
                            else:
                                backgrounds[i] |= merge(result, 0.2, shape, 3)
                            del masks, result
                        del prediction, scores
                del state, find, geometry
                inferred += b
                # Bound retained CPU arrays. Count only futures with both PNGs committed.
                for i, name in enumerate(batch_names):
                    with timed('upload_wait'):
                        while len(writer.futures) >= writer.capacity:
                            collect(timeout=1)
                    outputs = {}
                    if need_mask[i]:
                        outputs['mask_sam3'] = binary[i]
                    if need_semantic[i]:
                        weights = np.full_like(backgrounds[i], 200)
                        weights[backgrounds[i] > 0] = 100
                        weights[excluded[i] > 0] = 0
                        outputs['semantic_weights'] = weights
                    writer.submit(name, outputs)
                    submitted += 1
                    collect()
                for im in images:
                    im.close()
        with timed('upload_wait'):
            while writer.futures:
                collect(timeout=1)
        status('complete', done)
    except BaseException as exc:
        status('failed', done, error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        writer.close()


if __name__ == '__main__':
    main()
