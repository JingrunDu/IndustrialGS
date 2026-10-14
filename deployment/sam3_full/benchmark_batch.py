"""Local capacity/throughput test; never writes production masks or submits jobs."""
import argparse
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from generate import names_from_colmap
from sam3.model.data_misc import FindStage, interpolate
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--max-batch', type=int, default=128)
    ap.add_argument('--min-batch', type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    report = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  cuda=torch.version.cuda, resolution=1008, precision='bf16',
                  timing='preloaded PIL -> batched encoder + 4 batched prompts -> CPU maps; no OSS writes',
                  results=[])
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)

    def emit(record):
        print(json.dumps(record), flush=True)
        path.write_text(json.dumps(report, indent=2))

    names = names_from_colmap(Path(args.data) / 'sparse/0_full/images.bin')
    images = []
    for i in range(args.max_batch):
        name = names[i * len(names) // args.max_batch]
        with Image.open(Path(args.data) / 'images' / name) as im:
            images.append(im.convert('RGB'))
        if (i + 1) % 16 == 0:
            print(f'Loaded {i+1}/{args.max_batch} representative images', flush=True)
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=0.2)
    kernels = {r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1)) for r in (3, 5)}

    def merge(masks, scores, threshold, radius, shape):
        selected = masks[scores > threshold]
        if len(selected):
            out = selected.any(dim=0).squeeze(0).cpu().numpy().astype(np.uint8)
        else:
            out = np.zeros(shape, dtype=np.uint8)
        return cv2.dilate(out, kernels[radius])

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        texts = {p: model.backbone.forward_text([p], device='cuda')
                 for p in ('person', 'floor', 'wall', 'ceiling')}

        def infer(batch):
            b = len(batch)
            state = processor.set_image_batch(batch)
            find = FindStage(img_ids=torch.arange(b, device='cuda'),
                             text_ids=torch.zeros(b, dtype=torch.long, device='cuda'),
                             input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                             input_points=None, input_points_mask=None)
            geometry = model._get_dummy_prompt(num_prompts=b)
            people, binary, background = [], [], [np.zeros((im.height, im.width), np.uint8) for im in batch]
            for prompt in texts:
                state['backbone_out'].update(texts[prompt])
                out = model.forward_grounding(backbone_out=state['backbone_out'],
                                             find_input=find, find_target=None, geometric_prompt=geometry)
                assert out['pred_masks'].shape[0] == b
                scores = (out['pred_logits'].sigmoid() * out['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
                for i, im in enumerate(batch):
                    keep = scores[i] > 0.2
                    s = scores[i][keep]
                    masks = interpolate(out['pred_masks'][i][keep].unsqueeze(1),
                                        (im.height, im.width), mode='bilinear', align_corners=False).sigmoid() > 0.5
                    shape = (im.height, im.width)
                    if prompt == 'person':
                        binary.append(merge(masks, s, 0.3, 5, shape) * 255)
                        people.append(merge(masks, s, 0.2, 3, shape))
                    else:
                        background[i] |= merge(masks, s, 0.2, 3, shape)
                    del masks
                del out, scores
            weights = []
            for i in range(b):
                w = np.full_like(background[i], 200)
                w[background[i] > 0] = 100
                w[people[i] > 0] = 0
                weights.append(w)
            return binary, weights

        print('Warming up complete inference, batch=1', flush=True)
        infer(images[:1])
        torch.cuda.synchronize()
        # Verify the new batch decoder against the original processor for one image.
        candidate = infer(images[:1])
        state = processor.set_image(images[0])
        ref_background = np.zeros((images[0].height, images[0].width), np.uint8)
        for prompt in texts:
            result = processor.set_text_prompt(prompt=prompt, state=state)
            if prompt == 'person':
                ref_binary = merge(result['masks'], result['scores'], 0.3, 5, ref_background.shape) * 255
                ref_person = merge(result['masks'], result['scores'], 0.2, 3, ref_background.shape)
            else:
                ref_background |= merge(result['masks'], result['scores'], 0.2, 3, ref_background.shape)
        ref_weights = np.full_like(ref_background, 200)
        ref_weights[ref_background > 0] = 100
        ref_weights[ref_person > 0] = 0
        assert np.array_equal(candidate[0][0], ref_binary)
        assert np.array_equal(candidate[1][0], ref_weights)
        del state, result, candidate
        report['batch1_matches_original'] = True
        if len(images) >= 4:
            batched = infer(images[:4])
            fractions = []
            for i in range(4):
                single = infer(images[i:i+1])
                fractions.append({
                    'mask_different_fraction': float(np.mean(batched[0][i] != single[0][0])),
                    'weights_different_fraction': float(np.mean(batched[1][i] != single[1][0])),
                })
            report['batch4_vs_single_pixel_difference'] = fractions
            del batched, single
            emit(dict(batch4_vs_single_pixel_difference=fractions))

        def test(b):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f'Testing complete inference batch={b}', flush=True)
            try:
                durations = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    output = infer(images[:b])
                    torch.cuda.synchronize()
                    durations.append(time.perf_counter() - start)
                    assert len(output[0]) == b and len(output[1]) == b
                    del output
                seconds = float(np.median(durations))
                record = dict(batch=b, ok=True, seconds=seconds, images_per_second=b/seconds,
                              peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            except torch.cuda.OutOfMemoryError:
                record = dict(batch=b, ok=False, error='CUDA OOM',
                              peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            report['results'].append(record)
            emit(record)
            gc.collect()
            torch.cuda.empty_cache()
            return record['ok']

        low, high, b = 0, None, args.min_batch
        while b <= args.max_batch:
            if not test(b):
                high = b
                break
            low = b
            b *= 2
        if high is not None:
            while high-low > 1:
                mid = (high+low)//2
                if test(mid):
                    low = mid
                else:
                    high = mid
        report['largest_passed_batch'] = low
        report['first_failing_batch'] = high
        emit(dict(finished=True, largest_passed_batch=low, first_failing_batch=high))


if __name__ == '__main__':
    main()
