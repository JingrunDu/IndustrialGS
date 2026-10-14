"""Render shared training views and build cross-experiment comparison panels."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--oss-output')
parser.add_argument('--per-camera', type=int, default=6)
parser.add_argument('--models', nargs='+', default=['E01_T01_seed0', 'E05_T05_seed0', 'E07_T07_seed0', 'E08_T08_seed0'])
a = parser.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]
root = Path('/data/oss_bucket_0/beifen/data/3dgs_bushu/experiments_20260923')
names = a.models
if not names or len(names) != len(set(names)) or any(Path(name).name != name for name in names):
    parser.error('Expected distinct experiment directory names')
labels = ['GT']
for name in names:
    spec = json.loads((root / name / 'experiment.json').read_text())
    labels.append(f'{spec["id"]}: {spec["num_images"]} images, seed {spec["seed"]}')
manifests = []
for name in names:
    with (out / (name + '.log')).open('w') as log:
        subprocess.run([sys.executable, '-u', str(repo / 'deployment/huanliufa_nebula/render_comparison.py'),
                        '--model', str(root / name), '--output', str(out / name), '--per-camera', str(a.per_camera)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest = json.loads((out / name / 'manifest.json').read_text())
    manifests.append(manifest)
    print(name, 'mean masked PSNR', manifest['mean_psnr'], flush=True)
assert all([v['image'] for v in m['views']] == [v['image'] for v in manifests[0]['views']] for m in manifests)
panels = out / 'panels'
panels.mkdir(exist_ok=True)
rows = []
for index, view in enumerate(manifests[0]['views']):
    key = Path(view['comparison']).stem
    images = [Image.open(out / names[0] / 'gt' / (key + '.png')).convert('RGB')]
    images += [Image.open(out / name / 'render' / (key + '.png')).convert('RGB') for name in names]
    w, h = images[0].size
    canvas = Image.new('RGB', (len(labels)*w, h+64), '#202020')
    draw = ImageDraw.Draw(canvas)
    for j, (label, im) in enumerate(zip(labels, images)):
        canvas.paste(im, (j*w, 64))
        draw.text((j*w+8, 6), label, fill='white')
        if j:
            draw.text((j*w+8, 24), 'Masked PSNR %.2f dB' % manifests[j-1]['views'][index]['valid_psnr'], fill='white')
    draw.text((8, 43), view['image'], fill='white')
    canvas.save(panels / f'{index:02d}.jpg', quality=95)
    thumb = canvas.copy()
    thumb.thumbnail((1800, 360))
    rows.append(thumb)
overview = Image.new('RGB', (1800, sum(im.height for im in rows)), '#303030')
y = 0
for im in rows:
    overview.paste(im, (0, y)); y += im.height
overview.save(out / 'overview.jpg', quality=93)
(out / 'index.html').write_text('<meta charset="utf-8"><title>3DGS comparison</title><h1>' + ' / '.join(labels) + '</h1><p>Shared training views. Metrics exclude people. Not held-out evaluation. Model iterations are recorded in each manifest.json.</p>' + ''.join(f'<p>{v["image"]}</p><a href="panels/{i:02d}.jpg"><img style="width:100%" src="panels/{i:02d}.jpg"></a>' for i,v in enumerate(manifests[0]['views'])))
(out / 'summary.json').write_text(json.dumps({name: m['mean_psnr'] for name,m in zip(names,manifests)}, indent=2))
if a.oss_output:
    shutil.copytree(out, a.oss_output, dirs_exist_ok=True)
    print('OSS output:', a.oss_output, flush=True)
print('Gallery:', out / 'index.html', flush=True)
