from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

root = Path('/home/dujingrun.djr/code/huanliufa_bushu/third_party_test/动态更新渲染')
selected = json.loads((root / 'selected_views.json').read_text())
archive = Path('/data/oss_bucket_0/beifen/data/3dgs_bushu/dynamic_update/20261005_173800')
archived = [archive / ('test%d/visualization/after' % test) for test in [1, 2]]
archive_manifests = [json.loads((folder / 'manifest.json').read_text()) for folder in archived]
local = [root / ('test%d_after' % test) for test in [1, 2]]
views = json.loads((root / '精选视角.json').read_text())
chosen = [[(root / ('精选test%d' % view['test']), view['image']) for view in views[i:i+2]]
          for i in range(0, 6, 2)]
assert len({name for row in chosen for folder, name in row}) == 6
rows = []
records = []
for pairs in chosen:
    images = []
    for folder, name in pairs:
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert 'iteration_997500' in manifest['model']
        view = next(v for v in manifest['views'] if v['image'] == name)
        key = Path(view['comparison']).stem
        records.append(dict(image=name, model=manifest['model']))
        for category in ['gt', 'render']:
            with Image.open(folder / category / (key + '.png')) as im:
                images.append(im.convert('RGB'))
    assert all(im.size == images[0].size for im in images)
    w, h = images[0].size
    row = Image.new('RGB', (w*4, h), 'white')
    for column, im in enumerate(images):
        row.paste(im, (column*w, 0))
    rows.append(row)

width = rows[0].width
header, gap = 54, 24
canvas = Image.new('RGB', (width, header + sum(row.height for row in rows) + gap*2), 'white')
font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 24)
draw = ImageDraw.Draw(canvas)
for column, label in enumerate(['GT', 'Render (Test 1)', 'GT', 'Render (Test 2)']):
    draw.text((column*width//4 + 14, 12), label, font=font, fill='#222222')
y = header
for row in rows:
    canvas.paste(row, (0, y))
    y += row.height + gap
output = root / '动态更新_GT渲染_6对_精选版'
canvas.save(output.with_suffix('.jpg'), quality=97, subsampling=0)
canvas.save(output.with_suffix('.png'))
(root / '6对不同视角对应表.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print(output.with_suffix('.jpg'))
print('size:', canvas.size, 'images:', 12, 'pairs:', 6)
