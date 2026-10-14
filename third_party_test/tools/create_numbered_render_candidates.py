from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json

root = Path('/home/dujingrun.djr/code/huanliufa_bushu')
source = root / 'experiments/next_16/review_665000/renders_E01_E09/panels'
out = root / 'third_party_test/渲染候选图'
out.mkdir(exist_ok=True)
font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 24)
small = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 17)
tiles = []
records = []
paths = list(sorted(source.glob('*.jpg')))
additional = root / 'third_party_test/新增视角渲染/panels'
if additional.exists():
    paths += [additional / ('%02d.jpg' % i) for i in [2, 3, 6, 8, 10, 16, 18, 20]]
second_batch = root / 'third_party_test/新增视角渲染_第二批/panels'
if second_batch.exists():
    paths += [second_batch / ('%02d.jpg' % i) for i in [2, 8, 9, 11, 13, 15, 22, 23, 29, 34]]
for index, path in enumerate(paths, 1):
    number = '%02d' % index
    with Image.open(path) as original:
        scene = original.crop((0, 64, original.width, original.height)).convert('RGB')
    panel = Image.new('RGB', (scene.width, scene.height + 54), 'white')
    panel.paste(scene, (0, 54))
    draw = ImageDraw.Draw(panel)
    for column, label in enumerate(['GT', 'Test 1', 'Test 2']):
        draw.text((column * scene.width // 3 + 14, 12), '#%s | %s' % (number, label), font=font, fill='#222222')
    filename = '候选%s.jpg' % number
    panel.save(out / filename, quality=95)
    records.append({'number': number, 'file': filename, 'source_panel': str(path.relative_to(root))})
    tile = Image.new('RGB', (720, 430), '#eeeeee')
    thumb = panel.copy()
    thumb.thumbnail((700, 380))
    tile.paste(thumb, ((720-thumb.width)//2, 42+(380-thumb.height)//2))
    ImageDraw.Draw(tile).text((16, 10), 'Candidate %s' % number, font=font, fill='#222222')
    tiles.append(tile)

overview = Image.new('RGB', (1440, 430 * ((len(tiles) + 1) // 2)), 'white')
for i, tile in enumerate(tiles):
    overview.paste(tile, ((i % 2) * 720, (i // 2) * 430))
overview.save(out / '候选总览.jpg', quality=92)
if len(tiles) >= 20:
    extra = Image.new('RGB', (1440, 430 * 4), 'white')
    for i, tile in enumerate(tiles[12:20]):
        extra.paste(tile, ((i % 2) * 720, (i // 2) * 430))
    extra.save(out / '新增候选13-20总览.jpg', quality=92)
if len(tiles) >= 30:
    extra = Image.new('RGB', (1440, 430 * 5), 'white')
    for i, tile in enumerate(tiles[20:30]):
        extra.paste(tile, ((i % 2) * 720, (i // 2) * 430))
    extra.save(out / '新增候选21-30总览.jpg', quality=92)
(out / '编号对应表.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
html = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>渲染候选图</title><style>body{margin:24px;font-family:sans-serif;color:#222}section{margin-bottom:32px}img{display:block;max-width:100%;height:auto}h2{font-size:20px}</style><h1>渲染候选图</h1><p>从左到右：实拍图、第一次测试、第二次测试。</p>']
for item in records:
    html.append('<section><h2>候选%s</h2><a href="%s"><img src="%s"></a></section>' % (item['number'], item['file'], item['file']))
html.append('</html>')
(out / 'index.html').write_text('\n'.join(html), encoding='utf-8')
print(out)
print('%d numbered candidates and overview generated' % len(records))
