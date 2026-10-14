from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

root = Path('/home/dujingrun.djr/code/huanliufa_bushu/third_party_test/动态更新渲染')
selected = json.loads((root / 'selected_views.json').read_text())
font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 24)
html = ['<!doctype html><meta charset="utf-8"><title>动态更新渲染对比</title><h1>动态更新渲染对比</h1><p>从左到右：实拍图、更新前、更新后。视角依次为原候选14、11、29。</p>']
metrics = []
for test, name in [(1, 'E01_T01_seed0'), (2, 'E09_T01_seed1')]:
    after_root = root / ('test%d_after' % test)
    after = json.loads((after_root / 'manifest.json').read_text())
    assert 'iteration_997500' in after['model']
    rows = []
    for item in selected:
        before_root = Path(item['source']) / name
        before = json.loads((before_root / 'manifest.json').read_text())
        assert 'iteration_665000' in before['model']
        old = next(view for view in before['views'] if view['image'] == item['image'])
        new = next(view for view in after['views'] if view['image'] == item['image'])
        oldkey, newkey = Path(old['comparison']).stem, Path(new['comparison']).stem
        paths = [after_root / 'gt' / (newkey+'.png'), before_root / 'render' / (oldkey+'.png'), after_root / 'render' / (newkey+'.png')]
        images = [Image.open(path).convert('RGB') for path in paths]
        assert all(image.size == images[0].size for image in images)
        w, h = images[0].size
        row = Image.new('RGB', (w*3, h), 'white')
        for col, image in enumerate(images):
            row.paste(image, (col*w, 0))
        rows.append(row)
        metrics.append(dict(test=test, view=item['number'], image=item['image'], before_psnr=old['valid_psnr'], after_psnr=new['valid_psnr']))
    width = rows[0].width
    canvas = Image.new('RGB', (width, 54+sum(row.height for row in rows)+24*(len(rows)-1)), 'white')
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(['GT', 'Before update', 'After update']):
        draw.text((col*width//3+14, 12), label, font=font, fill='#222222')
    y = 54
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + 24
    basename = '第%d次动态更新_对比大图' % test
    canvas.save(root / (basename+'.jpg'), quality=97, subsampling=0)
    canvas.save(root / (basename+'.png'))
    html.append('<h2>第%d次动态更新</h2><a href="%s.png"><img style="max-width:100%%" src="%s.jpg"></a>' % (test, basename, basename))
(root / 'index.html').write_text('\n'.join(html), encoding='utf-8')
(root / '对比指标.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
print(root / 'index.html')
