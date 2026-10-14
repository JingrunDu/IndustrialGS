"""Build GT / before / after panels for the same camera views."""
import json
from pathlib import Path
import sys
from PIL import Image, ImageDraw


def main(directory):
    root = Path(directory)
    before = json.loads((root / 'before/manifest.json').read_text())
    after = json.loads((root / 'after/manifest.json').read_text())
    if [v['image'] for v in before['views']] != [v['image'] for v in after['views']]:
        raise ValueError('Camera views differ')
    panels = root / 'panels'
    panels.mkdir(exist_ok=True)
    html = ['<meta charset="utf-8"><title>GT / Before / After</title>']
    for i, view in enumerate(before['views']):
        key = Path(view['comparison']).stem
        images = [Image.open(root / label / folder / (key + '.png')).convert('RGB')
                  for label, folder in [('before', 'gt'), ('before', 'render'), ('after', 'render')]]
        w, h = images[0].size
        canvas = Image.new('RGB', (w*3, h+48), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, (label, im) in enumerate(zip(['GT', 'Before update', 'After update'], images)):
            canvas.paste(im, (j*w, 48))
            draw.text((j*w+8, 12), label, fill='black')
        filename = '%02d.jpg' % (i+1)
        canvas.save(panels / filename, quality=95)
        html.append('<p>%s</p><img style="max-width:100%%" src="panels/%s">' % (view['image'], filename))
    (root / 'index.html').write_text('\n'.join(html))


if __name__ == '__main__':
    main(sys.argv[1])
