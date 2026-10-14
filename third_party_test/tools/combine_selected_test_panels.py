from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path('/home/dujingrun.djr/code/huanliufa_bushu/third_party_test')
paths = [root / '渲染候选图' / ('候选%s.jpg' % number) for number in ['14', '11', '29']]
images = []
for path in paths:
    with Image.open(path) as original:
        images.append(original.crop((0, 54, original.width, original.height)).convert('RGB'))
font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 24)
width = max(im.width for im in images)
gap = 24
header = 54
height = header + sum(im.height for im in images) + gap * (len(images) - 1)
canvas = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(canvas)
for column, label in enumerate(['GT', 'Test 1', 'Test 2']):
    draw.text((column * width // 3 + 14, 12), label, font=font, fill='#222222')
y = header
for im in images:
    canvas.paste(im, ((width - im.width) // 2, y))
    y += im.height + gap
output = root / '渲染对比大图_14_11_29.jpg'
canvas.save(output, quality=97, subsampling=0)
canvas.save(output.with_suffix('.png'))
print(output)
print('size:', canvas.size)
