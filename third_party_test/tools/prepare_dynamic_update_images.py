from pathlib import Path
import json

root = Path('/home/dujingrun.djr/code/huanliufa_bushu')
out = root / 'third_party_test/动态更新渲染'
out.mkdir(exist_ok=True)
mapping = json.loads((root / 'third_party_test/渲染候选图/编号对应表.json').read_text())
records = []
for number in ['14', '11', '29']:
    entry = next(item for item in mapping if item['number'] == number)
    panel = root / entry['source_panel']
    source = panel.parent.parent
    index = int(panel.stem)
    manifest = json.loads((source / 'E01_T01_seed0/manifest.json').read_text())
    records.append(dict(number=number, image=manifest['views'][index]['image'], source=str(source)))
(out / 'selected_views.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
(out / 'images.txt').write_text('\n'.join(item['image'] for item in records) + '\n')
print(out)
