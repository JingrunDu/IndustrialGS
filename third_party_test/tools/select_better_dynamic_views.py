import json
from pathlib import Path

repo = Path('/home/dujingrun.djr/code/huanliufa_bushu')
root = repo / 'third_party_test/动态更新渲染'
mapping = json.loads((repo / 'third_party_test/渲染候选图/编号对应表.json').read_text())
records = []
for index, number in enumerate(['14', '18', '15', '24', '16', '27']):
    item = next(row for row in mapping if row['number'] == number)
    panel = repo / item['source_panel']
    manifest = json.loads((panel.parent.parent / 'E01_T01_seed0/manifest.json').read_text())
    view = manifest['views'][int(panel.stem)]
    records.append(dict(number=number, image=view['image'], test=1+index%2))
assert len({row['image'] for row in records}) == 6
(root / '精选视角.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
for test in [1, 2]:
    (root / ('精选test%d.txt' % test)).write_text('\n'.join(row['image'] for row in records if row['test'] == test)+'\n')
print(root / '精选视角.json')
