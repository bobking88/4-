"""Verify document preservation and render the Word-exported PDF for visual QA."""
import hashlib
import json
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from lxml import etree
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/report_errata_20261111_v2"
before = Document(ROOT / "结题/历史版本/技术报告_正式版_20261111_指标勘误前.docx")
path = OUT / "report_candidate.docx"
after = Document(path)
manifest = json.loads((HERE / "change_manifest.json").read_text(encoding="utf-8"))
def math_node(node):
    name = etree.QName(node).localname
    if name.endswith('Pr'):
        return None
    if node.tag == qn('m:r'):
        return ''.join(n.text or '' for n in node.iter(qn('m:t')))
    children = []
    for child in node:
        item = math_node(child)
        if item is None:
            continue
        if isinstance(item, str) and children and isinstance(children[-1], str):
            children[-1] += item
        else:
            children.append(item)
    return [name, children]

# Word coalesces adjacent math runs on save. Compare text plus fraction/script
# structure instead of requiring the original run segmentation.
math = lambda d: [math_node(x) for x in d._element.xpath('//m:oMath')]
assert math(before) == math(after) and len(math(after)) == 29
body = lambda d: [[[c.text for c in r.cells] for r in t.rows[1:]] for t in d.tables]
assert body(before) == body(after) and len(after.tables) == 65
for row in manifest['paragraph_changes']:
    assert row['new'] in [p.text for p in after.paragraphs], row['paragraph']
assert not any('门控选对' in p.text or '门控选择正确专家' in p.text for p in after.paragraphs)
assert not any('门控选对' in c.text or '门控选择正确率' in c.text for t in after.tables for r in t.rows for c in r.cells)

pdf = pdfium.PdfDocument(OUT / 'report_candidate.pdf')
pages = OUT / 'pages'
pages.mkdir(exist_ok=True)
index = []
for i in range(len(pdf)):
    page = pdf[i]
    text = page.get_textpage().get_text_range()
    assert 'Error! Reference source not found' not in text
    assert '错误!未定义书签' not in text.replace(' ', '')
    page.render(scale=1.5).to_pil().convert('RGB').save(pages / f'page-{i+1:03}.png')
    index.append({'page': i+1, 'characters': len(text), 'text': text})
for start in range(0, len(pdf), 8):
    canvas = Image.new('RGB', (1800, 1320), '#dddddd')
    draw = ImageDraw.Draw(canvas)
    for j in range(start, min(start+8, len(pdf))):
        im = Image.open(pages / f'page-{j+1:03}.png')
        im.thumbnail((440, 625))
        x, y = ((j-start) % 4)*450, ((j-start)//4)*660
        canvas.paste(im, (x, y+25))
        draw.text((x+10, y+5), f'Page {j+1}', fill='black')
    canvas.save(OUT / f'overview-{start+1:03}.png')
(OUT / 'page_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
verification = {
    'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'pdf_sha256': hashlib.sha256((OUT/'report_candidate.pdf').read_bytes()).hexdigest(),
    'pages': len(pdf), 'native_omml': 29, 'math_text_and_fraction_script_structure_preserved': True,
    'all_65_table_bodies_preserved': True, 'all_revised_paragraphs_present': True,
    'visual_review': 'pending', 'new_training_or_statistics': False,
}
(HERE / 'verification.json').write_text(json.dumps(verification, indent=2)+'\n', encoding='utf-8')
print(json.dumps(verification, indent=2))
for row in index:
    keys = [s for s in ['条件一致', '0.0194', '0.0051', '图 22', '图 23', '图 26', 'R12', 'B-9'] if s in row['text']]
    if keys:
        print(row['page'], keys)
