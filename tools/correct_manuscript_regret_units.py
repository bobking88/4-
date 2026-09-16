"""Correct historical x100 loss displays without changing probability metrics."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLACEMENTS = {
    '平均路由后悔 |': '平均路由后悔 / nat |',
    '9.86% ± 0.96% |': '0.0986 ± 0.0096 |',
    '8.09% ± 0.17% |': '0.0809 ± 0.0017 |',
    '11.77% ± 0.63% |': '0.1177 ± 0.0063 |',
    '8.24% ± 0.80% |': '0.0824 ± 0.0080 |',
    '平均路由后悔降低 1.77 个百分点': '平均路由后悔降低 0.0177 nat',
    '平均路由后悔相对 HRGV 降低 1.77 个百分点': '平均路由后悔相对 HRGV 降低 0.0177 nat',
    '[−2.86, −0.69] 个百分点': '[−0.0286, −0.0069] nat',
    '路由后悔进一步降低 3.53 个百分点': '路由后悔进一步降低 0.0353 nat',
    '[−4.75, −2.17] 个百分点': '[−0.0475, −0.0217] nat',
    '路由后悔降低 3.33 个百分点': '路由后悔降低 0.0333 nat',
    '[−4.86, −1.98] 个百分点': '[−0.0486, −0.0198] nat',
    '平均路由后悔增加 1.94 个百分点': '平均路由后悔增加 0.0194 nat',
    '[1.30, 2.60] 个百分点': '[0.0130, 0.0260] nat',
    '路由后悔相对完整 RSG 的差异为 −0.20 个百分点，区间 [−0.62, 0.12]': '路由后悔相对完整 RSG 的差异为 −0.0020 nat，区间 [−0.0062, 0.0012] nat',
    '路由后悔下降 0.51 个百分点，区间 [−0.85, −0.16]': '路由后悔下降 0.0051 nat，区间 [−0.0085, −0.0016] nat',
    'RSG−HRGV 为 −3.53 个百分点': 'RSG−HRGV 为 −0.0353 nat',
    '平均路由后悔差异为 −1.77 个百分点，95%区间为 [−2.86, −0.69]': '平均路由后悔差异为 −0.0177 nat，95%区间为 [−0.0286, −0.0069] nat',
    '摄影者留出确认中为 −3.53 个百分点，95%区间为 [−4.75, −2.17]': '摄影者留出确认中为 −0.0353 nat，95%区间为 [−0.0475, −0.0217] nat',
    '主干替换中为 −3.33 个百分点，95%区间为 [−4.86, −1.98]': '主干替换中为 −0.0333 nat，95%区间为 [−0.0486, −0.0198] nat',
}


def main():
    for name in ('manuscript_core_draft.md', 'theoretical_appendix.md'):
        path = ROOT / 'docs/paper_v1' / name
        text = path.read_text(encoding='utf-8')
        count = 0
        for old, new in REPLACEMENTS.items():
            count += text.count(old)
            text = text.replace(old, new)
        if count:
            path.write_text(text, encoding='utf-8')
        print(f'{name}: {count} replacements')


if __name__ == '__main__':
    main()
