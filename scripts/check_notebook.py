"""Check notebook syntax and release metadata, or refresh the embedded setup guide."""
import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/'notebooks/Noise2Inverse360_HPC_v3.5.ipynb'


def setup_text():
    return (ROOT/'docs/SETUP_NERSC.md').read_text(encoding='utf-8').replace(
        '(NAFNET_CT.md)','(../docs/NAFNET_CT.md)')


def check(refresh=False):
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    setup = next(c for c in nb['cells'] if c['id']=='installation-and-workspace')
    if refresh:
        setup['source'] = setup_text().splitlines(keepends=True)
        NOTEBOOK.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    if ''.join(setup['source'])!=setup_text():
        raise ValueError('Setup guide changed. Run scripts/check_notebook.py --refresh-setup.')
    numbers = []
    for cell in nb['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type']=='code':
            ast.parse(source,feature_version=(3,8))
            if cell['outputs'] or cell['execution_count'] is not None:
                raise ValueError('Release notebooks must have empty execution outputs.')
            match = re.match(r'# (\d+)\. ',source)
            if not match:
                raise ValueError('Code cell needs a numbered title: '+cell['id'])
            numbers.append(int(match[1]))
        elif sum(line.startswith('```') for line in source.splitlines())%2:
            raise ValueError('Unbalanced Markdown code fence: '+cell['id'])
    if numbers!=list(range(1,len(numbers)+1)):
        raise ValueError('Cell numbers must be sequential.')
    print('Notebook checks passed:',len(numbers),'code cells; no actions executed.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-setup',action='store_true')
    check(parser.parse_args().refresh_setup)
