import hashlib
import importlib.metadata
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = json.loads((HERE / 'config.json').read_text())


def read_tasks(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    old = json.loads((HERE / 'legacy_sources.json').read_text())
    core = {name: sha(ROOT / name) for name in old['expected_core_sha256']}
    return {
        'config': CONFIG,
        'experiment_sources': {str(p.relative_to(HERE)): sha(p)
                               for p in HERE.rglob('*') if p.suffix in ('.py', '.json', '.md')},
        'core_sources': core,
        'core_matches_legacy': {name: core[name] == digest
                                for name, digest in old['expected_core_sha256'].items()},
        'qp_source': 'legacy/repo/aav_mpc/condensed_mpc.py (private frozen copy)',
        'packages': {name: importlib.metadata.version(name)
                     for name in ('numpy', 'torch', 'scipy', 'osqp', 'casadi')},
    }
