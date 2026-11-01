"""Cheap read-only audit validation; default notebook never loads models/data."""
import ast
import json
from .storage import ROOT,file_hash,digest
from .protocol import guard

REQUIRED=['protocol.json','SARA_REPRODUCTION_AUDIT.md','sara_configuration_crosswalk.csv',
'literature_root_cause_matrix.csv','retriever_candidates.csv','compression_candidates.csv','reader_candidates.csv',
'dataset_confirmation_plan.md','experiment_matrix.csv','resource_feasibility.json','root_cause_hypotheses.json',
'execution_plan.md','W10_ROOT_CAUSE_PREFLIGHT.md']

# These notebook switches authorize execution; method settings remain in the
# separately hashed config and execution freeze. They must be editable normally.
EXECUTION_CONTROLS = {
    'RUN_AUDIT', 'RUN_RETRIEVER_FULL', 'RUN_COMPRESSION_FULL',
    'RUN_READER_FULL', 'RUN_DATASET_ANALYSIS_FULL', 'RUN_POLICY_EXPERIMENT',
    'RUN_TINY_MODEL_SMOKE', 'RUN_REPORT', 'SMOKE_ONLY', 'REVIEW_PHASE',
    'REVIEWED_RETRIEVER', 'REVIEW_REASON',
}

class ExecutionControls(ast.NodeTransformer):
    def visit_Assign(self, node):
        if all(isinstance(t, ast.Name) and t.id in EXECUTION_CONTROLS for t in node.targets):
            node.value = ast.Constant(value='<manual execution control>')
        return self.generic_visit(node)


def notebook_source_hash(path):
    """Pin instructions and executable logic, excluding Jupyter execution state.

    AST canonicalization ignores code formatting and comments. Only explicitly
    allowlisted execution-switch values are mutable; other code changes fail.
    Markdown text, cell order and cell types remain part of the identity.
    """
    notebook = json.loads(path.read_text())
    cells = []
    for cell in notebook['cells']:
        source = cell.get('source', '')
        if isinstance(source, list): source = ''.join(source)
        if cell['cell_type'] == 'code':
            tree = ExecutionControls().visit(ast.parse(source))
            source = ast.dump(tree, include_attributes=False)
        cells.append(dict(cell_type=cell['cell_type'], source=source,
                          attachments=cell.get('attachments', {})))
    return digest(cells)


def artifact_hash(path, scheme='sha256-bytes'):
    if scheme == 'notebook-source-v1': return notebook_source_hash(path)
    if scheme != 'sha256-bytes': raise ValueError('Unknown artifact hash scheme: '+scheme)
    return file_hash(path)


def validate_artifacts(cfg):
    guard(cfg['split']);base=ROOT/cfg['output_root'];manifest=json.loads((base/'manifest.json').read_text())
    missing=[name for name in REQUIRED if not (base/name).exists()]
    schemes=manifest.get('artifact_hash_schemes', {})
    changed=[name for name,sha in manifest['artifacts'].items() if not (ROOT/name).is_file() or artifact_hash(ROOT/name,schemes.get(name,'sha256-bytes'))!=sha]
    dependency_changes=[name for name,sha in manifest['read_only_dependencies'].items() if not (ROOT/name).is_file() or file_hash(ROOT/name)!=sha]
    if missing or changed or dependency_changes:raise ValueError(dict(missing=missing,changed=changed,dependency_changes=dependency_changes))
    return dict(audit_valid=True,split=cfg['split'],full_experiments_started=False,GO_NEXT=True,
        go_scope='synthetic checks and manually qualified Phase R only; no completed root-cause findings',
        remaining_gates=['representative GPU memory smoke','faithful SeleCom adapter','strong learned textual compressor',
            'reader panel feasibility/freeze','fresh external confirmation exposure audit'])
