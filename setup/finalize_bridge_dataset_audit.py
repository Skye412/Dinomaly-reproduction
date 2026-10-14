"""Normalize numeric-ID scope notes from the initial audit pass, preserving raw evidence."""
import json
from pathlib import Path

folder = Path('/home/skye/data/Skye/Dinomaly/reproduction/audits/bridge-datasets-20261014')
path = folder / 'cif_audit.json'
report = json.loads(path.read_text())
old_id_notes = [e for e in report['errors'] if e.startswith('Cross-split image_id overlap: ')]
if old_id_notes:
    (folder / 'cif_initial_pass.json').write_text(json.dumps(report, indent=2))
    report['errors'] = [e for e in report['errors'] if not e.startswith('Cross-split image_id overlap: ')]
notes = old_id_notes + report.pop('id_overlap_notes', [])
report['numeric_id_reuses_across_splits'] = len(notes)
report['numeric_id_reuse_examples'] = notes[:20]
report['image_id_scope_note'] = 'Numeric IDs are reused between official splits. This is not proof of duplicate images. Use (split, image_id) or file_name as the sample key; filename overlaps are checked separately.'
report['official_files_match'] = len(report['files']) == 498 and all(f['matches_official'] for f in report['files']) and not report['missing'] and not report['extra']
report['passed'] = not (report['errors'] or report['missing'] or report['extra'])
path.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))
