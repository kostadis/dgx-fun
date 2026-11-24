"""Build one full-module CROW wiki with the existing local Inkling, Clef and embeddings."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback
from llmw2.batch import Item
from runtime import Capture, REPO, TRIAL, SOURCE, PIN, load, save, sha, SERVICES

p = argparse.ArgumentParser(); p.add_argument('--output', required=True, type=Path); a = p.parse_args()
root = a.output.resolve(); assert root.is_relative_to(TRIAL)
root.mkdir(parents=True, exist_ok=True); assert not (root / 'ingest-progress.json').exists()
state = {'stage': 'preflight', 'passed': False, 'started_unix': time.time()}; save(root / 'ingest-progress.json', state)
capture = None
try:
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == PIN
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
    before = SERVICES.services(); SERVICES.check_services(before, before)
    assert before['load']['server_load'] == 0
    save(root / 'services-before.json', before)
    protected = {str(f.relative_to(TRIAL)):sha(f) for f in TRIAL.rglob('*.json') if not f.is_relative_to(root)}
    save(root / 'protected-inputs.json', protected)
    corpus = load(TRIAL / 'frozen/corpus.json'); assert len(corpus['documents']) == 13
    inputs = {k:sha(TRIAL / ('frozen/' + k + '.json')) for k in ['corpus', 'cases']}
    save(root / 'protocol.json', {'upstream_commit': PIN, 'upstream_license': 'AGPL-3.0-or-later',
        'input_sha256': inputs, 'source_documents': 13, 'ingestion_model': 'inkling-2-mini',
        'classifier': 'clef', 'classifier_endpoint': 'http://192.168.1.121:8002/v1/systemone',
        'embedding': 'qwen3-embedding:0.6b', 'all_local': True, 'source_code_sha256':
        {f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
        'granularity': 'Same thirteen frozen full-module source documents as Cognee; wiki chooses notes and merges.',
        'cognee_ingestion_model': 'qwen3.8-flash-next', 'ingestion_times_are_not_model_matched': True})
    capture = Capture(root); capture.phase = 'ingest'
    with capture.wiki(root / 'wiki') as wiki:
        save(root / 'wiki-config.json', wiki.cfg.model_dump(mode='json'))
        wiki.init()
        items = [Item(filename=d['id'] + ' ' + d['title'] + '.md', text=d['text'],
            resource='module://' + d['id'], fields={'document_id':d['id']}) for d in corpus['documents']]
        def progress(item, result):
            row = {'file':item.filename, 'result':result.to_dict() if hasattr(result, 'to_dict') else {'error':repr(result)}}
            save(root / 'ingest-documents' / (str(len(list((root / 'ingest-documents').glob('*.json'))) if (root / 'ingest-documents').exists() else 0) + '.json'), row)
            state.update(stage='ingest', completed=len(list((root / 'ingest-documents').glob('*.json'))), last_file=item.filename)
            save(root / 'ingest-progress.json', state); print('filed', item.filename, flush=True)
        tick = time.monotonic(); report = wiki.ingest_many(items, concurrency=1, on_result=progress)
        save(root / 'ingest-report.json', report.to_dict()); state['ingest_seconds'] = time.monotonic()-tick
        assert report.stopped is None and not report.not_done and len(report.results) == 13
        assert not any(isinstance(r, Exception) for _,r in report.results)
        notes = wiki.store.load().all_notes()
        save(root / 'notes.json', [{'path':n.rel, 'id':n.id, 'text':n.full_text(), 'body':n.main_body(), 'frontmatter':n.frontmatter} for n in notes])
        state.update(notes=len(notes))
    capture.check('after-ingest')
    state['passed'] = True
except Exception:
    state['error'] = traceback.format_exc(); print(state['error'], flush=True)
finally:
    if capture:capture.close()
    state.update(finished_unix=time.time(), stage='complete' if state['passed'] else 'failed')
    save(root / 'ingest-progress.json', state)
raise SystemExit(0 if state['passed'] else 1)
