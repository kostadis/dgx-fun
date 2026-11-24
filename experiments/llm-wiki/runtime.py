"""Local-only capture and configuration for the isolated Phandelver wiki benchmark."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import threading
import time
import uuid

import httpx
from llmw2 import Wiki
from llmw2.config import WikiConfig, LLMConfig, ClassifierConfig, EmbedderConfig
from llmw2.models.llm import LLM
from llmw2.models.classifier import Classifier
from llmw2.models.usage import UsageTracker

REPO = Path(__file__).resolve().parents[2]
TRIAL = REPO / 'experiments/adaptive-chunking/raw/pabtso/cognee-20261120'
MODEL = 'inkling-2-mini'
NATIVE = 'http://192.168.1.121:8004/v1'
CLEF = 'http://192.168.1.121:8002/v1'
OLLAMA = 'http://192.168.56.1:11434'
EMBED = 'qwen3-embedding:0.6b'
PIN = '3333c71485bf271a89544090647a91abe25c70ca'
SOURCE = Path('/tmp/dgx-llmwiki-v2-20261124')


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):h.update(b)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


SERVICES = module('wiki_stock_service_checks', REPO / 'experiments/inkling/stock-cognee/run.py')
SEQ = module('wiki_existing_agent', REPO / 'experiments/adaptive-chunking/cognee/sequential.py')
QUESTION_SYSTEM = ('Answer using only the retrieved evidence. Preserve negation. Say unknown when '
                   'the evidence does not establish the answer. Be concise (at most 120 words).')


class Capture:
    def __init__(self, root):
        self.root = root; self.phase = 'startup'; self.rows = []; self.lock = threading.Lock()
        self.client = httpx.Client(timeout=900, event_hooks={'request': [self.before], 'response': [self.after]})
        self.embedding_digest = next(m['digest'] for m in self.client.get(OLLAMA + '/api/tags').json()['models'] if m['name'] == EMBED)
        assert self.embedding_digest == load(TRIAL / 'current/manifest.json')['embedding_digest']

    def before(self, request):
        assert (request.url.host, request.url.port) in {('192.168.1.121', 8004), ('192.168.1.121', 8002), ('192.168.56.1', 11434)}
        body = json.loads(request.content) if request.content else None
        request.extensions['audit'] = {'id': uuid.uuid4().hex, 'phase': self.phase, 'url': str(request.url),
            'started_unix': time.time(), 'request': body, 'tick': time.monotonic()}
        if request.url.port == 8004 and request.method == 'POST':
            assert body['model'] == MODEL and body['temperature'] == 0 and body['seed'] == 604
            assert body['reasoning_effort'] == 'none' and body['bad_words'] == ['<|content_thinking|>']
            assert body['chat_template_kwargs']['enable_thinking'] is False

    def after(self, response):
        response.read(); row = response.request.extensions['audit']
        row['seconds'] = time.monotonic() - row.pop('tick'); row['status'] = response.status_code
        try:row['response'] = response.json()
        except ValueError:row['response'] = {'unparsed': response.text}
        kind = {8004: 'llm', 8002: 'classifier', 11434: 'embedding'}[response.request.url.port]
        row['kind'] = kind
        # Keep full source-bearing model captures only in the ignored experiment directory.
        save(self.root / 'calls' / (row['id'] + '.json'), row)
        with self.lock:self.rows.append(row)
        if kind == 'llm' and response.is_success:
            m = row['response']['choices'][0]['message']
            assert not (m.get('reasoning_content') or m.get('reasoning'))
            assert row['response']['usage'].get('completion_tokens_details', {}).get('reasoning_tokens', 0) == 0

    def post(self, url, body):
        r = self.client.post(url, json=body); r.raise_for_status(); return r.json()

    def options(self, cap):
        return {'reasoning_effort': 'none', 'bad_words': ['<|content_thinking|>'],
                'chat_template_kwargs': {'enable_thinking': False}, 'max_tokens': cap,
                'cache_salt': self.root.name + '-' + self.phase.split('/')[0]}

    def complete(self, messages, cap=4096, response_format=None):
        body = {'model': MODEL, 'messages': messages, 'temperature': 0, 'seed': 604, **self.options(cap)}
        if response_format is not None:body['response_format'] = response_format
        d = self.post(NATIVE + '/chat/completions', body)
        assert d['choices'][0]['finish_reason'] == 'stop', 'Retain truncated response as failure'
        return d['choices'][0]['message']['content']

    def embed(self, texts):
        out = []
        for i in range(0, len(texts), 4):
            batch = texts[i:i+4]
            d = self.post(OLLAMA + '/api/embed', {'model': EMBED, 'input': batch, 'truncate': False})
            vectors = d['embeddings']; assert len(vectors) == len(batch)
            assert all(len(v) == 1024 and all(math.isfinite(x) for x in v) and any(v) for v in vectors)
            out.extend(vectors)
        return out

    def wiki(self, bundle, *, steps=None, rerank=True, embedding=True, cap=16384):
        cfg = WikiConfig(bundle=bundle, mode='crow', concurrency=1,
            llm=LLMConfig(provider='custom', base_url=NATIVE, model=MODEL, temperature=0, seed=604,
                timeout=900, read_chars=100000, extra_body=self.options(cap)),
            classifier=ClassifierConfig(provider='custom', base_url=CLEF, model='clef', timeout=120,
                request_chars=24000), embedder=EmbedderConfig(model=EMBED))
        cfg.crow.rerank = rerank
        tracker = UsageTracker(self.root / (self.phase.replace('/', '-') + '-usage.jsonl'))
        llm = LLM(cfg.llm, tracker, client=self.client)
        classifier = Classifier(cfg.classifier, tracker, client=self.client)
        return Wiki(cfg, llm=llm, classifier=classifier, steps=steps, embedder=self.embed if embedding else None)

    def check(self, name):
        current = SERVICES.services(); SERVICES.check_services(current, load(self.root / 'services-before.json'))
        assert current['guard_stats']['container_id'] == current['native']['id']
        save(self.root / ('services-' + name + '.json'), current)

    def close(self):self.client.close()
