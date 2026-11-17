import copy
import hashlib
import json
import os
import threading
import uuid
from pathlib import Path
from .core import Invalid


class Conflict(Invalid):
    pass


class Store:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'assets').mkdir(exist_ok=True)
        self.lock = threading.RLock()

    def create(self, title):
        identifier = uuid.uuid4().hex
        doc = {'id': identifier, 'title': str(title)[:100] or 'Lens session', 'revision': 0,
               'board': {'columns': 8, 'rows': 6, 'square_mm': 25.0},
               'images': [], 'profiles': [], 'outputs': [], 'snapshots': [], 'active_profile': None,
               'stereo': []}
        self.write(doc)
        return doc

    def path(self, identifier):
        if not isinstance(identifier, str) or len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
            raise Invalid('工程ID不正确')
        return self.root / (identifier + '.json')

    def read(self, identifier):
        with self.lock:
            path = self.path(identifier)
            if not path.exists():
                raise Invalid('工程不存在')
            return json.loads(path.read_text())

    def write(self, doc):
        path = self.path(doc['id'])
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(doc, ensure_ascii=False, allow_nan=False))
        os.replace(temporary, path)

    def mutate(self, identifier, revision, operation):
        with self.lock:
            doc = self.read(identifier)
            if revision is not None and revision != doc['revision']:
                raise Conflict('工程已改变，请重新加载')
            operation(doc)
            doc['revision'] += 1
            self.write(doc)
            return copy.deepcopy(doc)

    def asset(self, raw):
        key = hashlib.sha256(raw).hexdigest()
        path = self.root / 'assets' / key
        if not path.exists():
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(raw)
            os.replace(temporary, path)
        return key

    def bytes(self, key):
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise Invalid('资源ID不正确')
        path = self.root / 'assets' / key
        if not path.is_file():
            raise Invalid('资源不存在')
        return path.read_bytes()

    def listed(self):
        output = []
        for path in self.root.glob('*.json'):
            doc = json.loads(path.read_text())
            output.append({'id': doc['id'], 'title': doc['title'], 'revision': doc['revision'],
                           'images': len(doc['images']), 'profiles': len(doc['profiles'])})
        return output

    def find(self, doc, group, identifier):
        for item in doc.get(group, []):
            if item['id'] == identifier:
                return item
        raise Invalid('所选条目不存在')
