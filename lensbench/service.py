import copy
import io
import json
import uuid
import zipfile
from PIL import Image
from . import core


class Bench:
    def __init__(self, store):
        self.store = store

    def image(self, item):
        return core.decode(self.store.bytes(item['asset']))

    def add(self, identifier, raw, name, revision=None):
        image = core.decode(raw)
        asset = self.store.asset(core.png(image))
        def operation(doc):
            if len(doc['images']) >= 40:
                raise core.Invalid('每个工程最多40幅照片')
            doc['images'].append({'id': uuid.uuid4().hex, 'name': str(name)[:100], 'asset': asset,
                                  'size': list(image.size), 'quality': core.quality(image),
                                  'enabled': True, 'observation': None})
        return self.store.mutate(identifier, revision, operation)

    def demo(self):
        config, images = core.demo_images()
        doc = self.store.create('合成棋盘镜头演示')
        for index, image in enumerate(images):
            doc = self.add(doc['id'], core.png(image), 'synthetic-board-' + str(index + 1) + '.png')
        return doc

    def configure(self, identifier, body):
        config = core.board_settings(body)
        def operation(doc):
            doc['board'] = config
            for item in doc['images']:
                item['observation'] = None
        return self.store.mutate(identifier, body.get('revision'), operation)

    def detect(self, identifier, body):
        doc = self.store.read(identifier)
        selected = body.get('image_ids') or [item['id'] for item in doc['images'] if item['enabled']]
        results = {}
        for image_id in selected:
            item = self.store.find(doc, 'images', image_id)
            results[image_id] = core.detect(self.image(item), doc['board'])
        def operation(current):
            for item in current['images']:
                if item['id'] in results:
                    item['observation'] = results[item['id']]
        return self.store.mutate(identifier, doc['revision'], operation)

    def toggle(self, identifier, image_id, body):
        def operation(doc):
            item = self.store.find(doc, 'images', image_id)
            item['enabled'] = bool(body['enabled'])
        return self.store.mutate(identifier, body.get('revision'), operation)

    def calibrate(self, identifier, body):
        doc = self.store.read(identifier)
        selected = [item for item in doc['images'] if item['enabled'] and (item.get('observation') or {}).get('found')]
        result = core.solve([item['observation'] for item in selected], doc['board'])
        result['id'] = uuid.uuid4().hex
        result['name'] = str(body.get('name', '标定档案'))[:100]
        result['sources'] = [copy.deepcopy(item) for item in selected]
        result['source_revision'] = doc['revision']
        def operation(current):
            current['profiles'].append(result)
            current['active_profile'] = result['id']
        return self.store.mutate(identifier, doc['revision'], operation)

    def import_profile(self, identifier, body):
        result = core.validate_profile(body)
        result.update({'id': uuid.uuid4().hex, 'name': str(body.get('name', '导入镜头'))[:100], 'sources': [], 'views': []})
        def operation(doc):
            doc['profiles'].append(result)
            doc['active_profile'] = result['id']
        return self.store.mutate(identifier, body.get('revision'), operation)

    def select_profile(self, identifier, body):
        def operation(doc):
            self.store.find(doc, 'profiles', body['profile_id'])
            doc['active_profile'] = body['profile_id']
        return self.store.mutate(identifier, body.get('revision'), operation)

    def process(self, identifier, body):
        doc = self.store.read(identifier)
        source = self.store.find(doc, 'images', body['image_id'])
        profile = self.store.find(doc, 'profiles', body.get('profile_id') or doc['active_profile'])
        image = self.image(source)
        kind = body.get('kind', 'undistort')
        assets = {}
        if kind == 'undistort':
            output, mask, metadata = core.undistort(image, profile, body.get('alpha', 0))
            assets['valid_mask'] = self.store.asset(core.png(mask))
        elif kind == 'pose':
            output, metadata = core.pose(image, source['observation'], profile)
        elif kind == 'plane':
            output, metadata = core.planar_view(image, source['observation'], profile, body.get('ppm', 2))
        else:
            raise core.Invalid('未知图像处理')
        assets['image'] = self.store.asset(core.png(output))
        result = {'id': uuid.uuid4().hex, 'kind': kind, 'source': copy.deepcopy(source),
                  'profile': copy.deepcopy(profile), 'assets': assets, 'metadata': metadata,
                  'size': list(output.size), 'source_revision': doc['revision']}
        def operation(current):
            current['outputs'].append(result)
        return self.store.mutate(identifier, doc['revision'], operation)

    def snapshot(self, identifier, body):
        def operation(doc):
            content = copy.deepcopy({k: v for k, v in doc.items() if k != 'snapshots'})
            doc['snapshots'].append({'id': uuid.uuid4().hex, 'name': str(body.get('name', '快照'))[:100], 'content': content})
        return self.store.mutate(identifier, body.get('revision'), operation)

    def restore(self, identifier, snapshot_id, body):
        def operation(doc):
            item = self.store.find(doc, 'snapshots', snapshot_id)
            for key in ('title', 'board', 'images', 'profiles', 'outputs', 'active_profile'):
                doc[key] = copy.deepcopy(item['content'][key])
        return self.store.mutate(identifier, body.get('revision'), operation)

    def export(self, identifier):
        doc = self.store.read(identifier)
        assets = set()
        def walk(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == 'asset' and isinstance(child, str):
                        assets.add(child)
                    elif key == 'assets':
                        assets.update(child.values())
                    else:
                        walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(doc)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('lensbench.json', json.dumps({'format': 'lensbench-1', 'document': doc}, ensure_ascii=False))
            for key in sorted(assets):
                archive.writestr('assets/' + key, self.store.bytes(key))
        return stream.getvalue()

    def import_project(self, raw):
        if len(raw) > 96 * 1024 * 1024:
            raise core.Invalid('工程压缩包上限96MB')
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if sum(item.file_size for item in archive.infolist()) > 256 * 1024 * 1024:
                    raise core.Invalid('工程展开内容过大')
                wrapper = json.loads(archive.read('lensbench.json'))
                if wrapper['format'] != 'lensbench-1':
                    raise core.Invalid('工程格式不兼容')
                doc = wrapper['document']
                core.board_settings(doc['board'])
                for profile in doc['profiles']:
                    core.validate_profile(profile)
                for member in archive.namelist():
                    if member.startswith('assets/') and len(member) == 71:
                        content = archive.read(member)
                        key = self.store.asset(content)
                        if key != member[7:]:
                            raise core.Invalid('资源内容指纹不一致')
                for image in doc['images']:
                    if list(core.decode(self.store.bytes(image['asset'])).size) != image['size']:
                        raise core.Invalid('图片尺寸不一致')
                doc['id'] = uuid.uuid4().hex
                doc['revision'] = 0
                self.store.write(doc)
                return doc
        except core.Invalid:
            raise
        except Exception as error:
            raise core.Invalid('工程内容损坏或缺少资源') from error
