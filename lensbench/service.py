import copy
import io
import json
import uuid
import zipfile
import numpy as np
from PIL import Image
from . import core
from . import stereo


class Bench:
    def __init__(self, store):
        self.store = store
        self._stereo_cache = {}

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

    def _asset_image(self, key):
        return Image.open(io.BytesIO(self.store.bytes(key))).convert('RGB')

    def _stereo_ctx(self, record):
        with self.store.lock:
            cached = self._stereo_cache.get(record['id'])
            if cached:
                return cached
            settings = record['matcher']
            gray_left = np.asarray(self._asset_image(record['assets']['rect_left']).convert('L'))
            gray_right = np.asarray(self._asset_image(record['assets']['rect_right']).convert('L'))
            ctx = {'gray_left': gray_left, 'gray_right': gray_right,
                   'cl': stereo.census(gray_left), 'cr': stereo.census(gray_right),
                   'base_disp': stereo.read_disp(self.store.bytes(record['assets']['disp_base']), settings['min_disparity']),
                   'base_reason': np.asarray(self._asset_image(record['assets']['reason_base']).convert('L')),
                   'base_cost': stereo.read_npy(self.store.bytes(record['assets']['cost_base'])),
                   'disp': stereo.read_disp(self.store.bytes(record['assets']['disp']), settings['min_disparity']),
                   'reason': np.asarray(self._asset_image(record['assets']['reason']).convert('L'))}
            self._stereo_cache = {record['id']: ctx}
            return ctx

    def stereo_create(self, identifier, body):
        doc = self.store.read(identifier)
        left_item = self.store.find(doc, 'images', str(body.get('left_image', '')))
        right_item = self.store.find(doc, 'images', str(body.get('right_image', '')))
        if left_item['id'] == right_item['id']:
            raise core.Invalid('左右图必须是不同的照片')
        left_profile = self.store.find(doc, 'profiles', str(body.get('left_profile', '')))
        right_profile = self.store.find(doc, 'profiles', str(body.get('right_profile', '')))
        rig = stereo.validate_rig(body)
        settings = stereo.matcher_settings(body)
        result = stereo.compute(self.image(left_item), self.image(right_item), left_profile, right_profile, rig, settings)
        images = result['images']
        assets = {'rect_left': self.store.asset(core.png(images['left'])),
                  'rect_right': self.store.asset(core.png(images['right'])),
                  'mask_left': self.store.asset(core.png(Image.fromarray(images['valid_left'].astype(np.uint8) * 255))),
                  'mask_right': self.store.asset(core.png(Image.fromarray(images['valid_right'].astype(np.uint8) * 255))),
                  'depth_vis': self.store.asset(core.png(result['depth_vis'])),
                  'reason_vis': self.store.asset(core.png(result['reason_vis'])),
                  'disp_base': self.store.asset(stereo.write_disp(result['disp'], settings['min_disparity'])),
                  'reason_base': self.store.asset(core.png(Image.fromarray(result['reason']))),
                  'cost_base': self.store.asset(stereo.npy_bytes(result['best_c']))}
        assets['disp'] = assets['disp_base']
        assets['reason'] = assets['reason_base']
        assets['cost'] = assets['cost_base']
        record = {'id': uuid.uuid4().hex, 'name': str(body.get('name', '立体结果'))[:100], 'parent': None,
                  'created_revision': doc['revision'],
                  'left': copy.deepcopy(left_item), 'right': copy.deepcopy(right_item),
                  'left_profile': copy.deepcopy(left_profile), 'right_profile': copy.deepcopy(right_profile),
                  'rig': rig, 'matcher': settings, 'rectify': result['rectify'],
                  'assets': assets, 'confirmations': [], 'conflicts': [], 'stats': result['stats']}
        record['fingerprint'] = core.fingerprint({'base': [assets['disp_base'], assets['reason_base'], assets['cost_base']],
                                                  'confirmations': [], 'rig': rig, 'matcher': settings})
        def operation(current):
            current.setdefault('stereo', []).append(record)
        return self.store.mutate(identifier, body.get('revision'), operation)

    def stereo_probe(self, identifier, sid, u, v):
        doc = self.store.read(identifier)
        record = self.store.find(doc, 'stereo', sid)
        ctx = self._stereo_ctx(record)
        height, width = ctx['reason'].shape
        u = core.integer(u, '列', 0, width - 1)
        v = core.integer(v, '行', 0, height - 1)
        code = int(ctx['reason'][v, u])
        disparity = float(ctx['disp'][v, u])
        rectify = record['rectify']
        focal = rectify['focal_px']
        baseline = record['rig']['baseline_mm']
        output = {'u': u, 'v': v, 'valid': code == 0, 'reason': code,
                  'reason_text': stereo.REASONS[code], 'reason_key': stereo.REASON_KEYS[code],
                  'disparity': round(disparity, 3)}
        if code == 0 and disparity > 0.2:
            z = focal * baseline / disparity
            p1 = rectify['P1']
            point = np.array([(u - p1[0][2]) * z / focal, (v - p1[1][2]) * z / p1[1][1], z])
            original = np.asarray(rectify['R1'], np.float64).T @ point
            output.update({'depth_mm': float(z), 'distance_mm': float(np.linalg.norm(point)),
                           'point_mm': [float(x) for x in original]})
        output['candidates'] = stereo.candidates_at(ctx['cl'], ctx['cr'], u, v, record['matcher'], focal, baseline)
        half = record['matcher']['window'] // 2
        patch = ctx['gray_left'][max(0, v - half):v + half + 1, max(0, u - half):u + half + 1].astype(np.float32)
        output['texture_std'] = float(patch.std())
        return output

    def stereo_confirm(self, identifier, sid, body):
        doc = self.store.read(identifier)
        parent = self.store.find(doc, 'stereo', sid)
        ctx = self._stereo_ctx(parent)
        settings = parent['matcher']
        size = parent['rectify']['size']
        raw = body.get('confirmations') or []
        if not raw:
            raise core.Invalid('请至少提交一条确认')
        if len(raw) > 40:
            raise core.Invalid('单次最多40条确认')
        fresh = []
        for item in raw:
            conf = stereo.validate_confirmation(item, size, settings)
            conf['id'] = uuid.uuid4().hex
            fresh.append(conf)
        cumulative = copy.deepcopy(parent.get('confirmations', [])) + fresh
        shape = (size[1], size[0])
        disp, reason, cost, statuses, conflicts, applied_px = stereo.apply_confirmations(
            ctx['cl'], ctx['cr'], ctx['base_disp'], ctx['base_reason'], ctx['base_cost'], cumulative, settings, shape)
        for conf in cumulative:
            conf['status'] = statuses.get(conf['id'], 'applied')
            conf['applied_px'] = int(applied_px.get(conf['id'], 0))
        depth_vis, z = stereo.depth_visualization(disp, reason, parent['rectify']['focal_px'], parent['rig']['baseline_mm'])
        assets = dict(parent['assets'])
        assets['disp'] = self.store.asset(stereo.write_disp(disp, settings['min_disparity']))
        assets['reason'] = self.store.asset(core.png(Image.fromarray(reason)))
        assets['cost'] = self.store.asset(stereo.npy_bytes(cost))
        assets['depth_vis'] = self.store.asset(core.png(depth_vis))
        assets['reason_vis'] = self.store.asset(core.png(stereo.reason_visualization(reason)))
        record = {key: copy.deepcopy(parent[key]) for key in
                  ('left', 'right', 'left_profile', 'right_profile', 'rig', 'matcher', 'rectify')}
        record.update({'id': uuid.uuid4().hex,
                       'name': str(body.get('name') or parent['name'] + ' · 确认' + str(len(cumulative)))[:100],
                       'parent': parent['id'], 'created_revision': doc['revision'],
                       'assets': assets, 'confirmations': cumulative, 'conflicts': conflicts,
                       'stats': stereo.depth_stats(z, reason)})
        record['fingerprint'] = core.fingerprint({'base': [assets['disp_base'], assets['reason_base'], assets['cost_base']],
                                                  'confirmations': cumulative, 'rig': record['rig'], 'matcher': settings})
        def operation(current):
            current.setdefault('stereo', []).append(record)
        return self.store.mutate(identifier, body.get('revision'), operation)

    def stereo_export(self, identifier, sid, kind):
        doc = self.store.read(identifier)
        record = self.store.find(doc, 'stereo', sid)
        ctx = self._stereo_ctx(record)
        rectify = record['rectify']
        focal = rectify['focal_px']
        baseline = record['rig']['baseline_mm']
        prefix = 'stereo-' + sid[:8] + '-'
        if kind == 'depth':
            z = np.full(ctx['disp'].shape, np.nan, np.float32)
            valid = ctx['reason'] == 0
            z[valid] = focal * baseline / np.maximum(ctx['disp'][valid], 1e-6)
            return stereo.npy_bytes(z), 'application/octet-stream', prefix + 'depth-mm-float32.npy'
        if kind == 'invalid':
            mask = Image.fromarray(((ctx['reason'] != 0) * 255).astype(np.uint8))
            return core.png(mask), 'image/png', prefix + 'invalid-mask.png'
        if kind == 'cloud':
            rgb = np.asarray(self._asset_image(record['assets']['rect_left']))
            data, _, _ = stereo.ply_bytes(ctx['disp'], ctx['reason'], rectify, rgb, sid, record['fingerprint'])
            return data, 'application/octet-stream', prefix + 'cloud-mm.ply'
        if kind == 'manifest':
            manifest = {'result_id': sid, 'parent': record['parent'], 'fingerprint': record['fingerprint'],
                        'coordinate_frame': '原始左相机坐标系，单位毫米',
                        'files': {'depth': {'file': prefix + 'depth-mm-float32.npy', 'dtype': 'float32',
                                            'definition': '沿校正左相机光轴的Z（毫米）', 'invalid': 'NaN'},
                                  'invalid': {'file': prefix + 'invalid-mask.png', 'convention': '255=无效'},
                                  'cloud': {'file': prefix + 'cloud-mm.ply', 'invalid_excluded': True}},
                        'confirmations': len(record['confirmations']), 'conflicts': len(record['conflicts']),
                        'rig': record['rig'], 'matcher': record['matcher'], 'stats': record['stats'],
                        'reasons': stereo.REASONS}
            return json.dumps(manifest, ensure_ascii=False, indent=2).encode(), 'application/json', prefix + 'manifest.json'
        raise core.Invalid('未知导出类型')

    def snapshot(self, identifier, body):
        def operation(doc):
            content = copy.deepcopy({k: v for k, v in doc.items() if k != 'snapshots'})
            doc['snapshots'].append({'id': uuid.uuid4().hex, 'name': str(body.get('name', '快照'))[:100], 'content': content})
        return self.store.mutate(identifier, body.get('revision'), operation)

    def restore(self, identifier, snapshot_id, body):
        def operation(doc):
            item = self.store.find(doc, 'snapshots', snapshot_id)
            for key in ('title', 'board', 'images', 'profiles', 'outputs', 'active_profile', 'stereo'):
                if key in item['content']:
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
                doc.setdefault('stereo', [])
                for record in doc['stereo']:
                    core.validate_profile(record['left_profile'])
                    core.validate_profile(record['right_profile'])
                    stereo.validate_rig(record['rig'])
                    stereo.matcher_settings(record['matcher'])
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
