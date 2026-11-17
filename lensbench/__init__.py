import io
import json
from flask import jsonify, request, render_template, send_file
from .storage import Store, Conflict
from .service import Bench
from . import core


def install(app, directory):
    store = Store(directory)
    bench = Bench(store)
    app.extensions['lensbench'] = bench

    @app.get('/')
    def lens_home():
        return render_template('lens.html')

    @app.get('/lens/api/health')
    def lens_health():
        return jsonify({'ok': True, 'product': 'LensBench', 'format': 'lensbench-1'})

    @app.errorhandler(core.Invalid)
    def lens_invalid(error):
        return jsonify({'error': str(error)}), 409 if isinstance(error, Conflict) else 400

    @app.get('/lens/api/projects')
    def lens_projects():
        return jsonify(store.listed())

    @app.post('/lens/api/projects')
    def lens_create():
        return jsonify(store.create((request.json or {}).get('title', '镜头工程')))

    @app.post('/lens/api/demo')
    def lens_demo():
        return jsonify(bench.demo())

    @app.get('/lens/api/projects/<identifier>')
    def lens_project(identifier):
        return jsonify(store.read(identifier))

    @app.post('/lens/api/projects/<identifier>/images')
    def lens_upload(identifier):
        upload = request.files.get('file')
        if not upload:
            raise core.Invalid('请选择图片')
        return jsonify(bench.add(identifier, upload.read(), upload.filename))

    @app.post('/lens/api/projects/<identifier>/board')
    def lens_board(identifier):
        return jsonify(bench.configure(identifier, request.json or {}))

    @app.post('/lens/api/projects/<identifier>/detect')
    def lens_detect(identifier):
        return jsonify(bench.detect(identifier, request.json or {}))

    @app.post('/lens/api/projects/<identifier>/images/<image_id>')
    def lens_toggle(identifier, image_id):
        return jsonify(bench.toggle(identifier, image_id, request.json or {}))

    @app.post('/lens/api/projects/<identifier>/calibrate')
    def lens_calibrate(identifier):
        return jsonify(bench.calibrate(identifier, request.json or {}))

    @app.post('/lens/api/projects/<identifier>/profile')
    def lens_profile_import(identifier):
        return jsonify(bench.import_profile(identifier, request.json or {}))

    @app.post('/lens/api/projects/<identifier>/select-profile')
    def lens_select(identifier):
        return jsonify(bench.select_profile(identifier, request.json or {}))

    @app.post('/lens/api/projects/<identifier>/process')
    def lens_process(identifier):
        return jsonify(bench.process(identifier, request.json or {}))

    @app.post('/lens/api/projects/<identifier>/stereo')
    def lens_stereo_create(identifier):
        return jsonify(bench.stereo_create(identifier, request.json or {}))

    @app.get('/lens/api/projects/<identifier>/stereo/<sid>/probe')
    def lens_stereo_probe(identifier, sid):
        return jsonify(bench.stereo_probe(identifier, sid, request.args.get('u'), request.args.get('v')))

    @app.post('/lens/api/projects/<identifier>/stereo/<sid>/confirm')
    def lens_stereo_confirm(identifier, sid):
        return jsonify(bench.stereo_confirm(identifier, sid, request.json or {}))

    @app.get('/lens/api/projects/<identifier>/stereo/<sid>/export/<kind>')
    def lens_stereo_export(identifier, sid, kind):
        data, mimetype, name = bench.stereo_export(identifier, sid, kind)
        return send_file(io.BytesIO(data), mimetype=mimetype, as_attachment=True, download_name=name)

    @app.get('/lens/api/assets/<key>')
    def lens_asset(key):
        return send_file(io.BytesIO(store.bytes(key)), mimetype='image/png')

    @app.get('/lens/api/projects/<identifier>/overlay/<image_id>')
    def lens_overlay(identifier, image_id):
        doc = store.read(identifier)
        item = store.find(doc, 'images', image_id)
        image = core.observation_overlay(bench.image(item), item['observation'])
        return send_file(io.BytesIO(core.png(image)), mimetype='image/png')

    @app.get('/lens/api/projects/<identifier>/reprojection/<profile_id>/<int:index>')
    def lens_reprojection(identifier, profile_id, index):
        profile = store.find(store.read(identifier), 'profiles', profile_id)
        if not 0 <= index < len(profile['sources']):
            raise core.Invalid('观测序号不正确')
        source = profile['sources'][index]
        image = core.reprojection_overlay(bench.image(source), source['observation'], profile['views'][index])
        return send_file(io.BytesIO(core.png(image)), mimetype='image/png')

    @app.get('/lens/api/projects/<identifier>/field/<profile_id>')
    def lens_field(identifier, profile_id):
        profile = store.find(store.read(identifier), 'profiles', profile_id)
        image, _ = core.distortion_field(profile)
        return send_file(io.BytesIO(core.png(image)), mimetype='image/png')

    @app.get('/lens/api/projects/<identifier>/profile/<profile_id>')
    def lens_profile_export(identifier, profile_id):
        profile = store.find(store.read(identifier), 'profiles', profile_id)
        return send_file(io.BytesIO(json.dumps(profile, ensure_ascii=False, indent=2).encode()),
                         mimetype='application/json', as_attachment=True, download_name='lens-profile.json')

    @app.get('/lens/api/projects/<identifier>/pattern')
    def lens_pattern(identifier):
        image = core.pattern(store.read(identifier)['board'])
        return send_file(io.BytesIO(core.png(image)), mimetype='image/png', as_attachment=True, download_name='checkerboard.png')

    @app.post('/lens/api/projects/<identifier>/snapshots')
    def lens_snapshot(identifier):
        return jsonify(bench.snapshot(identifier, request.json or {}))

    @app.post('/lens/api/projects/<identifier>/restore/<snapshot_id>')
    def lens_restore(identifier, snapshot_id):
        return jsonify(bench.restore(identifier, snapshot_id, request.json or {}))

    @app.get('/lens/api/projects/<identifier>/export')
    def lens_export(identifier):
        return send_file(io.BytesIO(bench.export(identifier)), mimetype='application/zip', as_attachment=True, download_name='lensbench-project.zip')

    @app.post('/lens/api/import')
    def lens_import():
        upload = request.files.get('file')
        if not upload:
            raise core.Invalid('请选择工程ZIP')
        return jsonify(bench.import_project(upload.read()))
