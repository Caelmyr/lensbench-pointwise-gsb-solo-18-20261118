import io
import math
import cv2
import numpy as np
from PIL import Image
from . import core

# 无效原因编码：留空而不是猜，前端图例与此一一对应
REASONS = {0: '有效', 1: '校正有效区外', 2: '纹理不足', 3: '多候选歧义', 4: '遮挡或左右不一致',
           5: '确认后无邻近候选', 6: '视差超出范围'}
REASON_KEYS = {0: 'valid', 1: 'area', 2: 'texture', 3: 'ambiguous', 4: 'occlusion',
               5: 'nocandidate', 6: 'range'}
PALETTE = np.array([[34, 40, 48], [0, 0, 0], [72, 116, 214], [238, 168, 55],
                    [226, 82, 65], [168, 86, 196], [62, 196, 214]], np.uint8)
MAXCOST = 60000
LR_TOL = 1            # 左右一致性整数视差容差
CONFIRM_TAU = 2.0     # 人工确认视差的邻近容差（像素）
CONFIRM_MARGIN = 0.35 # 邻近候选代价相对最优的放宽比例
CONFLICT_DISP = 2.5   # 两条确认视差差超过该值且区域相交即矛盾
SLAB = 96


def validate_rig(body):
    rotation = body.get('rotation')
    rvec = body.get('rvec')
    if rotation is not None:
        try:
            matrix = np.asarray(rotation, np.float64)
        except (TypeError, ValueError):
            raise core.Invalid('旋转矩阵需要数值')
        if matrix.size != 9:
            raise core.Invalid('旋转矩阵需要9个数（行优先）')
        matrix = matrix.reshape(3, 3)
    elif rvec is not None:
        try:
            vector = np.asarray(rvec, np.float64).ravel()
        except (TypeError, ValueError):
            raise core.Invalid('旋转向量需要数值')
        if vector.size != 3:
            raise core.Invalid('旋转向量需要3个数')
        matrix, _ = cv2.Rodrigues(vector)
    else:
        raise core.Invalid('需要左→右旋转（矩阵9数或旋转向量3数）')
    if not np.isfinite(matrix).all():
        raise core.Invalid('旋转不能有非有限数')
    if np.abs(matrix.T @ matrix - np.eye(3)).max() > 1e-3 or abs(np.linalg.det(matrix) - 1) > 1e-3:
        raise core.Invalid('旋转矩阵不正交：需满足 RᵀR≈I 且 det≈1')
    try:
        translation = np.asarray(body.get('translation_mm'), np.float64).ravel()
    except (TypeError, ValueError):
        raise core.Invalid('平移需要数值')
    if translation.size != 3 or not np.isfinite(translation).all():
        raise core.Invalid('平移需要3个毫米数值')
    baseline = float(np.linalg.norm(translation))
    if not 1 <= baseline <= 20000:
        raise core.Invalid('基线长度需在1–20000毫米')
    return {'rotation': matrix.tolist(), 'translation_mm': translation.tolist(),
            'baseline_mm': baseline, 'convention': 'X_right = R · X_left + t（毫米）'}


def matcher_settings(body):
    settings = {'min_disparity': core.integer(body.get('min_disparity', 0), '最小视差', -128, 256),
                'num_disparities': core.integer(body.get('num_disparities', 128), '视差范围', 16, 512),
                'window': core.integer(body.get('window', 9), '聚合窗口', 5, 21),
                'uniqueness': core.number(body.get('uniqueness', 0.10), '唯一性阈值', 0.01, 0.9),
                'texture': core.number(body.get('texture', 3.0), '纹理阈值', 0.5, 60)}
    if settings['window'] % 2 == 0:
        raise core.Invalid('聚合窗口需为奇数')
    return settings


def rectify_pair(left, right, left_profile, right_profile, rig):
    core.compatible(left, left_profile)
    core.compatible(right, right_profile)
    if left.size != right.size:
        raise core.Invalid('双目两图宽高必须一致（当前按同一校正尺寸处理）')
    k1 = np.asarray(left_profile['camera_matrix'], np.float64)
    d1 = np.asarray(left_profile['distortion'], np.float64)
    k2 = np.asarray(right_profile['camera_matrix'], np.float64)
    d2 = np.asarray(right_profile['distortion'], np.float64)
    rotation = np.asarray(rig['rotation'], np.float64)
    translation = np.asarray(rig['translation_mm'], np.float64).reshape(3, 1)
    size = left.size
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(k1, d1, k2, d2, size, rotation, translation,
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    output = {'R1': r1.tolist(), 'R2': r2.tolist(), 'P1': p1.tolist(), 'P2': p2.tolist(),
              'Q': q.tolist(), 'size': list(size), 'focal_px': float(p1[0, 0]),
              'baseline_mm': rig['baseline_mm']}
    result = {}
    for tag, image, k, dist, r, p in (('left', left, k1, d1, r1, p1), ('right', right, k2, d2, r2, p2)):
        mx, my = cv2.initUndistortRectifyMap(k, dist, r, p, size, cv2.CV_32FC1)
        remapped = cv2.remap(np.array(image), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        valid = (mx >= 0) & (mx <= image.width - 1) & (my >= 0) & (my <= image.height - 1)
        result[tag] = Image.fromarray(remapped)
        result['valid_' + tag] = valid
        output['valid_fraction_' + tag] = float(valid.mean())
    return output, result


def census(gray):
    pad = np.pad(gray, 2, mode='edge')
    center = pad[2:-2, 2:-2]
    desc = np.zeros(gray.shape, np.uint32)
    bit = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            neighbor = pad[2 + dy:gray.shape[0] + 2 + dy, 2 + dx:gray.shape[1] + 2 + dx]
            desc |= (neighbor < center).astype(np.uint32) << bit
            bit += 1
    return desc


_POP = np.array([bin(i).count('1') for i in range(256)], np.uint8)


def _hamming(xor):
    return _POP[xor & 255] + _POP[(xor >> 8) & 255] + _POP[(xor >> 16) & 255]


def _cost_slice(cl, cr, d, window):
    # 左图 x 列与右图 x-d 列配对的 Census 汉明代价，盒滤波聚合
    _, w = cl.shape
    cost = np.full(cl.shape, MAXCOST, np.uint16)
    if d >= 0:
        if d >= w:
            return cost
        xor = cl[:, d:] ^ cr[:, :w - d]
        cost[:, d:] = cv2.boxFilter(_hamming(xor).astype(np.uint16), -1, (window, window), normalize=False)
    else:
        if -d >= w:
            return cost
        xor = cl[:, :w + d] ^ cr[:, -d:]
        cost[:, :w + d] = cv2.boxFilter(_hamming(xor).astype(np.uint16), -1, (window, window), normalize=False)
    return cost


def match(cl, cr, settings):
    # 分行带计算，只保留每像素最优/次优与右图最优，不保存完整代价体
    height, width = cl.shape
    dmin = settings['min_disparity']
    window = settings['window']
    margin = window // 2 + 2
    best_c = np.full((height, width), MAXCOST, np.uint16)
    best_d = np.zeros((height, width), np.int16)
    sec_c = np.full((height, width), MAXCOST, np.uint16)
    c_minus = np.zeros((height, width), np.uint16)
    c_plus = np.zeros((height, width), np.uint16)
    rbest_c = np.full((height, width), MAXCOST, np.uint16)
    rbest_d = np.zeros((height, width), np.int16)
    for r0 in range(0, height, SLAB):
        r1 = min(height, r0 + SLAB)
        a, b = max(0, r0 - margin), min(height, r1 + margin)
        cls, crs = cl[a:b], cr[a:b]
        rows = (r1 - r0, width)
        s_b1c = np.full(rows, MAXCOST, np.uint16)
        s_b1d = np.zeros(rows, np.int16)
        s_b2c = np.full(rows, MAXCOST, np.uint16)
        s_cm = np.zeros(rows, np.uint16)
        s_cp = np.zeros(rows, np.uint16)
        s_rbc = np.full(rows, MAXCOST, np.uint16)
        s_rbd = np.zeros(rows, np.int16)
        prev = None
        for d in range(dmin, dmin + settings['num_disparities']):
            c = _cost_slice(cls, crs, d, window)[r0 - a:r0 - a + rows[0]]
            plus = s_b1d == (d - 1)
            s_cp[plus] = c[plus]
            lower = c < s_b1c
            s_cm[lower] = prev[lower] if prev is not None else c[lower]
            promote = lower & (np.abs(d - s_b1d) > 2) & (s_b1c < s_b2c)
            s_b2c[promote] = s_b1c[promote]
            s_b1c[lower] = c[lower]
            s_b1d[lower] = d
            notbest = (~lower) & (np.abs(d - s_b1d) > 2) & (c < s_b2c)
            s_b2c[notbest] = c[notbest]
            rc = np.full_like(c, MAXCOST)
            if d >= 0:
                if d < width:
                    rc[:, :width - d] = c[:, d:]
            elif -d < width:
                rc[:, -d:] = c[:, :width + d]
            rlower = rc < s_rbc
            s_rbc[rlower] = rc[rlower]
            s_rbd[rlower] = d
            prev = c
        best_c[r0:r1] = s_b1c
        best_d[r0:r1] = s_b1d
        sec_c[r0:r1] = s_b2c
        c_minus[r0:r1] = s_cm
        c_plus[r0:r1] = s_cp
        rbest_c[r0:r1] = s_rbc
        rbest_d[r0:r1] = s_rbd
    return {'best_c': best_c, 'best_d': best_d, 'sec_c': sec_c,
            'c_minus': c_minus, 'c_plus': c_plus, 'rbest_d': rbest_d}


def finalize(volume, gray_left, valid_left, valid_right, settings):
    height, width = gray_left.shape
    best_c = volume['best_c'].astype(np.float32)
    best_d = volume['best_d']
    denom = volume['c_minus'].astype(np.float32) - 2 * best_c + volume['c_plus'].astype(np.float32)
    offset = np.zeros((height, width), np.float32)
    ok = denom > 0
    offset[ok] = 0.5 * (volume['c_minus'][ok].astype(np.float32) - volume['c_plus'][ok].astype(np.float32)) / denom[ok]
    np.clip(offset, -1, 1, out=offset)
    disp = best_d.astype(np.float32) + offset
    reason = np.zeros((height, width), np.uint8)
    reason[~valid_left] = 1
    gray = gray_left.astype(np.float32)
    window = settings['window']
    mean = cv2.boxFilter(gray, -1, (window, window), normalize=True)
    square = cv2.boxFilter(gray * gray, -1, (window, window), normalize=True)
    std = np.sqrt(np.maximum(square - mean * mean, 0))
    reason[(reason == 0) & (std < settings['texture'])] = 2
    ys, xs = np.mgrid[0:height, 0:width]
    xr = xs - best_d.astype(np.int32)
    inside = (xr >= 0) & (xr < width)
    clipped = np.clip(xr, 0, width - 1)
    reason[(reason == 0) & ~(valid_right[ys, clipped] & inside)] = 1
    edge = ((best_d <= settings['min_disparity'])
            | (best_d >= settings['min_disparity'] + settings['num_disparities'] - 1)
            | (volume['best_c'] >= MAXCOST))
    reason[(reason == 0) & edge] = 6
    reason[(reason == 0) & (disp <= 0.2)] = 6
    sec = volume['sec_c'].astype(np.float32)
    uniqueness = (sec - best_c) / (sec + best_c + 1e-6)
    reason[(reason == 0) & (uniqueness < settings['uniqueness'])] = 3
    rd = volume['rbest_d'][ys, clipped].astype(np.int32)
    mismatch = (~inside) | (np.abs(rd - best_d.astype(np.int32)) > LR_TOL)
    reason[(reason == 0) & mismatch] = 4
    return disp, reason


def depth_visualization(disp, reason, focal, baseline):
    z = np.full(disp.shape, np.nan, np.float32)
    valid = reason == 0
    z[valid] = focal * baseline / np.maximum(disp[valid], 1e-6)
    vis = np.full((*disp.shape, 3), (26, 31, 38), np.uint8)
    if valid.any():
        lo, hi = np.percentile(z[valid], [2, 98])
        hi = max(hi, lo + 1e-6)
        norm = np.clip((z - lo) / (hi - lo), 0, 1)
        mapped = np.zeros(disp.shape, np.uint8)
        mapped[valid] = ((1 - norm[valid]) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(mapped, cv2.COLORMAP_TURBO)[:, :, ::-1]
        vis[valid] = colored[valid]
    return Image.fromarray(vis), z


def reason_visualization(reason):
    return Image.fromarray(PALETTE[reason])


def depth_stats(z, reason):
    valid = reason == 0
    stats = {'valid_px': int(valid.sum()), 'total_px': int(reason.size),
             'valid_fraction': float(valid.mean()),
             'reasons': {REASON_KEYS[r]: int((reason == r).sum()) for r in REASON_KEYS},
             'depth_mm': None}
    if valid.any():
        stats['depth_mm'] = {'min': float(z[valid].min()), 'p50': float(np.median(z[valid])),
                             'max': float(z[valid].max())}
    return stats


def compute(left, right, left_profile, right_profile, rig, settings):
    rectify, images = rectify_pair(left, right, left_profile, right_profile, rig)
    gray_left = np.asarray(images['left'].convert('L'))
    gray_right = np.asarray(images['right'].convert('L'))
    volume = match(census(gray_left), census(gray_right), settings)
    disp, reason = finalize(volume, gray_left, images['valid_left'], images['valid_right'], settings)
    depth_vis, z = depth_visualization(disp, reason, rectify['focal_px'], rig['baseline_mm'])
    return {'rectify': rectify, 'images': images, 'disp': disp, 'reason': reason,
            'best_c': volume['best_c'], 'depth_vis': depth_vis,
            'reason_vis': reason_visualization(reason), 'z': z,
            'stats': depth_stats(z, reason)}


def candidates_at(cl, cr, u, v, settings, focal, baseline, limit=4):
    height, width = cl.shape
    half = settings['window'] // 2
    dmin = settings['min_disparity']
    count = settings['num_disparities']
    dy0 = max(-half, -v)
    dy1 = min(half + 1, height - v)
    curve = np.full(count, MAXCOST, np.float64)
    for i in range(count):
        xr = u - (dmin + i)
        if xr < 0 or xr >= width:
            continue
        dx0 = max(-half, -u, -xr)
        dx1 = min(half + 1, width - u, width - xr)
        a = cl[v + dy0:v + dy1, u + dx0:u + dx1]
        b = cr[v + dy0:v + dy1, xr + dx0:xr + dx1]
        curve[i] = _hamming(a ^ b).sum()
    best = curve.min()
    if best >= MAXCOST:
        return []
    picked = []
    for i in range(1, count - 1):
        if curve[i] < curve[i - 1] and curve[i] <= curve[i + 1] and curve[i] <= best * (1 + CONFIRM_MARGIN) + 2:
            picked.append((curve[i], dmin + i))
    picked.sort()
    output = []
    for cost, d in picked:
        if any(abs(d - p[1]) < 3 for p in output):
            continue
        i = d - dmin
        denom = curve[i - 1] - 2 * curve[i] + curve[i + 1]
        offset = float(np.clip(0.5 * (curve[i - 1] - curve[i + 1]) / denom, -1, 1)) if denom > 0 else 0.0
        sub = d + offset
        output.append((cost, d, sub))
        if len(output) >= limit:
            break
    return [{'cost': float(cost), 'disparity': round(float(sub), 3),
             'depth_mm': (float(focal * baseline / sub) if sub > 0.2 else None)} for cost, _, sub in output]


def validate_confirmation(body, size, settings):
    width, height = size
    pixel = body.get('pixel')
    if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
        raise core.Invalid('确认需要像素坐标')
    u = core.integer(pixel[0], '确认点列', 0, width - 1)
    v = core.integer(pixel[1], '确认点行', 0, height - 1)
    disparity = core.number(body.get('disparity'), '确认视差', settings['min_disparity'],
                            settings['min_disparity'] + settings['num_disparities'])
    polygon = body.get('polygon')
    if not isinstance(polygon, list) or not 3 <= len(polygon) <= 96:
        raise core.Invalid('圈选多边形需要3–96个顶点')
    points = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise core.Invalid('多边形顶点格式不正确')
        points.append([core.number(point[0], '顶点列', -2 * width, 3 * width),
                       core.number(point[1], '顶点行', -2 * height, 3 * height)])
    area = float(cv2.contourArea(np.asarray(points, np.float32)))
    if area < 25:
        raise core.Invalid('圈选区域过小')
    if area > 0.7 * width * height:
        raise core.Invalid('圈选区域不能超过画面70%')
    return {'pixel': [u, v], 'disparity': disparity, 'polygon': points}


def confirmation_region(cl, cr, conf, settings, base_cost, shape):
    # 确认视差邻近候选 ∩ 圈选多边形，再取与确认点连通的成分：不会跨过前景边缘改到别的物体
    height, width = shape
    window = settings['window']
    dmin = settings['min_disparity']
    dstar = conf['disparity']
    u, v = conf['pixel']
    poly = np.asarray(conf['polygon'], np.int32)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    mask[v, u] = 1
    x0 = max(0, int(poly[:, 0].min()))
    x1 = min(width - 1, int(poly[:, 0].max()))
    y0 = max(0, int(poly[:, 1].min()))
    y1 = min(height - 1, int(poly[:, 1].max()))
    lo = max(dmin, int(math.floor(dstar - CONFIRM_TAU - 1)))
    hi = min(dmin + settings['num_disparities'] - 1, int(math.ceil(dstar + CONFIRM_TAU + 1)))
    ds = list(range(lo, hi + 1))
    margin = window // 2 + 2
    a = max(0, y0 - margin)
    b = min(height, y1 + margin + 1)
    rows = slice(y0 - a, y0 - a + (y1 - y0 + 1))
    volume = np.stack([_cost_slice(cl[a:b], cr[a:b], d, window)[rows, x0:x1 + 1] for d in ds])
    near = [i for i, d in enumerate(ds) if abs(d - dstar) <= CONFIRM_TAU]
    sub = volume[near]
    near_min = sub.min(axis=0).astype(np.float32)
    base = base_cost[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    eligible = (near_min <= base * (1 + CONFIRM_MARGIN) + 2) & (mask[y0:y1 + 1, x0:x1 + 1] > 0)
    sx, sy = u - x0, v - y0
    eligible[sy, sx] = True
    _, labels = cv2.connectedComponents(eligible.astype(np.uint8), connectivity=8)
    seed = labels[sy, sx]
    if seed == 0:
        return None
    region = labels == seed
    if region.sum() < 4:
        return None
    index = np.argmin(sub, axis=0)
    chosen = np.asarray([ds[near[i]] for i in range(len(near))], np.int32)[index]
    rel = chosen - lo
    ys, xs = np.mgrid[0:region.shape[0], 0:region.shape[1]]
    c0 = volume[rel, ys, xs].astype(np.float32)
    cm = volume[np.clip(rel - 1, 0, len(ds) - 1), ys, xs].astype(np.float32)
    cp = volume[np.clip(rel + 1, 0, len(ds) - 1), ys, xs].astype(np.float32)
    denom = cm - 2 * c0 + cp
    offset = np.where(denom > 0, 0.5 * (cm - cp) / np.maximum(denom, 1e-6), 0).astype(np.float32)
    offset = np.clip(offset, -1, 1)
    offset[(rel == 0) | (rel == len(ds) - 1)] = 0
    return {'bbox': (x0, y0, x1, y1), 'mask': region,
            'disp': chosen.astype(np.float32) + offset, 'cost': c0.astype(np.uint16)}


def _region_overlap(first, second):
    ax0, ay0, ax1, ay1 = first['bbox']
    bx0, by0, bx1, by1 = second['bbox']
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x1 < x0 or y1 < y0:
        return 0
    a = first['mask'][y0 - ay0:y1 - ay0 + 1, x0 - ax0:x1 - ax0 + 1]
    b = second['mask'][y0 - by0:y1 - by0 + 1, x0 - bx0:x1 - bx0 + 1]
    return int((a & b).sum())


def apply_confirmations(cl, cr, base_disp, base_reason, base_cost, confirmations, settings, shape):
    disp = base_disp.copy()
    reason = base_reason.copy()
    cost = base_cost.copy()
    prepared = [(conf, confirmation_region(cl, cr, conf, settings, base_cost, shape)) for conf in confirmations]
    statuses = {}
    conflicts = []
    accepted = []
    for conf, region in prepared:
        if region is None:
            statuses[conf['id']] = 'empty'
            continue
        clash = False
        for other_conf, other_region in accepted:
            if abs(other_conf['disparity'] - conf['disparity']) <= CONFLICT_DISP:
                continue
            overlap = _region_overlap(other_region, region)
            if overlap > 0:
                conflicts.append({'a': other_conf['id'], 'b': conf['id'], 'overlap_px': overlap})
                clash = True
        if clash:
            statuses[conf['id']] = 'conflict'
            continue
        accepted.append((conf, region))
        statuses[conf['id']] = 'applied'
    applied_px = {}
    for conf, region in accepted:
        x0, y0, x1, y1 = region['bbox']
        mask = region['mask']
        sub_disp = disp[y0:y1 + 1, x0:x1 + 1]
        sub_disp[mask] = region['disp'][mask]
        sub_cost = cost[y0:y1 + 1, x0:x1 + 1]
        sub_cost[mask] = region['cost'][mask]
        sub_reason = reason[y0:y1 + 1, x0:x1 + 1]
        sub_reason[mask] = np.where(region['disp'][mask] > 0.2, 0, 6).astype(np.uint8)
        applied_px[conf['id']] = int(mask.sum())
    return disp, reason, cost, statuses, conflicts, applied_px


def npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def read_npy(raw):
    return np.load(io.BytesIO(raw), allow_pickle=False)


def write_disp(disp, dmin):
    return npy_bytes(np.round((disp - dmin) * 64).clip(0, 65535).astype(np.uint16))


def read_disp(raw, dmin):
    return read_npy(raw).astype(np.float32) / 64 + dmin


def ply_bytes(disp, reason, rectify, rgb, result_id, fingerprint):
    valid = reason == 0
    ys, xs = np.nonzero(valid)
    d = disp[valid].astype(np.float64)
    focal = rectify['focal_px']
    baseline = rectify['baseline_mm']
    p1 = np.asarray(rectify['P1'], np.float64)
    z = focal * baseline / d
    x = (xs - p1[0, 2]) * z / focal
    y = (ys - p1[1, 2]) * z / p1[1, 1]
    rotation = np.asarray(rectify['R1'], np.float64)
    points = rotation.T @ np.vstack([x, y, z])  # 回到原始左相机坐标系
    colors = rgb[valid]
    count = points.shape[1]
    note = 'full'
    if count > 8000000:
        index = np.linspace(0, points.shape[1] - 1, 8000000).astype(np.int64)
        points = points[:, index]
        colors = colors[index]
        count = points.shape[1]
        note = 'subsampled-8m'
    vertex = np.empty(count, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex['x'] = points[0]
    vertex['y'] = points[1]
    vertex['z'] = points[2]
    vertex['red'] = colors[:, 0]
    vertex['green'] = colors[:, 1]
    vertex['blue'] = colors[:, 2]
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment LensBench stereo result ' + result_id + '\n'
              'comment confirmation fingerprint ' + fingerprint + '\n'
              'comment frame: original left camera, millimeters\n'
              'comment sampling: ' + note + '\n'
              'element vertex ' + str(count) + '\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    return header.encode() + vertex.tobytes(), count, note
