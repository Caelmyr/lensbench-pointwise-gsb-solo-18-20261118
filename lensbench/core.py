import io
import json
import math
import hashlib
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw


class Invalid(ValueError):
    pass


def number(value, label, low, high):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise Invalid(label + '需要数值')
    if not math.isfinite(result) or not low <= result <= high:
        raise Invalid(label + '超出范围')
    return result


def integer(value, label, low, high):
    result = number(value, label, low, high)
    if result != int(result):
        raise Invalid(label + '需要整数')
    return int(result)


def board_settings(body):
    return {'columns': integer(body.get('columns', 8), '内角列数', 3, 18),
            'rows': integer(body.get('rows', 6), '内角行数', 3, 18),
            'square_mm': number(body.get('square_mm', 25), '方格毫米', .1, 500)}


def object_points(config):
    points = np.zeros((config['rows'] * config['columns'], 3), np.float64)
    points[:, :2] = np.mgrid[0:config['columns'], 0:config['rows']].T.reshape(-1, 2) * config['square_mm']
    return points


def decode(raw):
    if len(raw) > 24 * 1024 * 1024:
        raise Invalid('单张图片上限24MB')
    try:
        image = Image.open(io.BytesIO(raw))
        if image.format not in ('PNG', 'JPEG', 'TIFF', 'WEBP'):
            raise Invalid('支持PNG、JPEG、单页TIFF、WebP')
        if image.width * image.height > 12000000 or max(image.size) > 4096 or min(image.size) < 32:
            raise Invalid('图片边长32–4096，总像素不超过1200万')
        return ImageOps.exif_transpose(image).convert('RGB')
    except Invalid:
        raise
    except Exception as error:
        raise Invalid('不能读取图片') from error


def png(image):
    stream = io.BytesIO()
    image.save(stream, 'PNG')
    return stream.getvalue()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def detect(image, config):
    gray = np.asarray(image.convert('L'))
    found, corners = cv2.findChessboardCornersSB(gray, (config['columns'], config['rows']),
                                               flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return {'found': False, 'corners': [], 'board': config, 'size': list(image.size)}
    points = corners.reshape(-1, 2)
    hull = cv2.convexHull(points.astype(np.float32))
    area = float(cv2.contourArea(hull))
    centered = points - points.mean(axis=0)
    return {'found': True, 'corners': points.tolist(), 'board': config, 'size': list(image.size),
            'coverage': area / (image.width * image.height), 'center_px': points.mean(axis=0).tolist(),
            'spread_px': np.sqrt((centered * centered).mean(axis=0)).tolist()}


def observation_overlay(image, observation):
    canvas = np.array(image)
    if observation and observation['found']:
        b = observation['board']
        corners = np.asarray(observation['corners'], np.float32).reshape(-1, 1, 2)
        cv2.drawChessboardCorners(canvas, (b['columns'], b['rows']), corners, True)
        for index in (0, b['columns'] - 1, len(corners) - 1):
            point = tuple(np.rint(corners[index, 0]).astype(int))
            cv2.putText(canvas, str(index), point, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 70, 40), 1, cv2.LINE_AA)
    return Image.fromarray(canvas)


def quality(image):
    gray = np.asarray(image.convert('L'))
    return {'width': image.width, 'height': image.height,
            'mean': float(gray.mean()), 'contrast': float(gray.std()),
            'black_fraction': float((gray < 3).mean()), 'white_fraction': float((gray > 252).mean()),
            'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var())}


def validate_profile(body):
    size = body.get('size', [])
    if len(size) != 2:
        raise Invalid('镜头档案需要图像宽高')
    size = [integer(size[0], '宽', 32, 4096), integer(size[1], '高', 32, 4096)]
    try:
        matrix = np.asarray(body['camera_matrix'], np.float64)
        dist = np.asarray(body['distortion'], np.float64).ravel()
    except (KeyError, ValueError, TypeError):
        raise Invalid('镜头矩阵或畸变系数不正确')
    if matrix.shape != (3, 3) or dist.size not in (4, 5, 8, 12, 14):
        raise Invalid('针孔矩阵为3×3，畸变系数为4/5/8/12/14项')
    if not np.isfinite(matrix).all() or not np.isfinite(dist).all():
        raise Invalid('镜头档案不能有非有限数')
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
        raise Invalid('焦距必须为正且矩阵最后一行为0,0,1')
    if abs(matrix[0, 1]) > 1e-8 or abs(matrix[1, 0]) > 1e-8:
        raise Invalid('当前只接受无斜切针孔模型')
    return {'size': size, 'camera_matrix': matrix.tolist(), 'distortion': dist.tolist(), 'model': 'pinhole'}


def solve(observations, config):
    if len(observations) < 3 or len(observations) > 40:
        raise Invalid('需要3–40幅有效观测')
    size = observations[0]['size']
    expected = fingerprint(config)
    for observation in observations:
        if not observation['found'] or observation['size'] != size or fingerprint(observation['board']) != expected:
            raise Invalid('标定照片必须有相同图像尺寸和棋盘定义')
    objects = [object_points(config).astype(np.float32) for _ in observations]
    images = [np.asarray(o['corners'], np.float32).reshape(-1, 1, 2) for o in observations]
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(objects, images, tuple(size), None, None)
    views = []
    for index, (obj, observed, rotation, translation) in enumerate(zip(objects, images, rotations, translations)):
        projected, _ = cv2.projectPoints(obj, rotation, translation, matrix, distortion)
        residual = observed.reshape(-1, 2) - projected.reshape(-1, 2)
        errors = np.linalg.norm(residual, axis=1)
        views.append({'index': index, 'rvec': rotation.ravel().tolist(), 'tvec_mm': translation.ravel().tolist(),
                      'projected': projected.reshape(-1, 2).tolist(), 'residual': residual.tolist(),
                      'rms_px': float(np.sqrt((errors * errors).mean())), 'max_px': float(errors.max())})
    profile = validate_profile({'size': size, 'camera_matrix': matrix, 'distortion': distortion})
    return {**profile, 'rms_px': float(rms), 'board': config, 'views': views,
            'warning': '普通针孔最小二乘标定，不自动选择模型或剔除照片；低残差不代表视角充分。'}


def compatible(image, profile):
    if list(image.size) != profile['size']:
        raise Invalid('照片尺寸与镜头档案不同，不能隐式缩放内参')


def undistort(image, profile, alpha=0):
    compatible(image, profile)
    alpha = number(alpha, '保留视场', 0, 1)
    matrix = np.asarray(profile['camera_matrix'], np.float64)
    dist = np.asarray(profile['distortion'], np.float64)
    size = tuple(profile['size'])
    target, rectangle = cv2.getOptimalNewCameraMatrix(matrix, dist, size, alpha, size)
    mx, my = cv2.initUndistortRectifyMap(matrix, dist, None, target, size, cv2.CV_32FC1)
    result = cv2.remap(np.array(image), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = (mx >= 0) & (mx <= image.width - 1) & (my >= 0) & (my <= image.height - 1)
    return Image.fromarray(result), Image.fromarray(valid.astype(np.uint8) * 255), {
        'camera_matrix': target.tolist(), 'roi': [int(x) for x in rectangle], 'alpha': alpha,
        'valid_fraction': float(valid.mean()), 'source_profile': fingerprint(profile)}


def reprojection_overlay(image, observation, view):
    canvas = observation_overlay(image, observation)
    draw = ImageDraw.Draw(canvas)
    for measured, projected in zip(observation['corners'], view['projected']):
        x, y = projected
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(255, 30, 30), width=2)
        draw.line([tuple(measured), (x, y)], fill=(255, 210, 30), width=2)
    return canvas


def pose(image, observation, profile):
    compatible(image, profile)
    if not observation or not observation['found']:
        raise Invalid('先检测此图的棋盘')
    obj = object_points(observation['board'])
    measured = np.asarray(observation['corners'], np.float64)
    matrix = np.asarray(profile['camera_matrix'], np.float64)
    distortion = np.asarray(profile['distortion'], np.float64)
    success, rotation, translation = cv2.solvePnP(obj, measured, matrix, distortion)
    if not success or translation[2, 0] <= 0:
        raise Invalid('没有有效的正向棋盘姿态')
    axis = np.array([[0, 0, 0], [75, 0, 0], [0, 75, 0], [0, 0, -75]], np.float64)
    projected, _ = cv2.projectPoints(axis, rotation, translation, matrix, distortion)
    canvas = np.array(image)
    p = np.rint(projected.reshape(-1, 2)).astype(int)
    for index, color in enumerate([(255, 30, 30), (30, 255, 30), (30, 100, 255)], 1):
        cv2.line(canvas, tuple(p[0]), tuple(p[index]), color, 3, cv2.LINE_AA)
    return Image.fromarray(canvas), {'rvec': rotation.ravel().tolist(), 'tvec_mm': translation.ravel().tolist(),
                                    'axis_mm': 75, 'board': observation['board'], 'profile': fingerprint(profile)}


def planar_view(image, observation, profile, ppm=2):
    compatible(image, profile)
    if not observation or not observation['found']:
        raise Invalid('平面展开需要棋盘观测')
    ppm = number(ppm, '每毫米像素', .1, 5)
    matrix = np.asarray(profile['camera_matrix'], np.float64)
    distortion = np.asarray(profile['distortion'], np.float64)
    points = cv2.undistortPoints(np.asarray(observation['corners'], np.float64).reshape(-1, 1, 2), matrix, distortion, P=matrix)
    real = object_points(observation['board'])[:, :2]
    margin = 20
    dest = real * ppm + margin
    transform, inliers = cv2.findHomography(points.reshape(-1, 2), dest, cv2.RANSAC, 2)
    if transform is None:
        raise Invalid('无法求解棋盘平面')
    width, height = np.ceil(real.max(axis=0) * ppm + margin * 2).astype(int)
    if width * height > 12000000 or max(width, height) > 4096:
        raise Invalid('展开尺寸超出限制')
    corrected = cv2.undistort(np.array(image), matrix, distortion)
    result = cv2.warpPerspective(corrected, transform, (int(width), int(height)))
    return Image.fromarray(result), {'homography': transform.tolist(), 'pixel_per_mm': ppm,
                                    'origin_margin_px': margin, 'inlier_count': int(inliers.sum()),
                                    'note': '仅棋盘所在平面有效，不测量离开平面的物体。'}


def pattern(config, cell=48):
    cell = integer(cell, '方格像素', 16, 120)
    width, height = (config['columns'] + 1) * cell, (config['rows'] + 1) * cell
    canvas = Image.new('RGB', (width + 80, height + 100), 'white')
    draw = ImageDraw.Draw(canvas)
    for row in range(config['rows'] + 1):
        for column in range(config['columns'] + 1):
            if (column + row) % 2 == 0:
                x, y = 40 + column * cell, 40 + row * cell
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill='black')
    draw.text((15, height + 55), 'LensBench checkerboard / print at known square size', fill='black')
    return canvas


def distortion_field(profile):
    w, h = profile['size']
    matrix = np.asarray(profile['camera_matrix'], np.float64)
    distortion = np.asarray(profile['distortion'], np.float64)
    ys, xs = np.mgrid[20:h:40, 20:w:40]
    points = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    corrected = cv2.undistortPoints(points, matrix, distortion, P=matrix).reshape(-1, 2)
    canvas = Image.new('RGB', (w, h), (25, 32, 43))
    draw = ImageDraw.Draw(canvas)
    for original, target in zip(points.reshape(-1, 2), corrected):
        draw.line([tuple(original), tuple(target)], fill=(70, 210, 220), width=2)
        x, y = target
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 190, 80))
    displacement = np.linalg.norm(corrected - points.reshape(-1, 2), axis=1)
    return canvas, {'max_displacement_px': float(displacement.max()), 'mean_displacement_px': float(displacement.mean())}


def demo_images():
    config = board_settings({})
    source = pattern(config, 60).crop((40, 40, 580, 460))
    array = np.array(source)
    w, h = source.size
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    targets = [
        [[100, 85], [545, 100], [550, 425], [95, 440]],
        [[140, 65], [540, 115], [510, 430], [75, 365]],
        [[90, 110], [520, 45], [575, 390], [150, 435]],
        [[180, 95], [535, 150], [500, 425], [105, 350]],
        [[75, 60], [480, 95], [555, 375], [125, 450]],
        [[145, 100], [570, 65], [510, 420], [70, 360]],
    ]
    output = []
    for target in targets:
        transform = cv2.getPerspectiveTransform(corners, np.asarray(target, np.float32))
        result = cv2.warpPerspective(array, transform, (640, 480), borderValue=(155, 155, 155))
        output.append(Image.fromarray(result))
    return config, output
