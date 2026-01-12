import numpy as np
import cv2, base64, struct, re, json, math

MAX_EXPANSION, MAX_OVER_AVG, PADDING = 0.1, 1.3, 0.1
MAX_STROKE_NUM, STROKE_POINT_NUM, MAX_CHK_INTERVAL = 50, 2, 0.05
MIN_N_OVERLAP, PENALTY_ILLEGAL = 0.3, 0.1
FORMAT_FACTOR, MATCHING_FACTOR = 0.125, 1.0


def base64_to_image(str_base64):
    data = base64.b64decode(str_base64)
    h, w = struct.unpack('>II', data[:8])
    bits = np.unpackbits(np.frombuffer(data[8:], dtype=np.uint8), bitorder='little', count=h * w)
    image_bin = bits.reshape((h, w)).astype(np.uint8)
    return image_bin


def get_stroke_poly_seg(image, pt_st, pt_ed):
    if pt_st[0] < PADDING or pt_st[0] > 1 - PADDING or pt_st[1] < PADDING or pt_st[1] > 1 - PADDING:
        return None
    if pt_ed[0] < PADDING or pt_ed[0] > 1 - PADDING or pt_ed[1] < PADDING or pt_ed[1] > 1 - PADDING:
        return None
    h, w = image.shape
    pt_st, pt_ed = np.array(pt_st), np.array(pt_ed)

    length = np.linalg.norm(pt_ed - pt_st)
    if length == 0:
        return None

    pts = np.linspace(pt_st, pt_ed, max(int(np.ceil(np.linalg.norm(pt_ed - pt_st) / MAX_CHK_INTERVAL)) + 1, 3))
    v = (pt_ed - pt_st) / length
    n1, n2 = np.array([-v[1], v[0]]), np.array([v[1], -v[0]])

    edge_pts, widths = list(), list()
    for pt in pts:
        edge_pt_pair = list()
        for n in [n1, n2]:
            width = 0
            pos = np.array([pt[0], pt[1]])
            if image[int(pos[1] * h), int(pos[0] * w)] != 1:
                return None
            while width < min(MAX_EXPANSION, length / 2):
                pos = pos + n / math.sqrt(w ** 2 + h ** 2)
                width = width + 1 / math.sqrt(w ** 2 + h ** 2)
                if pos[0] < 0 or pos[0] >= 1 or pos[1] < 0 or pos[1] >= 1:
                    break
                if image[int(pos[1] * h), int(pos[0] * w)] == 0:
                    break
            edge_pt_pair.append(np.array([pos[0], pos[1]]))
            widths.append(width)
        edge_pts.append(edge_pt_pair)
    width_avg = sum(widths) / len(widths)
    widths = [width for width in widths if width <= width_avg * MAX_OVER_AVG]
    width_avg = sum(widths) / len(widths)
    for edge_pt_pair, pt in zip(edge_pts, pts):
        for idx, (edge_pt, n) in enumerate(zip(edge_pt_pair, [n1, n2])):
            width = np.linalg.norm(edge_pt - pt)
            edge_pt_pair[idx] = pt + min(width, width_avg * MAX_OVER_AVG) * n

    poly_pts = [pt_st - v * np.linalg.norm(edge_pts[0][1] - edge_pts[0][0]) / 2]
    poly_pts += [edge_pt_pair[0] for edge_pt_pair in edge_pts]
    poly_pts += [pt_ed + v * np.linalg.norm(edge_pts[-1][1] - edge_pts[-1][0]) / 2]
    poly_pts += [edge_pt_pair[1] for edge_pt_pair in reversed(edge_pts)]

    return [pt.tolist() for pt in poly_pts]


def reward_func_stroke(data_source, solution_str, ground_truth, extra_info=None):
    try:
        assert re.sub(r'<stroke>([^<>]*)</stroke>', '', solution_str, flags=re.DOTALL).strip() == ''
        strokes = [json.loads(stroke) for stroke in re.findall(r'<stroke>([^<>]*)</stroke>', solution_str, flags=re.DOTALL)]
        assert len(strokes) <= MAX_STROKE_NUM
        for stroke in strokes:
            assert isinstance(stroke, list)
            assert len(stroke) == STROKE_POINT_NUM
            for point in stroke:
                assert isinstance(point, list) and len(point) == 2
                assert isinstance(point[0], float) and isinstance(point[1], float)
                assert 0 <= point[0] <= 1 and 0 <= point[1] <= 1
        reward_format = 1.0
    except:
        strokes = list()
        reward_format = 0.0

    stroke_polies, illegal_path_cnt = list(), 0
    if reward_format == 1.0:
        image = base64_to_image(extra_info['base64'])
        for stroke in strokes:
            stroke_poly, illegal_stroke = list(), False
            for i in range(len(stroke) - 1):
                stroke_poly_seg = get_stroke_poly_seg(image, stroke[i], stroke[i + 1])
                if stroke_poly_seg is None:
                    illegal_path_cnt += 1
                    illegal_stroke = True
                    break
                else:
                    stroke_poly.append(stroke_poly_seg)
            if illegal_stroke == False:
                stroke_polies.append(stroke_poly)

        total = image.sum()
        h, w = image.shape
        stroke_polies, stroke_polies_preprocessed = list(), stroke_polies
        reward_matching_raw = list()
        for stroke_poly in stroke_polies_preprocessed:
            if len(stroke_poly) > 0:
                illegal_stroke = False
                for stroke_poly_seg in stroke_poly:
                    stroke_poly_seg_unnormalized = list()
                    for pt in stroke_poly_seg:
                        stroke_poly_seg_unnormalized.append([int(pt[0] * w), int(pt[1] * h)])
                    mask = np.zeros_like(image, dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(stroke_poly_seg_unnormalized, np.int32).reshape(-1, 1, 2)], 1)
                    if int((image & mask).sum()) / (int(mask.sum()) + 1e-12) < MIN_N_OVERLAP:
                        illegal_path_cnt += 1
                        illegal_stroke = True
                        break
                    rest_pre = image.sum()
                    image[mask == 1] = 0
                    rest_after = image.sum()
                if illegal_stroke == False:
                    reward_matching_raw.append((rest_pre - rest_after) / total)
                    stroke_polies.append(stroke_poly)

        if len(reward_matching_raw) > 0:
            reward_matching = sum(reward_matching_raw) * max(0, 1 - illegal_path_cnt * PENALTY_ILLEGAL)
        else:
            reward_matching = 0.0
    else:
        reward_matching_raw = list()
        reward_matching = 0.0

    reward_total = reward_format * FORMAT_FACTOR + reward_matching * MATCHING_FACTOR

    return reward_total
