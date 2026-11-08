"""Explain observed evidence, without inferring unobserved physical causes."""
import math


def describe_event(event):
    if event.get('kind') == 'motion_supported_pair_contact':
        return {'type': 'motion_supported_contact', 'title': 'Nghi va chạm giữa hai xe',
                'overlay': 'NGHI VA CHAM HAI XE',
                'evidence': ['Hai ID được quan sát tách nhau rồi tiến sát nhau.',
                             'Mô hình xác minh giảm chuyển động mạnh của một xe qua nhiều cửa sổ; xe còn lại đồng thời giảm chuyển động và đổi hướng.'],
                'limitation': 'Kết luận dựa trên chuyển động và tiếp xúc khung ảnh; bộ phân loại cặp xe chưa xác nhận. Chưa xác định bên gây va chạm hoặc nguyên nhân điều khiển xe.'}
    if event.get('kind') != 'single_vehicle_incident':
        return {'type': 'pair_contact', 'title': 'Nghi va chạm giữa hai xe',
                'overlay': 'NGHI VA CHAM HAI XE',
                'evidence': ['Hai ID được quan sát tách nhau trước khi khung xe tiến sát hoặc chồng lấn.',
                             'Cặp xe vượt ngưỡng xác minh chuyển động và hình ảnh.'],
                'limitation': 'Chưa xác định bên gây va chạm hoặc nguyên nhân điều khiển xe.'}
    motion = event.get('motion') or {}
    def value(key):
        raw = motion.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw):
            return float(raw)
        return None
    before, after = value('speed_before'), value('speed_after')
    angle, lateral, shape = (value(k) for k in ('course_change_degrees', 'lateral_change', 'shape_change'))
    evidence = []
    # Reuse the observable gates already used by the single-vehicle branch.
    # These describe appearance/motion, not brake failure, lane departure or blame.
    if angle is not None and shape is not None and angle >= 60 and shape >= .4:
        kind, title, overlay = 'strong_turn', 'Nghi xoay hướng mạnh', 'NGHI XOAY HUONG MANH'
        evidence.append(f'Hướng chuyển động trên ảnh thay đổi khoảng {angle:.0f}°; hình dạng khung xe cũng thay đổi rõ.')
    elif before is not None and after is not None and before > .6 and 0 <= after < .3 * before:
        kind, title, overlay = 'abrupt_slowdown', 'Nghi giảm chuyển động đột ngột', 'NGHI GIAM CHUYEN DONG DOT NGOT'
        evidence.append(f'Mức chuyển động trên ảnh sau sự kiện còn khoảng {100 * after / before:.0f}% so với trước đó.')
    elif all(v is not None for v in (before, lateral, shape)) and before > .15 and lateral > .15 and shape > .18:
        kind, title, overlay = 'trajectory_change', 'Nghi thay đổi quỹ đạo', 'NGHI THAY DOI QUY DAO'
        evidence.append('Quỹ đạo ngang và hình dạng khung xe thay đổi trong chuỗi quan sát.')
        if angle is not None:
            evidence.append(f'Hướng chuyển động trên ảnh thay đổi khoảng {angle:.0f}°.')
    else:
        kind, title, overlay = 'unspecified', 'Nghi sự cố phương tiện — chưa rõ dạng', 'NGHI SU CO PHUONG TIEN'
        evidence.append('Mô hình phát hiện bất thường nhưng chưa đủ đặc trưng để mô tả dạng sự cố.')
    anchors = set(event.get('support_window_anchor_frames', []))
    if len(anchors) >= 2:
        evidence.append('Dấu hiệu bất thường có hỗ trợ từ nhiều cửa sổ thời gian.')
    return {'type': kind, 'title': title, 'overlay': overlay, 'evidence': evidence,
            'limitation': 'Đây là dấu hiệu chuyển động trên ảnh, không phải kết luận nguyên nhân như mất lái, nổ lốp hoặc đâm vật cản. Số ID được đánh dấu không xác định tổng số xe thực tế liên quan.'}
