#!/usr/bin/env python3
"""Offline CPU vision smoke; run inside either policy or development image."""
import json

import cv2
import numpy as np
import torch
from PIL import Image


def main():
    # Match agent usage: RGB observation -> PNG bytes -> OpenCV BGR -> HSV mask.
    rgb = np.zeros((64, 96, 3), dtype=np.uint8)
    rgb[20:40, 30:60] = (255, 0, 0)
    bgr = cv2.cvtColor(np.asarray(Image.fromarray(rgb)), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode('.png', bgr)
    assert ok
    decoded = cv2.imdecode(np.frombuffer(encoded.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded, bgr)
    hsv = cv2.cvtColor(decoded, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 200, 200), (10, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1
    assert cv2.boundingRect(contours[0]) == (30, 20, 30, 20)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    assert count == 2 and stats[1, cv2.CC_STAT_AREA] == 600
    np.testing.assert_allclose(centroids[1], [44.5, 29.5])
    # Verify the existing NumPy/PyTorch bridge still works with this wheel.
    np.testing.assert_array_equal(torch.from_numpy(rgb).numpy(), rgb)
    print(json.dumps({'opencv': cv2.__version__, 'numpy': np.__version__,
                      'checks': ['png_bytes', 'rgb_bgr_hsv', 'morphology',
                                 'contours', 'connected_components', 'torch_numpy']}))


if __name__ == '__main__':
    main()
