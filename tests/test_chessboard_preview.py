# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from handeye_calib.chessboard import (
    PREVIEW_HIT_PERIOD_S,
    PREVIEW_MISS_PERIOD_S,
    find_chessboard_corners,
    preview_detect_due,
)


class PreviewDetectTests(unittest.TestCase):
    def test_preview_due_is_slower_after_a_miss(self) -> None:
        now = 10.0
        self.assertTrue(preview_detect_due(0.0, False, now))
        self.assertFalse(preview_detect_due(now - 0.05, False, now))
        self.assertTrue(preview_detect_due(now - PREVIEW_MISS_PERIOD_S - 1e-6, False, now))
        self.assertTrue(preview_detect_due(now - PREVIEW_HIT_PERIOD_S - 1e-6, True, now))
        self.assertFalse(preview_detect_due(now - 0.01, True, now))

    def test_empty_preview_search_returns_none_quickly(self) -> None:
        gray = np.full((720, 1280), 80, dtype=np.uint8)
        started = time.perf_counter()
        corners, method = find_chessboard_corners(gray, (11, 8), 1.0, mode="preview")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.assertIsNone(corners)
        self.assertEqual(method, "")
        self.assertLess(elapsed_ms, 80.0)

    def test_preview_finds_small_board_in_the_corner(self) -> None:
        square = 10
        squares_x, squares_y = 12, 9
        board = np.full((squares_y * square + 40, squares_x * square + 40), 255, dtype=np.uint8)
        for row in range(squares_y):
            for col in range(squares_x):
                if (row + col) % 2 == 0:
                    y0 = 20 + row * square
                    x0 = 20 + col * square
                    board[y0 : y0 + square, x0 : x0 + square] = 0
        gray = np.full((720, 1280), 200, dtype=np.uint8)
        height, width = board.shape
        gray[720 - height - 8 : 720 - 8, 1280 - width - 8 : 1280 - 8] = board
        corners, method = find_chessboard_corners(gray, (11, 8), 1.0, mode="preview")
        self.assertIsNotNone(corners)
        self.assertEqual(method, "tile/classic")
        self.assertEqual(len(corners), 11 * 8)

    def test_unknown_mode_is_rejected(self) -> None:
        gray = np.zeros((32, 32), dtype=np.uint8)
        with self.assertRaises(ValueError):
            find_chessboard_corners(gray, (3, 3), mode="slow")


if __name__ == "__main__":
    unittest.main()
