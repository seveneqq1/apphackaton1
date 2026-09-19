"""Capture chessboard images and save OpenCV camera calibration parameters."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the camera with a printed chessboard.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--corners-x", type=int, default=9, help="Inner corners across the chessboard")
    parser.add_argument("--corners-y", type=int, default=6, help="Inner corners down the chessboard")
    parser.add_argument("--square-mm", type=float, default=25.0, help="One chessboard square side in mm")
    parser.add_argument("--frames", type=int, default=20, help="Number of captures to use")
    parser.add_argument("--output", type=Path, default=Path("camera_calibration.npz"))
    args = parser.parse_args()

    if args.corners_x < 2 or args.corners_y < 2 or args.square_mm <= 0 or args.frames < 5:
        parser.error("Invalid chessboard dimensions, square size, or frame count")

    pattern_size = (args.corners_x, args.corners_y)
    object_points_template = np.zeros((args.corners_x * args.corners_y, 3), np.float32)
    object_points_template[:, :2] = np.mgrid[0 : args.corners_x, 0 : args.corners_y].T.reshape(-1, 2)
    object_points_template *= args.square_mm

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    last_capture_time = 0.0

    print("Move the chessboard around the frame. Press SPACE to capture a sharp detection; Q to cancel.")
    try:
        while len(object_points) < args.frames:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read a camera frame")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray,
                pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            preview = frame.copy()
            if found:
                refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                cv2.drawChessboardCorners(preview, pattern_size, refined, found)
            else:
                refined = None

            cv2.rectangle(preview, (0, 0), (preview.shape[1], 70), (20, 20, 20), -1)
            cv2.putText(preview, f"Captures: {len(object_points)}/{args.frames}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(preview, "SPACE capture  |  Q cancel", (15, 57),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 220, 255), 1, cv2.LINE_AA)
            cv2.imshow("ErgoNeck camera calibration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Calibration cancelled.")
                return
            if key == ord(" ") and refined is not None:
                object_points.append(object_points_template.copy())
                image_points.append(refined)
                image_size = (gray.shape[1], gray.shape[0])
                print(f"Captured {len(object_points)}/{args.frames}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    assert image_size is not None
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    np.savez(
        args.output,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        image_width=image_size[0],
        image_height=image_size[1],
        rms_error=rms,
        captures=len(object_points),
        corners_x=args.corners_x,
        corners_y=args.corners_y,
        square_mm=args.square_mm,
    )
    print(f"Saved calibration to {args.output.resolve()} (RMS reprojection error: {rms:.3f}px)")


if __name__ == "__main__":
    main()
