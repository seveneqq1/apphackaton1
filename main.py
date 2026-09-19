"""ErgoNeck: four-black-dot posture reminder prototype for a laptop camera."""

from __future__ import annotations

import argparse
import itertools
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Settings:
    dots_width_mm: float = 135.0
    dots_height_mm: float = 45.0
    dot_threshold: int = 75
    dot_min_area_px: float = 80.0
    dot_max_area_px: float = 5000.0
    forward_axis: int = 0  # x axis is normally the head nod axis for a front-facing camera.
    forward_sign: float = 1.0  # Change to -1 if looking down displays a negative angle.
    warning_angle: float = 15.0
    clear_angle: float = 10.0
    warning_seconds: float = 3.0
    clear_seconds: float = 2.0
    calibration_seconds: float = 2.0
    smoothing_samples: int = 5
    max_calibration_motion: float = 4.0


SETTINGS = Settings()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


FONT_SMALL = font(22)
FONT_MEDIUM = font(28)
FONT_LARGE = font(48)


def draw_text(frame: np.ndarray, text: str, position: tuple[int, int], fill: tuple[int, int, int], text_font) -> np.ndarray:
    """Draw Unicode text on an OpenCV BGR frame."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text(position, text, font=text_font, fill=fill)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def load_calibration(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Load calibration, scaling it to the active resolution; otherwise make an honest approximation."""
    if path.exists():
        data = np.load(path)
        camera_matrix = data["camera_matrix"].astype(np.float64)
        distortion = data["distortion_coefficients"].astype(np.float64)
        saved_width, saved_height = int(data["image_width"]), int(data["image_height"])
        camera_matrix[0, :] *= width / saved_width
        camera_matrix[1, :] *= height / saved_height
        return camera_matrix, distortion, True

    focal = float(max(width, height))
    camera_matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    return camera_matrix, np.zeros((5, 1), dtype=np.float64), False


def order_dot_corners(points: np.ndarray) -> np.ndarray:
    """Order four image points as top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float64)
    sums = points.sum(axis=1)
    differences = points[:, 0] - points[:, 1]
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmax(differences)]
    ordered[3] = points[np.argmin(differences)]
    return ordered


def select_dot_quad(candidates: list[tuple[np.ndarray, float]], expected_ratio: float) -> np.ndarray | None:
    """Choose four similarly sized circular blobs forming the glasses-dot quadrilateral."""
    if len(candidates) < 4:
        return None
    candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[:12]
    best_points: np.ndarray | None = None
    best_score = -float("inf")
    for group in itertools.combinations(candidates, 4):
        points = order_dot_corners(np.array([item[0] for item in group], dtype=np.float64))
        contour = points.astype(np.float32).reshape(-1, 1, 2)
        if not cv2.isContourConvex(contour):
            continue
        area = abs(cv2.contourArea(contour))
        sides = np.array([np.linalg.norm(points[(index + 1) % 4] - points[index]) for index in range(4)])
        if area < 500 or sides.min() < 12:
            continue
        observed_ratio = (sides[0] + sides[2]) / (sides[1] + sides[3])
        if not 0.35 * expected_ratio <= observed_ratio <= 2.8 * expected_ratio:
            continue
        dot_areas = np.array([item[1] for item in group])
        equal_size = 1.0 / (1.0 + float(np.std(dot_areas) / np.mean(dot_areas)))
        shape_match = 1.0 / (1.0 + abs(math.log(observed_ratio / expected_ratio)))
        score = area * equal_size * shape_match
        if score > best_score:
            best_score = score
            best_points = points
    return best_points


def find_four_dots(
    frame: np.ndarray, threshold: int, min_area: float, max_area: float, expected_ratio: float
) -> np.ndarray | None:
    """Find four solid black circular dots; their centres become the pose reference points."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, black = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[np.ndarray, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not min_area <= area <= max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        _, _, width, height = cv2.boundingRect(contour)
        aspect_ratio = width / height if height else 0
        if circularity < 0.62 or not 0.7 <= aspect_ratio <= 1.35:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        candidates.append((np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]), area))
    return select_dot_quad(candidates, expected_ratio)


def estimate_rotation(dot_points: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray, width_mm: float, height_mm: float) -> np.ndarray | None:
    half_width, half_height = width_mm / 2, height_mm / 2
    object_points = np.array(
        [
            [-half_width, half_height, 0],
            [half_width, half_height, 0],
            [half_width, -half_height, 0],
            [-half_width, -half_height, 0],
        ],
        dtype=np.float64,
    )
    image_points = dot_points.reshape(4, 2).astype(np.float64)
    success, rvec, _ = cv2.solvePnP(object_points, image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success or not np.isfinite(rvec).all():
        return None
    return cv2.Rodrigues(rvec)[0]


def rotation_distance_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    """Project the arithmetic average onto SO(3)."""
    average = np.mean(rotations, axis=0)
    u, _, vt = np.linalg.svd(average)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def signed_forward_angle(rotation: np.ndarray, reference: np.ndarray, axis: int, sign: float) -> float:
    """Return one configured component of the relative rotation vector in degrees.

    Unlike an image-edge slope, this uses a 3D pose from the four dot centres.
    """
    relative = rotation @ reference.T
    rvec, _ = cv2.Rodrigues(relative)
    return float(sign * math.degrees(rvec.flatten()[axis]))


class PostureState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.reference: np.ndarray | None = None
        self.recent_angles: deque[float] = deque(maxlen=settings.smoothing_samples)
        self.calibration_started: float | None = None
        self.calibration_rotations: list[np.ndarray] = []
        self.over_since: float | None = None
        self.back_since: float | None = None
        self.warning = False
        self.message = "Ожидание калибровки"

    def start_calibration(self, now: float) -> None:
        self.calibration_started = now
        self.calibration_rotations = []
        self.recent_angles.clear()
        self.over_since = None
        self.back_since = None
        self.warning = False
        self.message = "Калибровка: смотрите прямо и не двигайтесь"

    def dots_lost(self) -> None:
        self.over_since = None
        self.back_since = None
        self.warning = False
        # A newly visible dot set must not be mixed with values collected before an occlusion.
        self.recent_angles.clear()
        if self.calibration_started is not None:
            self.calibration_started = None
            self.calibration_rotations = []
            self.message = "Калибровка прервана: точки пропали. Нажмите C ещё раз"
        else:
            self.message = "Точки не видны"

    def update(self, rotation: np.ndarray, now: float) -> float | None:
        if self.calibration_started is not None:
            self.calibration_rotations.append(rotation)
            elapsed = now - self.calibration_started
            self.message = f"Калибровка: {max(0, self.settings.calibration_seconds - elapsed):.1f} с"
            if elapsed >= self.settings.calibration_seconds:
                reference = average_rotations(self.calibration_rotations)
                maximum_motion = max(rotation_distance_degrees(item, reference) for item in self.calibration_rotations)
                self.calibration_started = None
                self.calibration_rotations = []
                if maximum_motion > self.settings.max_calibration_motion:
                    self.message = "Слишком много движения. Нажмите C и повторите"
                else:
                    self.reference = reference
                    self.message = "Калибровка завершена"
            return None

        if self.reference is None:
            self.message = "Ожидание калибровки — нажмите C"
            return None

        raw_angle = signed_forward_angle(rotation, self.reference, self.settings.forward_axis, self.settings.forward_sign)
        self.recent_angles.append(raw_angle)
        angle = float(np.median(self.recent_angles))
        if angle > self.settings.warning_angle:
            self.back_since = None
            self.over_since = self.over_since or now
            seconds = now - self.over_since
            if seconds >= self.settings.warning_seconds:
                self.warning = True
                self.message = "ПОДНИМИТЕ ЭКРАН И ИЗМЕНИТЕ ПОЛОЖЕНИЕ ГОЛОВЫ"
            else:
                self.message = f"Наклон выше порога: {seconds:.1f} / {self.settings.warning_seconds:.0f} с"
        elif self.warning and angle <= self.settings.clear_angle:
            self.over_since = None
            self.back_since = self.back_since or now
            if now - self.back_since >= self.settings.clear_seconds:
                self.warning = False
                self.back_since = None
                self.message = "В пределах порога"
            else:
                self.message = "Возврат к норме…"
        else:
            self.over_since = None
            self.back_since = None
            if not self.warning:
                self.message = "В пределах порога"
        return angle


def draw_interface(frame: np.ndarray, state: PostureState, angle: float | None, calibrated: bool) -> np.ndarray:
    panel_bottom = 185
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_bottom), (18, 25, 34), -1)
    frame = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)
    frame = draw_text(frame, "ErgoNeck", (22, 18), (245, 245, 245), FONT_LARGE)
    angle_text = "Наклон: —" if angle is None else f"Наклон вперёд: {angle:+.1f}°"
    angle_color = (245, 245, 245) if angle is None or angle <= SETTINGS.warning_angle else (255, 185, 100)
    frame = draw_text(frame, angle_text, (26, 82), angle_color, FONT_MEDIUM)
    status_color = (255, 105, 105) if state.warning else (178, 230, 170)
    if not calibrated or angle is None:
        status_color = (180, 210, 240)
    frame = draw_text(frame, f"Состояние: {state.message}", (26, 122), status_color, FONT_SMALL)
    frame = draw_text(frame, "C — калибровка     Q — выход", (26, 155), (185, 195, 210), FONT_SMALL)
    if state.warning:
        warning_overlay = frame.copy()
        top = max(200, frame.shape[0] // 2 - 80)
        bottom = min(frame.shape[0] - 20, top + 150)
        cv2.rectangle(warning_overlay, (25, top), (frame.shape[1] - 25, bottom), (30, 20, 200), -1)
        frame = cv2.addWeighted(warning_overlay, 0.86, frame, 0.14, 0)
        frame = draw_text(frame, "Поднимите экран", (50, top + 28), (255, 255, 255), FONT_LARGE)
        frame = draw_text(frame, "и измените положение головы", (50, top + 88), (255, 255, 255), FONT_MEDIUM)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="ErgoNeck posture prototype")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--dots-width-mm", type=float, default=SETTINGS.dots_width_mm,
                        help="Distance between centres of left and right dots")
    parser.add_argument("--dots-height-mm", type=float, default=SETTINGS.dots_height_mm,
                        help="Distance between centres of upper and lower dots")
    parser.add_argument("--dot-threshold", type=int, default=SETTINGS.dot_threshold,
                        help="Pixels darker than this are treated as black (0-255)")
    parser.add_argument("--calibration", type=Path, default=Path("camera_calibration.npz"))
    parser.add_argument("--forward-sign", type=float, choices=(-1.0, 1.0), default=SETTINGS.forward_sign,
                        help="Use -1 when lowering the head gives a negative value")
    args = parser.parse_args()
    if args.dots_width_mm <= 0 or args.dots_height_mm <= 0 or not 0 <= args.dot_threshold <= 255:
        parser.error("dot dimensions must be positive and --dot-threshold must be between 0 and 255")

    settings = Settings(
        dots_width_mm=args.dots_width_mm,
        dots_height_mm=args.dots_height_mm,
        dot_threshold=args.dot_threshold,
        forward_sign=args.forward_sign,
    )
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру {args.camera}. Проверьте доступ к камере и индекс --camera.")

    state = PostureState(settings)
    camera_matrix: np.ndarray | None = None
    distortion: np.ndarray | None = None
    using_calibration = False
    sounded_warning = False
    print("ErgoNeck started. Press C to calibrate and Q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Не удалось получить кадр с камеры.")
            if camera_matrix is None:
                camera_matrix, distortion, using_calibration = load_calibration(args.calibration, frame.shape[1], frame.shape[0])
                source = "saved calibration" if using_calibration else "approximate intrinsics — run calibrate_camera.py for measurements"
                print(f"Camera model: {source}")

            now = time.monotonic()
            dots = find_four_dots(
                frame,
                settings.dot_threshold,
                settings.dot_min_area_px,
                settings.dot_max_area_px,
                settings.dots_width_mm / settings.dots_height_mm,
            )
            angle: float | None = None
            if dots is None:
                state.dots_lost()
            else:
                cv2.polylines(frame, [dots.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
                for index, point in enumerate(dots.astype(np.int32), start=1):
                    cv2.circle(frame, tuple(point), 8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, str(index), tuple(point + np.array([10, -10])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2, cv2.LINE_AA)
                rotation = estimate_rotation(
                    dots, camera_matrix, distortion, settings.dots_width_mm, settings.dots_height_mm
                )
                if rotation is None:
                    state.dots_lost()
                else:
                    angle = state.update(rotation, now)

            if state.warning and not sounded_warning:
                print("\a", end="", flush=True)
                sounded_warning = True
            if not state.warning:
                sounded_warning = False

            view = draw_interface(frame, state, angle, state.reference is not None)
            if not using_calibration:
                view = draw_text(view, "Тестовый режим: выполните калибровку камеры для точных углов", (24, view.shape[0] - 38),
                                 (50, 220, 255), FONT_SMALL)
            cv2.imshow("ErgoNeck", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C")):
                state.start_calibration(time.monotonic())
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
