"""ErgoNeck: ArUco-based posture reminder prototype for a laptop camera."""

from __future__ import annotations

import argparse
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
    marker_id: int = 0
    marker_size_mm: float = 45.0
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


def find_marker(frame: np.ndarray, marker_id: int):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters") else cv2.aruco.DetectorParameters_create()
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(frame)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    if ids is None:
        return None
    matches = np.where(ids.flatten() == marker_id)[0]
    return corners[int(matches[0])] if len(matches) else None


def estimate_rotation(corners: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray, marker_size_mm: float) -> np.ndarray | None:
    half = marker_size_mm / 2
    object_points = np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float64
    )
    image_points = corners.reshape(4, 2).astype(np.float64)
    success, rvec, _ = cv2.solvePnP(object_points, image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
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

    Unlike an image-edge slope, this uses 3D pose from solvePnP. The sign is
    intentionally configurable because the physical marker can be mounted in either orientation.
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

    def marker_lost(self) -> None:
        self.over_since = None
        self.back_since = None
        self.warning = False
        # A newly visible marker must not be mixed with values collected before an occlusion.
        self.recent_angles.clear()
        if self.calibration_started is not None:
            self.calibration_started = None
            self.calibration_rotations = []
            self.message = "Калибровка прервана: метка пропала. Нажмите C ещё раз"
        else:
            self.message = "Метки не видны"

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
    parser.add_argument("--marker-id", type=int, default=SETTINGS.marker_id)
    parser.add_argument("--marker-size-mm", type=float, default=SETTINGS.marker_size_mm,
                        help="INNER black-and-white square side; not its white border")
    parser.add_argument("--calibration", type=Path, default=Path("camera_calibration.npz"))
    parser.add_argument("--forward-sign", type=float, choices=(-1.0, 1.0), default=SETTINGS.forward_sign,
                        help="Use -1 when lowering the head gives a negative value")
    args = parser.parse_args()
    if args.marker_size_mm <= 0:
        parser.error("--marker-size-mm must be positive")

    settings = Settings(marker_id=args.marker_id, marker_size_mm=args.marker_size_mm, forward_sign=args.forward_sign)
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
            corners = find_marker(frame, settings.marker_id)
            angle: float | None = None
            if corners is None:
                state.marker_lost()
            else:
                cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
                rotation = estimate_rotation(corners, camera_matrix, distortion, settings.marker_size_mm)
                if rotation is None:
                    state.marker_lost()
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
