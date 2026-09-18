import numpy as np

from example_modular_usage import (
    Detection,
    get_detection_center,
    localize_object,
    project_pixel_to_3d,
    world_error_to_delta_action,
)


class StubDetector:
    """Return fixed detections keyed by camera name."""

    def __init__(self, detections: dict[str, Detection]):
        self.detections = detections

    def detect(
        self, image: np.ndarray, camera_name: str, object_name: str
    ) -> Detection:
        del image, object_name
        return self.detections[camera_name]


class PartiallyVisibleDetector(StubDetector):
    """Fail one view to exercise long-horizon occlusion handling."""

    def detect(
        self, image: np.ndarray, camera_name: str, object_name: str
    ) -> Detection:
        if camera_name == "right":
            raise RuntimeError("occluded")
        return super().detect(image, camera_name, object_name)


def test_detection_center_uses_nearest_mask_pixel() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True
    mask[1, 3] = True
    mask[3, 1] = True
    mask[3, 3] = True

    center = get_detection_center(Detection((1, 1, 3, 3), mask))

    assert mask[center[1], center[0]]


def test_detection_center_falls_back_to_bounding_box() -> None:
    detection = Detection((2, 4, 8, 10))

    assert get_detection_center(detection) == (5, 7)


def test_project_pixel_to_3d_applies_intrinsics_and_extrinsics() -> None:
    depth = np.full((5, 5), 2.0)
    intrinsics = np.asarray([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = [10.0, 20.0, 30.0]

    camera_point, world_point = project_pixel_to_3d(
        (3, 1), depth, intrinsics, camera_to_world
    )

    np.testing.assert_allclose(camera_point, [2.0, 0.0, 2.0])
    np.testing.assert_allclose(world_point, [12.0, 20.0, 32.0])


def test_localize_object_averages_projected_camera_points() -> None:
    detections = {
        "left": Detection((1, 1, 1, 1)),
        "right": Detection((1, 1, 1, 1)),
    }
    observation = {
        "left_image": np.zeros((3, 3, 3), dtype=np.uint8),
        "left_depth": np.ones((3, 3, 1)),
        "right_image": np.zeros((3, 3, 3), dtype=np.uint8),
        "right_depth": np.full((3, 3, 1), 3.0),
    }
    calibration = {
        camera_name: {
            "intrinsics": np.eye(3),
            "camera_to_world": np.eye(4),
        }
        for camera_name in detections
    }

    estimate = localize_object(
        StubDetector(detections),
        observation,
        calibration,
        ("left", "right"),
        "object",
    )

    np.testing.assert_allclose(estimate.world_point, [2.0, 2.0, 2.0])
    assert len(estimate.views) == 2


def test_localize_object_uses_remaining_visible_camera() -> None:
    detections = {
        "left": Detection((1, 1, 1, 1)),
        "right": Detection((1, 1, 1, 1)),
    }
    observation = {
        f"{camera}_{suffix}": (
            np.zeros((3, 3, 3), dtype=np.uint8)
            if suffix == "image"
            else np.ones((3, 3, 1))
        )
        for camera in detections
        for suffix in ("image", "depth")
    }
    calibration = {
        camera: {"intrinsics": np.eye(3), "camera_to_world": np.eye(4)}
        for camera in detections
    }

    estimate = localize_object(
        PartiallyVisibleDetector(detections),
        observation,
        calibration,
        ("left", "right"),
        "object",
    )

    assert len(estimate.views) == 1
    np.testing.assert_allclose(estimate.world_point, [1.0, 1.0, 1.0])


def test_world_error_to_delta_action_rotates_scales_and_clips() -> None:
    base_rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    action = world_error_to_delta_action(
        np.asarray([0.10, 0.025, -0.025]),
        base_rotation,
        np.full(3, 0.05),
    )

    np.testing.assert_allclose(action, [0.5, -1.0, -0.5, 0.0, 0.0, 0.0])
