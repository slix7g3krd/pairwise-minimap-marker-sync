"""六个小地图同步缺陷的回归测试。"""

from gameplay.minimap_marker_sync import (
    CameraFrame,
    FrameCommitQueue,
    MarkerPool,
    ProjectionCoordinator,
    ProjectionTask,
    TaskRegistry,
    camera_frame_guard,
    determinism,
    edge_direction,
    pause_refresh,
    stable_merge,
    submit_by_frame,
    task_unregister,
)


def test_camera_frame_number_guard_rejects_wrong_frame():
    assert camera_frame_guard(3, 3) is True
    assert camera_frame_guard(3, 4) is False

    current = CameraFrame(frame_id=8, generation=2)
    assert camera_frame_guard({"frame_id": 8, "generation": 2}, current).ok
    stale = camera_frame_guard({"frame_id": 7, "generation": 1}, current)
    assert not stale.ok
    assert "frame_mismatch" in stale.reasons
    assert "camera_generation_mismatch" in stale.reasons


def test_projection_results_commit_in_frame_order():
    results = [
        {"frame": 3, "id": "c"},
        (1, "a"),
        {"frame": 2, "id": "b"},
    ]
    ordered = submit_by_frame(results)
    assert ordered == ["a", {"frame": 2, "id": "b"}, {"frame": 3, "id": "c"}]
    assert ordered.frame_order == [1, 2, 3]

    queue = FrameCommitQueue(next_frame=1)
    assert queue.submit({"frame": 2}).committed == []
    assert queue.submit({"frame": 3}).committed == []
    committed = queue.submit({"frame": 1}).committed
    assert [item["frame"] for item in committed] == [1, 2, 3]


def test_edge_direction_uses_current_viewport_and_keeps_sign():
    assert edge_direction(120, 50, 100, 100) == "right"
    assert edge_direction(-20, 50, 100, 100) == "left"
    assert edge_direction(50, 120, 100, 100) == "bottom"
    assert edge_direction(50, -20, 100, 100) == "top"

    right = edge_direction(100, 0, (100, 100), relative=True)
    assert right == "right"
    assert right["point"] == (50.0, 0.0)

    # 缩放只是正倍数放大中心位移，方向不得翻转。
    base = edge_direction(10, 2, (100, 100), relative=True)
    zoomed = edge_direction(40, 8, (100, 100), relative=True)
    assert base.sign == zoomed.sign == (1, 1)

    # 视口尺寸变化后按新的宽高中心重算。
    resized = edge_direction(220, 100, 200, 200)
    assert resized == "right"
    assert resized["point"] == (200.0, 100.0)


def test_destroy_unregisters_tasks_and_detects_pool_reuse():
    registry = TaskRegistry()
    pool = MarkerPool(registry=registry)
    marker = pool.acquire("m-1", target_id="enemy-1")
    task = ProjectionTask(marker=marker, point=(1, 2), frame=CameraFrame(frame_id=4))
    registry.register(task)

    assert registry.validate(task).ok
    assert task_unregister(registry, marker) == 1
    assert not registry.validate(task).ok

    recycled = pool.acquire("m-1", target_id="other")
    assert recycled is marker
    new_task = ProjectionTask(marker=recycled, point=(3, 4), frame=CameraFrame(frame_id=5))
    registry.register(new_task)
    assert registry.validate(new_task).ok
    assert new_task.marker_revision != task.marker_revision


def test_stable_merge_is_independent_of_input_iteration_order():
    markers = [
        {"id": "icon-b", "target_id": "t1", "order": 1},
        {"id": "label-a", "target_id": "t1", "order": 0},
        {"id": "icon-c", "target_id": "t2"},
        {"id": "dot-a", "target_id": "t1", "order": 2},
    ]
    variants = [
        markers,
        list(reversed(markers)),
        [markers[2], markers[0], markers[3], markers[1]],
    ]

    flat_orders = []
    merged_orders = []
    for variant in variants:
        flat_orders.append([item["id"] for item in stable_merge(variant, combine=False)])
        merged = stable_merge(variant)
        merged_orders.append([item["id"] for item in merged])

    assert flat_orders[0] == flat_orders[1] == flat_orders[2]
    assert merged_orders[0] == merged_orders[1] == merged_orders[2]
    assert merged_orders[0] == ["label-a", "icon-c"]
    assert determinism()["deterministic"] is True


def test_paused_minimap_does_not_refresh_or_consume_primitives():
    from gameplay.minimap_marker_sync import Minimap

    minimap = Minimap(width=120, height=120)
    active = minimap.refresh([{"id": "a", "target_id": "t"}])
    assert active["performed"] is True
    assert active["primitives"] == 1

    minimap.set_paused(True)
    paused = minimap.refresh([{"id": "a"}, {"id": "b"}])
    assert paused["performed"] is False
    assert paused["primitives"] == 0
    assert minimap.refresh_controller.refresh_count == 1
    assert pause_refresh(False) is True
    assert pause_refresh(True) is False


def test_camera_switch_rejects_inflight_generation():
    coordinator = ProjectionCoordinator(viewport=(100, 100))
    marker = MarkerLike("marker-1")
    task = coordinator.request(marker, (10, 20), task_id="task-1")
    coordinator.run_worker(task)

    coordinator.switch_camera()
    report = coordinator.complete(task)
    assert not report.ok
    assert coordinator.results == []


class MarkerLike:
    def __init__(self, marker_id):
        self.id = marker_id
        self.alive = True
        self.revision = 0
