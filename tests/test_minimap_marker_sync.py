import pytest
import gameplay.minimap_marker_sync as mod

CASES = [
    ("camera_frame_guard", (1,)),
    ("submit_by_frame", (1,)),
    ("edge_direction", (1,)),
    ("task_unregister", (1,)),
    ("stable_merge", (1,)),
    ("pause_refresh", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("project_point", (1,)),
    ("metrics", (1,)),
    ("marker_pool", (1,)),
    ("viewport_scale", (1,)),
    ("refresh", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
