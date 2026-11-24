# pairwise-minimap-marker-sync

## 领域

游戏、图形与多媒体

## 主题

小地图标记投影同步

## 说明

相机帧号校验、帧序提交与标记池回收。

初始状态下 `gameplay/minimap_marker_sync.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
