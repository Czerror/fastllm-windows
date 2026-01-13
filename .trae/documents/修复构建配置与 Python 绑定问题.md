# 修复构建配置与 Python 绑定问题

通过对代码库的静态分析，发现了以下问题，建议按以下步骤进行修复：

## 1. 修复 setup.py 中的拼写错误
在 `pyfastllm/setup.py` 中，`CMakeExtension` 的名称被错误地拼写为 `pyfasltllm`（多了一个 'l'）。这可能导致生成的 Python 扩展模块名称不正确，从而无法被 `__init__.py` 正确导入。

**操作：**
- 修改 `pyfastllm/setup.py`，将 `name="pyfasltllm"` 修正为 `name="pyfastllm"`。

## 2. 修复 Python 绑定中的 Double Free Bug
在 `src/pybinding.cpp` 中，`ToNumpy` 函数存在 `Data` 对象和 Numpy 数组同时尝试释放 `cpuData` 内存的风险（Double Free）。代码中已有 `// TODO:fix data double free bug` 注释。

**操作：**
- 修改 `src/pybinding.cpp` 中的 `ToNumpy` 函数。
- 移除接管内存的 `py::capsule` deleter。
- 改为执行深拷贝（Deep Copy），将数据复制到 Numpy 数组中。虽然这会增加一次内存拷贝，但能彻底解决生命周期管理和重复释放的问题，确保稳定性。

## 其他发现（暂不处理）
- **GPU 内存泄露**：`src/pybinding.cpp` 中提到 `// TODO GPU内存不释放的bug`。这需要更深入的调试和上下文，暂不在此次变更中修复。
- **功能缺失**：`fastllm.cpp` 和 `basellm.cpp` 中包含多卡支持和多 Prompt 推理的 TODO，属于功能开发范畴。

请确认是否执行上述修复计划。
