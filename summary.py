import jax
import jax.numpy as jnp
from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModuleInfo:
    """存储模块信息的数据类"""

    name: str
    type: str
    params: int
    shape: Tuple[int, ...]
    children: List[str]


def get_param_count(params: Dict) -> int:
    """计算参数字典中的参数总数"""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def format_size(num: int) -> str:
    """将参数数量格式化为易读的字符串"""
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}P"


def print_model_summary(
    model: Any,
    input_shapes: Union[Tuple, Dict],
    batch_size: int = 1,
    show_param_shapes: bool = False,
) -> None:
    """
    打印 Flax 模型的结构摘要。

    Args:
        model: Flax 模型实例
        input_shapes: 输入形状（不包括批次维度）或形状字典
        batch_size: 用于初始化的批次大小
        show_param_shapes: 是否显示详细的参数形状
    """
    # 准备输入数据
    if isinstance(input_shapes, dict):
        inputs = {
            k: jnp.ones((batch_size,) + v, dtype=jnp.int32)
            for k, v in input_shapes.items()
        }
    else:
        inputs = jnp.ones((batch_size,) + input_shapes, dtype=jnp.int32)

    # 初始化模型
    key = jax.random.PRNGKey(0)
    variables = model.init(key, inputs)
    params = variables.get("params", {})

    # 收集模块信息
    module_dict = {}
    param_count = {"total": 0}  # 使用字典来存储总参数数量

    def traverse(params_dict: Any, parent_path: str = "") -> List[str]:
        children = []
        for name, value in params_dict.items():
            current_path = f"{parent_path}/{name}" if parent_path else name

            # 修改类型检查逻辑
            if hasattr(value, "items"):  # 检查是否是类字典对象
                # 递归处理子模块
                sub_children = traverse(value, current_path)
                if sub_children:
                    children.append(current_path)
                    module_dict[current_path] = ModuleInfo(
                        name=name,
                        type=value.__class__.__name__,
                        params=get_param_count(value),
                        shape=None,
                        children=sub_children,
                    )
            else:
                # 处理参数
                param_count_value = np.prod(value.shape)
                module_dict[current_path] = ModuleInfo(
                    name=name,
                    type=value.__class__.__name__,
                    params=param_count_value,
                    shape=value.shape,
                    children=[],
                )
                param_count["total"] += param_count_value

        return children

    traverse(params)
    total_params = param_count["total"]

    # 打印摘要
    logger.info("Model Summary:")
    logger.info("=" * 80)
    logger.info(
        f"{'Layer':<40} {'Type':<20} {'Params':<10} {'Shape' if show_param_shapes else ''}"
    )
    logger.info("-" * 80)

    def print_module(path: str, level: int = 0):
        info = module_dict[path]
        indent = "  " * level
        shape_str = f"{info.shape}" if show_param_shapes and info.shape else ""
        logger.info(
            f"{indent}{info.name:<{40-level*2}} {info.type:<20} {format_size(info.params):<10} {shape_str}"
        )

        for child in info.children:
            print_module(child, level + 1)

    # 打印根模块
    root_modules = [path for path in module_dict.keys() if "/" not in path]
    for root in root_modules:
        print_module(root)

    # 打印总结
    logger.info("=" * 80)
    logger.info(f"Total params: {format_size(total_params)} ({total_params:,})")
    logger.info(f"Input shape: {input_shapes}")
    if hasattr(model, "dtype"):
        logger.info(f"Model dtype: {model.dtype}")
    logger.info("=" * 80)
