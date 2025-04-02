"""
此脚本用于修改 ONNX 模型的输入和输出通道数。
若输入通道数为 1，会将其扩展为 3 通道；若输出通道数为 1，也会将其扩展为 3 通道。
修改后的模型会保存到指定的输出路径。

使用方法：
    python modify_onnx.py <input_model_path> [output_model_path]
    若未提供输出路径，会在输入模型文件名后添加 "-repeat3ch" 作为默认输出文件名。
"""

from onnx import helper, numpy_helper
from onnx import TensorProto
import numpy as np
import sys
import os
import onnx


def get_unique_name(base_name, existing_names):
    """
    生成唯一的名称，避免与现有名称冲突。

    参数:
        base_name (str): 基础名称。
        existing_names (set): 现有的名称集合。

    返回:
        str: 唯一的名称。
    """
    suffix = 1
    new_name = base_name
    while new_name in existing_names:
        new_name = f"{base_name}_{suffix}"
        suffix += 1
    return new_name


def modify_onnx_model(input_model_path, output_model_path):
    """
    修改 ONNX 模型的输入和输出通道数。

    参数:
        input_model_path (str): 输入 ONNX 模型的文件路径。
        output_model_path (str): 输出 ONNX 模型的文件路径。

    返回:
        None
    """
    # 1. 加载 ONNX 模型
    model = onnx.load(input_model_path)

    # 查找模型输入节点
    input_node = None
    input_shape = None

    # 获取模型输入信息
    for input in model.graph.input:
        input_shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        print(f"Input Name: {input.name}, Shape: {input_shape}")
        if input_node is None and len(input_shape) == 4:
            input_node = input
    if input_node is None:
        print("没有找到输入节点")
        return
    else:
        input_shape = [dim.dim_value for dim in input_node.type.tensor_type.shape.dim]

    output_node = None
    output_shape = None

    # 获取模型输出信息
    for output in model.graph.output:
        output_shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        print(f"Output Name: {output.name}, Shape: {output_shape}")
        if output_node is None and len(output_shape) == 4:
            output_node = output
    if output_node is None:
        print("没有找到输出节点")
        return
    else:
        output_shape = [dim.dim_value for dim in output_node.type.tensor_type.shape.dim]

    existing_names = {node.name for node in model.graph.node}
    existing_names.update([initializer.name for initializer in model.graph.initializer])
    existing_names.update([input.name for input in model.graph.input])
    existing_names.update([output.name for output in model.graph.output])
    for node in model.graph.node:
        existing_names.update(node.input)
        existing_names.update(node.output)

    print("========")
    if input_shape[1] == 3:
        print("输入通道数为 3，无需修改")
    elif input_shape[1] == 1:
        print("输入通道数为 1，开始修改")
        # 创建新的输入节点
        transfor_name_out = get_unique_name("transfor_input_out", existing_names)
        existing_names.add(transfor_name_out)
        transfor_name = get_unique_name("transfor_input", existing_names)
        existing_names.add(transfor_name)

        input_node_name = input_node.name
        input_node.type.tensor_type.shape.dim[1].dim_value = 3

        # 先找到第一个使用原始输入节点的节点的索引
        insert_index = 0
        for i, node in enumerate(model.graph.node):
            if input_node_name in node.input:
                insert_index = i
                break
        for node in model.graph.node:
            for i, input_name in enumerate(node.input):
                if input_name == input_node_name:
                    node.input[i] = transfor_name_out

        # 创建平均节点
        mean_node = helper.make_node(
            'ReduceMean',
            inputs=[input_node_name],
            outputs=[transfor_name_out],
            axes=[1],
            keepdims=1
        )
        # 在合适的位置插入新节点
        model.graph.node.insert(insert_index, mean_node)


    else:
        print(f"Error: 输入通道数{input_shape[1]}")
        return

    if output_shape[1] == 3:
        print("输出通道数为 3，无需修改")
    elif output_shape[1] == 1:
        print("输出通道数为 1，开始修改")
        # 创建新的输出节点
        transfor_name_in = get_unique_name("transfor_input_in", existing_names)
        existing_names.add(transfor_name_in)
        transfor_name = get_unique_name("transfor_output", existing_names)
        existing_names.add(transfor_name)

        output_node_name = output_node.name
        output_node.type.tensor_type.shape.dim[1].dim_value = 3

        # 找到第一个输出为原输出节点名的节点的索引
        insert_index = len(model.graph.node)
        for i, node in enumerate(model.graph.node):
            if output_node_name in node.output:
                insert_index = i + 1  # 新节点应插入到该节点之后
                break

        for node in model.graph.node:
            for i, output_name in enumerate(node.output):
                if output_name == output_node_name:
                    node.output[i] = transfor_name_in

        # 创建 repeats 张量
        repeats_tensor = numpy_helper.from_array(np.array([1, 3, 1, 1], dtype=np.int64),
                                                 name=get_unique_name("repeats_tensor", existing_names))
        model.graph.initializer.append(repeats_tensor)

        tile_node = helper.make_node(
            'Tile',
            inputs=[transfor_name_in, repeats_tensor.name],
            outputs=[output_node_name],
            name=get_unique_name(transfor_name, existing_names)
        )

        # 在合适的位置插入 Tile 节点
        model.graph.node.insert(insert_index, tile_node)

    else:
        print(f"Error: 输出通道数{output_shape[1]}")
        return

    print("========")
    onnx.checker.check_model(model)
    # 5. 保存修改后的模型
    onnx.save(model, output_model_path)
    print(f"模型已保存到: {output_model_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请输入需要修改的模型路径")
        sys.exit(0)

    input_model_path = sys.argv[1]
    if not os.path.exists(input_model_path):
        print(f"模型路径{input_model_path}不存在")
        sys.exit(0)

    output_model_path = ""
    if len(sys.argv) > 2:
        output_model_path = sys.argv[2]
        print(f"Input model: {input_model_path}")
        print(f"Output model: {output_model_path}")
    else:
        output_model_path = input_model_path.replace('.onnx', '') + "-repeat3ch.onnx"
        print(f"Input model: {input_model_path}")
        print(f"Output model (default): {output_model_path}")

    modify_onnx_model(input_model_path, output_model_path)