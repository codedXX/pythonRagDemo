#!/usr/bin/env python
"""入库前运行 uv run python scripts/selfcheck.py，核对外部服务。

实测向量模型的维度，避免用错误维度建库；用已知相关的段落检验重排顺序，并验证生成模型的凭据及流式响应。全部通过时退出码为 0，存在异常时为 1。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import sys  # 引入系统参数工具。
from pathlib import Path  # 从 pathlib 导入路径对象。

# 将源码目录加入搜索路径，方便直接运行脚本。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # 将源码路径置于导入搜索路径首位。

# 从 rag_demo.config 引入本模块使用的类型和工具。
from rag_demo.config import ConfigurationError, load_settings  # noqa: E402
# 从 rag_demo.selfcheck 引入本模块使用的类型和工具。
from rag_demo.selfcheck import format_report, run_selfcheck  # noqa: E402


def main(argv: list[str]) -> int:  # 执行命令行自检，并返回可用于部署判断的退出码。
    root = Path.cwd()  # 从启动目录定位项目。
    skip_llm = "--skip-llm" in argv  # 读取命令行开关，决定本次是否检查生成模型。

    try:  # 在此捕获后续操作可能产生的异常。
        settings = load_settings(root=root)  # 将配置设为加载配置的结果。
    except ConfigurationError as exc:  # 捕获配置校验异常。
        print(f"configuration error: {exc}", file=sys.stderr)  # 将配置校验错误写入标准错误流。
        return 1  # 配置无效时以失败状态结束。

    config_path = root / "config.yaml"  # 构造自检所用配置文件的默认路径。
    if not config_path.exists():  # 当不满足“文件存在的结果”时进入此分支。
        print(  # 向终端输出提示信息。
            f"note: {config_path} does not exist; using defaults and environment "  # 说明本次没有读取本地配置文件。
            "variables only. Copy config.example.yaml to config.yaml and fill in "  # 提示从示例文件复制本地配置。
            "the API keys.",  # 提醒填写访问模型所需的密钥。
            file=sys.stderr,  # 将配置提示写入标准错误流。
        )  # 结束当前多行表达式。

    print(f"checking dependencies (config: {config_path})")  # 显示即将检查的配置文件位置。
    print()  # 在终端输出空行以隔开检查标题和结果。
    report = run_selfcheck(settings, include_llm=not skip_llm, record_if_missing=True)  # 按命令行选项决定是否连同生成模型一起探测。
    print(format_report(report))  # 将各项检查状态以可读形式输出。

    return 0 if report.healthy else 1  # 所有依赖健康时返回零，否则返回一。


if __name__ == "__main__":  # 仅在直接执行脚本时启动自检。
    raise SystemExit(main(sys.argv[1:]))  # 用入口函数的结果作为进程状态码。