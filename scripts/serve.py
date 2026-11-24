#!/usr/bin/env python
"""运行 uv run python scripts/serve.py 启动接口服务。启动前检查外部依赖并记录结果，即使供应商配置有误，页面也能展示具体原因。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import sys  # 引入系统参数工具。
from pathlib import Path  # 从 pathlib 导入路径对象。

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # 把项目源码目录放在模块搜索路径首位，支持直接运行此脚本。

# 引入应用服务器。
import uvicorn  # noqa: E402

# 从 rag_demo.api.app 导入创建并初始化应用。
from rag_demo.api.app import create_and_initialise  # noqa: E402
# 从 rag_demo.config 引入本模块使用的类型和工具。
from rag_demo.config import ConfigurationError, load_settings  # noqa: E402


def main(argv: list[str]) -> int:  # 读取启动参数，创建应用并启动服务。
    root = Path.cwd()  # 用当前工作目录定位项目配置。
    try:  # 在此捕获后续操作可能产生的异常。
        settings = load_settings(root=root)  # 合并本地配置和环境变量。
    except ConfigurationError as exc:  # 捕获配置校验异常。
        print(f"configuration error: {exc}", file=sys.stderr)  # 将配置错误写到标准错误流。
        return 1  # 用非零退出码报告启动失败。

    if not (root / "config.yaml").exists():  # 没有本地配置文件时输出配置指引。
        print(  # 向终端输出提示信息。
            "note: config.yaml not found. Copy config.example.yaml to config.yaml "  # 提示从示例复制一份本地配置。
            "and fill in the API keys.",  # 提醒填写所需的服务凭据。
            file=sys.stderr,  # 将这条提示送到标准错误流。
        )  # 结束当前多行表达式。

    host = settings.api.host  # 取出当前使用的监听地址。
    port = settings.api.port  # 取出当前使用的监听端口。
    for argument in argv:  # 逐项读取命令行参数，允许覆盖监听设置。
        if argument.startswith("--port="):  # 识别指定端口的启动参数。
            port = int(argument.split("=", 1)[1])  # 取等号右侧的值并转成整数端口。
        elif argument.startswith("--host="):  # 识别指定监听地址的启动参数。
            host = argument.split("=", 1)[1]  # 取等号右侧的地址覆盖配置值。

    app = create_and_initialise(settings)  # 在启动监听前组装服务并初始化依赖。
    print(f"serving on http://{host}:{port}")  # 在终端显示实际监听地址与端口。
    uvicorn.run(app, host=host, port=port, log_level="info")  # 启动应用服务器，并以信息级别记录运行日志。
    return 0  # 服务正常退出时返回成功状态码。


if __name__ == "__main__":  # 直接运行脚本时进入命令行入口。
    raise SystemExit(main(sys.argv[1:]))  # 跳过脚本名，将入口返回值用作进程退出码。