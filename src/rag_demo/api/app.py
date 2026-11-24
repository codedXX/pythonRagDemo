"""创建 FastAPI 应用，由同一进程提供静态页面和接口。浏览器只访问 /api/ 下的服务，供应商配置留在服务端。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import logging  # 引入日志模块。
from pathlib import Path  # 从 pathlib 导入路径对象。

from fastapi import FastAPI  # 从 fastapi 导入接口应用。
from fastapi.responses import FileResponse  # 从 fastapi.responses 导入文件响应。
from fastapi.staticfiles import StaticFiles  # 从 fastapi.staticfiles 导入StaticFiles。

from rag_demo.api.routes import router  # 从 rag_demo.api.routes 导入路由器。
from rag_demo.api.services import Services, build_services, initialise  # 从 rag_demo.api.services 引入本模块使用的类型和工具。
from rag_demo.config import Settings, load_settings  # 从 rag_demo.config 引入本模块使用的类型和工具。

logger = logging.getLogger(__name__)  # 将日志记录器设为获取当前模块的日志记录器的结果。

STATIC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "static"  # 从模块位置定位项目中的静态资源目录。


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:  # 组装应用。
    """组装应用；测试可通过 services 注入替身，离线验证接口。"""
    logging.basicConfig(  # 配置进程的基础日志输出格式。
        level=logging.INFO,  # 默认记录信息级别及更严重的消息。
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",  # 依次输出时间、级别、模块名和具体消息。
    )  # 结束当前多行表达式。

    resolved = settings or load_settings(root=Path.cwd())  # 优先采用传入配置，否则从启动目录加载。
    graph = services or build_services(resolved)  # 优先使用传入服务，方便测试注入替身。

    app = FastAPI(  # 将应用设为接口应用的结果。
        title="RAG Demo",  # 传入标题参数。
        description="Hybrid retrieval over your own documents",  # 传入description参数。
        version="0.1.0",  # 传入版本参数。
    )  # 结束当前多行表达式。
    app.state.services = graph  # 把服务对象保存在应用状态中，供各路由读取。
    app.state.settings = resolved  # 保存本次启动实际使用的配置。
    app.include_router(router)  # 注册文档、问答和状态接口。

    if STATIC_DIR.is_dir():  # 确认静态目录存在后才挂载页面。
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")  # 将静态资源目录映射到浏览器请求路径。

        @app.get("/")  # 为下方定义应用 app.get 装饰器。
        def index() -> FileResponse:  # 返回问答界面的静态首页。
            return FileResponse(STATIC_DIR / "index.html")  # 访问首页时直接返回问答页面文件。

    return app  # 返回应用。


def create_and_initialise(settings: Settings | None = None) -> FastAPI:  # 组装应用并检查依赖。
    """组装应用并检查依赖；记录检查失败的原因，让页面仍可启动并展示故障。"""
    resolved = settings or load_settings(root=Path.cwd())  # 从传入对象或当前工作目录取得启动配置。
    graph = build_services(resolved)  # 根据配置装配存储、检索与生成服务。
    initialise(graph)  # 在接受请求前运行依赖检查并准备索引。
    return create_app(resolved, graph)  # 将已初始化的服务交给应用工厂。