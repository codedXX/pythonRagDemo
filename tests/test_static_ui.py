"""验证静态页面、控件和接口之间的约定，以及浏览器资源中没有供应商凭据。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

from pathlib import Path  # 从 pathlib 导入路径对象。

import pytest  # 引入测试框架。

STATIC = Path(__file__).resolve().parent.parent / "static"  # 将STATIC设为父块下的“static”。


@pytest.fixture(scope="module")  # 注册供用例复用的测试夹具。
def html() -> str:  # 定义页面源码的处理逻辑。
    return (STATIC / "index.html").read_text(encoding="utf-8")  # 返回read_text的结果。


@pytest.fixture(scope="module")  # 注册供用例复用的测试夹具。
def script() -> str:  # 定义页面脚本的处理逻辑。
    return (STATIC / "app.js").read_text(encoding="utf-8")  # 返回read_text的结果。


@pytest.fixture(scope="module")  # 注册供用例复用的测试夹具。
def styles() -> str:  # 定义styles的处理逻辑。
    return (STATIC / "style.css").read_text(encoding="utf-8")  # 返回read_text的结果。


def test_static_files_exist():  # 验证页面、样式和脚本资源都存在。
    for name in ("index.html", "app.js", "style.css"):  # 遍历组合值，逐项取得名称。
        assert (STATIC / name).is_file(), f"{name} is missing"  # 确认是普通文件的结果。


def test_page_references_assets_relative_to_static(html):  # 验证首页按静态资源路径加载文件。
    assert '/static/style.css' in html  # 确认“/static/style.css”包含在页面源码。
    assert '/static/app.js' in html  # 确认“/static/app.js”包含在页面源码。


def test_page_contains_the_qa_and_upload_affordances(html):  # 验证页面包含提问、上传和状态控件。
    assert 'id="question"' in html  # 确认“id="question"”包含在页面源码。
    assert 'id="dropzone"' in html  # 确认“id="dropzone"”包含在页面源码。
    assert 'id="messages"' in html  # 确认“id="messages"”包含在页面源码。
    assert 'id="doc-list"' in html  # 确认“id="doc-list"”包含在页面源码。
    assert 'id="health-panel"' in html  # 确认“id="health-panel"”包含在页面源码。


def test_upload_accepts_only_pdf_and_markdown(html):  # 验证文件选择器限制支持的文档格式。
    assert 'accept=".pdf,.md,.markdown"' in html  # 确认文本内容包含在页面源码。


def test_script_calls_the_endpoints_that_exist(script):  # 验证脚本请求已有的服务端接口。
    for endpoint in ("/api/chat", "/api/docs", "/api/health"):  # 遍历组合值，逐项取得endpoint。
        assert endpoint in script  # 确认endpoint包含在页面脚本。


def test_script_handles_every_chat_event(script):  # 每种推送事件都应有对应处理分支，避免消息被静默忽略。
    """每种推送事件都应有对应处理分支，避免消息被静默忽略。"""
    for event in ("status", "rewritten", "sources", "token", "debug", "error", "done"):  # 遍历组合值，逐项取得事件。
        assert f"'{event}'" in script, f"unhandled chat event: {event}"  # 确认格式化文本包含在页面脚本。


def test_script_handles_every_upload_event(script):  # 验证脚本处理每类上传事件。
    for event in ("progress", "done", "failed", "rejected"):  # 遍历组合值，逐项取得事件。
        assert f"'{event}'" in script, f"unhandled upload event: {event}"  # 确认格式化文本包含在页面脚本。


def test_no_secrets_or_upstream_endpoints_are_referenced(script, html, styles):  # 验证静态资源隐藏供应商地址和凭据。
    blob = script + html + styles  # 将合并文本设为页面脚本加上页面源码加上styles。
    # 供应商地址和凭据应留在服务端。
    assert "dashscope" not in blob.lower()  # 确认“dashscope”未包含在合并文本的小写形式。
    assert "8.134.19.201" not in blob  # 确认“8.134.19.201”未包含在合并文本。
    assert "api_key" not in blob.lower()  # 确认“api_key”未包含在合并文本的小写形式。
    assert "sk-" not in blob  # 确认“sk-”未包含在合并文本。


def test_markdown_output_is_escaped_before_tag_insertion(script):  # 上传内容不可信，展示时应按文本处理。
    """上传内容不可信，展示时应按文本处理。"""
    assert "escapeHtml" in script  # 确认“escapeHtml”包含在页面脚本。
    assert ".replace(/&/g, '&amp;')" in script  # 确认“.replace(/&/g, '&amp;')”包含在页面脚本。
    assert "escapeHtml(code" in script  # 确认“escapeHtml(code”包含在页面脚本。


def test_styles_provide_code_list_and_table_rendering(styles):  # 验证回答区提供代码、列表和表格样式。
    for selector in (".turn-answer pre", ".turn-answer ul", ".turn-answer table"):  # 遍历组合值，逐项取得selector。
        assert selector in styles  # 确认selector包含在styles。


def test_styles_handle_long_conversations(styles):  # 验证消息区支持长对话滚动。
    assert "overflow-y: auto" in styles  # 确认“overflow-y: auto”包含在styles。


def test_empty_state_is_dismissed_when_a_turn_starts(script):  # 验证开始提问时移除空页面提示。
    assert "empty-state" in script  # 确认“empty-state”包含在页面脚本。
    assert "empty.remove()" in script  # 确认“empty.remove()”包含在页面脚本。


def test_history_is_bounded_client_side(script):  # 服务端不保留会话，客户端需按相同的轮数限制裁剪历史。
    """服务端不保留会话，客户端需按相同的轮数限制裁剪历史。"""
    assert "while (history.length > 6)" in script  # 确认文本内容包含在页面脚本。


def test_debug_panel_is_opt_in(html):  # 验证检索调试默认关闭。
    assert 'id="debug-check"' in html  # 确认“id="debug-check"”包含在页面源码。
    # 调试开关默认关闭，由用户决定是否查看检索细节。
    assert 'id="debug-check" checked' not in html  # 确认“id="debug-check" checked”未包含在页面源码。