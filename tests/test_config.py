"""验证环境变量、本地文件和默认值的覆盖顺序。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import pytest  # 引入测试框架。

from rag_demo.config import (  # 从 rag_demo.config 引入本模块使用的类型和工具。
    ConfigurationError,  # 导入配置校验异常。
    load_settings,  # 导入加载配置。
    missing_secrets,  # 导入列出缺失凭据。
    redacted,  # 导入配置脱敏。
)  # 结束当前多行表达式。


def test_defaults_apply_when_no_file_and_no_env(tmp_path, monkeypatch):  # 验证无文件及环境覆盖时采用模型默认值。
    for name in list(__import__("os").environ):  # 遍历__import__('os').environ的列表形式，逐项取得名称。
        if name.startswith("RAG__"):  # 当前缀匹配的结果时进入此分支。
            monkeypatch.delenv(name, raising=False)  # 移除测试环境中对应变量。

    settings = load_settings(root=tmp_path, use_env=False)  # 将配置设为加载配置的结果。

    assert settings.llm.model == "claude-sonnet-5"  # 确认模型等于“claude-sonnet-5”。
    assert settings.milvus.consistency_level == "Strong"  # 确认一致性级别等于“Strong”。
    assert settings.retrieval.rrf_k == 60  # 确认倒数排名融合常数等于60。


def test_yaml_overrides_default(tmp_path, monkeypatch):  # 验证文件配置覆盖默认值。
    for name in list(__import__("os").environ):  # 遍历__import__('os').environ的列表形式，逐项取得名称。
        if name.startswith("RAG__"):  # 当前缀匹配的结果时进入此分支。
            monkeypatch.delenv(name, raising=False)  # 移除测试环境中对应变量。

    (tmp_path / "config.yaml").write_text(  # 将内容写入临时目录下的“config.yaml”。
        "retrieval:\n  rrf_k: 100\nllm:\n  model: from-yaml\n",  # 覆盖融合常数与生成模型以检验文件优先级。
        encoding="utf-8",  # 传入编码参数。
    )  # 结束当前多行表达式。

    settings = load_settings(root=tmp_path, use_env=False)  # 将配置设为加载配置的结果。

    assert settings.retrieval.rrf_k == 100  # 确认倒数排名融合常数等于100。
    assert settings.llm.model == "from-yaml"  # 确认模型等于“from-yaml”。
    # 未覆盖的字段应沿用默认值。
    assert settings.retrieval.dense_top_k == 40  # 确认向量召回条数等于40。


def test_env_overrides_yaml(tmp_path, monkeypatch):  # 验证环境变量优先于文件配置。
    (tmp_path / "config.yaml").write_text(  # 将内容写入临时目录下的“config.yaml”。
        "llm:\n  api_key: from-yaml\n  model: from-yaml\n",  # 在文件中提供模型与测试密钥，供环境覆盖用例对照。
        encoding="utf-8",  # 传入编码参数。
    )  # 结束当前多行表达式。
    monkeypatch.setenv("RAG__LLM__API_KEY", "from-env")  # 为本次用例临时设置环境变量。
    monkeypatch.setenv("RAG__RETRIEVAL__RRF_K", "77")  # 为本次用例临时设置环境变量。

    settings = load_settings(root=tmp_path)  # 将配置设为加载配置的结果。

    assert settings.llm.api_key == "from-env"  # 确认接口密钥等于“from-env”。
    assert settings.retrieval.rrf_k == 77  # 确认倒数排名融合常数等于77。
    # 环境未覆盖模型名称，因此应取文件中配置的模型。
    assert settings.llm.model == "from-yaml"  # 确认模型等于“from-yaml”。


def test_env_coerces_types(tmp_path, monkeypatch):  # 验证环境字符串能转成目标字段类型。
    monkeypatch.setenv("RAG__CHUNKING__SEMANTIC_ENABLED", "false")  # 为本次用例临时设置环境变量。
    monkeypatch.setenv("RAG__CHUNKING__PARENT_MAX_CHARS", "1500")  # 为本次用例临时设置环境变量。
    monkeypatch.setenv("RAG__LLM__TEMPERATURE", "0.9")  # 为本次用例临时设置环境变量。

    settings = load_settings(root=tmp_path)  # 将配置设为加载配置的结果。

    assert settings.chunking.semantic_enabled is False  # 确认语义切分开关为假。
    assert settings.chunking.parent_max_chars == 1500  # 确认父块字符上限等于1500。
    assert settings.llm.temperature == pytest.approx(0.9)  # 确认采样温度等于approx的结果。


def test_env_ignored_when_disabled(tmp_path, monkeypatch):  # 验证关闭环境覆盖后忽略环境变量。
    monkeypatch.setenv("RAG__LLM__MODEL", "from-env")  # 为本次用例临时设置环境变量。

    settings = load_settings(root=tmp_path, use_env=False)  # 将配置设为加载配置的结果。

    assert settings.llm.model == "claude-sonnet-5"  # 确认模型等于“claude-sonnet-5”。


def test_invalid_yaml_reports_configuration_error(tmp_path):  # 验证文件语法错误映射为配置异常。
    (tmp_path / "config.yaml").write_text("llm: [unclosed\n", encoding="utf-8")  # 将内容写入临时目录下的“config.yaml”。

    with pytest.raises(ConfigurationError):  # 验证下方操作会抛出配置校验异常。
        load_settings(root=tmp_path, use_env=False)  # 加载配置。


def test_invalid_types_reports_configuration_error(tmp_path):  # 验证字段类型错误映射为配置异常。
    (tmp_path / "config.yaml").write_text(  # 将内容写入临时目录下的“config.yaml”。
        "retrieval:\n  rrf_k: not-a-number\n", encoding="utf-8"  # 用非数字字符串模拟无效的整数字段。
    )  # 结束当前多行表达式。

    with pytest.raises(ConfigurationError):  # 验证下方操作会抛出配置校验异常。
        load_settings(root=tmp_path, use_env=False)  # 加载配置。


def test_redacted_hides_secrets(tmp_path):  # 验证脱敏输出隐藏密钥原文。
    (tmp_path / "config.yaml").write_text(  # 将内容写入临时目录下的“config.yaml”。
        "llm:\n  api_key: sk-super-secret\nembedding:\n  api_key: sk-other\n",  # 为两个服务填入测试凭据以核对脱敏输出。
        encoding="utf-8",  # 传入编码参数。
    )  # 结束当前多行表达式。
    settings = load_settings(root=tmp_path, use_env=False)  # 将配置设为加载配置的结果。

    view = redacted(settings)  # 将脱敏副本设为配置脱敏的结果。
    blob = repr(view)  # 将合并文本设为repr的结果。

    assert "sk-super-secret" not in blob  # 确认“sk-super-secret”未包含在合并文本。
    assert "sk-other" not in blob  # 确认“sk-other”未包含在合并文本。
    assert view["llm"]["api_key"] == "***set***"  # 确认脱敏副本中的生成模型中的接口密钥等于“***set***”。


def test_missing_secrets_names_only_absent_keys(tmp_path):  # 验证缺失凭据报告只列出未设置的键名。
    (tmp_path / "config.yaml").write_text(  # 将内容写入临时目录下的“config.yaml”。
        "llm:\n  api_key: present\n", encoding="utf-8"  # 只设置生成服务凭据，检验缺失字段报告。
    )  # 结束当前多行表达式。
    settings = load_settings(root=tmp_path, use_env=False)  # 将配置设为加载配置的结果。

    absent = missing_secrets(settings)  # 将缺失项设为列出缺失凭据的结果。

    assert "llm" not in absent  # 确认“llm”未包含在缺失项。
    assert absent["embedding"] == ["api_key"]  # 确认缺失项中的向量等于列表。
    assert absent["rerank"] == ["api_key"]  # 确认缺失项中的重排等于列表。
    # 本地向量库可以省略认证令牌。
    assert "milvus" not in absent  # 确认“milvus”未包含在缺失项。