# Contributing to NanoLM

感谢你对 NanoLM 的关注！我们欢迎所有形式的贡献。

## 贡献方式

- **Bug 报告** — 通过 [GitHub Issues](https://github.com/jeurtr/nanolm/issues) 提交，请附上环境信息和复现步骤
- **功能建议** — 在 Issues 中描述需求和场景
- **代码贡献** — Fork 仓库后通过 Pull Request 提交

## 开发环境

```bash
git clone https://github.com/jeurtr/nanolm.git
cd NanoLM
pip install -e ".[dev]"
```

## 代码风格

- Python 3.10+，遵循项目中已有的代码惯例
- Lint: `ruff check .`
- 类型检查: `mypy nanolm/`
- 测试: `pytest tests/ -v`

CI 会检查以上三项，PR 提交前请确保全部通过。

## Pull Request 流程

1. Fork 本仓库
2. 创建 feature 分支 (`git checkout -b feat/my-feature`)
3. 提交变更，遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范
4. 确保 CI 通过
5. 提交 PR，附上清晰的变更说明

## Commit 规范

```
feat: 新功能
fix: 修 Bug
docs: 文档变更
refactor: 重构
test: 测试
chore: 构建 / 工具链
```
