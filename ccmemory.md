# 项目上下文（给新对话用）

## 项目背景
本科毕设：基于LLM与大数据技术的NLP领域论文检索与分析系统
- 技术栈：ChromaDB、BGE-M3、Reranker、Pandas、Agno Agent
- 问题：代码过度工程化（用 vibe coding 完成，不符合本科水平）
- 目标：简化代码，保持功能，准备中期检查（2天后）

## 已完成的简化工作

### 1. 检索系统（-57% 代码）
- 新建 `src/search/` 模块（260行）：三种检索模式（semantic/metadata/hybrid）
- 删除 `agent/utils/search_backend.py`（重复实现）
- 备份 `retrieval/` → `retrieval_backup/`
- 新建 `retrieval/__init__.py` 桥接层（供 Agent 用）

### 2. 简化其他代码
- `data_process.py`: 删除手写 n-gram，只保留单词提取
- `updater_simple.py`: 删除备份恢复机制（-50%）

### 3. 测试验证
- ✅ 语义检索正常
- ✅ 混合检索正常
- ✅ 增量更新正常
- ✅ Agent 工作流程保持一致（LLM分析意图 → 提取元数据 → 检索 → LLM整理输出）

## 项目结构
```
src/
├── search/              # 核心检索（简化版）
├── retrieval/           # 桥接层（供Agent用）
├── retrieval_backup/    # 备份（复杂版本）
├── analysis/query.py    # 早期版本（保留作为证据）
├── agent/brain.py       # Agent主逻辑（未改动）
├── crawler/updater_simple.py  # 简化版更新脚本
```

## 关键文件路径
- 元数据：`src/analysis/cleaned_papers.parquet/`（分区目录）
- 向量库：`src/chroma_db/`
- 数据库：`data/arxiv_papers.db`

## 中期检查说辞
"最初在 `analysis/query.py` 实现基础检索，后来重构到 `search/` 模块支持多种模式。Agent 通过桥接层调用。探索过程中尝试了多种方案（`retrieval_backup/`），最终选择了简洁实现。"

## 待办（明天）
1. 应对中期检查
