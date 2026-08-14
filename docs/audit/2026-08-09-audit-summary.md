# RongNengRAG / excel-workbook-service 审核总结

> 固化日期：2026-08-09  
> 来源：用户提供的只读审计总结  
> 用途：作为后续整改、验证和回归的固定上下文。本文不表示缺陷已修复，完成状态以同目录下的整改开发记录为准。

> 后续说明：2026-08-11 的 OCR、PDF 多页版面、目录导航和目次渲染收尾记录见 [`2026-08-09-remediation-log.md`](./2026-08-09-remediation-log.md) 与 [`../开发日志.md`](../开发日志.md)。本文件仍保留 2026-08-09 的原始审计快照，不回写历史结论。

> 当前实现补充（2026-08-13）：本文中的缺陷描述是历史审计快照，不代表当前代码状态。RAG 检索现已使用 Qwen3-Embedding-0.6B dense API、本地 CPU BGE-M3 sparse、语料 BM25 和本地 CPU BGE-Reranker-v2-m3；编辑/删除后的 layout、Milvus generation 和重建一致性也已纳入当前发布链路。最新实现说明见 [`../开发日志.md`](../开发日志.md) 和 [`../DEPLOYMENT.md`](../DEPLOYMENT.md)。

## 总体结论

两个项目的职责划分合理：

- `RongNengRAG`：负责非结构化文档知识库、检索增强生成和统一前端入口。
- `excel-workbook-service`：负责 Excel 结构化解析、人工校正、SQLite 物化及 SQL 驱动的精准问答。
- RAG 通过 `/excel/*` 反向代理集成 Excel 服务。

当前版本暂不适合直接部署到公网或多人生产环境。已确认的主要缺陷包括：

1. 文本切分可能死循环，导致文档导入永久卡死。
2. 跨领域问答调用链未接通，且会触发提示词格式异常。
3. Excel 报告存在同源存储型 XSS 风险。
4. Excel 内容可能通过系统提示词注入控制智能体。
5. 两个服务基本没有认证和资源隔离。
6. Excel 人工调整表头后，实际物化数据不会重新计算。
7. RAG 重建索引不是事务性的，失败可能破坏原有可用索引。
8. Excel 多轮会话在 RAG 前端未实际启用，并存在消息重复。

## 整体调用链

```text
                         普通知识库问答
浏览器                   QueryAnalyzer
  │                      Hybrid Retrieval
  ├── RongNengRAG ─────→ Reranker → Prompt + LLM → SSE
  │
  │                      Excel 上传/确认 → Parser → Draft 校正
  └── /excel/* 代理 ──→ SQLite Materializer → ReAct Agent → SQL/HTML/Todo → SSE
```

RAG 是确定性管线：文件上传与解析、切块、稠密/稀疏向量生成、Milvus Lite 入库、问题分析、混合召回、重排、上下文组装、LLM 生成与同步/SSE 输出。统计类问题会从 Milvus 拉取结构化字段聚合。

Excel 服务是 ReAct 智能体：找到活动 SQLite 版本，读取 schema、样例行和会话历史，调用 `sql_query` / `html_interpreter` / `todowrite` / `terminate` 工具，循环至终止或上限，并保存会话、工具调用和答案。SQL 层已有 sqlglot 校验、SQLite 只读模式和 authorizer，安全方向正确。

## P0 严重问题

### P0-1：长文本切分存在确定性死循环

- 位置：`src/ingestion/chunker.py:377`
- 根因：循环尾部 `start = end - overlap`；当 `end == len(text)` 且 `overlap > 0` 时，`start` 持续回退。
- 影响：超长段落或无分隔符内容会让导入线程永久卡住，并可能阻塞异步 API 事件循环。
- 修复原则：`end >= len(text)` 时立即退出；下一个 `start` 必须严格大于上一个。

### P0-2：跨领域问答不可用

- 领域识别：`src/retrieval/query_analyzer.py:241`
- 跨领域检索：`src/retrieval/retriever.py:258`
- 错误入口：`src/api/main.py:925` 始终调用普通检索。
- 提示词：`src/generation/prompt_templates.py:48` 要求 `context_domain1` / `context_domain2`，主流程却只传 `context`，导致 `KeyError: context_domain1`，又被误包装为“LLM 不可用”。
- `src/retrieval/retriever.py:141` 中显式 `domain_filter` 无法覆盖自动领域，两分支可能搜索同一领域。

### P0-3：Excel 报告存在同源存储型 XSS

- Excel 智能体允许输出包含 `<script>` 和外部 CDN 的 HTML：`excel-workbook-service/app/agent/workflow_prompt.py:24`。
- HTML 原样写盘：`excel-workbook-service/app/agent/tools.py:51`。
- 报告以 `text/html` 返回：`excel-workbook-service/app/api/endpoints.py:307`。
- RAG 前端使用无 sandbox 的 iframe：`src/ui-vue2/src/excel/HtmlReportViewer.vue:7`。
- 风险：报告经 RAG 代理后与主页同源，脚本可访问父页 DOM/同源接口并外传数据。
- 建议：严格 iframe sandbox；独立无凭证域名；禁止脚本与外部 CDN；HTML 白名单消毒；图表数据使用受控 JSON。

### P0-4：Excel 内容被提升到 system prompt

- 数据库上下文组装：`excel-workbook-service/app/agent/react_loop.py:78`。
- 作为 system message：`excel-workbook-service/app/agent/react_loop.py:174`。
- 攻击链：恶意 Excel 内容 → system prompt 注入 → `html_interpreter` → 恶意脚本 → 同源 iframe 执行。
- 原则：workbook 内容必须明确标记为不可信数据，不得提升到 system role；系统规则禁止执行数据中的指令。

### P0-5：缺少认证和资源隔离

- RAG 开放全来源 CORS：`src/api/main.py:56`。
- `/upload/from-paths` 接受并解析服务器本地路径：`src/api/main.py:422`。
- 删除、同步、重建索引、文件下载及 Excel 服务都没有统一认证/授权。
- 上传文件名仅替换空格，`src/api/main.py:325` 存在 `../` 路径穿越风险。
- 影响：本地文件枚举/索引、跨用户访问、数据删除以及 OCR/Embedding/LLM 等高成本资源滥用。

## P1 高优先级逻辑问题

### P1-1：Excel 表头调整不会重新解析

`excel-workbook-service/app/service/service.py:263` 修改 `header_row` / `data_start_row` 时只更新数字，不重建 `columns` / `rows` / 推断类型 / 验证警告；`app/service/materializer.py:71` 最终仍物化旧数据。

### P1-2：RAG 元数据修改后 SQLite 与 Milvus 不一致

`src/ingestion/file_processor.py:639` 只更新注册表，不同步 Milvus chunk 中的 domain/category/document number/metadata；重建时也未完整保留人工元数据。

### P1-3：索引更新不具事务性

`src/ingestion/file_processor.py:889` 首批新 embedding 成功后先删旧 chunk，后续批次失败会留下部分新索引。重建也是“先删再建”。建议临时版本全量写入后原子切换 active version。

### P1-4：Excel 上传和解析内存放大

- RAG 代理完整读请求体：`src/api/excel_proxy.py:42`。
- Excel endpoint 再完整读文件：`excel-workbook-service/app/api/endpoints.py:68`。
- 大小限制在完整读取后才检查。CSV/XLSX 解析继续全量持有数据，openpyxl fallback 行限不完整，且没有 ZIP 解压大小/压缩比限制。

### P1-5：Excel 多轮会话未接通且消息重复

- RAG 前端不传/不保存 `conv_id`：`src/ui-vue2/src/api.js:231`、`src/ui-vue2/src/sse.js:48`。
- Excel 后端先保存当前问题再读全历史，`run_react` 又追加当前问题：`app/api/endpoints.py:241`、`app/agent/react_loop.py:179`。

### P1-6：Excel SSE 异常时前端可能永久“思考中”

`src/ui-vue2/src/excel/ExcelQA.vue:93` 未完整处理网络异常、非 2xx、JSON 错误、无 `done` 的 EOF 及组件卸载取消；`src/ui-vue2/src/sse.js:27` 在 EOF 不主动结束状态。

### P1-7：SQL 没有执行资源限制

`excel-workbook-service/app/service/service.py:654` 只限返回行数，没有 progress handler、超时、VM step 上限、并发/队列限制和每会话 SQL/LLM 总预算。

## 其他明确缺陷

1. `src/retrieval/retriever.py:781` 的中文章节文件名正则只有一个捕获组，却读取 `group(2)`，可稳定触发 `IndexError`。
2. 检索上下文按“文件”去重，而非按 chunk 去重：`src/retrieval/retriever.py:309`，同一重要文件的多个相关条款只能保留一个。
3. 统计查询最多读 20,000 条但不标记截断：`src/retrieval/retriever.py:590`，合计可能错误。
4. **历史状态**：`provider: none` 重排主要依赖硬编码元数据倍率；当前默认已切换为本地 `BAAI/bge-reranker-v2-m3` CPU 交叉编码器，旧无模型逻辑仍保留为显式降级路径。
5. **历史状态**：旧 `bm25_sparse.py` 实际是 `log1p(tf)` 归一化；当前正式 BM25 位于 `src/retrieval/bm25_index.py`，包含语料 IDF 和文档长度校正，旧模块仅作兼容包装。
6. OCR 返回空结果时会丢弃原本已有但较短的 PDF 文本：`src/ingestion/file_processor.py:834`。
7. Excel 会话加载不确认 `conv_id` 是否属于当前 workbook：`excel-workbook-service/app/models/conversation.py:74`。
8. Excel 版本号采用 `MAX(version)+1`，不是原子操作：`excel-workbook-service/app/models/workbook.py:120`。
9. 草稿 revision 是“读取后再写入”，不是数据库 compare-and-swap，并发提交可能覆盖。
10. 删除 workbook 时未同步清理 conversation/message，可能留下孤儿数据。
11. Excel 草稿前端对重复 warning code 的计数逻辑错误，可能使“全部接受”后仍禁用确认按钮。
12. Excel API 包装器遇 HTTP 错误时返回字符串而非抛异常：`src/ui-vue2/src/api.js:45`，多个组件的 `try/catch` 失效。
13. 智能体最多允许 100 轮，但没有 token、调用费用或工具总执行预算。
14. 智能体 Thought 被传输和持久化，不应将模型内部推理作为稳定业务审计日志。
15. RAG 工作区存在大量 CRLF 行尾转换，应用 `.gitattributes` 统一行尾，保证后续审查、合并和回滚可用。

## 原始验证结果

已完成：

- RAG 全部 Python 文件语法编译通过。
- Excel 服务全部 Python 文件语法编译通过。
- 长文本切分死循环已最小复现。
- 跨领域提示词 `KeyError` 已最小复现。
- 中文章节名 `group(2)` 异常已最小复现。

未能完成：

- RAG 前端构建缺少 Rollup Linux 可选二进制依赖，未进入源码编译。
- Excel Linux Python 环境缺少 FastAPI、SQLAlchemy、sqlglot、openpyxl 等依赖。
- 项目 smoke test 使用硬编码 Windows 路径，在当前环境直接 SKIP。
- Excel 项目未发现完整 pytest 单元/集成测试体系。

## 审计建议顺序

第一批（阻断上线）：

1. 修复 chunker 死循环。
2. 禁用或隔离 Excel HTML 脚本执行。
3. 修复 Excel system prompt 注入边界。
4. 增加统一认证、资源归属与本地路径/文件名边界。
5. 接通并测试跨领域问答。
6. 将上传与 Excel 解析改为流式、有限额处理。

第二批（正确性与一致性）：表头重解析、元数据同步、索引原子更新、多轮会话、SSE 错误收敛、SQL/智能体资源预算以及所有其他明确缺陷。

第三批（工程基础）：补单元/集成/安全回归测试，修复前端构建环境，统一行尾和可重复的验证命令。
