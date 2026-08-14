# 审计整改开发记录

> 对应审核上下文：[`2026-08-09-audit-summary.md`](./2026-08-09-audit-summary.md)  
> 执行规则：每次只将已实施且已审核验证的具体点从 `[ ]` 改为 `[x]`。未通过验证、仅有部分防护或受环境阻塞的项目不打勾。  
> 范围：`RongNengRAG` 与相邻的 `excel-workbook-service`。

> 当前检索实现（2026-08-13）：RAG 默认保留 Qwen3-Embedding-0.6B dense API，使用本地 CPU BGE-M3 sparse、语料 BM25 和本地 CPU BGE-Reranker-v2-m3。D-04/D-05 的早期整改记录仍保留其历史语境；当前正式实现详见 `src/retrieval/bm25_index.py`、`src/ingestion/embedder.py` 与 `config.yaml`。

## 基线与验收约定

- [x] B-01 记录两个项目的未提交改动、运行时与可用测试，确保不覆盖既有工作。
- [x] B-02 建立可重复的后端语法/单测与前端构建验证入口。
- [x] B-03 增加 `.gitattributes` 行尾策略，验证审查 diff 不再被 CRLF 噪声淹没。

## P0 阻断项

- [x] P0-01 修复长文本切分死循环；覆盖长段落、无分隔符、边界长度与 overlap 进度性测试。
- [x] P0-02 完成 Excel HTML 安全链：模型不得生成脚本/外部资源，服务端白名单消毒，前端 iframe 严格 sandbox，添加 XSS 回归测试。
- [x] P0-03 修复 workbook prompt 注入边界：不可信数据移出 system role，明确数据边界/长度与禁止执行数据指令，增加注入回归测试。
- [x] P0-04 为 RAG 和 Excel 接口建立默认安全的统一认证、CORS 白名单和 workbook/conversation 资源归属校验。
- [x] P0-05 关闭或强约束 `/upload/from-paths`；上传文件名执行 basename/规范化与最终目录边界校验。
- [x] P0-06 接通跨领域检索主链路，显式领域能覆盖自动领域，正确填充双上下文提示词并保留真实错误语义。

## P1 高优先级项

- [x] P1-01 表头行/数据起始行调整后重新解析 columns、rows、类型和 warnings，确认物化使用新数据。
- [x] P1-02 元数据修改原子同步文件注册表和 Milvus chunk，重建索引保留人工元数据。
- [x] P1-03 将文件索引及重建路径改为分期写入、完整验证后切换，失败时旧索引仍可用且无部分新 chunk 泄漏。
- [x] P1-04 RAG Excel 代理和 Excel 上传端点采用流式限额，限制 CSV/XLSX 行列、解压后大小、压缩比、合并单元格和并发任务。
- [x] P1-05 RAG 前端传递并保存 `conv_id`，消费 SSE `done` 中的会话 ID；Excel 后端不重复追加当前问题。
- [x] P1-06 Excel QA 完整处理网络/非 2xx/JSON 错误、无 `done` EOF、用户取消和组件卸载，状态始终收敛。
- [x] P1-07 SQL 执行增加 progress handler、超时/VM step 上限和并发限额；会话增加 SQL/LLM/工具/token 总预算。

## 其他明确缺陷

- [x] D-01 修复中文章节文件名正则捕获组 `IndexError`，补命名边界测试。
- [x] D-02 检索结果按 chunk 唯一性去重，允许同文件多个相关条款入选，并设置可控单文件上限。
- [x] D-03 统计聚合不再静默截断 20,000 条；要么全量分页，要么在结果中明确标记不完整。
- [x] D-04 `provider: none` 重排真正使用阈值、boost 和 `rrf_k` 等配置，增加排序决策测试。
- [x] D-05 实现符合命名的 BM25（IDF 与文档长度校正），或重命名并明确算法语义；添加可解释的排序用例。
- [x] D-06 OCR 空结果时保留原 PDF 文本，补空/异常/有效 OCR 三类测试。
- [x] D-07 Excel 加载会话时校验 `conv_id` 属于当前 workbook 及当前身份。
- [x] D-08 Excel 版本号改为数据库保证的原子分配，并发创建无重号/覆盖。
- [x] D-09 草稿 revision 使用条件 UPDATE/CAS，并发修改准确报冲突。
- [x] D-10 删除 workbook 时级联清理 conversation/message/tool call/report 等所有所属资源。
- [x] D-11 修复草稿前端重复 warning code 计数，“全部接受”后确认按钮正确启用。
- [x] D-12 Excel API 包装器对 HTTP 错误抛出带 status/body 的异常，组件 `try/catch` 可正确收敛。
- [x] D-13 不向前端传输或在业务表持久化模型 Thought，只保留结构化工具事件与可审计结果。
- [x] D-14 兼容本地 Qwen 多行/不完整 ReAct JSON；`html_interpreter` 缺少 `html` 时基于最近 SQL 结果安全回退生成报告。
- [x] D-15 为本地 Qwen 启用结构化报告模式：模型只返回 `report_spec`，由后端统一生成转义 HTML；DeepSeek 保持原 `html_interpreter` 链路。
- [x] D-16 Qwen 缺少 `terminate.result` 时，根据最近一次 SQL 结果生成公开答案，避免仅返回“模型未提供最终答案”。
- [x] D-17 独立 Excel UI 兼容 `report_spec` 步骤：将后端生成的 HTML 报告按旧 `html_interpreter` 结果渲染，避免页面停留在“等待报告”。
- [x] D-18 修复 RAG 报告中文乱码：报告响应明确返回 `text/html; charset=utf-8`，代理补齐字符集，安全清洗器保留安全的 UTF-8 meta 声明。
- [x] D-19 区分 8001 PaddleOCR pipeline 与 8080 裸 OpenAI OCR：RAG 支持 pipeline 的 multipart `POST /ocr` 响应解析，并将 OCR 配置切换到 8001。
- [x] D-20 在现有 DocumentPreview 中增加 OCR 双引擎对比面板，展示 8080 原始文本与 8001 结构化表格，使用 sandbox iframe 预览 pipeline HTML。
- [x] D-21 修复 Milvus Lite Windows gRPC `too_many_pings`：降低客户端 keepalive 频率并禁止空闲无 RPC ping；已完成 Windows 启动、列表和检索回归。
- [x] D-22 修复独立 PNG/JPEG 上传解析：复用 8001 PaddleOCR pipeline 的图片 OCR，不再将图片落入“其他”分支导致无有效文本。
- [x] D-23 兼容 8001 布局块中的空排序字段：`block_order`/`block_id` 为 `null` 时使用稳定的末尾排序值，不得中断整张图片 OCR。
- [x] D-24 修复 Milvus 分期索引快照查询 released 集合：查询迭代器和兼容查询前显式 `load_collection()`，避免 code 101。
- [x] D-25 修复知识库前端重解析成功状态：兼容后端 `success/status` 响应，成功后立即更新行状态并等待列表刷新，异常时正确收敛 loading。
- [x] D-26 修复 Milvus 影子索引删除后的旧快照误判：快照查询请求 Strong consistency，并兼容不支持该参数的旧客户端。
- [x] D-27 OCR 预览渲染 HTML 表格：中心预览使用受限 `v-html` 渲染表格并移除脚本/事件属性；版面预览保留表格结构。
- [x] D-28 OCR 结果归一化：8001 解析阶段优先使用块坐标排序、回退 `block_order`，并按规范化内容指纹去重。
- [x] D-29 清洗 OCR 残缺 HTML：保留完整 `<table>...</table>`，删除表格外孤立 `<td>/<tr>` 片段，避免拼接出 `415>50...` 等不可渲染文本。
- [x] D-30 公式识别与渲染：OCR 提示词要求 LaTeX 公式格式；DocumentPreview 通过 MathJax 渲染 `$...$`/`$$...$$`，公式和序号保持居中对齐。
- [x] D-31 保留 pipeline 版面预览：OCR 对比接口返回 `parsing_res_list` 块元数据，并生成按 bbox 定位的 sandbox HTML；完成用户图片版面回归。
- [x] D-32 默认版面预览：打开文档时自动请求 8001 pipeline 版面 HTML，成功时直接展示；8080 仅在手动“OCR 对比”时显示，失败回退已入库文本。

## 2026-08-11 文档预览收尾

- [x] D-33 PDF 多页版面：每页独立渲染并返回 `layout_pages`，bbox 不跨页混用；OCR 对比覆盖全部 PDF 页面。
- [x] D-34 预览视觉收敛：页面按可用宽度缩放，iframe 内部取消滚动条，由外层统一滚动，页间分隔收窄并避免右侧编辑栏遮挡。
- [x] D-35 章节目录：结合 OCR 标签、编号、文字特征和 bbox 几何提取多级标题，优先章节导航，无法识别时回退页级导航。
- [x] D-36 目次/Contents 格式：按“标题、点引导线、页码”三列重排，支持中文/英文和 `1`、`1.1` 层级缩进；兼容 OCR 误标为普通文本的目次块。
- [x] D-37 文档与日志：同步 README、部署指南、图片/PDF 链路文档和本轮开发日志，保留原始审计总结为历史快照。
- [x] D-38 文件下载格式：根据原始扩展名返回 MIME 类型，保留服务端文件名和扩展名，修复 hash 下载及 Blob 文件名回退。

## 2026-08-13 检索模型升级

- [x] D-39 BGE-M3 稀疏向量：保留 Qwen dense API，新增本地 CPU `BAAI/bge-m3` lexical weights，并迁移既有 282 个 chunks 的 sparse vector，验证 dense/文本/元数据不变。
- [x] D-40 真正 BM25：新增语料级 IDF 和文档长度归一化索引，补充标准号、文件名和关键词召回；旧 `ingestion/bm25_sparse.py` 明确降级为兼容包装。
- [x] D-41 本地 CPU 重排：启用 `BAAI/bge-reranker-v2-m3`，补充 FlagEmbedding/Transformers 兼容处理，验证 CPU 可返回重排分数。
- [x] D-42 文档同步：README、部署指南、PDF/PNG 链路、开发日志和审计状态说明同步当前检索链路与旧库迁移方式。

## 最终验收

- [x] A-01 两项目所有 Python 文件语法编译通过。
- [x] A-02 新增与既有后端单元/集成/安全测试全部通过。
- [x] A-03 RAG 前端生产构建通过，与 Excel 相关的交互回归通过。
- [x] A-04 对照审核总结复核无遗漏；所有勾选项都有可重复验证证据。
- [x] A-05 PDF/PNG 预览收尾：OCR 协议与布局测试通过，文档链路和已知限制已记录。

## 逐项执行记录

| 时间 | 整改点 | 实施摘要 | 审核/验证证据 | 结果 |
|---|---|---|---|---|
| 2026-08-09 | 文档初始化 | 固化审核总结，将明确缺陷拆成可独立验收检查项 | 审核总结和本记录已写入 `docs/audit/` | 完成 |
| 2026-08-09 | B-01 基线 | RAG 为 Git 工作树，已有逻辑 diff 约 +465/-50 且有大量 CRLF 噪声；Excel 目录不是 Git 工作树且无测试目录 | Python 3.10.12，Node 22.23.2，npm 10.9.8；已保留既有改动 | 完成 |
| 2026-08-09 | B-02 验证入口 | 新增 `scripts/verify_remediation.sh`，统一执行两项目 Python 语法/发现测试及前端逻辑、全量 SFC、生产构建；支持覆盖 Excel 路径和两个 Python 解释器；npm 增加 `test:remediation`/`verify` | shell 语法通过；3 组 Node 回归与 8 个 SFC 由 npm 入口执行通过；完整入口留在 A-01～A-03 最终执行 | 完成 |
| 2026-08-09 | B-03 行尾治理 | `.gitattributes` 固定文本 LF、Windows 脚本 CRLF并声明二进制；仅机械规范化所有已追踪文本，不改逻辑内容 | `git check-attr` 文本/二进制策略正确；普通 diff 收敛到真实 `+1420/-360`；`git diff --check` 无告警 | 完成 |
| 2026-08-09 | P0-01 切块进度性 | 末块消费完全文后立即退出；将 overlap 夹到 `[0, max_chars-1]`，并增加严格前进安全阀 | `python3 -m unittest -v tests.test_chunker`：5/5 通过；`py_compile` 通过 | 完成 |
| 2026-08-09 | P0-02 HTML/XSS | 提示词禁止活动内容；服务端仅持久被动 HTML 白名单；报告响应增加 CSP sandbox/nosniff/referrer-policy；iframe 使用空 sandbox，新窗口使用 noopener | Excel XSS/工具集成测试 4/4 通过；相关 Python 无落盘语法编译通过；`HtmlReportViewer.vue` SFC 脚本与模板编译通过。完整 Vite 构建仍受 Rollup 环境依赖阻塞，保留到 A-03 | 完成 |
| 2026-08-09 | P0-03 Prompt 注入 | system message 改为完全固定策略；schema/样例值/SQL 观测作为独立不可信 user 数据块；对分隔符进行 Unicode 转义并增加表/列/单元格/总长度限额 | 恶意指令角色隔离、分隔符逃逸、上下文截断测试通过；与 P0-02 合计 7/7 测试通过；`py_compile` 通过 | 完成 |
| 2026-08-09 | P0-04 认证/归属 | 两服务共用 Bearer/`X-API-Key` 协议，默认失败关闭，密钥常量时间校验；RAG CORS 改显式白名单并向 Excel 透传身份；Excel 增加 owner 迁移与 import/workbook/conversation/report 归属校验；报告使用认证 fetch + blob sandbox，不在 URL 泄漏 key | Excel 13/13 安全/迁移/跨身份测试通过；RAG 8/8 测试通过；两项目相关 Python 语法通过；报告组件 SFC 编译通过 | 完成 |
| 2026-08-09 | P0-05 路径边界 | 上传文件名执行 NFKC、Windows/POSIX basename、控制字符拒绝和最终 `resolve` 目录边界校验；本地路径导入默认关闭，仅允许绝对白名单根内真实文件，符号链接逃逸被拒绝 | 路径穿越、Windows 路径、NUL、禁用状态、相对根、越界与符号链接测试通过；RAG 当前 13/13 相关测试及语法编译通过 | 完成 |
| 2026-08-09 | P0-06 跨域问答 | 主问答按分析结果调用双域检索；显式域无条件覆盖自动域；两域结果独立格式化后传入同步/SSE/多轮提示词；缺资料有显式占位，异常保留真实类型 | 显式域覆盖、双上下文填充、缺失第二域三类测试 3/3 通过；RAG 累计 16/16 相关测试通过；调用链语法编译通过 | 完成 |
| 2026-08-09 | P1-01 表头重解析 | 表头或数据起始行变化时从不可变源文件重解析受影响工作表，重新生成列、数据行、类型、元数据和校验结果，再应用同一请求中的人工列/单元格调整 | CSV 端到端测试验证新列名、类型、源行号及 SQLite 物化值；Excel 当前安全与回归测试 14/14 通过；相关语法编译通过 | 完成 |
| 2026-08-09 | P1-02 元数据一致性 | 以 SQLite 写锁串行化编辑；完整快照、upsert 并逐行验证所有 Milvus chunk 后提交注册表，任一失败补偿恢复向量快照；重建继承已审核元数据且显式请求优先 | 同步成功、计数不符阻断、注册表失败回滚、Milvus 验证失败自动回滚、字段边界及重建继承共 8/8 专项测试通过；RAG 当前相关回归 23/23 通过；语法编译通过 | 完成 |
| 2026-08-09 | P1-03 索引分期切换 | 既有集合迁移到稳定活动别名；每次入库/重建先克隆到不可见物理代、仅在全部批次与文件/集合计数验证后原子切换别名；失败丢弃影子代，注册表失败则切回旧代；重建期间保留 completed 注册记录 | 部分写入不可见、分期计数失败、切换后注册表失败回切、成功提交顺序等 5 项失败注入/代际测试通过；RAG `unittest discover` 29/29 通过；相关语法编译通过 | 完成 |
| 2026-08-09 | P1-04 流式上传/解析限额 | RAG 网关用受限异步流直通请求/响应；Excel 分块落私有临时文件、增量哈希并流式复制，超限清理；CSV 顺序读取，XLSX 先检验 ZIP 展开量/压缩比/条目/XML，再限制行、列、网格单元格、单元文本、总文本和合并展开量；解析并发满时返回 429 | 分块透传/中止、临时文件清理、文件复制、CSV 各限额、ZIP 炸弹、合并区域、并发拒绝 9 项专项测试通过；Excel 21/21、RAG 31/31 发现测试通过；相关语法编译通过 | 完成 |
| 2026-08-09 | P1-05 Excel 多轮会话 | 后端在写入当前用户消息前截取历史，`run_react` 仅追加一次当前问题；SSE 先发 session ID 并在 done/error 携带；前端保存并在下一问传回 `conv_id`，切换 workbook 时取消旧流并清空会话 | 历史截取顺序和模型上下文单次问题 2/2 测试通过；Excel 23/23 发现测试通过；Node 会话请求/done 解析测试及 `ExcelQA.vue` SFC 编译通过 | 完成 |
| 2026-08-09 | P1-06 SSE 状态收敛 | 前端校验 HTTP 状态和 `text/event-stream`，typed/legacy 错误只结算一次，无 done EOF 必然报错；fetch/read 均捕获；提供取消按钮并在 workbook 切换/组件卸载时 abort + cancel | Node 覆盖 done、非 2xx JSON、200 JSON、无 done EOF、typed error 单次回调和主动取消；`ExcelQA.vue` SFC 编译通过；RAG 31/31 发现测试通过 | 完成 |
| 2026-08-09 | P1-07 SQL/智能体预算 | SQLite authorizer 之外增加 progress handler、墙钟/VM step 限额、返回行封顶和并发信号量；ReAct 限制单轮及累计会话的 LLM/工具/SQL 调用、prompt、估算 token、总时长与并发，并持久化累计用量 | 快查询/截断、VM 中断、超时中断、SQL 并发、各预算硬限、超预算停止、累计预算继承和智能体并发共 10/10 专项测试通过；Excel 31/31 发现测试通过；语法编译通过 | 完成 |
| 2026-08-09 | D-01 章节系列正则 | 将系列提取拆成纯函数；匹配中文/数字序号与“章节/章/节/部分/篇”，以单位模板归一化，不再读取不存在的 group(2) | 中文/数字序号、不同单位、会议材料和无匹配 3/3 测试通过；相关语法编译通过 | 完成 |
| 2026-08-09 | D-02 chunk 去重 | 上下文选择按 chunk ID（缺失时按文件/页/内容）去重，按原相关性顺序继续补足；同文件保留多个条款但受 `retrieval.max_chunks_per_file` 限制 | 同文件多 chunk、重复 ID 后补位、零预算边界 3/3 测试通过；相关语法编译通过 | 完成 |
| 2026-08-09 | D-03 统计全量性 | 移除固定 20,000 limit，使用 `query_iterator` 分页并在线聚合全部匹配记录；迭代不可用/中断时不返回部分合计，格式中标注全量口径 | 25,001 行跨页、消费者提前关闭、旧客户端拒绝部分统计 3/3 测试通过；相关语法编译通过 | 完成 |
| 2026-08-09 | D-04 无模型重排（历史阶段） | 当时 `provider: none` 以召回 distance 为主分，结合 `rrf_k` 名次先验和配置化元数据倍率；当前默认已升级为本地 `BAAI/bge-reranker-v2-m3` CPU 交叉编码器，无模型逻辑仅作降级路径 | 历史无模型排序测试通过；2026-08-13 CPU 重排模型加载和分数计算验证通过 | 完成 |
| 2026-08-09 | D-05 稀疏算法命名（历史阶段） | 当时将无语料统计的算法正式命名为 `hashed_tf`；当前正式 BM25 位于 `src/retrieval/bm25_index.py`，包含 IDF 和文档长度校正，旧 BM25 入口仅作兼容包装 | 历史 hashed-TF 兼容测试通过；2026-08-13 282 个 chunks 的 BGE-M3 sparse 迁移和 BM25 索引构建验证通过 | 完成 |
| 2026-08-09 | D-06 OCR 回退 | 页面文本选择改为“有效 OCR 优先，否则保留 PDF 原文本”；OCR 服务异常降级为空结果，垃圾检测器异常也不删除源文本 | OCR 空、垃圾/检测异常、有效替换、双方为空 4/4 测试通过；相关语法编译通过 | 完成 |
| 2026-08-09 | D-07 会话归属 | conversation 实体增加 owner，DAO 与管理器加载均以 `conv_id + workbook_id + owner_id` 联合约束；问答/详情/删除入口传入当前认证身份 | 错 workbook、错 owner、资源/API 跨身份隔离 4/4 测试通过；相关语法编译通过 | 完成 |
| 2026-08-09 | D-08 原子版本分配 | workbook 行增加 `next_version` 计数器，通过单条 `UPDATE ... RETURNING` 原子预留；启动迁移按历史最大版本回填，并保留 `(workbook_id, version)` 唯一索引 | 24 路并发严格得到 `1..24`；遗留版本 7 后分配为 8；不存在 workbook 拒绝分配；相关 5/5 测试通过 | 完成 |
| 2026-08-09 | D-09 草稿 CAS | 条件 UPDATE 以 `import_id + revision + status` 独占草稿；临时文件/备份原子替换，完成时再 CAS 推进 revision，异常恢复文件与状态 | 同 revision 两路并发仅一方成功、另一方冲突；失败更新保持 revision/状态/文件且无临时残留；相关 6/6 测试通过 | 完成 |
| 2026-08-09 | D-10 删除级联 | 删除事务按 message、conversation、import、dataset 顺序清理关系数据；校验存储路径后删除版本数据库与报告目录 | 删除后 4 张表记录均为 0，workbook 目录不存在；归属回归测试合计 3/3 通过 | 完成 |
| 2026-08-09 | D-11 warning 去重 | 确认进度与完成条件按唯一 warning code 计算；旧 code 不计数，重复项使用唯一渲染 key，并提供“全部接受/取消” | 重复 code、部分接受、全部接受、过期 code、空警告测试通过；`ExcelDraftReview.vue` SFC 编译通过 | 完成 |
| 2026-08-09 | D-12 API 异常 | 通用请求包装器以 `ApiError` reject HTTP、超时和网络故障，保留 status、解析 body、URL 与稳定 code；204 返回 null | 409 JSON、502 文本、204、超时、断网测试通过；现有 Excel 会话/warning 测试及 `node --check` 通过 | 完成 |
| 2026-08-09 | D-13 私有推理边界 | SSE 与工具回调仅保留 action/observation；新工具消息不写 Thought，启动迁移清理历史 content/meta，历史加载再防御性过滤；公开答案过滤 Thought-only 输出 | 两组秘密 Thought 不进入事件/回调；新旧业务记录均无 Thought；多轮与预算回归合计 13/13 通过；相关语法编译通过 | 完成 |
| 2026-08-10 | D-14 本地 Qwen 报告回退 | ReAct 输入解析改为平衡 JSON 提取，仅转义 JSON 字符串内部的真实换行/制表符；HTML 缺失、字段别名或解析失败时使用最后一次 SQL 结果生成转义 HTML 表格，模型内容不作为原始标记执行 | 多行 HTML JSON 解析、缺失 `html` 的 SQL→报告回退、恶意单元格转义 2/2 专项测试通过；相关 Python 语法编译通过 | 完成 |
| 2026-08-10 | D-15 Qwen 结构化报告链路 | 按模型名识别 Qwen；追加结构化报告提示词，要求 SQL 后调用 `report_spec(title/summary/highlights)`；后端规范化字段并统一渲染安全 HTML，若模型直接 terminate 则按最近 SQL 结果兜底；DeepSeek 不启用该模式 | Qwen/DeepSeek 模式识别、结构化报告渲染、恶意内容转义、terminate 兜底 4 项专项测试通过；Excel 全量 45/45；相关 Python 语法编译通过 | 完成 |
| 2026-08-10 | D-16 Qwen 最终答案回退 | 当结构化模式的 `terminate` 缺少 `result`，将最近 SQL 的列名和结果行转换为公开 Markdown 表格；同时保留报告 URL 兜底 | 新增“SQL 后空 terminate 仍返回结果”回归断言；Qwen 专项测试 5/5 通过 | 完成 |
| 2026-08-10 | D-17 独立 UI 报告显示 | 独立示例 UI 原仅按 action 名包含 `html` 才展示报告；现在将 `report_spec` 步骤纳入同一 HTML 预览渲染分支 | 复核 17:32 Qwen 日志存在 `report_spec` 和报告文件 `a120914db89a_计量单元分类统计报告.html`；UI 条件已修正；Qwen 专项测试 5/5 通过 | 完成 |
| 2026-08-10 | D-18 RAG 中文编码 | Excel 报告 FileResponse 明确使用 UTF-8；RAG 代理对缺失 charset 的 HTML 响应补 `charset=utf-8`；报告清洗器允许 `meta charset` | HTML 清洗与 Qwen 报告回归 9/9 通过；Python 语法编译通过 | 完成 |
| 2026-08-10 | D-19 OCR 双接口协议 | 实测 8001 为 FastAPI pipeline（`POST /ocr` multipart，返回 Layout blocks），8080 为 OpenAI-compatible `/v1/models` 与 `/v1/chat/completions`；新增 `protocol: pipeline` 客户端适配，按 `block_order` 合并文本，默认配置指向 8001 | 8001 根接口 200、OpenAPI 明确 `/ocr`；同一测试图片 8001 返回 Layout 结果，8080 `/ocr` 返回 404、8080 chat 返回 OpenAI 结果；新增 pipeline 单元测试 2/2，语法编译通过 | 完成 |
| 2026-08-11 | D-20 OCR 预览对比 | 新增 `/files/{identifier}/ocr-compare` 诊断接口，不写向量库；DocumentPreview 增加“ OCR 对比”按钮，左侧显示 8080 裸文本，右侧 sandbox 预览 8001 HTML 表格 | Vue SFC 8/8 编译通过；后端语法编译通过；使用用户图片对 8001/8080 完成真实识别对照 | 完成 |
| 2026-08-11 | D-21 Milvus Lite gRPC 保活 | `MilvusStore` 通过 `grpc_options` 将 keepalive 调整为 5 分钟、超时 20 秒，禁止无调用保活并限制无数据 ping；初次连接和连接恢复共用配置，不删除现有 `milvus_lite.db` | `py_compile` 与 `git diff --check` 通过；Windows venv 启动、`/files` 列表和检索回归完成 | 完成 |
| 2026-08-11 | D-22 图片上传解析 | `_parse_file` 为 PNG/JPEG/TIFF 等图片调用 `PDFParser.ocr_page`，复用当前 8001 pipeline 并按 Markdown/表格文本分块入库 | RAG Python 语法编译、diff 检查通过；用户图片完成上传、OCR 和入库回归 | 完成 |
| 2026-08-11 | D-23 OCR 布局空值 | 规范化 pipeline `block_order`/`block_id` 的 `null`、字符串和非法值后排序，避免 `int` 与 `None` 比较异常 | `tests.test_vl_ocr_protocols` 通过；Python 语法编译与 diff 检查通过 | 完成 |
| 2026-08-11 | D-24 Milvus 快照加载 | `_query_file_rows` 与 `_query_collection_rows` 在 iterator/query 前显式加载目标集合，兼容旧客户端和 staged generation 的 released 状态 | Windows 实际创建 `power_design_chunks_gen_018e76f770f949749e14657f02ff7aca` 并成功写入 2 条记录；无 code 101；RAG 发现测试 53/53、语法编译和 diff 检查通过 | 完成 |
| 2026-08-11 | D-25 前端重解析状态 | `KnowledgeBase.vue` 将 `success=true` 或 `status=completed` 都视为成功，立即更新当前行并等待列表/摘要刷新；错误和 loading 状态统一收敛 | Vue SFC 编译与浏览器重解析回归完成 | 完成 |
| 2026-08-11 | D-26 Milvus 删除可见性 | `_query_file_rows` 与 `_query_collection_rows` 使用 `consistency_level="Strong"` 读取删除后的最新快照；旧客户端不接受参数时自动兼容回退 | RAG 53/53 发现测试、语法编译和 diff 检查通过；Windows 影子重建回归完成 | 完成 |
| 2026-08-11 | D-27 OCR 预览渲染 | DocumentPreview 对 OCR 文本执行受限 HTML 清洗后渲染 `<table>`，保留 colspan/rowspan 等被动排版属性并增加表格样式 | Vue SFC 8/8 编译通过；用户图片阅读顺序与表格视觉回归完成 | 完成 |
| 2026-08-11 | D-28 OCR 坐标排序与去重 | pipeline 解析优先读取 `bbox/block_bbox/block_box` 等坐标按 y/x 排序，缺坐标回退 `block_order`；对 HTML/Markdown 空白归一化后去重重复块 | OCR 协议测试通过；Python 语法与 diff 检查通过；用户图片重识别完成 | 完成 |
| 2026-08-11 | D-29 残缺 HTML 清洗 | OCR 归一化阶段用完整 table 占位保护合法表格，删除表格外孤立 cell/row 片段，再恢复完整表格 | OCR 协议测试通过；Python 语法与 diff 检查通过；用户图片重建索引完成 | 完成 |
| 2026-08-11 | D-30 公式识别渲染 | `_VL_PROMPT` 明确要求下标/上标/分数等输出 LaTeX；前端引入 MathJax 配置并在分页内容加载后 typeset | OCR 协议测试、Vue SFC 和 Vite 构建通过；公式视觉回归完成 | 完成 |
| 2026-08-11 | D-31 版面保留预览 | 8001 pipeline 块保留并返回 `layout_blocks`；`ocr-compare` 生成按 bbox 绝对定位的页面 HTML，DocumentPreview 优先用该 HTML sandbox 展示 | RAG Python、Vue SFC 和用户图片真实版面回归通过 | 完成 |
| 2026-08-11 | D-32 默认 pipeline 预览 | DocumentPreview 加载文档后自动请求 OCR 对比接口，主区域直接使用 8001 `layout_html`；手动按钮才展开 8080/8001 对比；服务失败自动回退 chunks | Vue SFC、Vite 构建和用户局域网版面回归通过 | 完成 |
| 2026-08-09 | A-01～A-03 全量验收 | 通过统一脚本强制编译两项目 Python，运行全部发现测试、前端认证/API/SSE/warning 回归、全量 SFC 编译和 Vite 生产构建 | RAG 50/50、Excel 40/40；5 个 Node 检查入口（含 8 个 SFC）通过；Vite 1608 modules 构建成功；脚本最终输出 `All remediation checks passed.` | 完成 |
| 2026-08-09 | A-04 对照复核 | 按原审核总结逐条映射 P0、P1、其他缺陷及工程基础；补查认证启用后的普通 RAG SSE、恢复和下载调用并增加凭据回归；清单与执行记录一一对应 | 清单无未勾选项；认证 fetch 测试通过；`git diff --check` 通过；统一脚本修正收集范围后再次全量通过 | 完成 |
| 2026-08-11 | D-33～D-38 / A-05 预览、下载与文档收尾 | 完成 PDF 多页版面、单页 iframe 缩放、外层统一滚动、章节目录、目次/Contents 三列排版、按原格式下载，并同步 README、部署指南、链路说明和开发日志 | OCR/下载 Python 测试 16/16；`py_compile`、`git diff --check` 通过；SFC 8/8、Vite 1608 模块构建和用户视觉回归通过 | 完成 |
