# PDF 完整解析、入库与版面预览链路

> 文档状态：已按实际 PDF 重新入库验证
>
> 代码基线：2026-08-13
>
> 验证样本：`G25_GBT7409.1-2008_同步电机励磁系统_定义.pdf`

## 1. 目标与产物

一份 PDF 上传后同时产出两条互相独立、但使用同源 OCR 结果的链路：

1. **检索链路**：可检索文本 → 按页分块 → Qwen 稠密向量 + 本地 BGE-M3 稀疏向量 → Milvus；查询时再融合语料 BM25 并使用本地重排模型。
2. **版面预览链路**：每页 OCR 版面块、公式/表格/标题/图表裁剪资源 → layout cache → 前端逐页还原。

因此，预览不依赖向量检索结果重新拼版；向量入库失败也不会错误切换正式索引。成功状态只会在两类产物均完成并通过入库校验后写入文件注册表。

```text
浏览器选择 PDF
  → POST /upload 保存原文件、计算 SHA-256、写入 file_registry（processing）
  → PyMuPDF 原生文本提取 + 逐页渲染 PNG
  → 8001 OCR Pipeline（文本 + parsing_res_list 版面块）
  → 选择可用文本、清洗、按页分块
  ├→ 保存 data/parsed_cache/{file_hash}.layout.json（预览）
  └→ Qwen dense API + BGE-M3 sparse CPU → Milvus 影子集合写入与校验 → 原子切换正式别名（检索）
  → file_registry 更新 completed

预览：GET /files/{hash}/content
  → 读取 layout cache，生成每页 layout_html + outline
  → DocumentPreview 以独立 iframe 显示每一页
```

## 2. 入口、文件状态与持久化位置

| 环节 | 入口/位置 | 说明 |
|---|---|---|
| 上传 | `POST /upload` | `multipart/form-data`，后端校验扩展名和文件名后保存原文件。 |
| 重新入库 | `POST /files/{identifier}/reindex` | 对已注册 PDF 重跑解析、OCR、缓存和入库。 |
| 文件状态 | `GET /files/{identifier}` | 返回 `processing`、`completed` 或 `failed`、分块数、耗时和错误信息。 |
| 预览内容 | `GET /files/{identifier}/content` | 返回文本 chunks、`layout_pages`、`outline` 与每页 HTML。 |
| OCR 诊断 | `GET /files/{identifier}/ocr-compare` | 用于 OCR 对比，仅诊断，不写缓存或向量库。 |
| 图表资源补建 | `POST /files/{identifier}/rebuild-preview-assets` | 对历史 PDF cache 补建图表图片，不重新 OCR 或入库。 |
| 原始文件与注册表 | 上传目录、`data/file_metadata.db` | 保存原 PDF、文件 hash、状态、元数据和统计。 |
| 版面缓存 | `data/parsed_cache/{file_hash}.layout.json` | 按 0 起始页码保存 OCR 版面块和视觉资源。 |
| 检索索引 | Milvus `power_design_chunks_active` 别名 | 别名指向当前已验证的物理 generation collection。 |

主要实现文件：`src/api/main.py`、`src/ingestion/file_processor.py`、`src/ingestion/pdf_parser.py`、`src/ingestion/vl_ocr.py`、`src/ingestion/chunker.py`、`src/ingestion/embedder.py`、`src/ingestion/milvus_store.py`、`src/ui-vue2/src/DocumentPreview.vue`。

## 3. 上传与原生 PDF 解析

1. 前端将 PDF 以 `FormData` 上传至 `/upload`。
2. `FileProcessor` 保存原文件，计算稳定的 SHA-256 `file_hash`，并把注册表状态置为 `processing`。
3. `PDFParser` 使用 PyMuPDF（fitz）逐页读取：
   - 原生文本；
   - 页码、页数；
   - 是否需要以 OCR 文本作为检索文本的判定。
4. 对可提取原生文本的页，原生文本是可靠回退；扫描件或原生文本不足的页，优先使用 OCR 结果。若 OCR 调用异常，已有原生文本的 PDF 仍可继续处理，不会被空 OCR 覆盖。

当前 `ocr.always_pipeline=true` 时，每页仍会发送给 8001，以保证所有页面都有统一的版面缓存；但检索文本是否使用 OCR，仍按该页的原生文本质量判定，避免把质量更高的 PDF 原文替换掉。

## 4. 8001 OCR、版面块与图表资源

### 4.1 逐页渲染与 OCR

PDF 每页先按配置 DPI 渲染为 PNG（默认 DPI 150，并受最大图像边长限制），再以 multipart 请求发送到 8001 Pipeline 的 `/ocr`。Pipeline 返回：

- 识别文本；
- `parsing_res_list` 版面元素；
- 每个元素的 bbox、类型和结构化内容。

客户端会兼容空排序字段，按坐标进行阅读顺序排序，并清理重复块。PDF 的 OCR 任务按页执行，因此第一页、后续页均会参与版面解析；不会再出现仅以第一页的 OCR 结果作为整份 PDF 预览的情况。

### 4.2 layout cache

每页版面块写入 `{file_hash}.layout.json`。常见块类型包括标题、文本、公式、表格、页眉页脚，以及 `chart`、`image`、`figure`、`diagram` 等视觉块。缓存是预览的事实来源，检索只使用其中清洗后的文本。

识别到图表/图片类块时，系统从**实际发送给 OCR 的同一页 PNG**按 bbox 裁剪，写入：

- `visual_data_uri`：裁剪后的图片数据；
- `visual_asset_bbox`：在原页面上的坐标。

预览将图片放回对应 bbox，图内重复的 OCR 文本不在视觉层重复显示，但仍保留在缓存与检索文本中。旧缓存没有视觉资源时，可调用 `rebuild-preview-assets` 按既有 bbox 补建，避免全量重新入库。

## 5. 文本选择、分块和向量生成

1. 每页选择 OCR 或原生回退文本，经过空白、重复和无效内容清洗。
2. `Chunker.chunk_page_text()` 按页生成 chunks：短页通常形成一个 chunk，长页按长度拆分。chunk id 含 `file_hash` 和页码，保留 `page_num`、文件名、文档类别等元数据。
3. `Embedder` 对 chunks 分批生成两类向量：Qwen3-Embedding-0.6B 通过 OpenAI-compatible API 生成 1024 维 dense vector；本地 CPU `BAAI/bge-m3` 只生成 lexical sparse weights，不替换 dense 模型。
4. 文本、向量及元数据一起写入待验证的 Milvus 影子集合；layout cache 不写入向量库。

查询阶段除 Milvus 的 dense+BGE-M3 sparse 加权召回外，还会从当前 active generation 的文本和元数据构建 BM25 索引。BM25 使用语料 IDF 与文档长度归一化，补充标准号、文件名和关键词的精确召回；最终候选由本地 CPU `BAAI/bge-reranker-v2-m3` 交叉编码器精排。BM25 索引只在进程内缓存，generation 切换或 chunk 数变化后自动重建。

这样既能在问答中定位 PDF 页码，又能在预览中保持逐页版面，而不会因为跨页合并而破坏阅读顺序。

## 6. Milvus 影子集合入库与完整性保护

正式检索集合不直接修改。每次文件入库按以下顺序执行：

```text
当前 active 物理集合
  → 以 chunk_id 的 SHA-256 前缀分桶、普通 query() 复制旧数据
  → 创建 staging generation，并移除本次 file_hash 的旧记录
  → 写入本次 PDF 的向量 chunks
  → 精确查询本文件 chunk 数 + collection row_count 双重校验
  → 校验通过：将 active alias 原子指向 staging generation
  → 更新注册表为 completed

任一步失败
  → 删除 staging generation / 保持 active alias 不变
  → 注册表记录 failed 和具体错误
```

此处不使用 Milvus Lite 上会出现重复页数据的 `query_iterator` 来统计或复制全表。直接查询被限制为单次最多 16,000 条；若某一 hash 前缀桶满，则继续细分前缀。非 hash 主键、查询截断、重复 ID 或物理行数不一致都会使校验失败并阻止正式索引切换。

因此，单个 PDF 的解析、Embedding 或校验失败，不会污染已经可用的知识库，也不会把“部分入库”标为成功。

## 7. 后处理预览与章节目录

`GET /files/{hash}/content` 优先读取 layout cache，再由 `render_layout_html()` 输出每页独立 HTML：

- 页面根据可用阅读区居中缩放，外层只有一条阅读滚动；
- 每页使用独立 iframe，避免不同页的 bbox 坐标互相影响；
- 标题、章节标题居中；普通段落按 bbox 定位；
- 公式交由 MathJax 渲染，公式与右侧编号采用统一垂直对齐；
- 表格保留结构化 HTML；`目次`/`Contents` 以标题、引导点、页码三列展示；
- 图表、图片使用对应视觉裁剪资源，位置和尺寸跟随 OCR bbox；
- `extract_layout_outline()` 综合标题类型、编号规则、文本特征和坐标生成章节级目录；识别质量不足时才回退为页级导航。

如果某个历史文件没有 layout cache，接口仍可用向量 chunks 生成文本回退预览；这保证文档可读，但不等同于版面还原。要恢复版面，应重新入库或补建相应缓存。

## 8. 本次实际验证结果

2026-08-12 对以下 PDF 执行重新入库并完成全链路验证：

```text
G25_GBT7409.1-2008_同步电机励磁系统_定义.pdf
file_hash: f5d4388447ff10357b962ab91d77be7e3cc38448bb1303c76362a81fefa5442a
```

结果：

| 项目 | 实测值 |
|---|---:|
| 最终状态 | `completed` |
| 生成 chunks | 16 |
| 提取字符数 | 15,171 |
| 解析/OCR 耗时 | 127.03 秒 |
| Embedding 耗时 | 1.64 秒（历史样本，指当时的 dense 入库耗时） |
| 总耗时 | 132.66 秒 |
| 正式集合总 chunks | 249 → 265 |

该样本此前因 Milvus Lite 分页重复数据导致影子集合校验误判；修复后已完成重新入库，说明当前链路能覆盖多页 PDF 的解析、OCR 版面缓存、分块、Embedding、影子集合验证、正式切换和预览数据生成。

## 9. 运行检查与故障定位

处理完成后建议依次检查：

1. `GET /files/{hash}`：状态为 `completed`，并确认 `chunks_created`、`chars_extracted` 和错误字段。
2. `GET /files/{hash}/content`：应返回 `layout_pages` 和章节 `outline`；页数与 PDF 一致。
3. 前端预览：检查多页、目录、公式、表格与图表是否出现；历史 cache 缺图时调用视觉资源补建接口。
4. `GET /stats`：确认 active collection 总 chunk 数随成功入库增长。
5. 出现失败时先读取注册表中的 `error_message`。由于别名最后才切换，失败期间线上检索仍使用上一版正式集合，可安全修复后执行 `reindex`。

## 10. 当前边界

- 上传、OCR 与入库仍为同步请求；超大 PDF 尚未实现服务端任务队列和实时进度推送。
- OCR 和图表 bbox 的还原质量受 8001 模型输出影响；当模型未识别出图表块时，系统不会凭空生成可定位的视觉资源。
- MathJax 目前依赖前端可访问的资源；离线环境需要改为本地静态资源。
- 预览缓存采用 data URI 保存视觉裁剪图，复杂 PDF 的缓存和接口响应会增大；后续可拆分为独立静态资源文件并加入缓存版本号。
