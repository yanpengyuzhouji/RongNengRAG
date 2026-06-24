# 开发日志

## 2026-06-22

### RAG 检索质量修复 — 12题跨域诊断 + 4项代码修复

- **12题跨域测试** (变电/配电/送电输电/综合 各3题)，基线: 域分类准确率 67%，关键文件命中率 8%

**Bug修复 — 域分类子串误判** (`src/retrieval/query_analyzer.py`):
  - `if kw in query` 字符子串匹配 → jieba 分词 + 词边界精确匹配
  - "配电装置" 不再因含 "配电" 误判为配电域
  - 新增消歧规则: 变电/配电同时命中时，强变电信号("变电站""GIS"等)加权+2
  - 新增消歧规则: 送电输电强信号("架空线路""杆塔"等)加权+2
  - 域分类准确率: 67% → 75%

**Bug修复 — RRF 权重死代码** (`src/ingestion/milvus_store.py`):
  - `RRFRanker(k=60)` 完全忽略 dense_weight/sparse_weight → 改用 `WeightedRanker(dense_weight, sparse_weight)`
  - config.yaml: dense_weight 0.5→0.7, sparse_weight 0.5→0.3 (标准规范语义更重要)
  - 权重配置对检索排名真正生效

**优化 — 元数据加权从乘法改为加法** (`src/retrieval/reranker.py`):
  - `_compute_metadata_boost()`: 乘法叠加(最高1.58x) → 加法归一化+上限1.15x
  - 文件名匹配: 1.30x → +0.10; 域匹配: 1.05x → +0.05; 电压匹配: 1.10x → +0.05
  - 防止元数据统治语义评分

**Bug修复 — 完整文档注入撑爆 LLM 上下文** (`src/retrieval/retriever.py`):
  - 根因: query 含 "根据GB50052-2009" → FileRegistry 检测文档编号 → get_full_document() 注入 28000+ 字符全文档 → 超出 32768 token 窗口 → LLM 只输出 "基于" 2字
  - 修复: `build_context_with_file_injection()` 中添加上下文预算检查，完整文档超过 10000 字符自动截断
  - 修复前: answer="基于"(2字) → 修复后: answer=1342字+5条引用

**配置 — 送电输电关键词补强** (`config.yaml`):
  - 新增 "电力电缆""电磁环境""电缆敷设""架空电力线路" → 修复 S09 回归

### 技术文档全面更新

- **更新** `README.md` — 项目结构、架构图、文档类型支持、GPU 显存管理、评估框架
- **更新** `TECHNICAL.md` — 新增 OFD 提取器、评估框架、OCR 显存编排流程、communicate() 根因修复、超大图缩放
- **更新** `CHANGELOG.md` — 补充 2026-06-12 至 2026-06-22 开发日志
- **更新** `docs/资料分类与知识库入库说明.md` — OCR 参数更新、性能基准更新、OFD 格式说明

### OFD 文档自定义提取器

- **新增** `src/ingestion/ofd_extractor.py` — 国产版式文档 GB/T 33190 文本提取:
  - 纯标准库实现 (zipfile + xml.etree)，零额外依赖
  - 支持直接 Unicode 文本、GBK/GB18030 编码 TextCode
  - 支持 CID-keyed 字体 CMap 字形映射
  - ofdparser 作为回退方案
- **修改** `requirements.txt` — 添加 `xmltodict>=0.13.0` 和 `reportlab>=4.0.0` (ofdparser 依赖)

### OCR 超大图自动缩放

- **修改** `src/ingestion/ocr_engine.py` — `_render_page_to_png()` 新增 `max_dim` 参数:
  - PDF 渲染 PNG 后，边长超过 `max_image_dim` 时用 Pillow LANCZOS 缩放
  - 大规格工程图 (65"×46") 150DPI = 9744×6890px
  - 直接 OCR: 660s/页 + 12GB 显存
  - 缩放至 3000px: 2.6s/页 + ~10GB 显存 (254x 提速)
- **修改** `config.yaml` — 新增 `ocr.max_image_dim: 3000` 配置项

### OCR Turbo 进程池配置

- **修改** `config.yaml` — 新增 `ocr.turbo` 配置节:
  - `enabled: false` (默认关闭，安全模式)
  - `max_workers: 0` (0=VRAM 感知自动)
  - `pages_per_worker: 25`
  - `safety_margin_mb: 1500`
- **新增** `ocr_engine.py` — `ocr_pool_map()` VRAM 感知多子进程并行，大文件可提速 ~1.7-1.9x

---

## 2026-06-19

### 根因修复: subprocess stderr 管道死锁 → WDDM 显存爆炸

- **根因分析**:
  - PaddleOCR 初始化输出大量 ANSI 彩色日志到 stderr (~200KB/次)
  - 旧方案使用 `--stream` 模式 + `stderr=PIPE`，stderr 缓冲区仅 64KB (OS 管道)
  - 主进程未及时读取 stderr → 管道满 → 子进程 `write()` 阻塞 → 僵尸状态
  - 僵尸子进程持有 CUDA context (1.5-3GB)，WDDM 无法回收
  - 每轮 OCR 泄漏 1.5-3GB → 三轮后 12GB 全满 → OOM
- **修复** `src/ingestion/ocr_engine.py`:
  - `_spawn_ocr_worker()` — 从 `--stream` Popen+管道 改为 `Popen` + `communicate(input=json, timeout=N)`
  - `communicate()` 内部使用线程并发读取 stdout/stderr，**无死锁风险**
  - 超时后 `cmd //c taskkill /F /T /PID` 杀整个进程树 (含 PaddlePaddle CUDA 孙进程)
  - 移除 `--stream` 模式全部代码，简化为单次 communicate 调用

### 根本解决显存溢出: 单 OCR 子进程处理全部页面

- **修改** `src/ingestion/ocr_engine.py`:
  - 从双槽并行 (2 OCR 子进程) 回退到单子进程批量模式
  - 所有 OCR 页面在一个子进程中一次性批量处理
  - 通过 `_ocr_lock` 全局互斥锁确保同时最多 1 个 OCR 子进程
  - 防止 2+ PaddleOCR 实例同时占用 GPU 导致 OOM

### OCR 入库前卸载 BGE-M3: 防止两模型并存爆显存

- **修改** `src/ingestion/file_processor.py`:
  - `_process_pdf_progressive()` — OCR 前调用 `embedder.unload()`，OCR 后调用 `embedder.reload()`
  - 释放 ~2GB 显存给 PaddleOCR 使用
- **修改** `src/ingestion/embedder.py`:
  - 新增 `unload()` — `del self.model` → `gc.collect()` → `torch.cuda.empty_cache()`
  - 新增 `reload()` — 重新调用 `_ensure_loaded()` 加载 BGE-M3

### 降低 OCR 并行度 + 临时目录改 E 盘 + stderr 编码修复

- **修改** `config.yaml`:
  - `ocr.ocr_tmp_dir` 从 C 盘改为 `E:/RongNengRAG/data/ocr_tmp` — 避开系统盘
  - `ocr.gpu_memory.max_wait_s` 从 600 降至 300
- **修复** OCR 子进程 stderr 读取出错时 GBK 编码异常
- **效果**: 磁盘 I/O 不再与系统盘争抢，显存等待超时更合理

---

## 2026-06-12

### 检索评估框架建立

- **新增** `eval/` 目录 — 三层检索评估体系:
  - `README.md` — 评估框架说明和快速开始指南
  - `datasets/` — 评估数据集 (JSON 格式，含 domain + category + relevant_chunks)
  - `extract_dataset.py` — 从 `_test_results.json` 提取数据集
  - `metrics.py` — 指标计算纯函数 (Recall@K, MRR, NDCG@10, Precision@K)
  - `report_generator.py` — Markdown 报告生成器
  - `run_eval.py` — 主评测入口
  - `output/` — 评测报告和结构化数据输出
- **三层评估**:
  - Layer 1 (检索质量): Recall@K, MRR, NDCG@10, Precision@K
  - Layer 2 (召回质量): 文档召回率, Chunk召回率, 跨文档覆盖率, Domain/Category准确率
  - Layer 3 (重排序效果): Top-1提升率, MRR Delta, NDCG Delta, 退化检测
- **首轮评测** (送电输电/标准规范, 30题):
  - Recall@50: 0.833 | MRR: 0.406 | 关键词命中率: 55.5%
  - 重排序 MRR 提升: +0.333 | NDCG 提升: +0.370

### 答案质量对比评估

- **新增** `eval/output/answer_comparison_*.md` — LLM 答案质量对比报告
- 对比维度: 正确性、完整性、引用准确性、可读性

---

## 2026-06-11

### GPU 显存感知调度 — OCR双槽并行 + 嵌入流水线 + Reranker按需卸载

- **新增** `src/utils/gpu_monitor.py` — GPU 显存监控器:
  - 基于 pynvml 实时检测 GPU 显存使用
  - `is_ollama_busy()` 检测 Ollama 是否正在推理（/api/ps 端点）
  - `wait_for_vram(min_free_mb)` 背压等待：显存不足时检查 Ollama 状态，LLM 繁忙则等待释放，LLM 空闲则短等后强制执行
  - 全局单例 `get_gpu_monitor()`

- **重构** `src/ingestion/ocr_engine.py` — OCR 引擎完整重写:
  - **stdout/stderr 分离**: CLI 入口将 sys.stdout 重定向到 stderr，JSON 输出通过 `_orig_stdout_fd` 写原始文件描述符，彻底解决 PaddleOCR 初始化日志污染 JSON
  - **PaddleOCR 2.x/3.x 兼容**: 同时支持旧版二元组 `(box, (text, conf))` 和新版 `OCRResult` dict（`rec_texts`/`rec_scores` 键）
  - **GPU 子进程专用 Python**: `_get_ocr_python()` 使用 PPOCRLabel venv 的 Python（paddlepaddle-gpu 3.2.2 兼容 paddleocr 3.6.0）
  - **环境隔离**: `_get_ocr_env()` 清除 PYTHONPATH，防止 conda base 路径干扰 venv 包加载
  - 单页/批量/直接模式全部集成 `_wait_for_gpu_slot()` 背压
  - 大文件分批: 每批 ≤50 页，动态超时 `max(300, pages × 15s)`

- **重构** `src/ingestion/file_processor.py` — PDF 双槽并行入库:
  - `_process_pdf_progressive()` — 2 OCR 子进程 + 1 嵌入线程 同时跑:
    - OCR: `ThreadPoolExecutor(max_workers=2)` 提交全部批次，`as_completed` 消费
    - 嵌入: 独立 `threading.Thread` 从 `queue.Queue` 取 chunk 编码入库
    - 任意 OCR 完成 → 立即分块 → 入队，嵌入线程并行消费
    - 中断恢复: `delete_by_file_hash(file_hash)` 清理已插入 chunk
  - `_embed_and_insert()` — 嵌入+入库抽成独立方法
  - `process()` 路由: PDF+OCR → 渐进式，其他 → 通用路径
  - **docx 解析增强**: 遍历 `doc.tables` 提取表格文本，格式化为 `[表格N]\n列A | 列B`
    - 电力规范 docx 表格信息占比 20-80%，不再丢失
    - 异常日志: ImportError → "请安装 python-docx"，其他异常 → 打印文件名和错误

- **修改** `src/retrieval/reranker.py`:
  - 新增 `unload()` — `del self.model` + `torch.cuda.empty_cache()`，释放 ~2GB 显存
  - `_ensure_loaded()` 保持不变，下次 rerank 时自动重新加载（~5s）

- **修改** `src/retrieval/retriever.py`:
  - `search()` 完成后自动调用 `self.reranker.unload()`，reranker 用后即卸
  - 交叉编码器未使用时无需常驻显存

- **修改** `src/api/main.py`:
  - 移除启动时 OCR 模型预热 — PaddleOCR 4个子模型（检测/识别/方向/文档预处理）~3-4GB，与 BGE-M3(~2GB)+BGE-Reranker(~2GB) 同时加载撑爆 12GB
  - 新增 `GET /gpu` — GPU 显存状态端点
  - `POST /files/sync` — 一致性校验端点（扫描失效文件、清理孤记录）
  - `DELETE /files/{id}?remove_file=true` — 删除时可选清理 uploads 下的物理文件

- **修改** `src/ui/app.py`:
  - 新增「扫描失效文件」按钮: dry-run 扫描注册表中物理文件不存在的记录
  - 新增「清理失效文件」按钮: 执行 vector+registry 清理
  - 文件列表刷新时自动检测物理文件存在性，失效文件显示 `[~] 文件丢失`
  - 标题栏显示失效文件总数警告

- **修改** `config.yaml`:
  - `ocr.use_gpu: true` — 启用 GPU 推理 (~5-7s/页 vs CPU ~15-18s/页)
  - `ocr.page_delay_ms: 0` — GPU 模式无需冷却延迟
  - 新增 `ocr.gpu_memory` 配置: `min_free_vram_mb/poll_interval_s/max_wait_s`

- **效果对比**:
  | 指标 | 修改前 | 修改后 |
  |------|--------|--------|
  | 启动显存 | 11.9 GB (全预加载) | 1.7 GB (仅系统) |
  | 常态显存 | 11.9 GB | 3.8 GB (BGE-M3) |
  | 搜索峰值 | 11.9 GB | 5.8 GB (临时加载 Reranker) |
  | 入库 OCR 峰值 | 11.9 GB | 8 GB (2×PaddleOCR + BGE-M3) |
  | 空闲显存 | 0 | 8.4 GB (LLM 随便用) |
  | 138页全扫描 | 超时失败 | ~13 分钟 (双槽并行) |
  | CJJ61-2017 OCR | 子进程崩溃 RC=1 | 5.7s/页, 正常识别 |

### 文件注册表-向量库-物理文件三端一致性

- **修改** `src/ingestion/file_processor.py`:
  - `list_files()` 新增 `check_existence=True`，自动检测物理文件是否存在，返回 `file_exists` 字段
  - `delete()` 新增 `remove_file` 参数，仅删除 uploads/ 下文件（安全策略）
  - `sync_orphans()` 新增 — 扫描注册表中物理文件已消失的孤记录

### 检索质量评估体系设计

- **更新** `docs/送电输电标准规范检索测试报告.md` — 三层评估框架:
  - 检索质量: Recall@K, MRR, NDCG@10, Precision@K
  - 召回质量: 文档召回率, Chunk召回率, 跨文档覆盖率, 域/类目准确性
  - 重排效果: Top-1 提升率, MRR提升, NDCG提升, 退化检测
- 测试数据集规范: 每条题目必须带 `domain` + `category` + `relevant_chunks`

---

## 2026-06-09

### PaddleOCR 扫描件 PDF 识别

- **新增** `src/ingestion/ocr_engine.py` — PaddleOCR 引擎（子进程隔离模式）:
  - 通过子进程运行 OCR，解决 paddlepaddle protobuf 3.x 与 pymilvus protobuf 5.x 版本冲突
  - 主进程: `protobuf>=5.27.2` (pymilvus 正常) / OCR 子进程: `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
  - 支持 `--batch` 批量模式: 一次子进程加载模型，处理多张图片（4x 加速）
  - 单页模式: `ocr_page(image_path)` → 子进程单图 OCR
  - PDF 批量模式: `ocr_pdf_pages(pdf_path, pages)` → 渲染 PNG → 子进程批量识别
  - `OCREngine` 类: `use_subprocess=True` 自动隔离模式
  - 首次调用 ~10s (模型加载)，后续 ~1.5s/页 (CPU)
  - 子进程超时保护 600s / 120s
- **修改** `src/ingestion/pdf_parser.py`:
  - `__init__` 新增 `ocr_config` 参数
  - 新增 `_get_ocr_engine()` — 延迟加载 OCR 引擎
  - 新增 `ocr_pages(filepath, pages)` — 批量 OCR PDF 页面
  - 新增 `ocr_page(image_path)` — 单图 OCR
- **修改** `src/ingestion/file_processor.py`:
  - `__init__` — 传递 OCR 配置给 PDFParser
  - `_parse_file` PDF 分支 — 解析后检测 `needs_ocr_pages`，自动调用 OCR 并回填文本
  - OCR 文本与原 fitz 文本智能合并（有则追加）
- **修改** `src/api/main.py`:
  - 服务启动时异步预热 OCR 子进程（首次调用不等待）
- **修改** `config.yaml` — 新增 `ocr` 配置节:
  ```yaml
  ocr:
    enabled: true          # 启用 OCR
    provider: "paddleocr"  # PaddleOCR
    lang: "ch"             # 中文
    dpi: 200               # PDF 渲染 DPI
    min_text_chars: 50     # 触发 OCR 阈值
    use_subprocess: true   # 子进程隔离
  ```
- **效果**:
  - 208 页 PDF: 4 页标记为扫描件 → 批量 OCR 6s (1.5s/页)
  - 封面页识别: "国家电网有限公司 / 十八项电网重大反事故措施 / （修订版）" (99.8% 置信度)
  - 空白页正确跳过，不消耗额外 OCR 时间
  - 不影响原有文字型 PDF 的解析速度（仅在 `needs_ocr` 时触发）

### .wps 文件解析支持

- **修改** `src/ingestion/file_processor.py`:
  - `_parse_file` 新增 `.wps` 分支
  - 新增 `_parse_wps_file()` — 6 级回退解析管道
  - 新增 `_parse_wps_via_docx()` — 新版 WPS (ZIP+XML) 用 python-docx 读取
  - 新增 `_parse_wps_via_zip()` — 备选: 直接解压 ZIP 提取 XML 文本
  - 新增 `_parse_wps_via_wps_com()` — WPS Office COM 自动化
  - 后 3 级复用 .doc 管道 (LibreOffice / olefile / MS Word COM)
- **修改** `src/api/main.py` — 上传白名单添加 `.wps`

---

## 2026-06-08

### 文件注册表识别 + 完整文档注入

- **新增** `src/retrieval/file_registry.py` — `FileRegistry` 类:
  - 从 SQLite `file_registry` 表读取已入库文件元数据，60 秒缓存
  - `detect_files_in_query(query)` — 四阶段多策略文件名检测:
    - **策略1**: 正则提取 query 中的文件名候选（书名号、引号、扩展名模式、文档编号）
    - **策略2**: 双向匹配 — 文件名 in query + query 关键词 in 文件名（解决短 query 匹配长文件名问题）
    - **策略3**: 文档编号精确匹配（闽电发展〔2015〕241号、GB 50060-2008 等）
    - **策略4**: "会议材料之X" / "会议材料X" 系列匹配 — 含中文数字序数归一化（一→1）
  - 支持五种自然语言写法: `01会议材料之一` / `会议材料之一` / `01会议材料一` / `会议材料一` / `材料一`
  - 中文→阿拉伯数字映射: 一～二十 → 1~20，含 `zfill(2)` 对齐 "01" 格式
  - `_extract_query_phrases(query)` — 将 query 拆解为关键词片段并生成数字变体
  - 返回 `FileMatchResult`（含 `match_type`、`match_score`、`match_text`），去重 + 按分数降序

- **修改** `src/ingestion/milvus_store.py`:
  - 新增 `query_by_file_path(file_path)` — 按 `file_path` 精确查询文件的完整 chunks
  - 新增 `query_by_file_hash(file_hash)` — 按 `chunk_id LIKE "{hash}%"` 前缀查询（主方案，最可靠）
  - 返回结果按 `page_num` + `chunk_index` 排序

- **修改** `src/retrieval/retriever.py`:
  - `__init__` 新增 `self.file_registry` 实例
  - 新增 `detect_file_in_query(query)` — 委托 FileRegistry 返回最佳匹配
  - 新增 `get_full_document(file_path, file_hash)` — 从 Milvus 拉取完整文档、格式化输出（含页眉+页码导航）
  - 新增 `build_context_with_file_injection(query, search_results, max_chunks)` — LLM 上下文构建核心:
    - 检测到文件名: 完整文档前置 + 强化聚焦指令 + 补充检索（排除同系列 + 最多 2 个）
    - 未检测到: 正常检索格式
  - 新增 `_extract_series_key(filename)` — 从文件名提取"系列标识"
    - "01会议材料之一..." → `会议材料之`
    - 用于排除同系列文件干扰（查询材料01时排除02-07）
  - 聚焦指令强化: `"请只基于上述完整文档内容回答，不要引用其他文件"`
  - 补充检索结果标注: `"以下内容仅供背景了解，回答时请勿引用"`
  - 补充最多保留 2 个文件（原 5+ 个）
  - 新增 debug 日志输出匹配过程

- **修改** `src/api/main.py`:
  - `/ask` 和 `/ask/stream` 端点改用 `r.build_context_with_file_injection()` 替代旧版 `_build_focused_context`
  - 边界处理: 检索无结果但 query 含文件名时，从注册表拉取完整文档再送 LLM
  - 保留 `_build_focused_context_fallback` 作为纯文本回退方案

- **效果**:
  - 查询 "会议材料一总结" → 精确匹配 01 文件，排除 02-07，回答仅含材料一内容
  - 查询含文件名 → score=1.0 精确匹配，注入完整文档（22k+ chars）
  - 查询无文件名 → 不触发，正常检索流程
  - 同系列文件 100% 排除，不再混淆

### Qwen3 强制思考链优化 — 终极方案

- **根因**: Qwen3 (4b/8b/14b) 和 DeepSeek-R1 是推理模型，强制生成 thinking tokens（占 85-95% 输出），`enable_thinking=false` 和 `disable_cot=true` 均对 Qwen3 无效
- **方案演进**:
  1. Ollama OpenAI兼容API `/v1/chat/completions` → reasoning 字段被 openai 库丢弃 → 改用原生 `/api/chat`
  2. `disable_cot=true` 顶层参数 → Qwen3 不接受，反而更慢
  3. `ollama create qwen3:4b-nothink` 自定义模板移除 think 标签 → 有效但只是隐藏，仍在生成
  4. **最终方案**: 升级到 Ollama 最新版 + `qwen3.5:4b`，原生支持 `think: false` → 完全关闭思维链
- **修改**: `ollama_provider.py` — 新增 `think` 参数，`payload["think"]=false` 关闭思考；流式直接输出无需缓冲
- **预期效果**: 32k 上下文下 25-35 t/s，单次 RAG 问答 15-25 秒（vs qwen3:4b 的 70-90 秒）

### LLM Provider 模式重构 + 百炼 API 支持

- **新增** `src/generation/providers/` 包:
  - `base.py` — `BaseProvider` 抽象基类 (`generate` + `generate_stream`)
  - `ollama_provider.py` — 迁移原有 Ollama 逻辑到独立 Provider，改用原生 `/api/chat` 端点
  - `bailian_provider.py` — 阿里云百炼 DashScope，OpenAI 兼容接口
- **修改** `llm_engine.py` — 重构为 Provider 模式 facade，支持 `generate_chat()` 多轮对话 + `generate_chat_stream()` 流式
- **修改** `config.py` — `load_config()` 自动加载 `.env` 环境变量
- **新增** `.env.example` — API Key 模板

### 多轮对话 + 上下文记忆系统

- **新增** `src/generation/conversation_manager.py`:
  - 会话管理: create/list/get/delete，每会话独立历史
  - 压缩策略: 超窗口时保留最近 3 轮详细信息，旧消息压缩为摘要
  - 所有时间戳使用北京时间 (Asia/Shanghai)
- **修改** `api/main.py`:
  - `AskRequest` 添加 `conversation_id` 字段
  - 新增 `POST/GET/DELETE /conversations` CRUD 端点
  - `/ask` 支持多轮上下文加载
- **修改** `prompt_templates.py` — 新增 `get_system_prompt()` 系统提示词

### SSE 流式生成 + 会话侧边栏 + 会话隔离

- **修改** `api/main.py` — 新增 `POST /ask/stream` SSE 端点:
  - 检索完成后立即发送 `searching` → `thinking` 心跳
  - 模型产出 token 实时推送到前端
- **修改** `ui/app.py` — 大幅重构:
  - **新布局**: 左侧会话侧边栏 + 主区域 Chatbot
  - 会话管理: 新建/切换/删除，`gr.State` 维护会话隔离
  - 流式展示: 接收 SSE 事件实时更新 Chatbot
  - 北京时间显示: 每条消息带时间戳
  - 保留原有文件管理/检索/统计功能
- **新增** API 流式心跳机制: thinking 阶段显示 "💭 思考中..." 动画

### 检索置信度计算

- **修改** `reranker.py` — 为每个结果附加 `_rerank_score` 并映射为 `confidence` (0-1)
- **修改** `retriever.py` — `RetrievalResult` 新增 `confidence: float` 字段
- **修改** `api/main.py` — `SearchResultItem` 新增 `confidence` 字段
- **修改** `ui/app.py` — 搜索结果显示置信度百分比 + 可视化进度条
- **修改** `config.yaml` — 新增 `retrieval.confidence` 配置

### .doc 文件解析

- **修改** `file_processor.py`:
  - 新增 `_parse_doc_file()` 方法，按优先级: Word COM → LibreOffice → olefile → antiword → docx2txt
  - 新增 `_parse_doc_via_win32()` — 通过 `win32com` 自动化本机 Word 提取文本
  - `_parse_file()` 添加 `.doc` 分支
- **效果**: 测试文件 "附件1福建省变配电工程消防设计技术要点.doc" → 37 chunks, 79,664 chars

### 文件详情查看修复

- **修改** `api/main.py` — 新增 `GET /files/{identifier}` 端点
- **修改** `ui/app.py` — `view_file_details()` 直接调 API 查找，不再依赖过期缓存

### 文档列表类目换行修复

- **修改** `ui/app.py` `_build_file_tree()` — 每个类目标题前插入空行，避免 markdown 渲染连在一起

### 文件管理标签页加载修复

- **修改** `ui/app.py` — 文件管理/检索/统计 Tab 补齐 `demo.load` 事件，页面打开自动加载

---

## 2026-06-05

### 路径配置项目化

- **新增** `src/config.py` — 统一配置加载模块
  - 自动检测项目根目录（向上查找 `config.yaml`）
  - 所有相对路径自动解析为绝对路径
  - `ensure_data_dirs()` 自动创建数据目录
  - `get_config_path()` 返回默认配置文件路径
- **修改** `config.yaml` — 所有路径改为相对路径
  - `knowledge_base: ""` 暂时留空，批量导入时再配置
  - `metadata_db: "data/file_metadata.db"`
  - `milvus_db: "data/milvus_lite.db"`
  - `uploads_dir: "data/uploads"`
  - `parsed_cache: "data/parsed_cache"`
  - `models_dir: "data/models"`
- **更新 8 个模块** — 全部改用 `from config import load_config`
  - `milvus_store.py`, `embedder.py`, `file_processor.py`, `chunker.py`
  - `file_walker.py`, `query_analyzer.py`, `reranker.py`, `retriever.py`
  - `llm_engine.py` (llama_cpp 路径也改为项目相对)
  - `api/main.py` (添加 startup 事件初始化数据目录)
- **数据存储位置**: `D:/rag-system/` → 项目内 `data/`
- **部署简化**: 克隆项目后无需创建外部目录或复制配置文件

### 文档与项目管理

- **新增** `README.md` — 项目概述、技术栈、快速开始、项目结构、架构图
- **新增** `TECHNICAL.md` — 详细技术文档（模块详解、API 接口、数据流、性能参数）
- **新增** `CHANGELOG.md` — 本开发日志

---

## 2026-06-04

### 启动项目与依赖修复

- **安装缺失依赖**: `pymilvus`, `milvus-lite`, `FlagEmbedding`, `PyMuPDF`, `python-pptx`, `xlrd`, `rarfile`, `faiss-cpu`
- **创建数据目录**: `D:/rag-system/data/` (uploads, parsed_cache, milvus_lite.db)
- **配置 LLM 模型**: 将 `qwen2.5:7b-instruct-q4_K_M` 改为已安装的 `qwen3:8b`
- **复制配置文件**: `config.yaml` → `D:/rag-system/config.yaml`

### HuggingFace 离线迁移 — 全部改为本地 Ollama

#### 嵌入引擎 (`ingestion/embedder.py`)
- 默认使用 Ollama `bge-m3:latest` 的 `/api/embed` 批量 API
- 保留 `sentence_transformers` 作为回退方案
- 添加 `_init_ollama()` 连接验证
- 修复 `from chunker import Chunk` 无效导入问题

#### 重排序器 (`retrieval/reranker.py`)
- 默认使用 Ollama 嵌入 + 余弦相似度 + 元数据加权
- 批量嵌入 Query + 候选文档计算相似度
- 保留 FlagEmbedding 交叉编码器作为回退
- 添加 `rerank_without_model` 纯元数据排序兜底
- 修复所有 emoji → ASCII 以兼容 Windows GBK 终端

#### 配置文件 (`config.yaml`)
- `embedding.provider`: `"ollama"` (默认)
- `reranker.provider`: `"ollama"` (默认)
- `device`: `"cpu"` → `"cuda"` (GPU 加速)

### Milvus Lite 多进程冲突修复

#### 架构变更: Gradio UI → HTTP 客户端模式
- **根因**: Gradio 和 FastAPI 两个独立 Python 进程同时打开 Milvus Lite，LMDB 排他锁冲突
- **修复**: `ui/app.py` 完全重写为 FastAPI HTTP 客户端
  - 问答: `POST /ask`
  - 检索: `POST /search`
  - 上传: `POST /upload`
  - 管理: `GET /files`, `DELETE /files/{id}`
  - 统计: `GET /files/summary`

### 空集合检索报错修复

- `milvus_store.py:hybrid_search()`: 集合不存在时返回空列表而非抛异常
- `milvus_store.py:delete_by_file_hash()`: 集合不存在时跳过删除
- `milvus_store.py:insert()`: 添加 `_ensure_collection()` 自动创建集合
- `milvus_store.py:_connect()`: 连接失败时自动清理残留 `LOCK` 文件并重试

### Pydantic 校验错误修复
- `file_processor.py:_build_file_meta()`: 所有 `None` 值加 `or ""` / `or 0` 兜底
- `api/main.py:UploadResponse`: `domain`, `category`, `doc_number` 接收 `None` 导致 422 错误

### pymilvus 3.0 API 迁移
- `hybrid_search(rerank=...)` → `hybrid_search(ranker=...)`

### WinError 183 根本性修复
- **根因**: milvus-lite 3.0 在 Windows 上使用 `os.rename(tmp, target)` 写入 manifest.json，Windows 的 `os.rename` 不能覆盖已存在文件
- **修复**: `milvus_store.py` 顶部 monkey-patch `os.rename` → `_safe_rename`（内部用 `os.replace` 原子替换）
- **影响范围**: create_collection、create_index、insert、delete 全部不再受此 bug 影响

### 文件上传错误修复
- Gradio `upload_and_index()` 返回 2 值但 click 只定义 1 个输出 → 改为只返回结果字符串
- `upload_and_index()` 使用 `os.path.basename(f.name)` 提取原始文件名，避免乱码

### 检索结果数修复
- `retriever.py`: 构建响应时添加 `ranked[:top_k]` 硬限制
- 重排序失败时自动回退到 `rerank_without_model`

### 文件管理 UI 重写
- 新增文件夹树状视图：按域→类目→文件三级展示
- 新增文件选择器下拉框，支持按文件名搜索
- 新增"查看详情"功能：文件属性完整展示
- "刷新列表"默认排除已删除文件，显示文件数+chunk总数
- 页面加载时自动刷新文件列表
- 删除后自动清空下拉选择器并刷新树

### 搜索状态修复
- `milvus_store.py:insert()` 末尾添加 `load_collection()` 防止集合变为 `released` 状态
- `milvus_store.py:hybrid_search()` 开头添加 `load_collection()` 兜底
- `ui/app.py:_api()` 添加空响应检查和超时处理

### Windows 兼容性修复汇总
- 所有 emoji 打印替换为 ASCII 标签 (`[OK]`, `[WARN]`, `[info]`, `[drop]`, `[retry]`)
- `PYTHONIOENCODING=utf-8` 环境变量用于启动命令
- `os.rename` → `os.replace` 补丁
- LOCK 文件自动清理

---

## 2026-06-04 (Initial)

### 初始版本

- 完整的 RAG 系统脚手架
- 四域知识库结构 (变电/配电/送电输电/综合)
- 格式感知分块引擎 (语义/单页/全文)
- BGE-M3 嵌入 + 稀疏向量混合搜索
- 三阶段检索管道 (分析→召回→精排)
- 五种 LLM 提示词模板 (技术问答/文档查找/跨域对比/数值查规/通用)
- FastAPI REST API + Gradio UI
- 命令行索引工具 (`scripts/build_index.py`)
- SQLite 文件注册表 + Milvus Lite 向量库
