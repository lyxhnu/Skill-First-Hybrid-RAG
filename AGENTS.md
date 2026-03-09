# AGENTS.md

> 面向 AI Agent 的当前实现规范。本文档描述的是仓库里已经落地的 RAG 系统能力、硬约束和后续开发规则，不是纯设计草案。

## 1. 当前项目状态

当前仓库已经实现了一套可运行的 `Skill-first + FAISS Hybrid Retrieval + LangGraph` RAG 系统，核心特征如下：

1. `LangGraph` 负责编排完整问答流程。
2. `.agent/skills/` 下的 skills 已接入注册、选择和执行，不再只是静态文档。
3. `rag-skill` 是主检索技能，负责知识库目录路由、文件类型约束和检索执行。
4. `vector` 是补召回层，不是默认主路径。
5. 默认向量后端已经切到 `FAISS`，采用 `dense embedding + BM25 sparse` 的本地混合检索。
6. 已实现 `跨会话长期记忆 + SummaryBlocks + SlidingWindow` 的分层记忆架构。
7. 记忆层通过 `memory://sessions/{session_id}/turns/{turn_id}` 路径索引做无损压缩与按需回溯。
8. 模型层已支持多厂商切换，当前默认组合为：
   - `chat`: `zhipu / glm-5`
   - `embedding`: `bailian / text-embedding-v4`
   - `rerank`: `bailian / qwen3-rerank`
   - `vector backend`: `faiss`
9. 已实现电商客服问答闭环：未命中电商客服问题会自动进入待处理池，人工补答后回写 FAQ 并触发索引更新。
10. 系统已经暴露 API 和浏览器聊天页。

## 2. 当前主链路

当前查询主链路为：

`query -> load_memory_context -> analyze_query -> route_by_skill_index -> refine_query_plan -> run_skill_retrieval -> assess_evidence -> run_vector_retrieval(if needed) -> fuse_evidence -> rerank_evidence -> generate_answer -> verify_citations -> persist_memory`

对应实现文件：

1. `src/rag_graph/graph/workflow.py`
2. `src/rag_graph/service.py`

### 2.1 主链路行为

1. `load_memory_context`
   - 读取 `SlidingWindow`
   - 读取 `SummaryBlocks`
   - 读取跨会话长期记忆
   - 由模型判断是否需要把追问改写为更完整的 `effective_query`
   - 产出 `effective_query`、`memory_context`、`memory_trace`
2. `analyze_query`
   - 产出 `query_constraints`
   - 选择候选 skills
   - 记录当前生效模型
3. `route_by_skill_index`
   - 基于 `knowledge/*/data_structure.md` 做目录级和文件级路由
4. `refine_query_plan`
   - 结合候选文件再次细化 soft terms
5. `run_skill_retrieval`
   - 先执行 `rag-skill` 或其他选中 skills
6. `assess_evidence`
   - 判断 Skill 证据是否足够
7. `run_vector_retrieval`
   - 仅在 `hybrid/vector` 模式下运行
   - 默认受 `candidate_files` 限制，不允许无约束全库乱搜
   - 当 `vector_backend=faiss` 时，执行 `FAISS dense + BM25 sparse` 的本地混合检索
8. `fuse_evidence`
   - 以 `skill_weight / vector_weight` 融合 skill 与 vector 证据
9. `rerank_evidence`
   - 对融合后的证据再做重排
10. `generate_answer`
   - 没有证据时直接拒答，不调用模型硬编答案
11. `verify_citations`
   - 引用必须来自当前 evidence pool
12. `persist_memory`
   - 追加当前 turn
   - 达到阈值时生成 `SummaryBlocks`
   - 仅把高重要度、持久化项目/用户上下文写入跨会话长期记忆

## 3. 已实现架构

### 3.1 Skill Registry

系统会自动发现 `.agent/skills/*/SKILL.md` 并注册 skills。

当前相关实现：

1. `src/rag_graph/skill_runtime/registry.py`
2. `src/rag_graph/skill_runtime/manager.py`

当前可通过接口使用：

1. `GET /skills`
2. `POST /skills/execute`

### 3.2 Skill Router

目录路由与文件路由由 `SkillRouter` 实现：

1. 先基于 `data_structure.md` 做 lexical 路由
2. 如有 embedding 能力，再做语义重排
3. 返回 `candidate_dirs` 和 `candidate_files`

相关实现：

1. `src/rag_graph/skill_runtime/router.py`

### 3.3 Skill Retriever

`SkillRetriever` 当前负责：

1. 在已收敛候选文件内做 chunk 检索
2. PDF 命中页的相邻页补充
3. 对 PDF / Excel 证据附加 references 元数据
4. 对 JSON 记录支持字段级加权命中

相关实现：

1. `src/rag_graph/skill_runtime/retriever.py`

### 3.4 Excel Structured Analyzer

当前 Excel 分析已经不是“只看行文本”的简单检索，而是：

1. 先强制读取 `excel_reading.md`
2. 再强制读取 `excel_analysis.md`
3. 把这两份 references 注入模型 prompt
4. 由模型生成结构化分析计划
5. 用 `pandas` 执行计划
6. 输出结构化 evidence

当前已支持的通用操作：

1. `filter`
2. `extreme`（如 `max / min`）
3. `aggregate`（如 `count / sum / avg / max / min`）
4. `group_aggregate`

相关实现：

1. `src/rag_graph/skill_runtime/excel_analyzer.py`

### 3.5 Vector Store

当前向量补召回层默认使用 `FAISS + BM25` 本地混合索引。

特点：

1. 当前仅支持 `faiss`，其余残留旧配置值应视为无效并归一化到 `faiss`
2. 稠密检索使用 `FAISS IndexFlatIP`
3. 稀疏检索使用本地 BM25 统计索引
4. 查询时同时执行：
   - dense embedding 召回
   - BM25 稀疏召回
5. 混合排序支持：
   - `weighted`
   - `rrf`
6. 查询仍然支持 `allowed_files` 限制范围
7. 索引保存在 `storage/vector/faiss_hybrid_index.pkl`
8. 若本机未成功导入 `faiss`，系统会保留索引数据并回退到 `numpy` 稠密检索，但默认目标仍是使用 FAISS

相关实现：

1. `src/rag_graph/vector_store/index.py`

### 3.6 Model Layer

当前模型层包含：

1. `ModelGateway`
2. `EmbeddingGateway`
3. `RerankGateway`

支持情况：

1. `chat`
   - `builtin`
   - `openai/openai-compatible`
   - `bailian`
   - `zhipu`
   - `anthropic`
   - `gemini`
2. `embedding`
   - `local-hash`
   - `openai/openai-compatible`
   - `bailian`
   - `zhipu`
   - `gemini`
3. `rerank`
   - `builtin lexical`
   - `bailian rerank`
   - `openai/zhipu/gemini` 的 LLM fallback rerank

相关实现：

1. `src/rag_graph/models/providers.py`
2. `src/rag_graph/config.py`

### 3.7 Memory Layer

当前分层记忆由 `MemoryStore + MemoryManager` 实现。

层次如下：

1. `SlidingWindow`
   - 保存最近若干轮原始 turn
   - 用于最近上下文连续性
2. `SummaryBlocks`
   - 按 turn block 压缩会话
   - 每个 block 保留 `turn_ids` 与 `source_paths`
   - `source_paths` 使用 `memory://sessions/{session_id}/turns/{turn_id}` 作为路径索引
3. `LongTermMemory`
   - 面向 `actor_id` 跨会话持久化
   - 当前采取保守写入策略，只保留高重要度、持久化的用户/项目上下文

当前相关实现：

1. `src/rag_graph/memory/store.py`
2. `src/rag_graph/memory/manager.py`

### 3.8 当前记忆行为

1. `session_id` 表示会话边界
2. `actor_id` 表示跨会话身份边界
3. `effective_query` 是记忆上下文化后的检索查询
4. 原始 `query` 仍用于最终回答展示
5. `SummaryBlocks` 是压缩索引，不是最终事实来源
6. 需要事实级回溯时，系统会通过路径索引回捞原始 turns
7. 普通知识库答案默认停留在 `SlidingWindow + SummaryBlocks`，不会自动进入跨会话长期记忆

## 4. Skill 规则是硬约束

`rag-skill/SKILL.md` 的要求已经被视为系统硬约束，而不是参考建议。

### 4.1 PDF 必须先读 references

处理 PDF 前，必须先读取：

1. `.agent/skills/rag-skill/references/pdf_reading.md`

这条约束当前已在两层执行：

1. `ingest` 解析 PDF 前读取并写入 metadata
2. `rag-skill` 检索 PDF 证据前读取并写入 metadata

### 4.2 Excel 必须先读 references

处理 Excel 前，必须先读取：

1. `.agent/skills/rag-skill/references/excel_reading.md`
2. `.agent/skills/rag-skill/references/excel_analysis.md`

这条约束当前已在三层执行：

1. `ingest` 解析 Excel 前读取并写入 metadata
2. `rag-skill` 检索 Excel 证据前读取并写入 metadata
3. `ExcelStructuredAnalyzer` 真正分析 Excel 前，把两份 references 注入模型 prompt

### 4.3 References 可追踪

当前 PDF/Excel 相关 evidence metadata 会记录：

1. `references_loaded`
2. `reference_hashes`

后续变更不得删掉这两个字段。

## 5. 当前文件类型能力

### 5.1 Markdown / TXT

当前能力：

1. `ingest` 时分块
2. 行号 / 行区间可追踪
3. skill 检索走 lexical match
4. 可进入 `FAISS + BM25` 混合索引

### 5.2 PDF

当前能力：

1. 解析优先级：同名 `.txt` sidecar > `pypdf`
2. 保留页码定位
3. skill 检索命中后可补相邻页
4. 查询前和解析前都必须读取 `pdf_reading.md`
5. 可进入 `FAISS + BM25` 混合索引

当前限制：

1. 还没有通用的“PDF 结构化分析执行器”
2. 目前仍以页级 / chunk 级检索为主
3. 如果 `pypdf` 抽取失败且无 sidecar，则证据可能不足

### 5.3 Excel

当前能力：

1. `ingest` 时保留行级 chunk
2. skill 查询时优先走结构化分析器
3. 结构化分析由 references + 模型计划 + pandas 执行组成
4. 行级 chunk 也会进入 `FAISS + BM25` 混合索引
5. 可回答典型表分析问题，例如：
   - 哪个员工工资最高
   - 销售部门有哪些职工
   - 哪些商品库存不足

当前限制：

1. 还没有 SQL 执行层
2. 目前是 `pandas plan execution`，不是 `xlsx -> sqlite`
3. 跨多个工作簿 / 多表 join 的通用能力仍有限

### 5.4 JSON

当前能力：

1. `ingest` 已支持 `.json` 文件扫描、解析与入索引
2. 对 FAQ 风格的 JSON 记录按“单条记录 -> 单个 chunk”处理
3. 默认保留 `label / question / answer / url` 等字段到 evidence metadata
4. skill 检索已支持 JSON 字段级加权，优先提升 `question` 与 `label` 的命中
5. 可进入 `FAISS + BM25` 混合索引，并继续受 `candidate_files` 约束

当前限制：

1. 当前默认按 FAQ 风格 JSON 设计，复杂嵌套 JSON 仅提供保守 fallback
2. 还没有通用 JSON schema 自适应分析器
3. 当前没有为 JSON 引入类似 PDF / Excel 的强制 references 机制

## 6. 当前检索与回答规则

### 6.1 Skill-first

默认原则不变：

1. 先 Skill
2. 再决定是否补 vector
3. 不允许默认直接全库 vector 搜索

### 6.2 Vector-augmented

向量检索当前只作为补召回层，主要规则：

1. `mode=skill` 不跑 vector
2. `mode=vector` 直接跑 vector
3. `mode=hybrid` 下，若 skill 命中不足才跑 vector
4. vector 默认受 `candidate_files` 约束
5. 默认 vector 后端为 `FAISS + BM25` 本地混合检索
6. 混合检索会同时结合：
   - dense embedding 召回
   - BM25 稀疏召回
7. 混合检索返回结果之后，仍然进入本系统的 `fuse -> rerank -> generate` 链路

### 6.3 无证据拒答

当前系统是硬门控：

1. 没有 evidence 时直接返回“无法回答”
2. 不允许模型在没有引用支撑时自由发挥

### 6.4 Confidence 说明

当前 `confidence` 仍是检索链路里的近似分数，不等于答案正确率。

它更接近：

1. Skill 顶部证据强度
2. 而不是最终答案可靠性评分

后续如修改 `confidence` 语义，必须同步更新文档和前端显示。

## 7. 当前 API 与前端

已实现接口：

1. `GET /`
   - 聊天页
2. `GET /chat`
   - 聊天页
3. `GET /health`
4. `POST /ingest`
5. `POST /query`
6. `POST /evaluate`
7. `GET /skills`
8. `POST /skills/execute`
9. `GET /customer-service/gaps`
10. `POST /customer-service/gaps/{gap_id}/resolve`

相关实现：

1. `src/rag_graph/api/main.py`
2. `src/rag_graph/api/static/index.html`

### 7.1 当前聊天页能力

当前聊天页支持：

1. 发送 `/query`
2. 切换 `skill/hybrid/vector`
3. 调整 `top_k`
4. 点击“重建索引”
5. 创建新会话
6. 浏览器本地保存 `actor_id` 和当前 `session_id`
7. 查看健康状态和命中统计
8. 展示当前 `vector_backend`

### 7.2 /query 会话字段

当前 `/query` 支持可选字段：

1. `session_id`
2. `actor_id`

若不显式传入，则使用默认值：

1. `default-session`
2. `default-user`

## 8. 当前可观测性与诊断字段

### 8.1 /query 返回字段

当前 `/query` 会返回：

1. `query_constraints`
2. `selected_skills`
3. `candidate_dirs`
4. `candidate_files`
5. `effective_query`
6. `memory_trace`
7. `confidence`
8. `evidence_trace`
9. `selected_models`
10. `customer_service_feedback`
11. `answer_support`
12. `evidence_preview`

### 8.2 /health 返回字段

当前 `/health` 会返回：

1. `pid`
2. `startup_time`
3. `project_root`
4. `service_file`
5. `vector_backend`
6. `vector_index_path`
7. `vector_index_metadata`
8. `memory_dir`
9. `customer_service_feedback`

这些字段是为了快速判断：

1. 当前连到的是不是正确进程
2. 当前进程是否加载了正确代码
3. 当前向量后端是不是预期的 `faiss`
4. 当前索引是否与 embedding 配置匹配
5. 当前记忆存储是否指向正确目录
6. 当前电商客服待处理池是否在持续积压

## 9. 当前运行要求

### 9.1 解释器一致性

启动和调试必须使用同一个 Python 解释器。

这是硬要求，因为：

1. 不同 conda 环境会导致服务行为与本地脚本行为不一致
2. 未设置 `PYTHONPATH=src` 时，可能根本导不进当前工作区代码

### 9.2 FAISS 依赖要求

默认 `vector_backend=faiss`，因此运行前必须满足：

1. 已安装 `faiss-cpu`
2. `RAG_VECTOR_BACKEND` 保持为 `faiss`
3. 已执行一次 `/ingest` 构建本地混合索引

### 9.3 重启要求

每次改动以下任一内容后，都必须重启服务：

1. `src/rag_graph/**/*.py`
2. `.agent/skills/rag-skill/references/*.md`
3. `config.py`
4. `AGENTS.md` 变更后如果前端/接口依赖了新增约定，也必须同步重启验证

### 9.4 重建索引要求

出现以下情况时，必须重新执行 `/ingest`：

1. `knowledge/` 文件变化
2. embedding provider 或 embedding model 变化
3. BM25 参数变化
4. FAISS 混合排序参数变化
5. 需要把最新 PDF/Excel references metadata 写入 parsed chunks

## 10. 当前开发边界

以下能力当前还没有完成，后续开发时必须视为扩展项，而不是现成功能：

1. 通用 SQL 表分析执行层
2. 通用 PDF 结构化规划执行器
3. 基于 skill references 的 PDF planner
4. 多表 join / 多工作簿联合分析
5. 基于最终答案质量的统一 confidence
6. 激进型的跨会话长期事实记忆
7. 更强的中文 BM25 分词优化与可插拔 analyzer

### 10.1 当前记忆边界

当前分层记忆已经可用，但要明确边界：

1. `SlidingWindow` 与 `SummaryBlocks` 已经是主路径能力
2. `LongTermMemory` 当前是保守写入
3. 长期记忆更适合保存用户偏好、约束、目标、身份、持续项目上下文
4. 普通知识库问答结果不应默认写入长期记忆

## 10. 当前电商客服问答闭环

### 10.1 触发条件

仅当同时满足以下条件时，才会自动记录缺口问题：

1. 当前问题已经完成一次完整检索
2. 本轮最终没有检索到任何明确可用的知识证据

### 10.2 存储位置

1. 待处理池：`storage/feedback/customer_service_gaps.json`
2. FAQ 问答库：`knowledge/E-commerce Data/faq.json`

### 10.3 当前闭环行为

1. `RAGService.query` 在工作流执行结束后判断是否需要记录客服知识缺口
2. 当前判定基于 `answer_support.explicit`，不再要求必须路由到 `faq.json`
3. 相同问题按归一化后的 `question_key` 去重，只累计 `hits`
4. 人工补答后调用 `POST /customer-service/gaps/{gap_id}/resolve`
5. 新答案会写回 `faq.json`
6. 默认自动执行一次 `/ingest`
7. 若同一问题已经被人工补答，但后续仍然没有明确证据支撑，则记为 `resolved_regression`

### 10.4 当前边界

1. 当前“是否属于电商客服问题”的判断基于现有路由结果，不额外引入新的 LLM 分类器
2. 当前待处理池使用本地 JSON 文件，不包含跨进程文件锁
3. 当前回写目标固定为 `knowledge/E-commerce Data/faq.json`

## 11. 后续变更硬约束

### 11.1 不能绕开 skill assets

后续开发不得：

1. 绕开 `.agent/skills/rag-skill/SKILL.md`
2. 绕开 `references/pdf_reading.md`
3. 绕开 `references/excel_reading.md`
4. 绕开 `references/excel_analysis.md`

如果引入新能力，必须说明它如何与现有 skill 规则兼容。

### 11.2 不能破坏 skill-first

后续任何改动都不能把系统改成：

1. 默认全库向量召回
2. 向量优先、skill 兜底
3. 无引用生成

### 11.3 不能破坏 FAISS 混合检索约束

任何涉及向量层的改动，都必须满足：

1. 默认后端仍为 `faiss`
2. `mode=hybrid` 下的向量补召回仍受 `candidate_files` 约束
3. 混合检索仍同时考虑稠密向量与 BM25 稀疏检索
4. 向量后端 metadata 必须记录当前 provider / model / backend
5. 不得把系统改回单纯 embedding 点积检索

### 11.4 PDF / Excel 变更规则

任何涉及 PDF / Excel 的代码变更，都必须满足：

1. 处理前加载对应 references
2. evidence metadata 可证明 references 已加载
3. 保持可追溯引用
4. 不允许为了性能删掉 references 读取步骤

### 11.5 记忆层变更规则

任何涉及分层记忆的改动，都必须满足：

1. 不得删除 `session_id` / `actor_id` 这两个边界概念
2. 不得删除 `effective_query`
3. 不得删除 `memory_trace`
4. 不得破坏 `SummaryBlocks` 的 `source_paths`
5. 不得把 `memory://sessions/{session_id}/turns/{turn_id}` 路径索引改成不可回溯格式
6. 不得把普通知识库答案无门槛刷入跨会话长期记忆
7. 需要事实回溯时，必须优先回捞原始 turns，而不是只依赖摘要文本

### 11.6 API 兼容性

以下接口默认视为稳定接口，不应随意删除：

1. `POST /query`
2. `POST /ingest`
3. `GET /health`
4. `POST /evaluate`
5. `GET /skills`
6. `POST /skills/execute`
7. `GET /customer-service/gaps`
8. `POST /customer-service/gaps/{gap_id}/resolve`

### 11.7 诊断字段不能随意删

以下字段默认保留：

1. `query_constraints`
2. `selected_skills`
3. `candidate_dirs`
4. `candidate_files`
5. `effective_query`
6. `memory_trace`
7. `evidence_trace`
8. `pid`
9. `startup_time`
10. `project_root`
11. `service_file`
12. `vector_backend`
13. `vector_index_metadata`
14. `memory_dir`
15. `customer_service_feedback`

### 11.8 不能破坏电商客服问答闭环

任何涉及电商 FAQ、拒答逻辑或客服域路由的改动，都必须满足：

1. 电商客服未命中问题仍能被记录到待处理池
2. 人工补答后仍能回写 `knowledge/E-commerce Data/faq.json`
3. 回写 FAQ 后仍默认触发索引更新
4. `/query` 返回中的 `customer_service_feedback` 字段不得删除
5. 不得把待处理池实现成只留日志、不支持回写的单向流程

### 11.9 每次改动的最小回归集合

后续每次改动，至少回归以下问题：

1. `三一重工的前三大股东是谁？`
2. `第二个是谁？`
3. `那第一个是谁`
4. `哪个员工的工资最高`
5. `他在哪个部门`
6. `销售部门有哪些职工`
7. `哪些商品库存不足`

另外，凡是改动了记忆层，还必须验证：

1. `effective_query` 是否正确包含追问上下文
2. `SummaryBlocks` 是否生成
3. `SummaryBlocks.source_paths` 是否可回溯到原始 turn
4. 普通业务问答是否没有被错误写入跨会话长期记忆

如果其中任意一个退化，必须先修回归，再做新功能。

## 12. 推荐后续方向

后续如果继续增强系统，优先顺序建议如下：

1. Excel 从 `pandas plan execution` 升级到 `xlsx -> sqlite -> SQL execution`
2. PDF 从 chunk retrieval 升级到 `references-driven planner + extraction`
3. 统一最终 confidence 计算
4. 更强的 skill policy 抽取，而不是继续堆题型特判
5. 长期记忆从当前保守写入升级到更可靠的 durable-project-memory 抽取
6. BM25 分词与 analyzers 做成可插拔

## 13. 完成定义

当前仓库的完成定义不是“所有 RAG 功能都完备”，而是：

1. Skill-first 主链路可运行
2. Hybrid 补召回可运行
3. 默认 `FAISS + BM25` 混合检索可运行
4. PDF / Excel references 约束已真正进入执行链路
5. Excel 已具备通用结构化分析能力的基础版本
6. 分层记忆已接入主链路，支持跨轮追问与压缩摘要回溯
7. API、前端、索引、模型网关、诊断字段可协同工作

后续任何变更都应以不破坏上述七点为前提。
