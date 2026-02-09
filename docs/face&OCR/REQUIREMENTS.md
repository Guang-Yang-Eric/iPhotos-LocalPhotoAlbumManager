# iPhotron — 人脸识别 / OCR 文字识别 需求文档

> **版本**: v1.0.0
> **状态**: Draft
> **最后更新**: 2026-02-08
> **所属模块**: `face_recognition` / `ocr_text`
> **依赖**: OpenCV 4.x, CUDA Toolkit >= 11.8, dlib / face_recognition, PaddleOCR / Tesseract

---

## 目录

1. [概述](#1-概述)
2. [术语与缩写](#2-术语与缩写)
3. [功能性需求](#3-功能性需求)
4. [非功能性需求](#4-非功能性需求)
5. [数据库设计](#5-数据库设计)
6. [队列与 Worker 架构](#6-队列与-worker-架构)
7. [接口定义](#7-接口定义)
8. [约束与假设](#8-约束与假设)
9. [参考](#9-参考)

---

## 1. 概述

本文档定义 iPhotron 本地相册管理器中 **人脸识别 / 聚类** 与 **OCR 文字识别** 两大子系统的完整需求。

设计目标：

| 目标 | 说明 |
|------|------|
| **离线优先** | 所有模型推理在本地完成，不依赖云端 API |
| **非侵入式** | 人脸 / OCR 处理在独立队列中运行，不阻塞主库索引流水线 |
| **独立存储** | 人脸数据和 OCR 数据分别存储在 `face_index.db` 和 `ocr_index.db` 中，与主库 `global_index.db` 物理隔离 |
| **GPU 可选** | 自动探测 CUDA 环境，有 GPU 时启用加速，无 GPU 时回退 CPU |
| **可扩展** | 支持未来替换检测 / 识别 / OCR 后端（如 InsightFace, EasyOCR 等） |

参考项目与行业惯例：

- **digiKam** — KDE 开源相册，内置人脸识别与标签管理
- **PhotoPrism** — Go 语言开源相册，TensorFlow 人脸检测 + 聚类
- **Immich** — 自托管相册，基于机器学习的人脸识别管线
- **Apple Photos / Google Photos** — 商业级人脸聚类与 OCR 搜索体验

---

## 2. 术语与缩写

| 缩写 / 术语 | 说明 |
|---|---|
| **Face Embedding** | 人脸特征向量（128 维 / 512 维浮点数组），用于衡量两张人脸的相似度 |
| **Cluster** | 聚类，一组被算法判定为同一人物的人脸集合 |
| **Person** | 用户确认 / 命名后的人物实体，可关联一个或多个 Cluster |
| **ROI** | Region of Interest，人脸在原图中的矩形区域 |
| **OCR** | Optical Character Recognition，光学字符识别 |
| **BM25** | 一种经典的全文检索排序算法 |
| **FTS5** | SQLite 全文搜索扩展 |
| **DFS** | Deficit Fair Scheduling，赤字公平调度 |
| **WAL** | Write-Ahead Logging，SQLite 并发写入模式 |

---

## 3. 功能性需求

### 3.1 人脸检测

| ID | 需求 | 优先级 |
|---|---|---|
| FR-DET-01 | 支持基于 OpenCV DNN 模块加载人脸检测模型（如 `YuNet`, `SSD`, `SCRFD`） | P0 |
| FR-DET-02 | 对每张图片返回零到多个人脸边界框（Bounding Box），含置信度分数 | P0 |
| FR-DET-03 | 支持最小人脸尺寸阈值设置（默认 >= 40x40 px），过滤过小人脸 | P1 |
| FR-DET-04 | 支持人脸关键点检测（5 点 / 68 点 Landmark），用于对齐预处理 | P1 |
| FR-DET-05 | 检测结果写入 `face_index.db` 的 `faces` 表 | P0 |
| FR-DET-06 | 支持 CUDA 加速推理（通过 `cv2.dnn.DNN_BACKEND_CUDA`） | P0 |
| FR-DET-07 | 支持对视频帧采样后执行人脸检测（默认每 2 秒取 1 帧） | P2 |

### 3.2 人脸特征提取与编码

| ID | 需求 | 优先级 |
|---|---|---|
| FR-ENC-01 | 使用 OpenCV DNN 或 dlib 提取 128-D / 512-D 人脸嵌入向量 | P0 |
| FR-ENC-02 | 人脸图像在提取前进行对齐（仿射变换到标准 112x112 或 160x160） | P0 |
| FR-ENC-03 | 嵌入向量以 BLOB 格式存储在 `face_embeddings` 表中 | P0 |
| FR-ENC-04 | 支持批量编码（Batch Inference），提升 GPU 利用率 | P1 |
| FR-ENC-05 | 编码后向量需进行 L2 归一化（单位向量），便于后续余弦相似度计算 | P0 |

### 3.3 人脸聚类

| ID | 需求 | 优先级 |
|---|---|---|
| FR-CLU-01 | 基于 DBSCAN / Chinese Whispers / HAC 算法，对人脸嵌入向量进行无监督聚类 | P0 |
| FR-CLU-02 | 聚类阈值可配置（默认余弦距离 <= 0.6 视为同一人） | P0 |
| FR-CLU-03 | 新照片入库后支持增量聚类（不必全量重算） | P1 |
| FR-CLU-04 | 聚类结果写入 `face_clusters` 表，并更新 `faces.cluster_id` | P0 |
| FR-CLU-05 | 提供手动触发全量重聚类的接口 | P2 |
| FR-CLU-06 | 噪声点（未能归入任何聚类的人脸）标记为 `cluster_id = NULL` | P0 |

### 3.4 聚类管理操作

| ID | 需求 | 优先级 |
|---|---|---|
| FR-MGT-01 | 用户可为聚类命名（关联到 `persons` 表） | P0 |
| FR-MGT-02 | 用户可合并两个或多个聚类为一个人物（Merge Clusters） | P0 |
| FR-MGT-03 | 用户可将单张人脸从一个聚类移动到另一个聚类 | P0 |
| FR-MGT-04 | 用户可将人脸标记为「非人脸 / 误检」，从聚类中排除 | P1 |
| FR-MGT-05 | 合并 / 移动操作需要更新聚类中心向量（Centroid） | P1 |
| FR-MGT-06 | 所有聚类管理操作需记录审计日志到 `face_audit_log` 表 | P2 |
| FR-MGT-07 | 人物列表视图按人脸数量降序排列，支持缩略图预览 | P0 |

### 3.5 OCR 文字识别

| ID | 需求 | 优先级 |
|---|---|---|
| OCR-01 | 支持基于 PaddleOCR / Tesseract / EasyOCR 对图片进行文字识别 | P0 |
| OCR-02 | 识别结果包含：文字内容、置信度、文字区域坐标（多边形） | P0 |
| OCR-03 | 支持中文、英文、日文等多语言识别 | P0 |
| OCR-04 | 识别结果写入 `ocr_index.db` 的 `ocr_results` 表 | P0 |
| OCR-05 | 对图片全文建立 SQLite FTS5 全文索引（`ocr_fts` 虚拟表） | P0 |
| OCR-06 | 支持 CUDA 加速（PaddleOCR GPU 模式 / Tesseract + cuDNN） | P1 |
| OCR-07 | 支持旋转 / 倾斜文字检测与矫正 | P2 |
| OCR-08 | 支持对视频关键帧进行 OCR | P2 |

### 3.6 文字搜索图片

| ID | 需求 | 优先级 |
|---|---|---|
| OCR-SEARCH-01 | 用户在搜索栏输入文字，返回包含该文字的图片列表 | P0 |
| OCR-SEARCH-02 | 搜索基于 FTS5 全文索引，支持分词匹配与前缀匹配 | P0 |
| OCR-SEARCH-03 | 搜索结果按 BM25 相关度排序，返回匹配的文字片段高亮 | P1 |
| OCR-SEARCH-04 | 搜索结果中显示匹配文字在图片中的位置区域 | P2 |
| OCR-SEARCH-05 | 搜索延迟 <= 200ms（10 万张图片库） | P1 |

---

## 4. 非功能性需求

### 4.1 CUDA GPU 加速

| ID | 需求 | 说明 |
|---|---|---|
| NF-GPU-01 | 启动时自动探测 CUDA 可用性（`cv2.cuda.getCudaEnabledDeviceCount()`） | 无 CUDA 时静默回退 CPU |
| NF-GPU-02 | GPU 显存使用上限可配置（默认使用可用显存的 70%） | 避免与用户其他 GPU 任务冲突 |
| NF-GPU-03 | 支持多 GPU 环境下指定使用的设备 ID | 通过配置文件 `cuda_device_id` 设定 |
| NF-GPU-04 | OpenCV DNN 模块使用 `DNN_BACKEND_CUDA` + `DNN_TARGET_CUDA` | 人脸检测与编码模型加速 |
| NF-GPU-05 | PaddleOCR 使用 `use_gpu=True` + `gpu_mem` 参数 | OCR 推理加速 |
| NF-GPU-06 | 批量推理大小根据 GPU 显存自适应调整 | 默认 batch_size=16，显存不足时自动减半 |

### 4.2 多队列架构与资源调度

```
+----------------------------------------------------------------------+
|                         主进程 (Main Process)                        |
|                                                                      |
|  +---------------+    事件总线 (EventBus)                            |
|  | 文件扫描器    |------------------------------------+              |
|  | FileScanner   |                                    |              |
|  +------+--------+                                    |              |
|         |                                             v              |
|         | AssetScannedEvent              +---------------------+     |
|         |                                |   队列分发器        |     |
|         v                                |   QueueDispatcher   |     |
|  +---------------+                       +----+-------+--------+     |
|  | 主队列        |                            |       |              |
|  | MainQueue     |                            v       v              |
|  | (入主库)      |                     +--------+ +--------+        |
|  +------+--------+                     |Face    | |OCR     |        |
|         |                              |Queue   | |Queue   |        |
|         v                              +---+----+ +---+----+        |
|  +---------------+                         |          |              |
|  |global_index   |                         v          v              |
|  |    .db        |                    +---------+ +---------+       |
|  +---------------+                    |face_    | |ocr_     |       |
|                                       |index.db | |index.db |       |
|                                       +---------+ +---------+       |
+----------------------------------------------------------------------+
```

| ID | 需求 | 说明 |
|---|---|---|
| NF-QUEUE-01 | 主队列（MainQueue）负责将资产元数据写入 `global_index.db`，拥有最高优先级 | 主库入库不可被人脸 / OCR 任务阻塞 |
| NF-QUEUE-02 | 人脸队列（FaceQueue）独立运行，负责人脸检测、编码、聚类任务 | 写入 `face_index.db` |
| NF-QUEUE-03 | OCR 队列（OCRQueue）独立运行，负责文字识别与全文索引构建 | 写入 `ocr_index.db` |
| NF-QUEUE-04 | 各队列使用独立的 Worker 线程 / 进程 | 线程 / 进程数可配置 |
| NF-QUEUE-05 | 新增资产时，主队列入库完成后，向人脸队列和 OCR 队列发送任务 | 通过事件总线 `AssetIndexedEvent` 触发 |
| NF-QUEUE-06 | 次队列（人脸 / OCR）在 CPU 密集型任务中使用 `ProcessPoolExecutor` | 绕过 Python GIL 限制 |

### 4.3 性能指标

| 场景 | 目标 | 说明 |
|---|---|---|
| 单张图片人脸检测（GPU） | <= 30ms | YuNet @ 640x640 输入 |
| 单张图片人脸检测（CPU） | <= 150ms | YuNet @ 640x640 输入 |
| 单张人脸编码（GPU） | <= 10ms | ArcFace / SFace 模型 |
| 单张人脸编码（CPU） | <= 80ms | ArcFace / SFace 模型 |
| 10000 张人脸聚类 | <= 5s | DBSCAN on CPU |
| 单张图片 OCR（GPU） | <= 100ms | PaddleOCR v4 |
| 单张图片 OCR（CPU） | <= 500ms | PaddleOCR v4 |
| FTS5 文字搜索（10 万条记录） | <= 200ms | BM25 排序 |
| 主库入库吞吐量 | >= 500 张/s | 不受人脸/OCR影响 |

### 4.4 可靠性与容错

| ID | 需求 | 说明 |
|---|---|---|
| NF-REL-01 | 人脸 / OCR 处理失败不影响主库数据完整性 | 独立数据库、独立队列 |
| NF-REL-02 | 单张图片处理失败后标记为 `status = 'error'`，不阻塞队列 | 支持后续手动重试 |
| NF-REL-03 | Worker 异常退出后自动重启（最多 3 次），超过后进入降级模式 | 降级模式下暂停该队列，等待用户干预 |
| NF-REL-04 | 数据库使用 WAL 模式，支持读写并发 | 避免锁竞争 |
| NF-REL-05 | 支持断点续处理：记录每张图片的处理状态，中断后从上次位置恢复 | `processing_status` 表 |
| NF-REL-06 | 所有数据库操作使用事务，确保原子性 | 批量写入时使用 `BEGIN IMMEDIATE` |

---

## 5. 数据库设计

### 5.1 数据库拓扑

```
<library_root>/
|-- global_index.db          <-- 主库（现有）
|-- face_index.db            <-- 人脸数据库（新增）
+-- ocr_index.db             <-- OCR 数据库（新增）
```

三个数据库物理独立，通过 `asset_rel`（资产在库中的相对路径）字段进行逻辑关联。这与主库中 `assets` 表的 `rel` 主键保持一致。

> **设计理由**：
> - 独立数据库允许人脸 / OCR 模块独立升级、迁移、重建，不影响主库
> - 避免主库膨胀（人脸嵌入向量为二进制大字段）
> - 不同数据库可分别使用不同的 WAL checkpoint 策略
> - 参考 digiKam 的 `recognition.db` + `thumbnails.db` 分库策略

---

### 5.2 人脸数据库 `face_index.db`

#### 表 `faces` — 人脸检测结果

| 列名 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `face_id` | TEXT | PRIMARY KEY | 人脸唯一 ID（UUID） |
| `asset_rel` | TEXT | NOT NULL, INDEX | 关联主库 `assets.rel`，资产相对路径 |
| `bbox_x` | INTEGER | NOT NULL | 人脸边界框左上角 X 坐标（像素） |
| `bbox_y` | INTEGER | NOT NULL | 人脸边界框左上角 Y 坐标（像素） |
| `bbox_w` | INTEGER | NOT NULL | 人脸边界框宽度（像素） |
| `bbox_h` | INTEGER | NOT NULL | 人脸边界框高度（像素） |
| `confidence` | REAL | NOT NULL | 检测置信度 (0.0 ~ 1.0) |
| `landmarks` | TEXT | | JSON 格式的关键点坐标数组 |
| `cluster_id` | TEXT | INDEX, FK -> face_clusters | 所属聚类 ID（NULL = 噪声点/未聚类） |
| `person_id` | TEXT | INDEX, FK -> persons | 所属人物 ID（NULL = 未命名） |
| `is_excluded` | INTEGER | DEFAULT 0 | 1 = 用户标记为误检，排除出聚类 |
| `source_type` | TEXT | DEFAULT 'image' | 来源类型：'image' / 'video_frame' |
| `frame_timestamp` | REAL | | 若来源为视频，提取帧的时间戳（秒） |
| `detected_at` | TEXT | NOT NULL | 检测时间 ISO 8601 |
| `model_name` | TEXT | NOT NULL | 使用的检测模型名称（如 'yunet_v2'） |

```sql
CREATE INDEX idx_faces_asset_rel ON faces(asset_rel);
CREATE INDEX idx_faces_cluster_id ON faces(cluster_id);
CREATE INDEX idx_faces_person_id ON faces(person_id);
CREATE INDEX idx_faces_confidence ON faces(confidence DESC);
```

#### 表 `face_embeddings` — 人脸特征向量

| 列名 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `face_id` | TEXT | PRIMARY KEY, FK -> faces | 人脸 ID |
| `embedding` | BLOB | NOT NULL | 128-D / 512-D 浮点向量（numpy -> bytes） |
| `embedding_dim` | INTEGER | NOT NULL | 向量维度（128 或 512） |
| `model_name` | TEXT | NOT NULL | 编码模型名称（如 'sface_v1', 'arcface_r50'） |
| `norm` | REAL | | L2 范数（归一化后应约等于 1.0） |
| `encoded_at` | TEXT | NOT NULL | 编码时间 ISO 8601 |

#### 表 `face_clusters` — 聚类信息

| 列名 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `cluster_id` | TEXT | PRIMARY KEY | 聚类 UUID |
| `person_id` | TEXT | FK -> persons | 关联的人物 ID（NULL = 未命名聚类） |
| `face_count` | INTEGER | DEFAULT 0 | 聚类中的人脸数量（冗余字段，加速查询） |
| `centroid` | BLOB | | 聚类中心向量（所有成员嵌入的均值） |
| `centroid_dim` | INTEGER | | 中心向量维度 |
| `representative_face_id` | TEXT | FK -> faces | 代表人脸 ID（距中心最近的人脸，用于缩略图） |
| `created_at` | TEXT | NOT NULL | 创建时间 |
| `updated_at` | TEXT | NOT NULL | 最后更新时间 |
| `algorithm` | TEXT | NOT NULL | 使用的聚类算法（如 'dbscan', 'chinese_whispers'） |
| `threshold` | REAL | | 聚类时使用的距离阈值 |

#### 表 `persons` — 人物信息

| 列名 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `person_id` | TEXT | PRIMARY KEY | 人物 UUID |
| `name` | TEXT | NOT NULL | 用户给定的人物名称 |
| `avatar_face_id` | TEXT | FK -> faces | 头像人脸 ID |
| `face_count` | INTEGER | DEFAULT 0 | 该人物关联的总人脸数 |
| `is_hidden` | INTEGER | DEFAULT 0 | 用户隐藏此人物 |
| `created_at` | TEXT | NOT NULL | 创建时间 |
| `updated_at` | TEXT | NOT NULL | 最后更新时间 |

```sql
CREATE INDEX idx_persons_name ON persons(name);
```

#### 表 `face_audit_log` — 操作审计日志

| 列名 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `log_id` | TEXT | PRIMARY KEY | 日志 UUID |
| `action` | TEXT | NOT NULL | 操作类型：'merge_clusters' / 'move_face' / 'rename_person' / 'exclude_face' / 'recluster' |
| `source_cluster_id` | TEXT | | 来源聚类 ID |
| `target_cluster_id` | TEXT | | 目标聚类 ID |
| `face_id` | TEXT | | 涉及的人脸 ID |
| `person_id` | TEXT | | 涉及的人物 ID |
| `detail` | TEXT | | JSON 格式详细信息 |
| `created_at` | TEXT | NOT NULL | 操作时间 |

#### 表 `face_processing_status` — 处理状态追踪

| 列名 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `asset_rel` | TEXT | PRIMARY KEY | 资产相对路径 |
| `detection_status` | TEXT | DEFAULT 'pending' | 检测状态：'pending' / 'processing' / 'done' / 'error' / 'skipped' |
| `encoding_status` | TEXT | DEFAULT 'pending' | 编码状态 |
| `clustering_status` | TEXT | DEFAULT 'pending' | 聚类状态 |
| `face_count` | INTEGER | DEFAULT 0 | 检测到的人脸数 |
| `error_message` | TEXT | | 错误信息 |
| `retry_count` | INTEGER | DEFAULT 0 | 重试次数 |
| `started_at` | TEXT | | 开始处理时间 |
| `completed_at` | TEXT | | 完成处理时间 |
| `updated_at` | TEXT | NOT NULL | 最后更新时间 |

```sql
CREATE INDEX idx_face_status_detection ON face_processing_status(detection_status);
CREATE INDEX idx_face_status_encoding ON face_processing_status(encoding_status);
```

---

### 5.3 OCR 数据库 `ocr_index.db`

#### 表 `ocr_results` — OCR 识别结果

| 列名 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `ocr_id` | TEXT | PRIMARY KEY | OCR 结果 UUID |
| `asset_rel` | TEXT | NOT NULL, INDEX | 关联主库 `assets.rel` |
| `text_content` | TEXT | NOT NULL | 识别出的文字内容 |
| `confidence` | REAL | NOT NULL | 识别置信度 (0.0 ~ 1.0) |
| `bbox_points` | TEXT | NOT NULL | JSON 格式多边形坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] |
| `language` | TEXT | | 检测到的语言代码（'zh', 'en', 'ja' 等） |
| `line_index` | INTEGER | DEFAULT 0 | 同一图片中的文字行序号 |
| `source_type` | TEXT | DEFAULT 'image' | 来源类型：'image' / 'video_frame' |
| `frame_timestamp` | REAL | | 若来源为视频，提取帧的时间戳（秒） |
| `detected_at` | TEXT | NOT NULL | 识别时间 |
| `model_name` | TEXT | NOT NULL | 使用的 OCR 模型名称 |

```sql
CREATE INDEX idx_ocr_asset_rel ON ocr_results(asset_rel);
CREATE INDEX idx_ocr_confidence ON ocr_results(confidence DESC);
CREATE INDEX idx_ocr_language ON ocr_results(language);
```

#### 虚拟表 `ocr_fts` — 全文搜索索引 (FTS5)

```sql
CREATE VIRTUAL TABLE ocr_fts USING fts5(
    asset_rel,          -- 资产路径（用于 JOIN 查询）
    text_content,       -- 可搜索的文字内容
    content='ocr_results',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
    -- 中文环境可替换为: tokenize='unicode61' 或使用 simple/jieba 分词器
);
```

> **分词器说明**：
> - `unicode61`：SQLite 内置 Unicode 分词器，支持基于 Unicode 标准的断词
> - 中文全文搜索推荐使用 simple 分词器或 jieba 预分词后写入
> - 也可考虑使用 trigram tokenizer: `tokenize='trigram'` 支持子串匹配

#### 同步触发器

```sql
-- 插入时同步到 FTS 索引
CREATE TRIGGER ocr_fts_insert AFTER INSERT ON ocr_results BEGIN
    INSERT INTO ocr_fts(rowid, asset_rel, text_content)
    VALUES (NEW.rowid, NEW.asset_rel, NEW.text_content);
END;

-- 删除时同步到 FTS 索引
CREATE TRIGGER ocr_fts_delete AFTER DELETE ON ocr_results BEGIN
    INSERT INTO ocr_fts(ocr_fts, rowid, asset_rel, text_content)
    VALUES ('delete', OLD.rowid, OLD.asset_rel, OLD.text_content);
END;

-- 更新时同步到 FTS 索引
CREATE TRIGGER ocr_fts_update AFTER UPDATE ON ocr_results BEGIN
    INSERT INTO ocr_fts(ocr_fts, rowid, asset_rel, text_content)
    VALUES ('delete', OLD.rowid, OLD.asset_rel, OLD.text_content);
    INSERT INTO ocr_fts(rowid, asset_rel, text_content)
    VALUES (NEW.rowid, NEW.asset_rel, NEW.text_content);
END;
```

#### 表 `ocr_processing_status` — 处理状态追踪

| 列名 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `asset_rel` | TEXT | PRIMARY KEY | 资产相对路径 |
| `status` | TEXT | DEFAULT 'pending' | 状态：'pending' / 'processing' / 'done' / 'error' / 'skipped' |
| `text_block_count` | INTEGER | DEFAULT 0 | 识别出的文字块数 |
| `error_message` | TEXT | | 错误信息 |
| `retry_count` | INTEGER | DEFAULT 0 | 重试次数 |
| `started_at` | TEXT | | 开始处理时间 |
| `completed_at` | TEXT | | 完成处理时间 |
| `updated_at` | TEXT | NOT NULL | 最后更新时间 |

```sql
CREATE INDEX idx_ocr_status ON ocr_processing_status(status);
```

#### 表 `ocr_asset_text` — 每资产聚合文本（加速搜索）

| 列名 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `asset_rel` | TEXT | PRIMARY KEY | 资产相对路径 |
| `full_text` | TEXT | NOT NULL | 该图片中所有 OCR 文字拼接后的完整文本 |
| `language_summary` | TEXT | | 主要语言（出现最多的语言代码） |
| `total_confidence` | REAL | | 所有文字块的平均置信度 |
| `updated_at` | TEXT | NOT NULL | 最后更新时间 |

---

### 5.4 主库关联字段

主库 `global_index.db` 的 `assets` 表 **不修改**。人脸 / OCR 数据通过 `asset_rel` (= `assets.rel`) 进行跨库逻辑关联。

查询示例（使用 `ATTACH DATABASE`）：

```sql
-- 附加人脸数据库
ATTACH DATABASE 'face_index.db' AS face_db;

-- 查询某人物的所有照片
SELECT a.*
FROM assets a
JOIN face_db.faces f ON f.asset_rel = a.rel
JOIN face_db.persons p ON p.person_id = f.person_id
WHERE p.name = '张三';

-- 附加 OCR 数据库
ATTACH DATABASE 'ocr_index.db' AS ocr_db;

-- 文字搜索图片
SELECT a.*, snippet(ocr_db.ocr_fts, 1, '<b>', '</b>', '...', 32) AS matched_text
FROM ocr_db.ocr_fts
JOIN assets a ON a.rel = ocr_db.ocr_fts.asset_rel
WHERE ocr_db.ocr_fts MATCH '发票'
ORDER BY rank;
```

---

## 6. 队列与 Worker 架构

### 6.1 队列拓扑

```
AssetScannedEvent
       |
       v
+------------------+
|  QueueDispatcher  |
|  (事件监听器)      |
+--+-------+-------++
   |       |       |
   v       v       v
+------+ +------+ +------+
|Main  | |Face  | |OCR   |
|Queue | |Queue | |Queue |
|      | |      | |      |
|Pri:0 | |Pri:1 | |Pri:1 |
|(最高) | |(次要) | |(次要) |
+--+---+ +--+---+ +--+---+
   |        |        |
   v        v        v
+------+ +------+ +------+
|Main  | |Face  | |OCR   |
|Worker| |Worker| |Worker|
|x1    | |xN    | |xM    |
+--+---+ +--+---+ +--+---+
   |        |        |
   v        v        v
+--------+ +--------+ +--------+
|global_ | |face_   | |ocr_    |
|index.db| |index.db| |index.db|
+--------+ +--------+ +--------+
```

### 6.2 Worker 资源调度策略

| 参数 | 默认值 | 说明 |
|---|---|---|
| `main_workers` | 1 | 主库 Worker 数（通常 1 即可，SQLite 写串行） |
| `face_workers` | 2 | 人脸 Worker 数（CPU 密集，使用 ProcessPool） |
| `ocr_workers` | 2 | OCR Worker 数（CPU 密集，使用 ProcessPool） |
| `gpu_workers` | 1 | GPU Worker 数（共享 GPU，通常 1 个即可） |
| `max_cpu_percent` | 60% | 次队列（人脸 + OCR）总 CPU 使用上限 |
| `max_gpu_mem_percent` | 70% | GPU 显存使用上限 |
| `batch_size` | 16 | 批处理大小（GPU 推理） |
| `queue_max_size` | 10000 | 各队列最大积压任务数 |

### 6.3 优先级与公平调度

采用 **加权公平队列 (Weighted Fair Queuing, WFQ)** 结合 **赤字公平调度 (Deficit Fair Scheduling, DFS)** 策略：

```python
# 资源权重分配
QUEUE_WEIGHTS = {
    "main":  1.0,   # 主队列：不限制，优先保障
    "face":  0.5,   # 人脸队列：占次要资源的 50%
    "ocr":   0.5,   # OCR 队列：占次要资源的 50%
}
```

**调度规则**：

1. **主队列优先**: 主队列有任务时，确保主 Worker 满负载运行，不被次队列抢占
2. **次队列公平分配**: 人脸队列与 OCR 队列按权重 1:1 分享剩余 CPU / GPU 资源
3. **动态让步**: 当主队列积压超过阈值（默认 100 条）时，次队列 Worker 暂停处理，释放 CPU 资源
4. **GPU 分时**: 人脸和 OCR 共享 GPU 时，使用互斥锁确保同一时刻只有一个队列占用 GPU
5. **背压机制**: 队列满时（`queue_max_size`），新任务进入溢出缓冲区（持久化到 SQLite `processing_status` 表），待队列空闲时自动填充
6. **空闲回收**: 无主队列任务时，人脸和 OCR Worker 可使用全部可用资源

```python
class ResourceGovernor:
    """资源调度器 - 控制各队列 Worker 的资源使用"""

    def should_pause_secondary(self) -> bool:
        """主队列积压时暂停次队列"""
        return self.main_queue.qsize() > self.backpressure_threshold

    def acquire_gpu(self, queue_name: str) -> bool:
        """GPU 互斥访问"""
        return self._gpu_lock.acquire(timeout=self.gpu_timeout)

    def get_batch_size(self, queue_name: str) -> int:
        """根据可用显存动态调整 batch size"""
        free_mem = self._get_free_gpu_memory()
        return min(self.max_batch_size, free_mem // self.per_sample_mem)
```

---

## 7. 接口定义

### 7.1 领域层接口 (Domain Repositories)

```python
class IFaceRepository(ABC):
    """人脸数据仓储接口"""

    @abstractmethod
    def save_face(self, face: FaceDetection) -> None: ...

    @abstractmethod
    def save_faces_batch(self, faces: List[FaceDetection]) -> None: ...

    @abstractmethod
    def get_faces_by_asset(self, asset_rel: str) -> List[FaceDetection]: ...

    @abstractmethod
    def get_faces_by_cluster(self, cluster_id: str) -> List[FaceDetection]: ...

    @abstractmethod
    def get_faces_by_person(self, person_id: str) -> List[FaceDetection]: ...

    @abstractmethod
    def update_cluster_assignment(self, face_id: str, cluster_id: str) -> None: ...

    @abstractmethod
    def exclude_face(self, face_id: str) -> None: ...


class IFaceClusterRepository(ABC):
    """聚类数据仓储接口"""

    @abstractmethod
    def save_cluster(self, cluster: FaceCluster) -> None: ...

    @abstractmethod
    def merge_clusters(self, source_ids: List[str], target_id: str) -> None: ...

    @abstractmethod
    def get_all_clusters(self) -> List[FaceCluster]: ...

    @abstractmethod
    def delete_empty_clusters(self) -> None: ...


class IPersonRepository(ABC):
    """人物数据仓储接口"""

    @abstractmethod
    def save_person(self, person: Person) -> None: ...

    @abstractmethod
    def get_person(self, person_id: str) -> Optional[Person]: ...

    @abstractmethod
    def find_by_name(self, name: str) -> List[Person]: ...

    @abstractmethod
    def list_all(self, include_hidden: bool = False) -> List[Person]: ...


class IOcrRepository(ABC):
    """OCR 数据仓储接口"""

    @abstractmethod
    def save_ocr_result(self, result: OcrResult) -> None: ...

    @abstractmethod
    def save_ocr_results_batch(self, results: List[OcrResult]) -> None: ...

    @abstractmethod
    def get_by_asset(self, asset_rel: str) -> List[OcrResult]: ...

    @abstractmethod
    def search_text(self, query: str, limit: int = 50) -> List[OcrSearchResult]: ...

    @abstractmethod
    def delete_by_asset(self, asset_rel: str) -> None: ...
```

### 7.2 应用层用例 (Use Cases)

```python
class DetectFacesUseCase:
    """检测单张图片中的人脸"""
    def execute(self, asset_rel: str, image_path: Path) -> List[FaceDetection]: ...

class EncodeFacesUseCase:
    """对已检测到的人脸生成嵌入向量"""
    def execute(self, face_ids: List[str]) -> None: ...

class ClusterFacesUseCase:
    """对所有未聚类的人脸执行聚类"""
    def execute(self, algorithm: str = 'dbscan', threshold: float = 0.6) -> int: ...

class MergeClustersUseCase:
    """合并多个聚类到一个人物"""
    def execute(self, cluster_ids: List[str], person_name: str) -> Person: ...

class MoveFaceUseCase:
    """移动单张人脸到另一个聚类"""
    def execute(self, face_id: str, target_cluster_id: str) -> None: ...

class RecognizeTextUseCase:
    """对单张图片执行 OCR"""
    def execute(self, asset_rel: str, image_path: Path) -> List[OcrResult]: ...

class SearchByTextUseCase:
    """根据文字搜索图片"""
    def execute(self, query: str, limit: int = 50) -> List[OcrSearchResult]: ...
```

### 7.3 事件定义

```python
@dataclass
class FaceDetectedEvent(Event):
    """人脸检测完成"""
    asset_rel: str
    face_count: int

@dataclass
class FaceClusteredEvent(Event):
    """聚类完成"""
    new_cluster_count: int
    updated_face_count: int

@dataclass
class ClustersMergedEvent(Event):
    """聚类合并"""
    source_cluster_ids: List[str]
    target_cluster_id: str
    person_id: Optional[str]

@dataclass
class OcrCompletedEvent(Event):
    """OCR 识别完成"""
    asset_rel: str
    text_block_count: int

@dataclass
class AssetIndexedEvent(Event):
    """资产已入主库（触发人脸/OCR处理）"""
    asset_rel: str
    media_type: str
```

---

## 8. 约束与假设

| # | 约束 / 假设 | 说明 |
|---|---|---|
| 1 | SQLite 单写者限制 | 每个数据库同一时间只能有一个写连接，通过 WAL 模式支持并发读 |
| 2 | OpenCV >= 4.8 | 需要 `cv2.FaceDetectorYN` 和 `cv2.FaceRecognizerSF` API |
| 3 | CUDA Toolkit >= 11.8 | GPU 加速需要，非必须依赖 |
| 4 | 人脸模型文件 | `yunet.onnx`、`face_recognition_sface_2021dec.onnx` 需手动下载或首次启动时自动下载 |
| 5 | PaddleOCR 或 Tesseract | 至少安装其中一个 OCR 后端 |
| 6 | 最大图片尺寸 | 人脸检测前将图片缩放至长边 <= 1920px，避免 GPU 显存溢出 |
| 7 | 文件系统 | 资产文件在处理期间不应被删除或移动 |
| 8 | Python >= 3.12 | 与现有项目保持一致 |

---

## 9. 参考

| 参考 | 链接 |
|---|---|
| OpenCV Face Detection (YuNet) | https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html |
| OpenCV Face Recognition (SFace) | https://docs.opencv.org/4.x/da/d60/tutorial_face_main.html |
| dlib Face Recognition | http://dlib.net/face_recognition.py.html |
| PaddleOCR | https://github.com/PaddlePaddle/PaddleOCR |
| SQLite FTS5 | https://www.sqlite.org/fts5.html |
| DBSCAN 聚类算法 | https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html |
| Chinese Whispers 聚类 | https://github.com/ageitgey/face_recognition |
| digiKam 人脸识别架构 | https://www.digikam.org/about/ |
| PhotoPrism 人脸识别 | https://docs.photoprism.app/developer-guide/media/face-recognition/ |
| Immich 机器学习管线 | https://immich.app/docs/features/ml |
