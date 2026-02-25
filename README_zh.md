## 结合评论有用性与相关性的信息检索（IR 方向）

### 数据集

- **数据来源**：YouTube Data API v3（关键词搜索视频 + 视频评论接口）。
- **规模**：最多 10 个视频，每视频最多 100 条评论；汇总为单个 CSV 文件保存。
- **字段说明**：CSV 包含 `video_id`、`video_title`、`comment_text`、`comment_likes` 等列；其中 `comment_text` 为评论正文，`comment_likes` 为点赞数，用于后续有用性融合。

### 索引与表示

- **文档单位**：每条评论为一条文档；文档文本由 `comment_text` 与 `video_title` 拼接而成，便于同时匹配评论内容与视频标题。
- **分词与预处理**：按空格切分、小写、去停用词（NLTK 英文停用词）；**词干提取**使用 Porter's Algorithm（NLTK PorterStemmer）对每个 token 做词干化，文档与查询共用同一词干空间；**查询扩展**在查询侧对词干化后的 token 做同义词/相近词扩展（手写同义词表，词干形式），再参与 BM25 检索。
- **索引**：基于 BM25（rank_bm25 库）在**词干化后**的 tokenized 文档上建索引；运行时内存索引，数据来自上述 CSV，无持久化倒排文件。

### 检索模型

- **BM25**：使用标准 BM25 公式，参数 k1=1.5、b=0.75；top_k 取 min(200, 文档总数)。
- **有用性信号**：点赞数经 log(1+x) 后做 min-max 归一化，与 BM25 分数（对当前命中做 min-max 归一化）线性融合；融合公式为 `FinalScore = λ·BM25_norm + (1-λ)·likes_norm`，λ=0.7。
- **查询处理**：文本规范化、去停用词、Porter 词干化、同义词/相近词扩展（expand_query_tokens），再用扩展后的 token 列表调用 BM25 检索。

### 评估

- **暂无评估**：当前无标注数据集与离线评估脚本；可后续增加 precision@k、recall、或人工抽样评估等。

### 搜索界面

- **首页**：关键词搜索（调用 YouTube API 拉取视频与评论）、历史记录选择（按 CSV 文件名）；分页列表展示评论，含点赞数、视频链接、视频标题等。
- **右侧 BM25 检索**：输入查询、选择当前 CSV 文件，返回按融合分排序的结果；展示 snippet、like_count、bm25_score、fused_score；支持关键词高亮与排序说明（λ·BM25 + (1-λ)·likes）。

---

作者：nikk909 (yinghua253659@163.com)
