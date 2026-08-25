# 大模型技术知识题库

本文档涵盖互联网大厂大模型岗位面试的高频技术考点，按模块分类。

## 目录

1. [Attention机制](#1-attention机制)
2. [Transformer架构](#2-transformer架构)
3. [预训练技术](#3-预训练技术)
4. [Tokenizer与文本处理](#4-tokenizer与文本处理)
5. [大模型架构](#5-大模型架构)
6. [微调技术](#6-微调技术)
7. [推理优化](#7-推理优化)
8. [分布式训练](#8-分布式训练)
9. [RAG与Agent](#9-rag与agent)
10. [向量数据库](#10-向量数据库)
11. [🆕 数据构建与清洗](#11-数据构建与清洗)
12. [🆕 SFT 有监督微调实战](#12-sft-有监督微调实战)
13. [🆕 RL 对齐训练](#13-rl-对齐训练)
14. [🆕 DPO 进阶与偏好优化家族](#14-dpo-进阶与偏好优化家族)
15. [🆕 OPD On-Policy Distillation](#15-opd-on-policy-distillation)

---

## 1. Attention机制

### Q1: 简述Self-Attention的计算过程

**答案：**

Self-Attention通过三个线性变换生成Q（Query）、K（Key）、V（Value）矩阵：

```
Q = X · Wq
K = X · Wk
V = X · Wv
```

核心计算：
```
Attention(Q,K,V) = softmax(QK^T / √dk) · V
```

其中√dk是缩放因子，防止点积值过大导致梯度消失。

### Q2: Multi-Head Attention的作用是什么？

**答案：**

- **关注不同子空间**：每个head学习不同的注意力模式（句法、语义、位置等）
- **增强模型表达能力**：多个head的输出concat后通过线性变换
```
MHA = Concat(head_1, ..., head_h) · Wo
```
- **实践建议**：LLaMA使用GQA（Grouped Query Attention）减少计算量

### Q3: FlashAttention是如何加速的？

**答案：**

**核心思想**：IO-aware计算，减少HBM（显存）和SM（计算单元）间的数据搬运

**关键技术**：
1. **Tiling**：将注意力计算分块处理，块内完整计算，块间累积
2. **Recomputation**：前向传播不保存完整注意力矩阵，反向时重新计算（空间换时间）

**效果**：
- 显存复杂度从O(N²)降至O(N)
- 训练速度提升2-4倍
- 实际精度无损

### Q4: MHA vs MQA vs GQA的区别？

**答案：**

| 方案 | Q头数 | K/V头数 | 特点 |
|------|-------|---------|------|
| MHA | 8 | 8 | 标准方案，显存消耗大 |
| MQA | 8 | 1 | K/V共享，显存省但效果可能下降 |
| GQA | 8 | 2-4 | 分组共享，平衡效果与效率 |

**应用场景**：
- LLaMA-2使用GQA（4组K/V）
- ChatGLM-2使用MQA
- Falcon使用GQA

### Q5: Attention有哪些常见变体？

**答案：**

| 变体 | 核心改进 | 适用场景 |
|------|----------|----------|
| FlashAttention | IO优化 | 所有场景 |
| StreamingLLM | 保留sink tokens | 流式推理 |
| Longformer | 滑动窗口+全局Attention | 长文档 |
| BigBird | 稀疏+全局+随机 | 超长序列 |
| Performer | Linear attention | 超长序列 |
| ReZero | 动态注意力权重 | 收敛加速 |

---

## 2. Transformer架构

### Q6: LayerNorm有哪些类型？有什么区别？

**答案：**

| 类型 | 计算范围 | 特点 |
|------|----------|------|
| Pre-LN | 残差前 | 训练更稳定，无需warm-up |
| Post-LN | 残差后 | 效果通常更好，但训练不稳定 |
| RMSNorm | 仅均值 | 去掉均值计算，效率更高 |

LLaMA使用Pre-LN + RMSNorm。

### Q7: Positional Encoding有哪些实现方式？

**答案：**

| 方式 | 公式/特点 | 代表模型 |
|------|----------|----------|
| Sinusoidal | PE(pos,2i)=sin(pos/10000^(2i/d)) | 原始Transformer |
| Learned | 可学习的position embedding | BERT |
| RoPE | 旋转矩阵实现相对位置 | LLaMA, ChatGLM |
| ALiBi | 线性偏置，不引入位置ID | Bloom |
| Relative PE | 相对位置编码 | Transformer-XL |

**RoPE核心思想**：通过旋转矩阵将位置信息融入Q/K，使得attention score只与相对位置有关。

### Q8: 为什么Transformer需要残差连接？

**答案：**

1. **梯度流动**：缓解深层网络的梯度消失问题
2. **特征融合**：低层特征可以直接传递到高层
3. **恒等映射**：理论上有利于网络学习恒等函数
4. **训练稳定性**：使得网络可以堆叠更深的层数

---

## 3. 预训练技术

### Q9: 预训练任务有哪些？

**答案：**

| 任务 | 描述 | 代表模型 |
|------|------|----------|
| MLM | Masked Language Model | BERT |
| NSP | Next Sentence Prediction | BERT |
| SOP | Sentence Order Prediction | ALBERT |
| CLM | Causal Language Model | GPT |
| GLM | Generalized LM | ChatGLM |
| UL2 | 混合去噪任务 | Flan-UL2 |

**大模型趋势**：大多采用Causal LM（GPT式），因为：
- 更适合生成任务
- 便于做Zero-shot/Few-shot

### Q10: RLHF的三个步骤是什么？

**答案：**

```
Step 1: SFT (Supervised Fine-Tuning)
├── 用人工标注的问答数据微调基座模型
├── 数据量：万级~十万级
└── 目标：学习对话格式和基础能力

Step 2: Reward Model Training
├── 收集人类偏好数据（chosen vs rejected）
├── 训练Reward Model预测人类偏好
└── 目标：学习什么回答是"好"的

Step 3: RL Training (PPO)
├── 使用Reward Model作为奖励信号
├── 约束项：KL散度限制与SFT模型的偏离
└── 目标：最大化期望奖励
```

**PPO算法核心**：
```
Reward = R(π_θ) - β · KL(π_θ || π_SFT)
```

### Q11: DPO相比PPO有什么优势？

**答案：**

| 方面 | PPO+RM | DPO |
|------|--------|-----|
| 训练复杂度 | 三阶段（RM+SFT+PPO） | 单阶段 |
| 显存占用 | 需要额外RM | 只训练策略网络 |
| 超参敏感性 | 多个（KL系数、PPO clip等） | 相对更少 |
| 调参难度 | 高 | 中等 |
| 效果 | 理论最优 | 基本持平 |

**DPO核心思想**：将reward modeling和RL合并，直接优化偏好损失。

### Q12: LLaMA/LLaMA-2的训练细节？

**答案：**

| 方面 | LLaMA | LLaMA-2 |
|------|-------|---------|
| 上下文 | 2K | 4K |
| 分组注意力 | 无 | GQA |
| 训练数据 | 1.4T tokens | 2T tokens |
| 位置编码 | RoPE | RoPE |
| 训练长度 | 1T tokens | 1.4T tokens |
| RLHF | 无 | PPO+DPO |

---

## 4. Tokenizer与文本处理

### Q13: BPE vs WordPiece vs SentencePiece的区别？

**答案：**

| 方案 | 切分策略 | 特点 | 代表 |
|------|----------|------|------|
| BPE | 最高频的相邻子词 | 简单高效 | GPT |
| WordPiece | 基于似然增益选择 | 需要语言学知识 | BERT |
| SentencePiece | 统一处理，特殊token化 | 语种无关 | LLaMA, ChatGLM |

### Q14: 为什么大模型都用SentencePiece？

**答案：**

1. **语言无关**：不依赖特定语言的预分词器
2. **端到端**：将空格也作为token处理
3. **灵活性**：支持BPE、Unigram等多种算法
4. **一致性**：训练和推理完全一致

### Q15: 如何处理多语言场景的tokenizer？

**答案：**

| 策略 | 做法 | 优缺点 |
|------|------|--------|
| 统一词表 | 所有语言共享 | 词表大，可能效率低 |
| 独立词表 | 每种语言独立 | 需要语言检测 |
| 层级tokenizer | 语言无关+语言特定 | 灵活但复杂 |

**推荐实践**：如XGLM使用扩大的多语言词表（250K tokens）。

---

## 5. 大模型架构

### Q16: 主流开源大模型架构对比

**答案：**

| 模型 | 架构特点 | 位置编码 | 注意力 | 开源协议 |
|------|----------|----------|--------|----------|
| LLaMA | 类GPT | RoPE | GQA | OpenRAIL |
| ChatGLM | GLM架构 | RoPE | MQA | BYAIL |
| Mistral | 类GPT | RoPE | GQA | Apache |
| Qwen | 类GPT+MoE | RoPE | GQA | Tongyi Qianwen |
| Bloom | 类GPT | ALiBi | MHA | BigScience |
| InternLM | 类GPT | RoPE | GQA | Apache |

### Q17: MoE（混合专家）架构原理

**答案：**

**核心思想**：每个token只激活少数expert（专家网络）

```
output = Σ(g_i(x) · expert_i(x)) / Σg_i(x)
```

其中g_i是门控网络输出的稀疏激活权重。

**代表模型**：
- Mixtral 8x7B：8个experts，每次激活2个
- DBRL：GShard论文实现

**优势**：
- 参数量大但计算量小
- 稀疏激活更高效

**挑战**：
- 负载均衡（避免某些expert被忽视）
- 通信瓶颈（分布式训练）

### Q18: GQA（Grouped Query Attention）的实现细节

**答案：**

**核心代码逻辑**：
```python
# Q有num_heads个head，K/V只有num_kv_heads个head
# Q按num_groups分组，每组共享一个K/V head
num_groups = num_heads // num_kv_heads

# 计算时将K/V扩展到Q的维度
for i in range(num_groups):
    k_expanded[i*group_size:(i+1)*group_size] = k[i]
    v_expanded[...] = v[i]
```

**效果对比**：
- LLaMA-7B：32 Q-heads，32 K/V heads
- LLaMA-2-70B：8 Q-heads，2 K/V heads（GQA）
- 显存节省约60%，效果几乎持平

---

## 6. 微调技术

### Q19: LoRA的原理和实现

**答案：**

**核心思想**：冻结预训练权重，只训练低秩分解的 adapter

```
W' = W + ΔW = W + BA
其中 B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)
```

**实现要点**：
```python
class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=1):
        self.original = original_layer
        self.lora_A = nn.Parameter(original.weight[:r, :])  # 简化的初始化
        self.lora_B = nn.Parameter(torch.zeros(original.weight.shape[0], r))
    
    def forward(self, x):
        # 原权重冻结
        base_output = self.original(x)
        # LoRA增量
        lora_output = x @ self.lora_A.T @ self.lora_B.T * (alpha/rank)
        return base_output + lora_output
```

**主流配置**：
| 参数 | 常见值 |
|------|--------|
| rank r | 4-16 |
| alpha | 2r 或 r |
| target_modules | q_proj, v_proj, k_proj, o_proj |

### Q20: QLoRA的创新点

**答案：**

**核心创新**：
1. **4-bit NormalFloat (NF4)**：针对正态分布权重量化
2. **Double Quantization**：对量化常数也量化
3. **Paged Optimizers**：分页管理优化器状态

**显存节省**：
```
7B模型全量微调：~28GB GPU
QLoRA微调：~6GB GPU
```

### Q21: Full Fine-tuning vs LoRA vs Prefix Tuning

**答案：**

| 方案 | 可训练参数 | 显存占用 | 效果 | 适用场景 |
|------|------------|----------|------|----------|
| Full FT | 100% | 最高 | 最好 | 小数据集+充足算力 |
| LoRA | 0.1-1% | 中等 | 接近Full | 推荐默认选择 |
| Prefix | <0.1% | 低 | 中等 | 极端资源受限 |

---

## 7. 推理优化

### Q22: KV Cache的作用和实现

**答案：**

**作用**：避免重复计算已生成token的K/V

**公式**：
```
P(t+1) = Attention(Q(t+1), K(1:t+1), V(1:t+1))
       = Attention(Q(t+1), concat(K_cache, K(t+1)), ...)
```

**显存问题**：
- 上下文越长，KV Cache越大
- 13B模型单token KV Cache约1.6MB
- 100K上下文约需160GB

**优化方案**：
- PagedAttention（vLLM）：分页管理KV Cache
- FlashAttention：IO优化
- StreamingLLM：只保留sink tokens

### Q23: Continuous Batching（迭代级调度）

**答案：**

**问题**：静态batch要求所有序列同时结束，GPU利用率低

**Solution**：迭代级调度
```
# 伪代码
while pending_requests:
    # 动态添加新请求
    for req in new_requests:
        if has_slot():
            batch.add(req)
    
    # 批量前向传播
    outputs = model(batch)
    
    # 处理完成
    for req in batch:
        if req.is_finished():
            batch.remove(req)
            pending_requests.remove(req)
```

**效果**：Throughput提升5-10倍

### Q24: 模型量化方案对比

**答案：**

| 方案 | 精度 | 显存压缩 | 速度 | 代表 |
|------|------|----------|------|------|
| FP16 | 16bit | 1x | baseline | - |
| INT8 | 8bit | 2x | 1.2-1.5x | GPTQ, AWQ |
| INT4 | 4bit | 4x | 2-3x | GPTQ, QAA |
| NF4 | 4bit | 4x | 2-3x | bitsandbytes |

**GPTQ流程**：
1. 使用校准数据集获取权重分布
2. 按chunk逐层量化，优化重建误差
3. 推理时用INT4权重，恢复FP16计算

### Q25: TensorRT-LLM的优化技术

**答案：**

| 优化 | 原理 | 效果 |
|------|------|------|
| Kernel融合 | 合并多个小算子 | 减少访存 |
| FP8推理 | 低精度计算 | 2x加速 |
| In-flight Batching | 迭代级调度 | 高吞吐 |
| Attention优化 | FlashAttention | 减少显存 |
| Speculative Decoding | 投机解码 | 降低延迟 |

---

## 8. 分布式训练

### Q26: ZeRO优化器的三个Stage

**答案：**

| Stage | 分片内容 | 显存节省 | 通信量 |
|-------|----------|----------|--------|
| ZeRO-1 | Optimizer States | 4x | 中等 |
| ZeRO-2 | + Gradients | 8x | 中等 |
| ZeRO-3 | + Parameters | Nx | 高 |

**DeepSpeed实现**：
```python
# ZeRO-3配置示例
{
    "zero_optimization": {
        "stage": 3,
        "stage3_param_persistence_threshold": 1e4,
        "stage3_gather_16bit_weights_on_model_save": True
    }
}
```

### Q27: Pipeline Parallel的并行策略

**答案：**

**问题**：Tensor Parallel通信过于密集

**Solution**：按层划分pipeline

```
GPU0: Layer0-8   → GPU1: Layer9-16  → GPU2: Layer17-24  → GPU3: Layer25-32
  ↓micro_batch1    ↓micro_batch1       ↓micro_batch1        ↓micro_batch1
  ↓micro_batch2    ↓micro_batch2       ↓micro_batch2        ↓micro_batch2
  ...               ...                  ...                  ...
```

**问题**：Pipeline Bubble（流水线气泡）

**优化**：PipeDream采用1F1B（One-Forward-One-Backward）调度

### Q28: Megatron-LM的Tensor Parallel实现

**答案：**

**核心**：按attention head或FFN维度切分

```
# Column Parallel Linear
Y = X @ W  →  Y = [X@W1, X@W2]  # K/V heads分到不同GPU

# Row Parallel Linear
Y = X @ W  →  Y = X@[W1;W2]     # 结果汇总
```

**AllReduce通信**：
- 每个transformer layer需要2次AllReduce
- 通信量与tensor size成正比

---

## 9. RAG与Agent

### Q29: RAG的核心流程

**答案：**

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Document  │ →  │   Chunking   │ →  │ Embedding   │
└─────────────┘    └──────────────┘    └──────────────┘
                                             ↓
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Response  │ ←  │   Generator  │ ←  │   Retriever │
└─────────────┘    └──────────────┘    └─────────────┘
```

**关键组件**：
1. **Document Loader**：PDF、Markdown、Web等
2. **Text Splitter**：语义分块（100-500 tokens）
3. **Embedding Model**：text-embedding-ada-002、bge等
4. **Vector Store**：Pinecone、Milvus、Chroma
5. **Retrieval**：Top-k + Reranker
6. **Generator**：ChatGPT、Claude、本地模型

### Q30: 如何提升RAG效果？

**答案：**

| 优化方向 | 具体方法 |
|----------|----------|
| 检索质量 | 混合检索（向量+关键词）、重排序 |
| 分块策略 | 语义分块、重叠窗口 |
| 上下文压缩 | RecursiveCharacterTextSplitter |
| 迭代优化 | Query改写、HyDE |
| 路由 | 根据问题类型路由到不同知识库 |
| 评估 | RAGAS: faithfulness, answer relevancy, context recall |

### Q31: Agent的核心组件

**答案：**

```
┌──────────────────────────────────────────────────┐
│                    Agent                          │
├──────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Planning │  │  Memory  │  │    Tools     │   │
│  │          │  │          │  │              │   │
│  │ - ReAct  │  │ - Short  │  │ - Search     │   │
│  │ - CoT    │  │ - Long   │  │ - Calculator │   │
│  │ - ToT    │  │          │  │ - Code Exec  │   │
│  └──────────┘  └──────────┘  └──────────────┘   │
└──────────────────────────────────────────────────┘
```

**ReAct范式**：
```
Thought: 我需要先搜索相关信息
Action: search[关键词]
Observation: 搜索结果...
Thought: 根据结果...
Action: calculator[计算]
...
```

### Q32: LangChain的核心模块

**答案：**

| 模块 | 功能 | 关键类 |
|------|------|--------|
| Model I/O | 模型调用封装 | ChatOpenAI, PromptTemplate |
| Retrieval | 文档检索 | VectorStoreRetriever |
| Chains | 链式调用 | LLMChain, RetrievalQA |
| Agents | 工具调用 | Agent, Tool |
| Memory | 对话记忆 | ConversationBufferMemory |

---

## 10. 向量数据库

### Q33: 主流向量数据库对比

**答案：**

| 数据库 | 实现语言 | 索引算法 | 特点 |
|--------|----------|----------|------|
| Milvus | Go | HNSW, IVF, PQ | 功能全面，工业级 |
| Qdrant | Rust | HNSW | 性能高，支持过滤 |
| Weaviate | Go | HNSW | 原生支持混合搜索 |
| Chroma | Python | HNSW (LanceDB) | 轻量，适合研究 |
| Pinecone | 云服务 | 闭源 | 托管方便 |
| FAISS | C++ | IVF, HNSW | Meta开源，经典方案 |

### Q34: HNSW算法原理

**答案：**

**核心思想**：分层可导航小世界图

```
Layer 3:  ●─────────────●
          │             │
Layer 2:  ●───●───●─────●───●
          │   │   │     │   │
Layer 1:  ●───●───●───●─●───●───●
          │   │   │   │ │   │   │
Layer 0:  ●───●───●───●─●───●───●───●  (全量数据)
```

**搜索过程**：
1. 从顶层入口开始
2. 贪心搜索最近邻
3. 下降到下一层
4. 重复直到最底层

**参数**：
- M：每层连接数（通常16-64）
- efConstruction：构建时的动态列表大小

---

---

## 11. 数据构建与清洗

### Q35: 大模型训练数据的来源有哪些？

**答案：**

| 来源 | 占比 | 典型 | 特点 |
|------|------|------|------|
| 网页抓取 Common Crawl | 60-70% | CC-MAIN-2024 | 量最大，噪声最多 |
| 代码仓库 | 10-15% | GitHub, The Stack | 高质量，需过滤License |
| 书籍 | 5-8% | Books3, Gutenberg | 知识密度高，长文 |
| 论文/百科 | 3-5% | arXiv, Wikipedia | 事实性强 |
| 问答/论坛 | 3-5% | StackOverflow, Reddit | 对话风格，事实需核实 |
| 合成数据 | 5-15% | Self-Instruct, Evol-Instruct | 可控但有"模型味" |

**LLaMA-3 数据配比**：50% 英文网页、25% 代码、10% 多语种、8% 学术、7% 书籍。

### Q36: 如何做数据去重（Document-level）？

**答案：**

| 方法 | 原理 | 适用场景 |
|------|------|----------|
| Hash去重 | MD5/SHA 完全匹配 | 复制粘贴内容 |
| MinHash | Jaccard 相似度估算 | 近似重复检测 |
| SimHash | 局部敏感哈希 | 海量数据快速去重 |
| SemDeDup | Embedding + 聚类 | 语义级重复 |
| URL规范化 | URL去重 | 网页爬取 |

**MinHash核心思想**：
```
签名 = min_hash(M)  # M是文档k-shingle集合
Jaccard(A,B) ≈ similarity(sig(A), sig(B))
```
保留概率 < 0.8 的文档对，约可砍掉 20-30% 数据。

### Q37: 文本质量过滤（Quality Filtering）的常用方法？

**答案：**

```
层级一：启发式规则（Heuristic）
├── 长度过滤：剔除 < 100 chars 或 > 100K chars 的文档
├── 词比例过滤：字母/标点比、停用词比例
├── 语言检测：fastText 置信度 > 0.65
└── 符号过滤：乱码、HTML残留、不可见字符

层级二：分类器打分（Classifier）
├── 训练一个 quality classifier（基于BERT）
├── 标注 10K 高质量 vs 低质量样本
└── 预测每个文档的 quality score，保留 top-K

层级三：困惑度过滤（PPL-based）
├── 用预训练模型（如KenLM）计算 PPL
├── PPL 极高 → 无意义文本
├── PPL 极低 → 重复模板
└── 保留中间 70% 的样本
```

**实战工具**：
- **Gopher 规则**（DeepMind）：启发式规则集
- **DSIR**（Stanford）：基于 importance resampling
- **DataComp-LM**：大规模数据过滤比赛

### Q38: PII（个人隐私信息）脱敏怎么做？

**答案：**

| 方案 | 工具 | 覆盖 |
|------|------|------|
| 正则匹配 | 邮箱、电话、身份证 | 弱结构化PII |
| NER模型 | Presidio, spaCy | 姓名、地址 |
| LLM识别 | GPT-4 + few-shot prompt | 长上下文PII |
| 哈希化 | SHA256(email) | 用于关联而非训练 |

**原则**：宁可误杀也不漏杀。训练数据中的PII会在推理时泄漏（如手机号、邮箱），且涉及 GDPR/CCPA 合规。

### Q39: 如何构造 SFT 指令数据？

**答案：**

**主流构造方法**：

```
1. Self-Instruct（Wang 2022）
   └── 给 GPT-4 175个种子指令 → 生成更多 → 过滤

2. Evol-Instruct（WizardLM）
   └── 在已有指令上"进化"：加深、加宽、加约束、改格式

3. Backtranslation（Humpback）
   └── 用基础模型生成回答 → 用GPT-4反向生成问题

4. Human-in-the-loop
   └── 众包平台 Labelbox/Scale AI 人工标注

5. Rejection Sampling FT（Llama-2-chat）
   └── 采样多个回答 → RM打分 → 只保留最优
```

**配比建议**（LLaMA-2-chat 论文）：

| 数据类型 | 占比 | 来源 |
|----------|------|------|
| 通用对话 | 50% | ShareGPT, Wizard |
| 知识问答 | 20% | Natural Questions |
| 推理（数学/代码） | 15% | MATH, HumanEval |
| 安全/价值观 | 10% | Anthropic HH |
| 长文档理解 | 5% | BookQA |

### Q40: 合成数据有哪些风险？

**答案：**

| 风险 | 现象 | 缓解 |
|------|------|------|
| 模型坍缩 (Model Collapse) | 迭代训练导致输出多样性下降 | 混入人类数据 |
| 事实幻觉 | 看起来对但编造 | 与真实数据混合 |
| "模型味" | 风格过于同质 | 多源 prompt 模板 |
| 自我偏见 | 偏好 GPT-4 风格 | 多模型 ensemble |
| 版权争议 | 训练数据本身争议 | 人工二次过滤 |

---

## 12. SFT 有监督微调实战

### Q41: SFT 的训练损失如何计算？

**答案：**

**核心**：仅对 response 部分的 token 计算损失，prompt 部分 mask 掉。

```python
def sft_loss(model, input_ids, labels, prompt_lengths):
    # labels[i] = -100 表示忽略
    for i, pl in enumerate(prompt_lengths):
        labels[i, :pl] = -100
    
    logits = model(input_ids).logits  # [B, T, V]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = CrossEntropyLoss()(shift_logits.view(-1, V), shift_labels.view(-1))
    return loss
```

**关键点**：
- shift 操作：预测 `t+1` 用 `t` 的 logits
- 不计算 prompt 损失，避免模型"学 prompt"
- 部分实现会对所有 token 计算损失（Full LM Loss），效果略差

### Q42: Chat Template 的重要性？

**答案：**

**三种主流格式**：

| 模型 | 格式 |
|------|------|
| ChatML (Qwen, ChatGLM) | `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` |
| LLaMA-3 | `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n...<|eot_id|>` |
| Mistral | `[INST] ... [/INST]` |

**踩坑点**：
1. 训练用 ChatML，推理用 raw → 灾难性遗忘
2. 多轮对话需要正确拼接 `assistant` 边界
3. Special token 在 tokenizer 中必须注册

### Q43: SFT 中如何处理多轮对话？

**答案：**

**方案一：全量计算损失**（简单但效果差）
- 把每轮都当独立样本
- 上下文长度易爆炸

**方案二：仅最后一轮计算损失**（LLaMA-2 风格）
- 前 N-1 轮全部 mask
- 只算最后一轮 response 的损失
- 训练效率高，但学不到多轮一致性

**方案三：Assistant 部分全算损失**（推荐）
- 对每轮 assistant 的 token 计算损失
- user/system 部分 mask
- 兼顾效率和效果

### Q44: 数据配比（Data Mixing）的策略？

**答案：**

**核心原则**：

```
"不要让主导 domain 压制长尾 domain"
```

**常用方法**：

1. **按比例上采样**：稀有domain重复多遍
2. **按比例下采样**：主导domain随机丢弃
3. **DoReMi 域权重**（Microsoft 2023）：
   - 用 proxy model 学习最优域权重
4. **离线调参**：通过 eval loss/benchmark 反推

**LLaMA-2 配方**：SFT阶段数据 27,540 条标注 + 1M Rejection Sampling 生成。

### Q45: SFT 数据量多少合适？

**答案：**

| 场景 | 推荐数据量 | 来源 |
|------|-----------|------|
| 行业小模型微调（客服/FAQ） | 5K-20K | 业务真实数据 |
| 通用指令微调 | 50K-200K | Self-Instruct + 人工 |
| RLHF 第一阶段 | 10K-50K | 高质量人工标注 |
| 持续学习（Continue Training） | 100K-1M | 行业语料 |

**经验法则**：数据质量 > 数据数量。10万条精心标注 > 100万条粗糙数据。

### Q46: 训练时遇到 Loss 异常如何处理？

**答案：**

```
症状1：Loss = NaN
├── 排查：数据中是否有异常 token、超长序列
├── 处理：clip grad norm=1.0, fp32 master weights
└── 工具：torch.autograd.detect_anomaly()

症状2：Loss 不下降
├── 排查：lr 太小？数据全 mask？标签错位？
├── 检查：tokenize 后 ids 是否与 labels 一一对应
└── 处理：换 AdamW，warmup ratio=0.03

症状3：Loss 震荡
├── 排查：batch size 太小？lr 太大？
└── 处理：增大 batch，cosine lr schedule

症状4：Eval loss 下降但 benchmark 差
├── 排查：训练/测试分布不一致
└── 处理：人为构造 in-domain eval set
```

---

## 13. RL 对齐训练

### Q47: 什么是 Reward Hacking？如何避免？

**答案：**

**定义**：模型学会了"骗过"Reward Model 但实际质量下降的现象。

**常见模式**：

```
1. Length Bias
   └── 模型生成越来越长的回答，RM 偏好长文本
   └── 缓解：长度归一化、length penalty

2. Verbosity Bias
   └── 堆砌废话显得"详尽"
   └── 缓解：RM 训练时人工剔除"为长而长"的回答

3. Repetition
   └── 反复重写同一观点
   └── 缓解：n-gram penalty，重复率监控

4. Sycophancy 谄媚
   └── "您说得对！完全同意！"
   └── 缓解：RM 中加入对抗样本

5. Goodhart's Law
   └── "当度量成为目标时，它就不再是好度量"
   └── 缓解：多维度 RM，定期人工校准
```

### Q48: Reward Model 的训练细节？

**答案：**

**损失函数**（Bradley-Terry 模型）：
```
L = -log σ(r_chosen - r_rejected)
```

**实现要点**：

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        self.backbone = base_model
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids):
        hidden = self.backbone(input_ids).last_hidden_state
        # 取最后一个 token 的隐状态
        reward = self.reward_head(hidden[:, -1, :])
        return reward

# 训练
chosen_reward = rm(prompt + chosen_response)
rejected_reward = rm(prompt + rejected_response)
loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
```

**数据要求**：
- 单条 prompt 对应 K 个 response（K≥4）
- chosen vs rejected 必须质量差距明显
- 引入 "margin"：差 < 0.5 的 pair 弃用

### Q49: PPO 在 LLM 训练中的核心公式？

**答案：**

**目标函数**：
```
J(θ) = E[ R(y) - β · KL(π_θ || π_ref) ]
```

**实际优化用 PPO-Clip 目标**：
```
L^CLIP(θ) = E[ min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t) ]

其中 r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
      A_t = advantage = R_t - V(s_t)
```

**实战技巧**：
- `clip_epsilon = 0.2`
- `GAE lambda = 0.95`
- `mini_batch = 1`（显存允许时）
- KL 系数 β 通常 0.05-0.1

### Q50: PPO 训练的工程挑战有哪些？

**答案：**

| 挑战 | 原因 | 解决 |
|------|------|------|
| 显存爆炸 | 4个模型：policy, ref, value, reward | LoRA + frozen ref |
| 训练不稳定 | reward 信号噪声大 | 优势归一化、KL early stop |
| 收敛慢 | online 采样 + 同步 | 异步生成、Adaptive KL |
| off-policy 失效 | 数据分布漂移 | importance ratio clipping |

**核心工具**：
- **TRL**（HuggingFace）：最常用 SFT/PPO/DPO 库
- **OpenRLHF**：高性能 RLHF 框架
- **DeepSpeed-Chat**：分布式 RLHF

### Q51: 什么是 GRPO（Group Relative Policy Optimization）？

**答案：**

**提出者**：DeepSeekMath, DeepSeek-R1

**核心思想**：不需要 critic/value model，用 group 内相对优势。

```
对每个 prompt 采样 G 个回答 {y_1, ..., y_G}
A_i = (R_i - mean(R_1..G)) / std(R_1..G)

L = -E[ π_θ(y_i|x) / π_old(y_i|x) · A_i ] + β · KL
```

**优势**：
- 显存省：无需 value model
- 训练稳：group 内归一化降低方差
- 效果强：DeepSeek-R1 用 GRPO + RL 训练推理能力

### Q52: Rejection Sampling Fine-tuning（RSF）的原理？

**答案：**

**流程**：
```
1. 用 SFT 模型对每个 prompt 采样 K 个回答
2. 用 RM 评分
3. 只保留得分最高的 1-N 个作为训练数据
4. 用这些"自我精华"再做一次 SFT
```

**LLaMA-2-chat 的做法**：
- 每 2 个 epoch 做一次 RS
- 500K 迭代下来产生 1M 高质量样本
- 配合 PPO 效果叠加

**优点**：
- 简单稳定（纯 SFT，不需要 RL 库）
- 不需要 reward hacking
- 与 SFT pipeline 无缝衔接

---

## 14. DPO 进阶与偏好优化家族

### Q53: DPO 的核心数学推导？

**答案：**

**出发点**：PPO 的最优策略有闭式解
```
π*(y|x) ∝ π_ref(y|x) · exp(R(x,y)/β)
```

**取对数 + 配对化简**，得到 DPO 损失：
```
L_DPO = -log σ( β · log(π_θ(y_w|x)/π_y(y_w|x)) 
                  - β · log(π_θ(y_l|x)/π_y(y_l|x)) )

其中：
  - y_w = chosen (preferred)
  - y_l = rejected
  - π_y = 推理时"冻结"的策略（实践中用 ref model）
```

**直觉**：直接增大 chosen 的似然，减小 rejected 的似然，但用 reference model 锚定。

### Q54: DPO 的常见问题及变种？

**答案：**

| 问题 | 表现 | 变种方案 |
|------|------|----------|
| 长度偏差 | 模型学会"答得长 = 好" | R-DPO 加长度归一化 |
| 概率坍缩 | π_ref 概率被压到 0 | IPO 加正则 |
| 单一偏好 | 仅二元 (chosen/rejected) | KTO 用前景理论 |
| 推理慢 | 仍需加载 ref model | CPO 取消 ref |
| 缺乏多样性 | 生成模式单一 | SimPO 简化目标 |

### Q55: 主流偏好优化变种对比

**答案：**

| 算法 | 全称 | 核心改动 | 代表论文 |
|------|------|----------|----------|
| DPO | Direct Preference Optimization | 闭式解偏好 | NeurIPS 2023 |
| IPO | Identity Preference Optimization | 加 ` (Δ/2β)²` 项 | 2023 |
| KTO | Kahneman-Tversky Optimization | 用前景理论，无须 pair | 2024 |
| CPO | Contrastive Preference Optimization | 取消 ref model | 2024 |
| SimPO | Simple Preference Optimization | 去掉 ref，用 length-norm | 2024 |
| ORPO | Odds Ratio Preference Optimization | SFT loss + odds ratio | 2024 |
| RLOO | REINFORCE Leave-One-Out | 用 REINFORCE+LOO 估计 baseline | 2022 |

**选择建议**：
```
数据量大、效果优先 → DPO + R-DPO
显存紧张、推理快 → CPO / SimPO
数据无 rejected → KTO
显存紧张 + 效果优先 → ORPO
```

### Q56: 偏好数据集如何构造？

**答案：**

**开源偏好数据集**：

| 数据集 | 规模 | 来源 |
|--------|------|------|
| Anthropic HH-RLHF | 170K | 人类标注 |
| UltraFeedback | 204K | GPT-4 打分 |
| OpenAssistant | 90K | 众包 |
| HelpSteer2 | 21K | NVIDIA 多维标注 |
| Magpie | 1M | Self-Instruct 衍生 |

**构造流程**：
```
Prompt 来源
├── Human-written：用户真实问题（最稀缺）
├── Generated：用种子 prompt LLM 扩展
└── Mixed：两者配比

Response 生成
├── Multi-model sampling：3-5 个模型生成
├── Human ranking：标注员排序
└── Auto-ranking：GPT-4/Claude 当裁判

Quality Check
├── 人工抽检 5-10%
├── 标注员一致性（IAA > 0.7）
└── Margin filtering
```

**踩坑点**：
- 拒绝"明显错误"的回答 → 数据偏负面
- 标注员偏好"漂亮话" → 风格偏差
- "Tie" pair 占比过高 → 信号弱

### Q57: DPO 训练的关键超参？

**答案：**

```yaml
# 关键超参（Llama-Factory 默认）
beta: 0.1                 # KL 系数，越大越保守
learning_rate: 5e-7       # 比 SFT 低 1-2 个数量级
lr_scheduler: cosine
warmup_ratio: 0.1
batch_size: 128           # 偏好 batch（含 chosen+rejected）
epochs: 2-3
loss_type: sigmoid        # 或 ipo, kto, simpo
max_length: 1024          # prompt + response
max_prompt_length: 512
```

**调参经验**：
- `beta` ↑ → 模型更保守，但容易"答非所问"
- `beta` ↓ → 模型更激进，但易 reward hacking
- `lr` 不能太大，否则破坏 SFT 已有能力
- 推荐先用 SimPO 跑基线

---

## 15. OPD On-Policy Distillation

### Q58: 什么是 OPD（On-Policy Distillation）？

**答案：**

**定义**：把 on-policy RL 与 SFT 蒸馏结合，让学生模型从教师模型的 on-policy 轨迹中学习。

**核心动机**：
```
传统 SFT 蒸馏：学生模仿"教师在各种 prompt 上的 offline 回答"
                 → 学生学不到教师的"决策过程"

OPD：学生自己生成回答 → 教师给每个回答打 reward
     → 学生学习"什么样的回答能得高分"
```

**DeepSeek-R1 的应用**：
- 教师：DeepSeek-R1（强推理模型）
- 学生：DeepSeek-V3 / Qwen-7B（更小）
- 数据：800K 高质量 on-policy 轨迹

### Q59: OPD vs SFT 蒸馏 vs PPO？

**答案：**

| 方案 | 数据来源 | 教师参与 | 训练目标 |
|------|----------|----------|----------|
| SFT 蒸馏 | 教师 offline 输出 | 一次性 | 模仿 KL(学生||教师) |
| PPO | 学生 on-policy | RM 评分 | 最大奖励 |
| OPD | 学生 on-policy | 教师 RM | 模仿 + 奖励 |

**公式**：
```
L_OPD = α · KL(π_student || π_teacher) 
      + (1-α) · (-R(y) · log π_student(y|x))

其中 y 是学生自己生成的样本
```

**优势**：
1. 比纯 SFT 蒸馏效果更好（学生"消化"过）
2. 比 PPO 训练更稳定（不需要 critic）
3. 可以看作"带教师指导的 SFT"

### Q60: OPD 的实战 pipeline？

**答案：**

```
┌──────────────────────────────────────────────────────┐
│                OPD Training Pipeline                  │
├──────────────────────────────────────────────────────┤
│                                                       │
│  Step 1: 生成 on-policy 样本                         │
│  ┌──────────────────────────┐                       │
│  │ 学生模型采样 y ~ π_s(·|x)│  × N samples/prompt   │
│  └──────────────────────────┘                       │
│             ↓                                         │
│  Step 2: 教师评分                                     │
│  ┌──────────────────────────┐                       │
│  │ 教师 RM: r = R_T(x, y)   │                       │
│  │ 或教师模型打分            │                       │
│  └──────────────────────────┘                       │
│             ↓                                         │
│  Step 3: 加权 SFT                                     │
│  ┌──────────────────────────┐                       │
│  │ L = -E[ w(y) · log π_s ]│  w(y) = softmax(r/τ)  │
│  └──────────────────────────┘                       │
│             ↓                                         │
│  Step 4: 多轮迭代                                     │
│  ┌──────────────────────────┐                       │
│  │ 学生已更新 → 重新采样     │                       │
│  │ 重复 Step 1-3            │                       │
│  └──────────────────────────┘                       │
└──────────────────────────────────────────────────────┘
```

### Q61: OPD 在 DeepSeek-R1 中的关键创新？

**答案：**

1. **冷启动数据**：先用 6000 条高质量 SFT 数据冷启动
2. **拒绝采样 + RL**：生成 + GRPO 筛选，再做 SFT
3. **多阶段蒸馏**：
   - Stage 1: R1 → 蒸馏到 V3-base
   - Stage 2: V3-base → 进一步小尺寸 (如 7B)
4. **保留推理痕迹**：要求模型输出完整 CoT，不删"嗯/啊"等
5. **数据配比**：推理数据 60%，通用数据 40%

**结论**：DeepSeek-R1-Distill 系列在多项推理 benchmark 上超过 GPT-4o。

### Q62: OPD 面试高频追问点？

**答案：**

**Q: OPD 相比 DPO 的优势？**

| 维度 | DPO | OPD |
|------|-----|-----|
| 教师使用 | 静态偏好数据 | 动态教师打分 |
| 训练信号 | chosen vs rejected | 教师 RL 轨迹 |
| 数据效率 | 中等 | 更高（on-policy 匹配学生分布） |
| 推理能力 | 中等 | 强（可继承 R1 推理模式） |

**Q: OPD 在哪些场景特别有效？**
- 推理任务（数学、代码）
- 长 CoT 蒸馏
- 教师昂贵不能全量微调时

**Q: OPD 的失败模式？**
- 教师 RM 本身有 bias → 学生继承
- on-policy 采样成本高
- 训练不稳定（on-policy 方差大）

---

## 面试高频追问

### 深度追问方向

1. **Attention计算复杂度**：O(N²·d)，如何优化？
2. **LLM幻觉问题**：有哪些缓解方法？
3. **Context Length扩展**：如何支持百万token上下文？
4. **长上下文Attention稀疏化**：哪些方法？
5. **模型评测**：有哪些benchmark？各有什么问题？
6. **指令遵循**：如何训练？数据如何构建？
7. **思维链（CoT）**：为什么有效？
8. **涌现能力**：什么是涌现？为什么会出现？
9. **🆕 数据质量vs数量**：Scaling Law 在数据上的体现？
10. **🆕 SFT数据配比**：如何平衡多domain？
11. **🆕 Reward Hacking**：如何检测和缓解？
12. **🆕 DPO的局限**：哪些情况下DPO不如PPO？
13. **🆕 GRPO vs PPO**：显存和效果的trade-off？
14. **🆕 OPD vs 传统蒸馏**：为什么on-policy重要？
15. **🆕 DeepSeek-R1的训练流程**：能否完整讲一遍？

---

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [LLaMA Paper](https://arxiv.org/abs/2302.13971)
- [LLaMA-2 Paper](https://arxiv.org/abs/2307.09288)
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [RAG Survey](https://arxiv.org/abs/2312.10997)
- [Self-Instruct Paper](https://arxiv.org/abs/2212.10560)
- [WizardLM Evol-Instruct](https://arxiv.org/abs/2304.12244)
- [LLaMA-2-Chat RLHF](https://arxiv.org/abs/2307.09288)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [IPO Paper](https://arxiv.org/abs/2310.12036)
- [KTO Paper](https://arxiv.org/abs/2402.01306)
- [SimPO Paper](https://arxiv.org/abs/2405.14734)
- [ORPO Paper](https://arxiv.org/abs/2403.07691)
- [GRPO Paper (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948)
- [DataComp-LM (Data Filtering)](https://arxiv.org/abs/2406.04170)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
