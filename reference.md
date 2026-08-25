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
