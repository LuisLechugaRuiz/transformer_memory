## External Memory-augmented Transformer

### Overview

Traditional transformer architectures have shown impressive results across various tasks. However, their capacity to remember long-term dependencies or historical context can be limited by the fixed size of the input window. To tackle this limitation, we introduce an external memory cell to the transformer, allowing it to access and leverage historical information, making it particularly suitable for tasks with temporal significance.

### Core Concepts

- **External Memory Cell (MM)**: An external memory matrix that acts as a reservoir of information, akin to the LSTM's cell state. This memory matrix allows the transformer to access historical information, enabling a richer context beyond the current input.
    
- **Interaction with Inputs**: Before the inputs are processed by the transformer layers, they interact with MM. This process can be perceived as the input being "pre-processed" with historical context. The nature of this interaction—whether it's an addition, concatenation, gating, etc.—can be explored and fine-tuned for optimal results.
    
- **Attention-based Memory Update**:
    
    1. **Hierarchical Attention**:
        - **Layer-Level Attention**: Each intermediate layer's output computes an attention score over the memory, yielding a weighted memory representation for each layer.
        - **Global Attention**: Using the above representations, a global attention over the memory is computed. This singular representation encompasses the importance of each layer's contribution concerning the memory.
    2. **Layer Selection**: Rather than utilizing all layers, only the outputs from the top few (e.g., top 2 or 3) layers are considered. These often encapsulate more abstract and holistic information, potentially making them more pertinent for updating the memory.
    3. **Pooling Over Layers**: Each layer's output undergoes max or average pooling to reduce its sequence length before hierarchical attention. This step might enable the model to concentrate on the most pivotal parts of the sequences.
    4. **Gated Memory Update**: A gate, derived from the final attention-weighted representation and processed through a sigmoid activation, dictates which components of the memory are updated and by how much. This ensures selective and relevant updating of the memory.
- **Relevance to Time-sensitive Tasks**: The introduced architecture shines in scenarios where the order and temporal aspect are paramount. For instance, in robotic tasks where the sequence of actions can dictate success, having a memory of prior actions or observations enables the model to make better-informed decisions.