# Dual Cross-Attention Learning for Fine-Grained Visual Categorization and Object Re-Identification

## Introduction

In this work, we investigate how to extend self-attention modules to better learn subtle feature embeddings for recognizing fine-grained objects, e.g., different bird species or person identities. 

We adopt a different way to incorporate local information based on Vision Transformer. To this end, we propose global-local cross-attention (GLCA) to enhance the interactions between global images and local high-response regions. Specifically, we compute the cross-attention between a selected subset of query vectors and the entire set of key-value vectors. By coordinating with self-attention learning, GLCA can help reinforce the spatial-wise discriminative clues to recognize fine-grained objects.

We propose pair-wise cross-attention (PWCA) to regularize the attention learning of an image by treating another image as distractor. Specifically, we compute the cross-attention between query of an image and combined key-value from both images. By introducing confusion in key and value vectors, the attention scores are diffused to another image so that the difficulty of the attention learning of the current image increases. Such regularization allows the network to discover more discriminative regions and alleviate overfitting to sample-specific features. It is noted that PWCA is only used for training and thus does not introduce extra computation cost during inference. 

## Proposed Approach

### Global-Local Cross-Attention


Self-attention treats each query equally to compute global attention scores. In other words, each local position of the image is interacted with all the positions in the same manner. For recognizing fine-grained objects, we expect to mine discriminative local information to facilitate the learning of subtle features. To this end, we propose global-local cross-attention to emphasize the interaction between global images and local high-response regions.

To identify these high-response regions, we use an approach called **attention rollout** to track information flow through Transformer layers, accounting for residual connections:

$$\bar{S} = 0.5S + 0.5E$$

The accumulated attention from input to layer $i$ is:
$$\hat{S}_i = \bar{S}_i \otimes \bar{S}_{i-1} \cdots \otimes \bar{S}_1$$

We select the top $R$ tokens with highest accumulated attention as local queries $Q^l$.

Finally, we compute the cross-attention between these selected local queries and the global set of key-value pairs as below.

$$
    \hat{S}_i = \bar{S}_i \otimes \bar{S}_{i-1} \cdots \otimes \bar{S}_1
    \tag{eq:rollout}
$$

where $\bar{S}=0.5S+0.5E$ means the re-normalized attention weights using an identity matrix $E$ to consider residual connections, $\otimes$ means the matrix multiplication operation. In this way, we track down the information propagated from the input layer to a higher layer. Then, we use the aggregated attention map to mine the high-response regions. According to Eq. [eq:rollout], the first row of $\hat{S}_i = [\hat{s}_{i,j}]_{(N+1)\times (N+1)}$ means the accumulated weights of class embedding $\hat{\texttt{CLS}}$. We select top $R$ query vectors from $Q_i$ that correspond to the top $R$ highest responses in the accumulated weights of $\hat{\texttt{CLS}}$ to construct a new query matrix $Q^l$, representing the most attentive local embeddings. Finally, we compute the cross attention between the selected local query and the global set of key-value pairs as below.

$$
f_{\text{GLCA}}(Q^l,K^g,V^g)=\text{softmax}(\frac{Q^l{K^g}^T}{\sqrt{d}})V^g 
\tag{eq:glca}
$$

In self-attention, all the query vectors will be interacted with the key-value vectors. In our GLCA (Eq. [eq:glca]), only a subset of query vectors will be interacted with the key-value vectors. We observe that GLCA can help reinforce the spatial-wise discriminative clues to promote recognition of fine-grained classes. Another possible choice is to compute the self-attention between local query $Q^l$ and local key-value vectors ($K^l$, $V^l$). However, through establishing the interaction between local query and global key-value vectors, we can relate the high-response regions with not only themselves but also with other context outside of them.

### Pair-Wise Cross-Attention

The scale of fine-grained recognition datasets is usually not as large as that of general image classification, e.g., ImageNet contains over 1 million images of 1,000 classes while CUB contains only 5,994 images of 200 classes for training. Moreover, smaller visual differences between classes exist in FGVC and Re-ID compared to large-scale classification tasks. Fewer samples per class may lead to network overfitting to sample-specific features for distinguishing visually confusing classes in order to minimize the training error. 

To alleviate the problem, we propose pair-wise cross attention to establish the interactions between image pairs. PWCA can be viewed as a novel regularization method to regularize the attention learning. Specifically, we randomly sample two images ($I_1$, $I_2$) from the same training set to construct the pair. The query, key and value vectors are separately computed for both images of a pair. For training $I_1$, we concatenate the key and value matrices of both images, and then compute the attention between the query of the target image and the combined key-value pairs as follows:

$$
    f_{\text{PWCA}}(Q_1,K_c,V_c) =\text{softmax}(\frac{Q_1 K_c^T}{\sqrt{d}})V_c
\tag{eq:pwca}
$$

where $K_c=[K_1;K_2] \in \mathbb{R}^{(2N+2)\times d}$ and $V_c=[V_1;V_2] \in \mathbb{R}^{(2N+2)\times d}$. For a specific query from $I_1$, we compute $N+1$ self-attention scores within itself and $N+1$ cross-attention scores with $I_2$ according to Eq. [eq:pwca]. All the $2N+2$ attention scores are normalized by the softmax function together and thereby contaminated attention scores for the target image $I_1$ are learned. 

We note that the softmax function is applied for each query separately (i.e., over each row of input matrix). Thus, the normalized attention scores are computed based on $(N+1)\times 2$ values, and the final attention output is also computed with a weighted sum of $(N+1)\times 2$ values. That leads to contaminated attention scores for the target image $I_1$. 

Optimizing this noisy attention output increases the difficulty of network training and reduces the overfitting to sample-specific features. Note that PWCA is only used for training and will be removed for inference without consuming extra computation cost.  

## Experiments

### Experimental Setting

**Datasets.**

We conduct extensive experiments on two fine-grained recognition tasks: fine-grained visual categorization (FGVC) and object re-identification (Re-ID). For FGVC, we use three standard benchmarks for evaluations: CUB-200-2011, Stanford Cars, FGVC-Aircraft.
 
For Re-ID, we use four standard benchmarks: Market1501, DukeMTMC-ReID, MSMT17 for Person Re-ID and VeRi-776 for Vehicle Re-ID. In all experiments, we use the official train and validation splits for evaluation.

**Baselines.** 

**Baselines:** 

We evaluate on DeiT and ViT with architectures: DeiT-T/S/B/16, ViT-B/16, and R50-ViT-B/16 with $L=12$ SA blocks for evaluation.

Attention map is generated using [vit-explain](vit-explain.md) and the selected high-response patches.

**Implementation Details.**

We coordinate the proposed two types of cross-attention with self-attention in the form
of multi-task learning. We build $L = 12$ SA blocks, $M = 1$ GLCA blocks and $T = 12$ PWCA blocks as the overall architecture for training. The PWCA branch shares weights with the SA branch while GLCA does not share weights with SA.

To balance the different loss terms during collaborative optimization, we adopt a dynamic loss weighting strategy based on the uncertainty loss method (as used in FairMOT, following Kendall et al., 2018). This approach introduces learnable parameters to automatically adjust the contribution of each loss term, avoiding manual hyper-parameter search. The total loss is formulated as:

$$
    L_{\text{total}} = \frac{1}{2}\left(\frac{1}{e^{w_1}}L_1 + \frac{1}{e^{w_2}}L_2 + w_1 + w_2\right)
$$

where $L_1$ and $L_2$ are the individual task losses: self-attention and cross-attention branches, and $w_1$, $w_2$ are learnable parameters that balance the two tasks. This formulation allows the network to dynamically balance the losses during training.

The PWCA branch has the same ground truth target as the SA branch since we treat another image as distractor.

For FGVC, we resize the original image into 448 $\times$ 448 for training. The sequence length of input embeddings for self-attention baseline is $28\times 28=784$. We select input embeddings with top $R=10\%$ highest attention responses as local queries. We apply stochastic depth regularization, randomly dropping layers during training while using the full network at inference. We use Adam optimizer with weight decay of 0.05 for training. The learning rate is initialized as ${\rm lr}_{scaled}=\frac{5e-4}{512}\times batchsize$ and decayed with a cosine policy. We train the network for 100 epochs with batch size of 16 using the standard cross-entropy loss. 

For Re-ID, we resize the image into 256 $\times$ 128 for pedestrian datasets, and 256 $\times$ 256 for vehicle datasets. We select input embeddings with top $R=30\%$ highest attention responses as local queries. We use SGD optimizer with a momentum of 0.9 and a weight decay of 1e-4. The batch size is set to 64 with 4 images per ID. The learning rate is initialized as 0.008 and decayed with a cosine policy. We train the network for 120 epochs using the cross-entropy and triplet losses.

All of our experiments are conducted on PyTorch with Nvidia Tesla V100 GPUs. Our method costs 3.8 hours with DeiT-Tiny backbone for training using 4 GPUs on CUB, and 9.5 hours with ViT-Base for training using 1 GPU on MSMT17. During inference, we remove all the PWCA modules and only use the SA and GLCA modules. We add class probabilities output by classifiers of SA and GLCA for prediction for FGVC, and concat two final class tokens of SA and GLCA for prediction for Re-ID. A single image with the same input size as training is used for test. 

### Results on Fine-Grained Visual Categorization

We evaluate our method on three standard FGVC benchmarks, results shown in Table 1, particularly, with the R50-ViT-Base backbone, DCAL reaches 92.0\%, 95.3\% and 93.3\% top-1 accuracy on CUB-200-2011, Stanford Cars and FGVC-Aircraft benchmarks, respectively. Our method can consistently improve different vision Transformer baselines on all the three benchmarks, e.g., surpassing the pure Transformer (DeiT-Tiny) by 2.2\% and the hybrid structure of CNN and Transformer (R50-ViT-Base) by 1.3\% on Stanford Cars.

| Backbone     | CUB | Cars | Aircraft |
|--------------|-----|------|----------|
| DeiT-Tiny    | +2.5| +2.2 | +2.7     |
| DeiT-Small   | +1.8| +1.6 | +1.9     |
| DeiT-Base    | +0.8| +0.9 | +2.3     |
| ViT-Base     | +0.6| +0.9 | +1.5     |
| R50-ViT-Base | +0.7| +1.3 | +0.9     |

Table 1: Improvement over baselines (%) on FGVC benchmarks: CUB-200-2011, Stanford Cars and FGVC-Aircraft.

### Results on Object Re-ID

We evaluate our method on four standard Re-ID benchmarks in Table 2. Particularly, with the ViT-Base backbone, DCAL reaches 80.2\%, 64.0\%, 87.5\%, 80.1\% mAP on VeRi-776, MSMT17, Market1501, DukeMTMC, respectively. Similar to FGVC, our method can consistently improve different vision Transformer baselines, e.g., surpassing the light-weight Transformer (DeiT-Tiny) by 2.8\% and the larger Transformer (ViT-Base) by 2.4\% on MSMT17.

| Backbone     | VeRi-776 | MSMT17 | Market1501 | DukeMTMC |
|--------------|----------|--------|------------|----------|
| DeiT-Tiny    | +2.8 mAP | +2.8   | +1.9       | +2.2     |
| DeiT-Small   | +1.4 mAP | +1.8   | +1.0       | +1.7     |
| DeiT-Base    | +1.7 mAP | +1.8   | +0.6       | +1.1     |
| ViT-Base     | +2.1 mAP | +2.4   | +0.4       | +1.2     |
Table 2: mAP improvement over baselines (%) on Re-ID benchmarks: VeRi-776, MSMT17, Market1501, DukeMTMC. The input size is 256x128 for pedestrian datasets and 256x256 for vehicle datasets. * means results without side information for fair comparison.

### Ablation Study

**Contributions from Algorithmic Components.**

We examine the contributions from the two types of cross-attention modules using different vision Transformer baselines. We use DeiT-Tiny for FGVC and ViT-Base for Re-ID. With either GLCA or PWCA alone, our method can obtain higher performance than the baselines. With both cross-attention modules, we can further improve the results. We note that PWCA will be removed for inference so that it does not introduce extra parameters or FLOPs. We uses one GLCA module in our method, which only requires a small increase of parameters or FLOPs compared to the baseline. Both GLCA and PWCA individually improve performance over baselines. GLCA adds ~9% parameters and ~4% FLOPs, while PWCA adds no inference cost. Combined, they achieve best results across all datasets.

**Ablation Study on GLCA.** 

1. Cross-ViT is a most recent method based on cross-attention for general image classification. It constructs two Transformer branches to handle image tokens of different sizes and uses the class token from one branch to interact with patch tokens from another branch. We implement this idea using the same selected local queries and the same DeiT-Tiny backbone. The cross-token strategy obtains 82.1\% accuracy on CUB, which is worse than our GLCA by 1\%.
2. Another possible baseline to incorporate local information is computing the self-attention for the high-response local regions (i.e., local query, key and value vectors). This local self-attention baseline obtains 82.6\% accuracy on CUB using the DeiT-Tiny backbone, which is also worse than our GLCA (83.1\%). 
3. We conduct more ablation experiments to examine the effect of GLCA. We obtain 82.6\% accuracy on CUB by selecting local query randomly and obtain 82.8\% by selecting local query based on the penultimate layer only. Our GLCA outperforms both baselines, validating that mining high-response local query with aggregated attention map is effective for our cross-attention learning.


| Method                         | CUB | MSMT17 |
|--------------------------------|-----|--------|
|                                | Acc | mAP    |
| Baseline                       | 82.1| 61.6   |
| + PWCA                         | 83.1| 62.8   |
| + Adding noise in $I_1$        | 77.3| 56.0   |
| + Adding noise in label of $I_1$ | 81.6| 60.8   |
| + $I_2$ from noise             | 82.1| 62.1   |
| + $I_2$ from COCO              | 82.5| 62.2   |
| + $I_2$ from intra-class only  | 81.7| 62.2   |
| + $I_2$ from inter-class only  | 83.0| 62.7   |
| + $I_2$ from intra- & inter-class (1:1) | 83.0| 62.5   |

Table 4: Comparisons of different regularization methods. DeiT-Tiny is used for CUB and ViT-Base is used for MSMT17.

**Ablation Study on PWCA.**

We compare PWCA with different regularization strategies in Table 4 by taking $I_1$ as the target image. The results show that adding image noise or label noise without cross-attention causes degraded performance compared to the self-attention learning baseline. As the extra image $I_2$ used in PWCA can be viewed as distractor, we also test replacing the key and value embeddings of $I_2$ with Gaussian noise. Such method performs better than adding image / label noise, but still worse than our method. Moreover, sampling $I_2$ from a different dataset (i.e., COCO), sampling intra-class / inter-class pair only, or sampling intra-class \& inter-class pairs with equal probability performs worse than PWCA. We assume that the randomly sampled image pairs from the same dataset (i.e., natural distribution of the dataset) can regularize our cross-attention learning well.

**Amount of Cross-Attention Blocks.** 

For GLCA, the results show that $M=1$ performs best. We analyze that the deeper Transformer encoder can produce more accurate accumulated attention scores as the attention flow is propagated from the input layer to higher layer. Moreover, using one GLCA block only introduces small extra Parameters and FLOPs for inference. For PWCA, the results show that $T=12$ performs best. It implies that adding $I_2$ throughout all the encoders can sufficiently regularize the network as our self-attention baseline has $L=12$ blocks in total. Note that PWCA is only used for training and will be removed for inference without consuming extra computation cost.

### Limitations

Training time increases by ~80% due to joint optimization. GLCA adds ~9% parameters and ~2% FLOPs during inference.

## Different Inference Architectures

For FGVC, we add class probabilities output by classifiers of SA and GLCA for prediction. For Re-ID, we concat two final class tokens of SA and GLCA as the output feature for prediction. We also test two different inference architectures: 

1. ``SA``: using the last SA module for inference. 
2. ``GLCA``: using the GLCA module for inference. 

The results show that only using the SA or GLCA module can obtain similar performance with our default setting. It is also noted that ``SA`` has the same inference architecture with the baseline by removing all the PWCA and GLCA modules for inference, which does not introduce extra computation cost.

## Ablation Study on Effect of $R$

Performance is stable across different R values. We use R=10% for FGVC and R=30% for Re-ID.

## More Transformer Baselines

DCAL improves CaiT-XS24 by 1.2% and Swin-T by 0.9% on CUB.