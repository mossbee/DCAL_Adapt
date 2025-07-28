# PromptCAM: Making Vision Transformers Interpretable for Fine-Grained Analysis

**Illustration of PromptCAM.** By learning class-specific prompts for a pre-trained Vision Transformer (ViT), PromptCAM enables multiple functionalities. 
- PromptCAM achieves fine-grained image classification using the output logits from the class-specific prompts. 
- PromptCAM enables trait localization by visualizing the multi-head attention maps queried by the true-class prompt. 
- PromptCAM identifies common traits shared between species by visualizing the attention maps queried by another-class prompt. 
- PromptCAM can identify the most discriminative traits per species (e.g., distinctive yellow chest and black neck for ``Scott Oriole``) by systematically masking out the least important attention heads.

## Abstract

We present a simple approach to make pre-trained Vision Transformers (ViTs) interpretable for fine-grained analysis, aiming to identify and localize the traits that distinguish visually similar categories, such as bird species. Pre-trained ViTs, such as DINO, have demonstrated remarkable capabilities in extracting localized, discriminative features. However, saliency maps like Grad-CAM often fail to identify these traits, producing blurred, coarse heatmaps that highlight entire objects instead. We propose a novel approach, **Prompt Class Attention Map (PromptCAM)**, to address this limitation.  PromptCAM learns class-specific prompts for a pre-trained ViT and uses the corresponding outputs for classification. To correctly classify an image, the true-class prompt must attend to unique image patches not present in other classes' images  (\ie, traits). As a result, the true class's multi-head attention maps reveal traits and their locations. Implementation-wise, PromptCAM is almost a ``free lunch,`` requiring only a modification to the prediction head of Visual Prompt Tuning (VPT). This makes PromptCAM easy to train and apply, in stark contrast to other interpretable methods that require designing specific models and training processes. Extensive empirical studies on a dozen datasets from various domains (e.g., birds, fishes, insects, fungi, flowers, food, and cars) validate the superior interpretation capability of PromptCAM. The source code and demo are available at \url{https://github.com/Imageomics/Prompt_CAM}.

## Introduction

Vision Transformers (ViT) pre-trained on huge datasets have greatly improved vision recognition, even for fine-grained objects~. DINO  and DINOv2  further showed remarkable abilities to extract features that are localized and informative, precisely representing the corresponding coordinates in the input image. These advancements open up the possibility of using pre-trained ViTs to discover ``traits`` that highlight each category's identity and distinguish it from other visually close ones.

One popular approach to this is saliency maps, for example, Class Activation Map (CAM)~. After extracting the feature maps from an image, CAM highlights the spatial grids whose feature vectors align with the target class's fully connected weight. While easy to implement and efficient, the reported CAM saliency on ViTs is often far from expectation. It frequently locates the whole object with a blurred, coarse heatmap, instead of focusing on subtle traits that tell visually similar objects (e.g., birds) apart. One may argue that CAM was not originally developed for ViTs, but even with dedicated variants like attention rollout~, the issue is only mildly attenuated.

*What if we look at the attention maps?* ViTs rely on self-attention to relate image patches; the [CLS] token aggregates image features by attending to informative patches. As shown in~, the attention maps of the [CLS] token do highlight local regions inside the object. *However, these regions are not ``class-specific.``* Instead, they often focus on the same object regions across different categories, such as body parts like heads, wings, and tails of bird species. While these are where traits usually reside, they are not traits. For example, the distinction between ``Red-winged Blackbird`` and other bird species is the red spot on the wing, having little to do with other body parts.  

How can we leverage pre-trained ViTs, particularly their localized and informative patch features, to identify traits that are so special for each category?

Our proposal is to *prompt* ViTs with learnable ``class-specific`` tokens, one for each class, inspired by~. These ``class-specific`` tokens, once inputted into ViTs, *attend* to image patches via self-attention, similar to the [CLS] token. However, unlike the [CLS] token, which is ``class-agnostic,`` these ``class-specific`` tokens can *attend to the same image differently*, with the potential to highlight regions specific to the corresponding classes, \ie, traits.

We implement our approach, named **Prompt Class Attention Map (PromptCAM)**, as follows. Given a pre-trained ViT and a fine-grained classification dataset with $C$ classes, we add $C$ learnable tokens as additional inputs alongside the input image. To make these tokens ``class-specific,`` we collect their corresponding output vectors after the final Transformer layer and perform inner products with a shared vector (also learnable) to obtain $C$ ``class-specific`` scores, following~. One may interpret each class-specific score as how clearly the corresponding class's traits are visible in the input image. Intuitively, the input image's ground-truth class should possess the highest score, and we encourage this by minimizing a cross-entropy loss, treating the scores as logits. We keep the whole pre-trained ViT frozen and only optimize the $C$ tokens and the shared scoring vector. See  for details and variants.

For interpretation during inference, we input the image and the $C$ tokens simultaneously to the ViT to obtain the $C$ scores. One can then select a specific class (e.g., the highest-score class) and visualize its multi-head attention maps over the image patches. See  for an illustration and   for how to rank these maps to highlight the most discriminative traits. When the highest-score class is the ground-truth class, the attention maps reveal its traits. Otherwise, comparing the attention maps of the highest-score class with those of the ground-truth class helps explain why the image is misclassified. Possible reasons include the object being partially occluded or in an unusual pose, making its traits invisible, or the appearance being too similar to a wrong class, possibly due to lighting conditions ().

**PromptCAM is fairly easy to implement and train.**
*It requires no change to pre-trained ViTs and no specially designed loss function or training strategy*---just the standard cross-entropy loss and SGD. Indeed, building upon Visual Prompt Tuning (VPT)~, one merely needs to adjust a few lines of code and can enjoy fine-grained interpretation.
This simplicity sharply contrasts other interpretable methods like ProtoPNet~ and ProtoTree~. %, and TesNet~. 
Compared to INterpretable TRansformer (INTR) , which also featured simplicity, PromptCAM has three notable advantages.
First, PromptCAM is *encoder-only* and can potentially utilize any ViT encoder. In contrast, INTR is built upon an encoder-decoder model pre-trained on object detection datasets. As a result, PromptCAM can more easily leverage up-to-date pre-trained models.  Second, PromptCAM can be trained much faster---only the prompts and the shared vector need to be learned. In contrast, INTR typically requires full fine-tuning. Third, PromptCAM produces cleaner and sharper attention maps than INTR, which we attribute to the use of state-of-the-art ViTs like DINO~ or DINOv2~. Taken together, we view PromptCAM as a *simpler* yet more powerful interpretable Transformer. 


We validate PromptCAM on over a dozen datasets: CUB-200-2011~, Birds-525~, Oxford
Pet~, Stanford Dogs~, Stanford Cars~, iNaturalist-2021-Moths~, Fish Vista~, Rare Species~, Insects-2~, iNaturalist-2021-Fungi~, Oxford Flowers~, Medicinal Leaf~, Stanford Cars~, and Food 101~. PromptCAM can identify different traits of a category through multi-head attention and consistently localize them in images. *To our knowledge, PromptCAM is the only explainable or interpretable method for vision that has been evaluated on such a broad range of domains.* We further show PromptCAM's extendability by applying it to discovering taxonomy keys. Our contributions are two-fold. 

- We present **PromptCAM**, an easily implementable, trainable, and reproducible *interpretable* method that leverages the representations of pre-trained ViTs to identify and localize traits for fine-grained analysis.
- We conduct extensive experiments on more than a dozen datasets to validate **PromptCAM**'s interpretation quality, wide applicability, and extendability.  


**Comparison to closely related work.** Besides INTR~, our class-specific attentions are inspired by two other works in different contexts, MCTformer for weakly supervised semantic segmentation  and Query2Label for multi-label classification . Both of them learned class-specific tokens but aimed to localize visually distinct common objects (e.g., people, horses, and flights). In contrast, we focus on fine-grained analysis: supervised by class labels of visually similar objects (e.g., bird species), we aim to localize their traits (e.g., red spots on wings). One particular feature of PromptCAM is its *simplicity**, in both implementation and compatibility with pre-trained backbones, without extra modules, loss terms, and changes to the backbones, making it an almost plug-and-pay approach to interpretation. 

Due to space constraints, we provide a detailed related work section in the Supplementary Material (Suppl.).

## Approach

We propose **Prompt Class Attention Map (PromptCAM)** to leverage pre-trained Vision Transformers (ViTs)~ for fine-grained analysis. The goal is to identify and localize traits that highlight an object category’s identity. PromptCAM adds learnable class-specific tokens to prompt ViTs, producing class-specific attention maps that reveal traits. 
The overall framework is presented in.  *We deliberately follow the notation and naming of Visual Prompt Tuning (VPT)~ for ease of reference.*  

### Preliminaries

A ViT typically contains $N$ Transformer layers~. Each consists of a Multi-head Self-Attention (MSA) block, a Multi-Layer Perceptron (MLP)
block, and several other operations like layer normalization and residual connections. 

The input image $I$ to ViTs is first divided into $M$ fixed-sized patches. Each is then projected into a $D$-dimensional feature space with positional encoding, denoted by $e_0^{j}$, with $1\leq j \leq M$. We use $E_0=[e_0^{1}, \cdots, e_0^{M}]\in\R^{D\times M}$ to denote their column-wise concatenation.  

Together with a learnable [CLS] token $x_0\in\R^D$, the whole ViT is formulated as:
$$
[E_i, x_i] = L_i([E_{i-1}, x_{i-1} ]), \quad i = 1, \cdots, N, \nonumber
$$
where $L_i$ denotes the $i$-th Transformer layer. The final $x_N$ is typically used to represent the whole image and fed into a prediction head for classification. 

### Prompt Class Attention Map (PromptCAM)


Given a pre-trained ViT and a downstream classification dataset with $C$ classes, we introduce a set of $C$ learnable $D$-dimensional vectors to prompt the ViT. These vectors are learned to be ``class-specific`` by minimizing the cross-entropy loss, during which the ViT backbone is frozen. In the following, we first introduce the baseline version.

**PromptCAMS.** The $C$  class-specific prompts are injected into the first Transformer layer $L_1$. We denote each prompt by $p^{c}\in\R^D$, where $1\leq c\leq C$, and use $P = [p^{1},\cdots,p^{C}]\in\R^{D\times C}$ to indicate their column-wise concatenation. The prompted ViT is:
$$
[Z_1, E_1, x_1]  = L_1([P, E_{0}, x_{0}]) \nonumber\\
[Z_i, E_i, x_i]  = L_i([Z_{i-1},  E_{i-1}, x_{i-1}]), \quad  i = 2, \cdots, N, \nonumber
$$
where $Z_i$ represents the features corresponding to $P$, computed by the $i$-th Transformer layer $L_i$. The order among $x_{0}$, $E_{0}$, and $P$ does not matter since the positional encoding of patch locations has already been inserted into $E_{0}$. 

To make $P = [p^1,\cdots,p^C]$ class-specific, we employ a cross-entropy loss on top of the corresponding ViT's output, \ie, $Z_N = [z_N^{1}, \cdots, z_N^{C}]$. Given a labeled training example $(I, y\in\{1,\cdots, C\})$, we calculate the logit of each class by:

$$
    s[c] = w^{\top}z_N^{c}, \quad 1\leq c \leq C, \tag{eq:score\_rule}
$$

where $w\in\R^D$ is a learnable vector. $P$ can then be updated by minimizing the loss:

$$
-\log\left(\cfrac{\exp{\left(s[y]\right)}}{\sum_c \exp{\left(s[c]\right)}}\right). \tag{eq:loss}
$$

**PromptCAMD.** While straightforward, PromptCAMS has two potential drawbacks. First, the class-specific prompts attend to every layer's patch features, \ie, $E_i$,  $i = 0, \cdots,  N-1$. However, features of the early layers are often not informative enough but noisy for differentiating classes. Second, the prompts $p^1,\cdots,p^C$ have a ``double duty.`` Individually, each needs to highlight class-specific traits. Collectively, they need to adapt pre-trained ViTs to downstream tasks, which is the original purpose of VPT~. In our case, the downstream task is *a new usage of ViTs on a specific fine-grained dataset.*
  
To address these issues, we resort to the VPT-Deep's design while deliberately *decoupling* injected prompts' roles. VPT-Deep adds learnable prompts to every layer's input. Denote by $P_{i-1}=[p_{i-1}^1,\cdots,p_{i-1}^C]$ the prompts to the $i$-th Transformer layer, the deep-prompted ViT is formulated as:  
$$
[Z_i, E_i, x_i] = L_i([P_{i-1}, E_{i-1}, x_{i-1}]), \quad i = 1, \cdots,  N,\tag{eq:VPT-Deep}
$$
It is worth noting that the features $Z_i$ after the $i$-th layer are not inputted to the next layer, and are typically disregarded. 

In PromptCAMD, we repurpose $Z_N$ for classification, following~. As such, after minimizing the cross entropy loss in~, the corresponding prompts $P_{N-1}=[p_{N-1}^1,\cdots,p_{N-1}^C]$ will be *class-specific*. Prompts to the other layers' inputs, \ie, $P_{i}=[p_{i}^1,\cdots,p_{i}^C]$ for $i = 0, \cdots, N-2$, remain *class-agnostic*, because $p_{i}^c$ does not particularly serve for the $c$-th class, unlike $p_{N-1}^c$. *In other words, PromptCAMD learns both class-specific prompts for trait localization and class-agnostic prompts for adaptation.* The class-specific prompts $P_{N-1}$ only attend to the patch features $E_{N-1}$ inputted to the last Transformer layer $L_N$, further addressing the other issue in PromptCAMS. 

*In the following, we focus on PromptCAMD.*

### Trait Identification and Localization

During inference, given an image $I$, PromptCAMD extracts patch embeddings $E_0=[e_0^{1}, \cdots, e_0^{M}]$ and follows 
 to obtain $Z_N$ and  to obtain $s[c]$ for $c\in\{1,\cdots, C\}$. The predicted label $\hat{y}$ is:

$$
\hat{y} = \argmax_{c\in\{1,\cdots, C\}} s[c].
$$

**What are the traits of class $c$?** To answer this question, one could collect images whose true and predicted classes are both class $c$ (\ie, correctly classified) and visualize the multi-head attention maps queried by $p_{N-1}^c$ in layer $L_N$. 

Specifically, in layer $L_N$ with $R$ attention heads, the patch features $E_{N-1}\in\R^{D\times M}$ are projected into $R$ key matrices, denoted by $K_{N-1}^r\in\R^{D'\times M}$, $r = 1, \cdots, R$.
The $j$-th column corresponds to the $j$-th patch in $I$. Meanwhile, the prompt $p_{N-1}^c$ is projected into $R$ query vectors $q_{N-1}^{c,r}\in\R^{D'}$, $r = 1, \cdots, R$. Queried by $p_{N-1}^c$, the $r$-th head's attention map $\alpha^{c,r}_{N-1}\in\R^M$  is computed by:

$$
\alpha^{c,r}_{N-1} = \text{softmax} \left(\cfrac{{K_{N-1}^r}^{\top}q^{c,r}_{N-1}}{D'}\right)\in\R^M. \tag{eq:attention\_map}
$$

Conceptually, from the $r$-th head's perspective, the weight $\alpha^{c,r}_{N-1}[j]$ indicates how important the $j$-th patch is for classifying class $c$, hence localizing traits in the image. Ideally, each head should attend to different (sets of) patches to look for multiple traits that together highlight class $c$'s identity. By visualizing each attention map $\alpha^{c,r}_{N-1}$, $r = 1, \cdots, R$, 
instead of pooling them averagely, PromptCAM can potentially identify up to $R$ different traits for class $c$. 
 
**Which traits are more discriminative?** For categories that are so distinctive, like ``Red-winged Blackbird,`` a few traits are sufficient to distinguish them from others. To automatically identify these most discriminative traits, we take a greedy approach, *progressively blurring* the least important attention maps until the image is misclassified. The remaining ones highlight traits that are sufficient for classification.


Suppose class $c$ is the true class and the image is correctly classified. In each greedy step, for each of the unblurred heads indexed by $r'$, we iteratively replace $\alpha^{c,r'}_{N-1}$ with $\frac{1}{M}**1**$ and recalculate $s[c]$ in , where $**1**\in\R^M$ is an all-one vector. Doing so essentially blurs the $r'$-th head for class $c$, preventing it from focusing. The head with the *highest blurred $s[c]$* is thus the *least* important, as blurring it degrades classification the least. See Suppl.~for details.

**Why is an image wrongly classified?**
When $\hat{y}\neq y$ for a labeled image $(I,y)$, one could visualize both $\{\alpha^{y,r}_{N-1}\}_{r=1}^R$ and $\{\alpha^{\hat{y},r}_{N-1}\}_{r=1}^R$ to understand why the classifier made such a prediction. For example, some traits of class $y$ may be invisible or unclear in $I$; the object in $I$ may possess class $\hat{y}$'s visual traits, for example, due to light conditions. 

### Variants and Extensions

**Other PromptCAM designs.** Besides injecting class-specific prompts to the first layer (\ie, PromptCAMS) or the last (\ie, PromptCAMD), we also explore their interpolation. We introduce class-specific prompts like PromptCAMS to the $i$-th layer and class-agnostic prompts like PromptCAMD to the first $i-1$ layers. See the Suppl.~for a comparison.


**PromptCAM for discovering taxonomy keys.** So far, we have focused on a ``flat`` comparison over all the categories. In domains like biology that are full of fine-grained categories, researchers often have built hierarchical decision trees to ease manual categorization, such as taxonomy. The role of each intermediate ``tree node`` is to dichotomize a subset of categories into multiple groups, each possessing certain *group-level* characteristics (\ie, taxonomy keys).       

The *simplicity* of PromptCAM allows us to efficiently train multiple sets of prompts, one for each intermediate tree node, potentially *(re-)discovering* the taxonomy keys. One just needs to relabel categories of the same group by a single label, before training. In expectation, along the path from the root to a leaf node, each of the intermediate tree nodes should look at different group-level traits on the same image of that leaf node. See~ for a preliminary result.
 
### What is PromptCAM suited for?

As our paper is titled, PromptCAM is dedicated to fine-grained *analysis*, aiming to identify and, more importantly, *localize* traits useful for differentiating categories. This, however, does not mean that PromptCAM would excel in fine-grained classification *accuracy*. Modern neural networks easily have millions if not billions of parameters. How a model predicts is thus still an unanswered question, at least, not fully. It is known if a model is trained mainly to chase accuracies with no constraints, it will inevitably discover ``shortcuts`` in the collected data that are useful for classification but not analysis~. %For example, many prior works have shown that neural network models may learn *spurious* correlation, looking at the wrong things (from humans' perspectives) but still making the correct predictions. 
We thus argue:

To make a model suitable for fine-grained analysis, one must constrain its capacity, while knowing that doing so would unavoidably hurt its classification accuracy.


PromptCAM is designed with this mindset. Unlike conventional classifiers that employ a fully connected layer on top, PromptCAM follows~ and learns a shared vector $w$ in~. The goal of $w$ is NOT to capture class-specific information BUT to answer a ``binary`` question: *Based on where a class-specific prompt attends, does the class recognize itself in the input image?*

To elucidate the difference, let us consider a *simplified* single-head-attention Transformer layer with no layer normalization, residual connection, MLP block, and other nonlinear operations. Let $V = \{v^1, \cdots, v^M\}\in\R^{D\times M}$ be the $M$ input patches' value features, $\alpha^c\in\R^M$ be the attention weights of class $c$, and $\alpha^\star\in\R^M$ be the attention weights of the [CLS] token. Conventional models predict classes by:
$$
\hat{y} = \argmax_{c} w_c^\top (\sum_j \alpha^\star[j] \times v^j)\nonumber \\
=  \argmax_{c} \sum_j \alpha^\star[j] \times (w_c^\top v^j),\tag{eq:standard}
$$
where $w_c$ stores the fully connected weights for class $c$. We argue that this formulation allows for a potential ``detour,`` enabling the model to correctly classify an image $I$ of class $y$ even without meaningful attention weights. In essence, the model can choose to produce holistically discriminative value features from $I$ without preserving spatial resolution, such that $v^j$ aligns with $w_y$ but $v^j = v^{j'}, \forall j\neq j'$. In this case, regardless of the specific values of  $\alpha^\star$, as long as they sum to one---as is default in the $\text{softmax}$ formulation---the prediction remains unaffected. 

In contrast, PromptCAM predicts classes by:
$$
\hat{y} = \argmax_{c} w^\top (\sum_j \alpha^c[j] \times v^j)\nonumber\\
= \argmax_{c} \sum_j \alpha^c[j] \times (w^\top v^j),\tag{eq:INTR}
$$
where $w$ is the shared binary classifier. (For brevity, we assume no self-attention among the prompts.) While the difference between  and  is subtle at first glance, it fundamentally changes the model's behavior. In essence, it becomes less effective to store class discriminative information in the channels of $v^j$, because there is no $w_c$ to align with. Moreover, the model can no longer produce holistic features with no spatial resolution; otherwise, it cannot distinguish among classes since all of their scores $s[c]$ will be exactly the same, no matter what $\alpha^c$ is. 

In response, the model must be equipped with two capabilities to minimize the cross-entropy error:

- Generate localized features $v^j$ that highlight discriminative patches (e.g., the red spot on the wing) of an image. 
- Generate distinctive attention weights $\alpha^c$ across classes, each focusing on traits frequently seen in class $c$.

These properties are what fine-grained analysis needs.

In sum, PromptCAM discourages patch features from encoding class-discriminative holistic information (e.g., the whole object shapes or mysterious long-distance pixel correlations), even if such information can be ``beneficial`` to a conventional classifier. To this end, PromptCAM needs to *distill* localized, trait-specific information from the pre-trained ViT's patch features, which is achieved through the injected class-agnostic prompts in PromptCAMD.

## Experiments

### Experimental Setup

**Dataset.**
We comprehensively evaluate the performance of PromptCAM on **13** diverse fine-grained image classification datasets across three domains:  **(1) animal-based**:  CUB-200-2011 (*CUB*)~, Birds-525 (*Bird*)~,  Stanford Dogs (*Dog*)~, Oxford Pet (*Pet*)~, iNaturalist-2021-Moths (*Moth*)~, Fish Vista (*Fish*)~, Rare Species (*RareS.*)~ and Insects-2 (*Insects*)~; **(2) plant and fungi-based**: iNaturalist-2021-Fungi (*Fungi*)~, Oxford Flowers (*Flower*)~ and Medicinal Leaf (*MedLeaf*)~; **(3) object-based**: Stanford Cars (*Car*)~ and Food 101 (*Food*)~. We provide details about data processing and statistics in  Suppl.


**Model.**
We consider three pre-trained ViT backbones, DINO~, DINOv2~, and BioCLIP~ across different scales including ViT-B (the main one we use) and ViT-S. 
The backbones are kept completely frozen when applying PromptCAM. We mainly used DINO, unless stated otherwise. More details can be found in Suppl.


**Baseline Methods.**
We compared PromptCAM with explainable methods like  Grad-CAM~,  Layer-CAM~ and Eigen-CAM~ as well as with interpretable methods like ProtoPFormer~, TesNet~, ProtoConcepts~ and INTR~. More details are in Suppl. 