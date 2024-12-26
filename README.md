## Adaptive meta-path selection based heterogeneous spatial enhancement for circRNA-disease associations prediction

### Abstract

Circular RNAs (circRNAs) play a crucial role in human biological processes as miRNA sponges, regulating gene expression and affecting disease manifestations. Establishing heterogeneous nodal feature relationships through meta-paths can effectively enhance the predictive capability of models. Previous research primarily constructed associations between meta-paths manually, and excessive noise made it difficult to capture highly correlated hidden features. Equally important is learning more about feature distributions, which is key to improving the generalization ability of algorithms. To address these challenges, we propose an adaptive meta-path selection method named AdaMH. The core scheme introduces an adaptive path selection method that automatically identifies highly relevant heterogeneous meta-paths during iterative training rounds. Considering the sparsity of data distribution, we introduce controlled random noise into the data through graph contrastive learning to ensure an even distribution of features. Subsequently, a multi-head attention mechanism is utilized to capture relationships in the high-dimensional heterogeneous feature space, enhancing feature representation capability. Comparing with state-of-the-art (SOTA) algorithms, AdaMH is the only one that surpasses a performance threshold of 0.95 across seven evaluation metrics.

![adahpc v5](figures\AdaMH.jpg)

### contributions

- We propose an adaptive meta-path selection tactics, which trains an adaptive path selection matrix based on the historical loss computation of the reward $R$, and subsequently selects the meta-paths automatically, ensuring a high degree of correlation of information between the fused meta-paths.
- We propose a heterogeneous space feature enhancement method based on graph contrastive learning, i.e., adding controllable tiny random noise to each embedded node to enhance the feature capturing ability of the model, and then utilizing the multi-head attention mechanism to capture feature distributions in multiple feature spaces to achieve node representation-level enhancement in heterogeneous space.

- We demonstrated the effectiveness of the components of AdaMH through ablation studies. In addition, through case studies, we identified some valuable circRNAs that can provide scientific guidelines for wet experiments.

### Running steps

```
python main.py
```

### Others

**If you have any questions, please submit your issues.**