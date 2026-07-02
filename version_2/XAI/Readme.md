####
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAPE (SHapley Additive exPlanations)
- Counterfactual Explanations/contrastive explanations
  - Rashomon Effect
  - DiCE (Diverse Counterfactual Explanations)
- Layer-wise Relevance Propagation (LRP): which input is more important for output. 
  - Relevance Score: Last result of output based on neuron contribution.
  - Pixelwise Decomposition
  * - Rules of LRP
    - Upper layers: LRP-0 (Basic rule)
    - Middle layers: LRP- $\epsilon$ 
    - Lower layers: LRP- $\gamma$

- How to explain Graph Neural Networks (with XAI)?
  - Gradient-based: Use Grad CAM to find importance of input using back propagation
  - Perturbation: GNNExplainer, SubgraphX work based on Perturbation
  - Decomposition: Layerwise impact of input. Example: GNN-LRP
  - Surrogates: Example: GraphLIME
  - Explanation of model level: XGNN

- How to works GNNExplainer?
  - Computation graph
  - Counterfactual approach
  - Mutual information and masking
  - Result
  
