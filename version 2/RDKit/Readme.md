# Drug Discovery Environment

## Core Environment

| Package | Version |
|----------|----------|
| Python | 3.10.20 |
| PyTorch | 2.1.2 |
| RDKit | 2026.3.2 |
| TorchDrug | 0.2.1 |
| DeepChem | 2.8.0 |
| PyTorch Geometric | 2.5.3 |

---

## Purpose

This environment is configured for:

- Cheminformatics
- Molecular Property Prediction
- QSAR Modeling
- Molecular Graph Neural Networks (GNNs)
- Drug–Target Interaction (DTI)
- Drug Discovery Research
- Oncology AI Research

---

## Core Stack

```text
SMILES
   ↓
RDKit
   ↓
Descriptors / Fingerprints
   ↓
PyTorch Geometric
   ↓
Graph Neural Networks
   ↓
Drug Discovery
```

---

## Main Libraries

### RDKit

Used for:

- SMILES parsing
- SMARTS queries
- Molecular descriptors
- Molecular fingerprints
- Molecular visualization

### PyTorch

Used for:

- Deep Learning
- Neural Network Training
- GPU Acceleration

### PyTorch Geometric

Used for:

- Molecular Graph Construction
- Graph Neural Networks
- GCN
- GAT
- GraphSAGE
- Message Passing Networks

### TorchDrug

Used for:

- Drug Discovery Pipelines
- Molecular Graph Learning
- Protein Modeling
- Drug–Target Interaction (DTI)

### DeepChem

Used for:

- Molecular Featurization
- Benchmark Datasets
- QSAR
- Drug Discovery Workflows

---

## Compatibility

Tested Environment:

```text
Python 3.10.20
PyTorch 2.1.2
RDKit 2026.3.2
TorchDrug 0.2.1
DeepChem 2.8.0
PyTorch Geometric 2.5.3
```

## One-Shot Installation (Existing Environment)

```bash
uv pip install torch==2.1.2 torch-geometric==2.5.3 rdkit==2026.3.2 torchdrug==0.2.1 deepchem==2.8.0
```

# SMILES vs SMARTS

| Feature | SMILES | SMARTS |
|----------|----------|----------|
| Full Form | Simplified Molecular Input Line Entry System | SMILES Arbitrary Target Specification |
| Purpose | Represent a specific molecule | Define a molecular search pattern |
| Represents | Exact structure | Query / Pattern |
| Used For | Storing molecules | Substructure searching |
| Example | `CCO` | `[#6]-[#6]-[#8]` |
| Meaning | Ethanol | Any Carbon-Carbon-Oxygen pattern |
| Unique Molecule? | Yes | Not necessarily |
| RDKit Parser | `MolFromSmiles()` | `MolFromSmarts()` |
| Output Type | Molecule (Mol) | Query Molecule (QueryMol) |
| Question It Answers | "What molecule is this?" | "Does this molecule match this pattern?" |
| Analogy | Person's ID card | Police search criteria |
