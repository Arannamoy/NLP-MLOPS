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