# Code Distillation: From Black-Box to White-Box

> Reverse-engineering neural networks into human-readable Python code using LLM Agents.

## The Idea

**Core Insight**: Intelligence is compression. If a neural network has learned true patterns from data, those patterns should be expressible as simple, human-readable code.

This is inspired by:
- **Kolmogorov Complexity**: The shortest program that produces an output captures its essence
- **Demoscene**: Programmers create stunning 3D worlds in 4KB of code by encoding generative rules
- **Scientific Laws**: F=ma is more powerful than a lookup table of force values

### The Problem

Modern ML models (especially in biology) are black boxes:
- CellTypist: Predicts cell types from gene expression
- scVI: Learns latent representations of single-cell data
- scGPT: Foundation model for single-cell biology

These models work, but we don't understand *why*. We want to:
1. Extract the "rules" the model learned
2. Express them as readable Python code
3. Validate they capture the model's behavior

### The Solution

Use an LLM Agent as an "AI Scientist" that:
1. **Observes**: Loads the model, analyzes weights
2. **Hypothesizes**: "If CD3D > 1.5, then T cell"
3. **Experiments**: Knockout genes, sweep thresholds
4. **Synthesizes**: Converts validated rules to code
5. **Evaluates**: Tests fidelity, iterates if needed

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Code Distillation Agent                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│   │   Observe    │ -> │  Hypothesize │ -> │   Experiment │ │
│   │              │    │              │    │              │ │
│   │ Load model   │    │ "IF gene_A   │    │ Knockout     │ │
│   │ Get weights  │    │  > threshold │    │ Sweep        │ │
│   │ Find top     │    │  THEN class" │    │ Synthetic    │ │
│   │ features     │    │              │    │ samples      │ │
│   └──────────────┘    └──────────────┘    └──────────────┘ │
│          │                                       │          │
│          │         ┌──────────────┐              │          │
│          │         │   Evaluate   │              │          │
│          │         │              │              │          │
│          └─────────│ Fidelity     │<─────────────┘          │
│                    │ Complexity   │                          │
│                    │ Iterate?     │                          │
│                    └──────┬───────┘                          │
│                           │                                  │
│                    ┌──────▼───────┐                          │
│                    │  Synthesize  │                          │
│                    │              │                          │
│                    │ Generate     │                          │
│                    │ Python code  │                          │
│                    └──────────────┘                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install celltypist

# Run distillation
cd examples/code_distillation
python run.py
```

## Expected Output

The agent will generate code like:

```python
def predict_cell_type(expression: dict) -> str:
    """
    Distilled classifier for cell type prediction.
    Fidelity: 95.2% agreement with CellTypist
    Rules: 15, Features: 23
    """
    # T cells: CD3D is the master regulator
    if expression.get('CD3D', 0) > 1.5:
        # CD8+ cytotoxic T cells
        if expression.get('CD8A', 0) > 1.0:
            return 'CD8+ T cell'
        # CD4+ helper T cells
        if expression.get('CD4', 0) > 0.8:
            # Regulatory T cells have high FOXP3
            if expression.get('FOXP3', 0) > 1.2:
                return 'Treg'
            return 'CD4+ T cell'
        return 'T cell'

    # B cells: CD79A/B and MS4A1 (CD20)
    if expression.get('CD79A', 0) > 1.5 or expression.get('MS4A1', 0) > 1.5:
        return 'B cell'

    # Monocytes: CD14
    if expression.get('CD14', 0) > 2.0:
        return 'CD14+ Monocyte'

    return 'Unknown'
```

## Paper Planning

### Title Options

1. **Technical**: "Automated Program Synthesis for Neural Network Interpretability via Agentic Hypothesis Testing"
2. **Conceptual**: "Distilling Black Boxes into Executable Laws: An AI Scientist Approach to Model Interpretability"
3. **Applied**: "Extracting Human-Readable Gene Regulatory Rules from Single-Cell Models using LLM Agents"

### Target Venues

- Nature Methods (methodology + biological application)
- Nature Machine Intelligence (AI methods)
- Cell Systems (systems biology + computation)

### Key Contributions

1. **Framework**: Agentic approach to model interpretability
2. **Method**: Hypothesis-experiment-synthesis loop
3. **Metrics**: Fidelity-complexity trade-off (Pareto frontier)
4. **Application**: CellTypist distillation with biological validation

### Figure Planning

1. **Conceptual diagram**: Black box → Agent loop → White box code
2. **CellTypist case study**: Weights → Experiments → Rules
3. **Pareto frontier**: Code complexity vs. fidelity
4. **Biological validation**: Distilled rules match known markers

## Theoretical Foundation

### Kolmogorov Complexity

The Kolmogorov complexity K(x) of a string x is the length of the shortest program that produces x.

For a trained model M:
- K(M) = millions of parameters (complex)
- K(rules) = few lines of code (simple)

If K(rules) << K(M) but rules ≈ M (high fidelity), we've found the model's "essence".

### Minimum Description Length

Our optimization objective:

```
minimize: Error(code, model) + λ * Length(code)
```

Where:
- Error = 1 - Fidelity (disagreement rate)
- Length = Code complexity (lines, conditions, variables)
- λ = Trade-off parameter

### Connection to Scientific Discovery

This is the scientific method applied to neural networks:
1. Observation → Feature importance analysis
2. Hypothesis → Rule proposal
3. Experiment → Perturbation testing
4. Theory → Code synthesis
5. Validation → Fidelity measurement

## Directory Structure

```
code_distillation/
├── README.md                    # This file
├── run.py                       # Entry point
├── .pantheon/
│   ├── teams/
│   │   └── distillation.md     # Team definition
│   ├── agents/
│   │   └── distiller.md        # Agent definition
│   └── skills/
│       ├── SKILL.md            # Skill index
│       ├── model_inspection.md # Loading and analyzing models
│       ├── perturbation.md     # Experiment design
│       ├── code_synthesis.md   # Rule → code conversion
│       └── evaluation.md       # Fidelity and complexity
├── data/                        # Test datasets
└── results/                     # Output directory
```

## References

1. OpenAI (2023). "Language models can explain neurons in language models"
2. Anthropic (2023). "Sparse Autoencoders Find Highly Interpretable Features"
3. Anthropic (2024). "Mapping the Mind of a Large Language Model"
4. Sakana AI (2024). "The AI Scientist: Towards Fully Automated Scientific Discovery"
5. CellTypist: https://www.celltypist.org/
