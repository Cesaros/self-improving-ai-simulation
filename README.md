# Self-Improving AI Simulation

An educational toy model of recursive AI self-improvement using an
epsilon-greedy multi-armed bandit.

The simulation explores a positive-feedback loop:

> better AI → better research → better strategy selection → better AI

The resulting trajectory is deliberately stochastic. Improvement can include
flat periods, occasional jumps, changing research priorities, and diminishing
returns rather than smooth exponential growth at every generation.

## View the executed notebook

**[Open the complete Jupyter notebook](./self_improving_ai_5_strategies.ipynb)**

The notebook has already been executed and includes:

- The full mathematical model and Python implementation
- Summary statistics and the first 15 simulated generations
- Nine embedded plots with concise interpretations
- A general summary of the results
- Academic references and a human–AI authorship disclosure

## Model overview

The model gives an AI system a normalized capability level `A` between 0 and
1. In every generation, it selects one of five research strategies:

1. Prompt and scaffold improvement
2. Search and planning improvement
3. Synthetic data generation
4. Fine-tuning
5. Inference optimization

The agent initially does not know which strategy is most productive. It learns
from observed capability improvements with epsilon-greedy action selection and
a constant-step-size Q-value update.

Research success becomes more likely as capability increases. Successful
experiments improve capability according to a diminishing-returns function,
while failed experiments produce no improvement.

This is an explanatory simulation, not an empirical model or forecast of real
AI development.

## Repository contents

| File | Purpose |
| --- | --- |
| `self_improving_ai_5_strategies.ipynb` | Executed and documented Jupyter notebook |
| `self_improving_ai_5_strategies.py` | Standalone Python version of the simulation |
| `requirements.txt` | Python dependencies used by the project |
| `CITATION.cff` | Citation metadata for this repository |
| `LICENSE` | MIT license |

## Run locally

Python 3.11 or a compatible recent Python version is recommended.

```bash
git clone https://github.com/Cesaros/self-improving-ai-simulation.git
cd self-improving-ai-simulation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python self_improving_ai_5_strategies.py
```

To run the notebook, select `.venv/bin/python` as the Jupyter kernel in VS Code
or another Jupyter-compatible editor, then choose **Run All**.

The script and notebook generate:

- `self_improving_ai_results.csv`
- `self_improving_ai_q_values.csv`

## Authorship and AI assistance

This project was designed and created by **Cesar Osorio with AI assistance**.
Cesar specified the educational objective, model scope, strategy structure,
desired complexity, presentation style, interpretations, and revisions. An AI
assistant supported the Python implementation, notebook organization,
explanatory writing, execution, and presentation.

The project is an original educational synthesis built from standard methods
and cited research. Responsibility for reviewing, publishing, and maintaining
the final project remains with the human creator.

## Intellectual foundations

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction* (2nd ed.), Chapter 2. MIT Press.
- Srikanth, D., Zhao, B., Xu, D., Wu, Y., & Jiang, Z. (2026). “Recursive
  self-improvement of AI research agents.” arXiv:2609.26457.
- Chan, A. et al. (2026). “What if automating AI R&D triggers an intelligence
  explosion?” Cambridge Programme on AI Science & Policy.

The full citations, links, and an explanation of how this simulation differs
from AIDE² appear at the end of the notebook.

## License

The repository is available under the [MIT License](./LICENSE).

