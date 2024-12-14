# CAWF + Memory Web — A Proof of Concept

**Repository:** Resonant-Intelligence/P.O.C  
**File:** `CAWF + Memory Web.py`

---

## Overview

This script provides a **Proof of Concept** for the **Cognitive Amplitude Wave Function (CAWF)**—a wave‐inspired approach focusing exclusively on *cognitive* facets (omitting the ethical dimension). It integrates:

1. **CAWFCore:**  
   A class that models cognition as a superposition of wave facets. Each facet evolves over time with adaptive parameters (wavevectors `k`, frequencies `ω`, phases `φ`).

2. **Memory Web:**  
   A NetworkX‐based graph whose nodes store “cognitive activation.” When the wavefunction is computed, the *mean amplitude* updates these activations, providing a unique interpretability layer.

3. **scikit‐learn Integration:**  
   - `CAWFClassifier` is a scikit‐learn‐style wrapper enabling cross-validation, holdout testing, and direct comparison to baseline ML models (RandomForest, SVM, etc.).  

---

## Key Features

1. **Pure Cognitive Wave Function**  
   - **No** ethical dimension—just wave amplitude and phase representing the system’s cognitive state.  
   - An **adaptive rate** randomly perturbs wave parameters each iteration, imitating an evolving “cognitive dynamic.”

2. **Separate Memories for Classification vs. Visualization**  
   - **Classification Memory:** Accumulates single‐sample wave outputs for a memory effect that influences subsequent classifications.  
   - **Visualization Memory (optional):** Maintains consistent shape when you do large 2D wave plots.

3. **Visualization**  
   - **3D Plot** of \(\lvert \Psi \rvert\) over `(cognitive_dim1, cognitive_dim2)` if you use 2D cognitive input.  
   - **Memory Web** graph showing node “cognitive_activation” levels, re-drawn each simulation step.

4. **Proof of Concept**  
   - Demonstrates feasibility rather than high accuracy. On small synthetic data, cross-validation might yield modest results.  
   - The approach’s real strengths lie in interpretability (Memory Web) and the potential for emergent phenomena.

---

## Usage

1. **Dependencies**  
   - Python 3.7+, NumPy, Matplotlib, NetworkX, scikit-learn, Numba (optional for speed).

2. **Running**  
   - Clone this repository and open **`CAWF + Memory Web.py`** in a Jupyter environment.  
   - Run the script. You’ll see a cross-validation accuracy printout and a 3D wave plot plus a Memory Web visualization.

3. **Basic Steps**  
   - `CAWFClassifier.fit(X, y)` trains the wave-based classifier.  
   - `predict()` uses wave amplitude thresholds to produce a label.  
   - `update_memory_web()` modifies node activations based on wavefunction amplitude, providing an interpretability angle.

---

## Results & Next Steps

- **Initial performance** on toy data might vary (e.g. ~50–70% accuracy).  
- For deeper validation:
  1. Apply to more complex or real datasets.  
  2. Tweak parameters (e.g., `num_facets`, `adaptive_rate`) to explore emergent behaviors.  
  3. Compare CAWF to ECWF and UniWave frameworks in parallel experiments.

---

## License

MIT License

Copyright (c) 2024 captainkoopa420

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests if you’d like to refine the wave logic, memory web interpretability, or classification approach.

---

**Author / Contact**  
**William Robert Adams**  
*Part of the “Resonant Intelligence” initiative—pioneering wave‐based AI frameworks that integrate cognition, ethics, and emergent phenomena.*
