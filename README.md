# Lattice Visualizer

A physics-based visualizer for atomic lattice vibrations. This interactive simulation allows users to explore how atoms behave in a connected lattice structure under forces defined by Hooke's Law.

[**▶️ Click here to launch the simulation on GlowScript.org**](https://www.glowscript.org/#/user/X9Z3/folder/X9Z3Publications/program/Molecular-Lattice)


![image](https://github.com/user-attachments/assets/6ac8e7dd-330b-43bf-b741-d14c40227bd1)



## 🧪 About the Simulation

In the scene, a model simulates vibrations in a lattice of atoms. Several pre-designed lattice structures are available to choose from. Users can configure the number of simulated atoms—though higher node counts may impact performance.

If performance is a concern, the "Render: Static" option can be enabled to visualize thousands of atoms without real-time physics, ideal for exploring complex lattice configurations. Any changes to the simulation parameters require a reload: press the **Reload** button and left-click anywhere in the scene to restart.

### 💡 Key Features

- **Interactive Atoms**: Click and drag individual atoms to perturb the system and watch it respond in real time.
- **Cooling Mechanism**: Reduce system energy using the "energy conservation" slider to stabilize chaotic motion.
- **Lattice Generation**: 
  - Uses **Breadth-First Search (BFS)** to form dense, clustered structures by default.
  - Optionally switch to **Depth-First Search (DFS)** to generate long, tendril-like chains.
  - A **randomness slider** modifies the visitation order of lattice directions for more organic networks.
- **Camera Controls**:
  - Scroll to zoom
  - Right-click + drag to rotate
  - Shift + left-click + drag to pan

### ⚙️ Physics Engine

The model uses **Hooke’s Law** to compute forces based on displacement between atoms. Newton’s Second Law is applied via a **forward Euler integration scheme** to update atom positions over time. To enhance realism and stability:

- Opposing force vectors are subtracted to prevent torque-induced motion.
- The system maintains stiffness by suppressing whole-network rotation effects.

These approximations balance performance and physical accuracy, enabling smoother simulations of large systems.

## 🔧 Built With

- [**GlowScript VPython**](http://www.glowscript.org): A browser-based 3D visualization tool powered by WebGL. It allows for interactive physics simulations using Python-like syntax with minimal setup.

## 🙋‍♂️ Contributing

I welcome feedback, collaboration, or ideas for new lattice types. The current algorithm is designed to work with symmetric tileable lists of vectors shared across all nodes, which makes certain structures like Einstein solids tricky to implement. If you have suggestions or would like to help, please reach out!

---

**Created by Maximillian DeMarr**
