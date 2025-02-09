Introduction
Topology optimization for heat transfer aims to determine the optimal distribution of materials to minimize thermal resistance or achieve desired temperature distributions under given boundary conditions. This project employs neural networks to model the design domain, using a Fourier-mapped feature space to ensure smooth and manufacturable results.

How It Works:
Neural Networks Represent the Density Field

Instead of traditional element-wise optimization, a deep neural network (DNN) learns to represent the material distribution.
Finite Element Analysis (FEA) for Heat Transfer

The framework employs a JAX-based finite element solver to evaluate thermal performance.
Optimization via Backpropagation

The network is trained using a physics-based loss function, incorporating thermal conductivity interpolation and heat conduction constraints.
Features
Neural network-based material representation for topology optimization.
JAX-accelerated finite element solver (FEA) for heat transfer analysis.
Fourier-mapped input transformations for enhanced feature learning.
Support for symmetry, extrusion, and rotational constraints to enforce manufacturability.
SIMP and RAMP interpolation schemes for thermal conductivity.
Efficient gradient-based optimization using Adam and constraint-handling techniques.
Installation
To set up the project locally:

Clone the repository:

bash
Copy
Edit
git clone https://github.com/ULRA92/TOuNN-works.git
cd TOuNN-works
Set up a virtual environment (optional but recommended):

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Note: Ensure you have Python 3.7 or later installed.

Usage
Running Heat Transfer Topology Optimization
Prepare the configuration file (config.txt)

Define mesh settings, material properties, Fourier mapping settings, and optimization parameters.
Run the main script:

bash
Copy
Edit
python main_TOuNN.py
This will initialize the optimization process using the provided settings.
View the optimized topology:

Results will be saved as PDF plots and temperature distribution visualizations.
Project Structure
bash
Copy
Edit
TOuNN-works/
│── main_TOuNN.py            # Main script to run heat transfer topology optimization
│── TOuNN.py                 # Core class integrating neural networks with FEA
│── network.py                # Neural network architecture for topology optimization
│── FE_Solver.py              # Finite element solver for heat conduction analysis
│── projections.py            # Symmetry, rotational constraints, Fourier mapping
│── materialCoeffs.py         # Material properties and thermal conductivity models
│── requirements.txt          # Python dependencies
│── config.txt                # Configuration settings for optimization
│── docs/                     # Documentation and additional resources
│── results/                  # Output directory for optimized designs

Contributing
We welcome contributions to improve the framework. To contribute:

Fork the repository.
Create a new branch for your feature or bug fix.
Implement and test your changes.
Submit a pull request with a clear description.
For more details, see CONTRIBUTING.md.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
This project builds upon:

TOuNN: Aaditya Chandrasekhar and Krishnan Suresh, "TOuNN: Topology Optimization using Neural Networks", University of Wisconsin-Madison.
JAX: High-performance machine learning framework for differentiable programming.
Finite Element Methods (FEM) applied to heat transfer simulations.
For the original TOuNN paper and repository, visit:
UW-ERSL/TOuNN

