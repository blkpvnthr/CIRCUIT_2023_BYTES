# Johns Hopkins University Applied Physics Laboratory 
## 2023 CIRCUIT Undergraduate Research Internship BYTES Challenge


This project contains code for performing data analysis and machine learning tasks, including data preprocessing, visualization, classification, regression, and clustering. It uses Python and several popular libraries such as **pandas**, **numpy**, **matplotlib**, and **scikit-learn**.

---

## What's New

- **Policy Data Handling**: New workflow called `main.py` streamlines the different `.py` files that comprise this project and compute correlations.
- **Population & Deaths Merge**: Code now supports reading population and total deaths (e.g., from `pop.csv` or `deaths.csv`) to attach to policy data for further analysis.
- **Correlation with `% of Population Dead`**: Introduces `pct_of_pop_dead` to see how policy timings correlate with mortality rates.
- **Multi-Index Melt**: Optionally melts policy date offset columns so each row is `(POSTCODE, PolicyType, EffectiveOffset, …)`, facilitating advanced analyses.

---

## Getting Started

To run this project on your local machine, follow the instructions below.

### Prerequisites
<a href="https://www.python.org/downloads/" target="_blank">Python>=3.7</a> 

### Installation

1. **Clone** this repository to your local machine or download the source code as a ZIP file.
2. **Open a terminal** (e.g., Command Prompt, bash shell) and navigate to the project directory.
3. **Install project dependencies** using:
   ```bash
   pip install -r requirements.txt
   ```
This will install the required Python packages.


### Usage
  ```bash
python main.py
```
The code will perform data preprocessing, visualization, classification, regression, and clustering based on the provided functions and print the results to the terminal. You may customize the code and functions according to your specific requirements.


### Contributing
Contributions are welcome! If you find issues or want to add new features, open an issue or submit a pull request.

### Acknowledgments
This project was inspired by the need for a code template for data analysis and machine learning tasks.
Thanks to the creators and maintainers of the pandas, numpy, matplotlib, and scikit-learn libraries for providing powerful tools for data manipulation and analysis.
