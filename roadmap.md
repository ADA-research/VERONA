# VERONA Roadmap  

This roadmap outlines the planned development and priorities for VERONA.  

## 1. Vision  

*VERONA aims to become the standard framework for scalable and reproducible neural network robustness experiments.*  

## 2. Near-Term Goals (Sept 2025 – Jan 2026)  
- **Roadmap meeting** – Finalise this draft in an open meeting with all council members.  
- **Certified defense with randomized smoothing** – Support for certified defense (yielding statistical robustness certificates) methods for image classifiers.  
- **Vehicle integration for local robustness.**  

## 3. Mid-Term Goals (2026)  
- **Vehicle integration for complex properties** – Extend to properties with tree structures requiring GPU communication in parallel execution. Add an estimator for this.  
- **Reduce dependency on AutoVerify.**
- Building dedicated interfaces to support adv attacks from [foolbox](https://github.com/bethgelab/foolbox) and [adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch) 
- **Support for tree-based models** – Add support for decision trees and random forests (based on Marie’s bachelor project + student work).  
- **Docker support** – Provide Docker images for reproducibility and paper-specific setups.  

## 4. Long-Term Goals  
- **Maintain AutoVerify / Create lean version.**  
- **Platform independence for AutoVerify** – Currently Linux-only.  
- **Benchmarking** – Use VERONA as the benchmarking tool for VNN-COMP.  
- **Model card integration** – Add robustness distributions to Hugging Face model cards.  

## 5. Community & Contributions  
- [TODO] Add links to PRs open for contributions.  

## 6. Updates  
- PyPI release: `ada_verona` v1.0.3 on [17-11-2025] ([[pypi](https://pypi.org/project/ada-verona/)])  
- First draft of roadmap released on [DATE].  
