# Machine Moral Uncertainty

This repository contains code and data to reproduce the experiments from:

> **Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment**  
> Jea Kwon, Luiz Felipe Vecchietti, Sungwon Park, Meeyoung Cha  
> Max Planck Institute for Security and Privacy (MPI-SP) & KAIST  
> Accepted to AAAI 2026 AI alignment special track (AAAI 2026 AIA)

We study how **uncertainty** and **overconfidence** in large language models (LLMs) affect their behavior in **moral dilemmas**, using the classical trolley-style scenarios from the **Moral Machine** framework.  
The repo provides tools to:

- Generate trolley problem scenarios across **9 moral dimensions**
- Query **open-source LLMs** and extract **binary decisions** (“Case 1” vs “Case 2”)
- Compute uncertainty metrics (confidence, binary entropy, **total entropy**, **conditional entropy**, **mutual information**)
- Inject **attention dropout at inference time** and measure its effect on:
  - Model uncertainty
  - **Human–LLM moral alignment** (via AMCE-based scores / L2 distance)

---

## 📚 Citation

If you use this code or datasets in your work, please cite:

```bibtex
@inproceedings{kwon2026dropouts,
  title     = {Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment},
  author    = {Kwon, Jea and Vecchietti, Luiz Felipe and Park, Sungwon and Cha, Meeyoung},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

You may also want to cite the original Moral Machine experiment and the Moral Machine LLM framework:

```bibtex
@article{awad2018moralmachine,
  title   = {The Moral Machine Experiment},
  author  = {Awad, Edmond and Dsouza, Sohan and Kim, Richard and Schulz, Jonathan and Henrich, Joseph and Shariff, Azim and Bonnefon, Jean-Fran{\c{c}}ois and Rahwan, Iyad},
  journal = {Nature},
  volume  = {563},
  number  = {7729},
  pages   = {59--64},
  year    = {2018}
}

@article{takemoto2024moralmachineLLM,
  title   = {The Moral Machine Experiment on Large Language Models},
  author  = {Takemoto, Kazuya},
  journal = {Royal Society Open Science},
  year    = {2024}
}
```

---
