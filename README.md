# A Review of the Role of Causality in Developing Trustworthy AI Systems – Datasets and Packages

<p align="center">

| ![](images/causal_ai_competition_resized.png)|
|:---:|
| Special thanks to our colleague **Alina Nekrasova** for designing this great figure!| 
</p>

This repository is a curated list of datasets used for recent Causal Machine Learning (ML) publications we covered in our [survey](https://arxiv.org/abs/2302.06975). It also includes an overview of useful causal and non-causal tools and packages to assess different characteristics of ML models (e.g., robustness or fairness) and for use in healthcare. We also provide slides that summarize the use of Causality for each discussed aspect of Trustworthy AI (see [here](Slides) for more information).

Feedback is welcome, so don't hesitate to reach out if you have any questions or comments. Simply send an e-mail to dren.fazlija@l3s.de

If you find this overview helpful, please cite our corresponding survey:

    @misc{Ganguly_Causality_2023,
          author = {Ganguly, Niloy and Fazlija, Dren and Badar, Maryam and Fisichella, Marco and Sikdar, Sandipan and Schrader, Johanna and Wallat, Jonas and Rudra, Koustav and Koubarakis, Manolis and Patro, Gourab K. and Zai El Amri, Wadhah and Nejdl, Wolfgang},
          title = {A Review of the Role of Causality in Developing Trustworthy AI Systems},
          month = feb,
          year = 2023,
          publisher = {arXiv},
          doi = {10.48550/ARXIV.2302.06975},
          url = {https://arxiv.org/abs/2302.06975},
    }

    

## Introduction

As a result of our literature review on causality-based solutions for Trustworthy AI, a need for an extensive overview of relevant datasets and packages was observed. To make causal machine learning (ML) more accessible and to facilitate comparisons to non-causal methods, we created a curated list of datasets used for recent Causal ML publications. This appendix also includes an overview of useful causal and non-causal tools and packages to assess different trustworthy aspects of ML models (interpretability, fairness, robustness, privacy, safety and accountability). We also provide a similar overview for the healthcare domain. Each aspect has its dedicated section that is structured as follows:

1.  An overview of **publicly available real-world datasets** used in
    cited publications of our main survey

2.  Some **benchmarks and packages for Causal Machine Learning** that
    researchers could utilize
3.  A number of **well-established tools**, that allow for a better
    comparison to non-causal machine learning


We want to clarify that this curated list does not (and cannot) aim for
completeness. Instead, we want to provide researchers interested in
working on a selection of aspects of Trustworthy AI with a concise
overview of exciting avenues for experimenting with causal machine
learning. We highly encourage readers interested to seek additional related reading material, such as Chapter 9
of [[2](#2)] or the two Github repositories for
datasets[^1] and algorithms[^2] resulting from [[1](#1)].

## Aspects of Trustworthy AI and Application Domain


### [Interpretability](Interpretability/README.md)

### [Fairness](Fairness/README.md)

### [Robustness](Robustness/README.md)

### [Privacy](Privacy/README.md)

### [Auditing (Safety and Accountability)](Auditing/README.md)

### [Healthcare](Healthcare/README.md)


<div id="refs" class="references hanging-indent">

## References

<div id="ref-guo2020survey">

<a id="1">[1]</a> Guo, Ruocheng, Lu Cheng, Jundong Li, P Richard Hahn, and Huan Liu. 2020.
“A Survey of Learning Causality with Data: Problems and Methods.” *ACM
Computing Surveys (CSUR)* 53 (4): 1–37.

</div>

<div id="ref-kaddour_2022_causalmlsurvey">

<a id="2">[2]</a> Kaddour, Jean, Aengus Lynch, Qi Liu, Matt J Kusner, and Ricardo Silva.
2022. “Causal Machine Learning: A Survey and Open Problems.” *arXiv
Preprint arXiv:2206.15475*.

## Acknowledgements
This work has received funding from the European Union's Horizon 2020 research and innovation program under Marie Sklodowska-Curie Action "NoBIAS - Artificial Intelligence without Bias" (grant agreement number 860630) and Network of Excellence "TAILOR - A Network for Trustworthy Artificial Intelligence" (grant agreement number 952215), the Lower Saxony Ministry of Science and Culture under grant number ZN3492 within the Lower Saxony "Vorab "of the Volkswagen Foundation and supported by the Center for Digital Innovations (ZDIN), and the Federal Ministry of Education and Research (BMBF), Germany under the project "LeibnizKILabor" with grant No. 01DD20003.

<p align="center">

| NoBIAS  | TAILOR  | Leibniz AILab |
|:---:|:---:|:---:|
|[![alt text](images/nobias.jpg "NoBIAS – Artificial Intelligence without Bias")](https://nobias-project.eu/)   |  [![alt text](images/tailor.jpg "Developing the scientific foundations for Trustworthy AI through the integration of learning, optimisation and reasoning")](https://tailor-network.eu/) |  [![alt text](images/leibnizailab.png "Artificial intelligence – networked worldwide")](https://leibniz-ai-lab.de/) |

|  Future Lab "Work and Society" | L3S Research Center |
|:---:|:---:|
| [![alt text](images/zdin_zlga.jpg "Reliable and responsible approaches of artificial intelligence to support people in their daily work and to shape the working environment of the future.")](https://zdin.de/zukunftslabore/gesellschaft-arbeit) |  [![alt text](images/l3s.jpg "Trustworthy AI & Digital Transformation")](https://www.l3s.de) |

</p>

[^1]: <https://github.com/rguo12/awesome-causality-data>

[^2]: <https://github.com/rguo12/awesome-causality-algorithms>
