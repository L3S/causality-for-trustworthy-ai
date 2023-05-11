# Healthcare

## Aspects of Trustworthy AI and Application Domain

### [Interpretability](../Interpretability/README.md)

### [Fairness](../Fairness/README.md)

### [Robustness](../Robustness/README.md)

### [Privacy](../Privacy/README.md)

### [Auditing (Safety and Accountability)](../Auditing/README.md)

### Healthcare

<details>
<summary><h2>Datasets Used by Cited Publications (Click to expand)</h2></summary>

  - **[Alzheimer’s Disease Neuroimaging Initiative
    (ADNI)](https://adni.loni.usc.edu/data-samples/access-data/)** \[[15](#ref-petersen2010alzheimer)\]:
    A medical dataset containing multi-modal information of over 5,000
    volunteering subjects. ADNI includes clinical and genomic data,
    biospecimens, MRI, and PET images. Researchers need to apply for
    data access. <span>&#10230;</span> **Used
    by**: \[[20](#ref-Sanchez2022),[21](#ref-shen2020)\]

  - **[SARS-CoV-2 infected cells (Series
    GSE147507)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE147507)** \[[5](#ref-blanco2020imbalanced)\]:
    A genomic dataset that contains RNA-seq data from
    SARS-CoV-2-infected cells of both humans and ferrets. The data are
    publicly available on the NCBI Gene Expression Omnibus (GEO) server
    under the accession number GSE147507. <span>&#10230;</span> **Used
    by**: \[[4](#ref-belyaeva2021)\]

  - **[The Genotype-Tissue Expression (GTEx)
    project](https://gtexportal.org/home/)** \[[7](#ref-carithers2015novel)\]:
    An online medical platform that provides researchers with tissue
    data. Data samples stem from 54 non-diseased tissue sites across
    nearly 1000 individuals whose genomes were processed via sequencing
    methods such as WGS, WES, and RNA-Seq. <span>&#10230;</span> **Used
    by**: \[[4](#ref-belyaeva2021)\]

  - **[L1000 Connectivity Map (Series
    GSE92742)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)** \[[23](#ref-subramanian2017next)\]:
    A connectivity map (CMap) connects genes, drugs, and disease states
    based on their gene-expression signatures. The CMap provided
    by \[[23](#ref-subramanian2017next)\] includes over 1.3 million
    L1000 profiles of 25,200 unique perturbagens[^1]. The data are
    publicly available on the NCBI Gene Expression Omnibus (GEO) server
    under the accession number GSE92742.<span>&#10230;</span> **Used
    by**: \[[4](#ref-belyaeva2021)\]

  - **[iRefIndex](https://irefindex.vib.be/wiki/index.php/iRefIndex)** \[[17](#ref-razick2008irefindex)\]:
    This protein-protein interaction (PPI) network is a graph-based
    database of molecular interactions between proteins from over ten
    organisms. The current version of iRefIndex (version 19) contains
    over 1.6 million PPIs. <span>&#10230;</span> **Used
    by**: \[[4](#ref-belyaeva2021)\]

  - **[DrugCentral](https://drugcentral.org/)** \[[2](#ref-avram2021drugcentral)\]:
    An online platform that provides up-to-date drug information. Users
    can traverse the database online through the corresponding website
    or via an API. The platform currently contains information on almost
    5,000 active ingredients. <span>&#10230;</span> **Used
    by**: \[[4](#ref-belyaeva2021)\]

  - **[Colorectal Cancer Single-cell Data
    (GSE81861)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81861)** \[[11](#ref-li2017reference)\]:
    The authors provide two datasets. The first dataset contains 1,591
    single cells RNA-seq data from 11 colorectal cancer patients. The
    second dataset contains 630 single cells from seven cell lines and
    can be used to benchmark cell-type identification algorithms. The
    data are publicly available on the NCBI Gene Expression Omnibus
    (GEO) server under the accession number GSE81861.
    <span>&#10230;</span> **Used by**: \[[4](#ref-belyaeva2021)\]

  - **[Pulmonary Fibrosis Single-cell
    Data](https://www.nupulmonary.org/resources/)** \[[18](#ref-reyfman2019single)\]:
    This genomic dataset contains approximately 76,000 single-cell
    RNA-seq data from healthy lungs and lungs from patients with
    pulmonary fibrosis. The data are available online and comes with a
    cluster visualization based on marker gene expressions.
    <span>&#10230;</span> **Used by**: \[[4](#ref-belyaeva2021)\]

  - **[SARS-CoV-2 Host-Pathogen Interaction
    Map](https://www.ndexbio.org/#/network/5d97a04a-6fab-11ea-bfdc-0ac135e8bacf)** \[[10](#ref-gordon2020sars)\]:
    A PPI network that maps 27 SARS-CoV-2 proteins to human proteins
    through 332 high-confidence protein-protein interactions. The online
    data contain data from the initial study and the CORUM database
    \[[24](#ref-tsitsiridis2023corum)\]. <span>&#10230;</span> **Used
    by**: \[[4](#ref-belyaeva2021)\]

  - **[Lung Image Database Consortium image collection
    (LIDC-IDRI)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254)** \[[1](#ref-armato2011lung)\]:
    An image dataset comprising annotated thoracic CT Scans of more than
    1,000 cases. The data stem from seven academic centers and eight
    medical imagining companies. Four trained thoracic radiologists
    provided the image annotations. <span>&#10230;</span> **Used
    by**: \[[25](#ref-vanAmsterdam2019)\]

  - **[MEDLINE](https://www.nlm.nih.gov/medline/medline_overview.html)** \[[12](#ref-medline2023)\]:
    An online bibliographic database of more than 29 million article
    references from the field of life science (primarily in
    biomedicine). MEDLINE is a primary component of PubMed and is hosted
    and managed by the NLM National Center for Biotechnology Information
    (NCBI). <span>&#10230;</span> **Used by**: \[[27](#ref-ziff2015)\]

  - **[Cochrane Central Register of Controlled Trials
    (CENTRAL)](https://www.cochranelibrary.com/central)** \[[9](#ref-Cochrane2022)\]:
    A database of reports for randomized and quasi-randomized controlled
    trials collected from different online databases. Although it does
    not contain full-text articles, the CENTRAL includes bibliographic
    details and often an abstract of the report.
    <span>&#10230;</span> **Used by**: \[[27](#ref-ziff2015)\]

</details>

<details>
<summary><h2>Interesting Causal Tools (Click to expand)</h2></summary>

  - **[Causal
    Inference 360](https://github.com/BiomedSciAI/causallib)** \[[22](#ref-shimoni2019evaluation)\]:
    A Python package developed by IBM to infer causal effects from given
    data. Causal Inference 360 includes multiple estimation methods, a
    medical dataset, and multiple simulation sets. The provided methods
    can be used for any complex ML model through a scikit-learn-inspired
    API.

  - **[gCastle](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle)** \[[26](#ref-zhang2021gcastle)\]:
    An end-to-end causal structure learning toolbox that is equipped
    with 19 techniques for Causal Discovery. It also assists users in
    data generation and evaluating learned structures. Having a firm
    understanding of the causal structure is crucial for
    healthcare-related research.

  - **[Benchpress](https://github.com/felixleopoldo/benchpress)** \[[19](#ref-rios2021benchpress)\]:
    A benchmark for causal structure learning allowing users to compare
    their causal discovery methods with over 40 variations of
    state-of-the-art algorithms. The plethora of available techniques in
    this single tool could encourage more causality-based solutions for
    the healthcare domain.

  - **[CausalML](https://github.com/uber/causalml)** \[[8](#ref-Chen2020_CausalML_Uber)\]:
    The Python package enables users to analyze the Conditional Average
    Treatment Effect (CATE) or Individual Treatment Effect (ITE)
    observable in experimental data. The package includes tree-based
    algorithms, meta-learner algorithms, instrumental variable
    algorithms, and neural-network-based algorithms.

  - **[WhyNot](https://github.com/mrtzh/whynot)** \[[13](#ref-miller2020whynot)\]:
    A Python package that provides researchers with many simulation
    environments for analyzing causal inference and decision-making in a
    dynamic setting. It allows benchmarking of multiple decision-making
    systems on 13 different simulators. This set of simulators includes
    environments that simulate HIV treatment effects and system dynamics
    models of both the Zika epidemic and the US opioid epidemic.

</details>

<details>
<summary><h2>Prominent Non-Causal Tools (Click to expand)</h2></summary>

  - **[Medical Open Network for AI
    (MONAI)](https://github.com/Project-MONAI/MONAI)** \[[6](#ref-Cardoso_MONAI_An_open-source_2022)\]:
    A PyTorch-based framework that offers researchers pre-processing
    methods for medical imaging data, domain-specific implementations of
    machine learning architectures, and ready-to-use workflows for
    healthcare imaging. The actively maintained framework also provides
    APIs for integration into existing workflows.

  - **[DeepChem](https://github.com/deepchem/deepchem)** \[[16](#ref-Ramsundar-et-al-2019)\]:
    A Life Science toolbox that provides researchers with deep learning
    solutions for different fields of Life Science, such as Quantum
    Chemistry, Biology, or Drug Discovery (with the latter being
    particularly interesting for comparisons to causality-based
    solutions). Deepchem supports TensorFlow, PyTorch, and JAX and has
    an extensive collection of running examples.

  - **Curated Lists on
    Github** \[[3](#ref-Medical_Data_List_2016),[14](#ref-Awesome_Medical_List_2016)\]:
    \[[14](#ref-Awesome_Medical_List_2016)\] host an [up-to-date GitHub
    repository](https://github.com/kakoni/awesome-healthcare) of
    relevant open-source healthcare tools and resources.
    \[[3](#ref-Medical_Data_List_2016)\] provide an extensive [overview
    of valuable medical
    datasets](https://github.com/beamandrew/medical-data) that could be
    used to assess Causal ML healthcare solutions. Although this list
    has not been updated since 2020, we still believe it to be a helpful
    initial overview of relevant datasets.

</details>

<div id="refs" class="references">

## References

<div id="ref-armato2011lung">

\[1\] Samuel G Armato III, Geoffrey McLennan, Luc Bidaut, Michael F
McNitt-Gray, Charles R Meyer, Anthony P Reeves, Binsheng Zhao, Denise R
Aberle, Claudia I Henschke, Eric A Hoffman, and others. 2011. The lung
image database consortium (lidc) and image database resource initiative
(idri): A completed reference database of lung nodules on ct scans.
*Medical physics* 38, 2 (2011), 915–931.

</div>

<div id="ref-avram2021drugcentral">

\[2\] Sorin Avram, Cristian G Bologa, Jayme Holmes, Giovanni Bocci,
Thomas B Wilson, Dac-Trung Nguyen, Ramona Curpan, Liliana Halip, Alina
Bora, Jeremy J Yang, and others. 2021. DrugCentral 2021 supports drug
discovery and repositioning. *Nucleic acids research* 49, D1 (2021),
D1160–D1169.

</div>

<div id="ref-Medical_Data_List_2016">

\[3\] Andrew L. Beam and others. 2016. Medical Data for Machine
Learning. 

</div>

<div id="ref-belyaeva2021">

\[4\] Anastasiya Belyaeva, Louis Cammarata, Adityanarayanan
Radhakrishnan, Chandler Squires, Karren Dai Yang, G. V. Shivashankar,
and Caroline Uhler. 2021. Causal network models of sars-cov-2 expression
and aging to identify candidates for drug repurposing. *Nature
Communications* 12, 1 (2021).

</div>

<div id="ref-blanco2020imbalanced">

\[5\] Daniel Blanco-Melo, Benjamin E Nilsson-Payant, Wen-Chun Liu,
Skyler Uhl, Daisy Hoagland, Rasmus Møller, Tristan X Jordan, Kohei
Oishi, Maryline Panis, David Sachs, and others. 2020. Imbalanced host
response to sars-cov-2 drives development of covid-19. *Cell* 181, 5
(2020), 1036–1045.

</div>

<div id="ref-Cardoso_MONAI_An_open-source_2022">

\[6\] M. Jorge Cardoso, Wenqi Li, Richard Brown, Nic Ma, Eric Kerfoot,
Yiheng Wang, Benjamin Murray, Andriy Myronenko, Can Zhao, Dong Yang,
Vishwesh Nath, Yufan He, Ziyue Xu, Ali Hatamizadeh, Wentao Zhu, Yun Liu,
Mingxin Zheng, Yucheng Tang, Isaac Yang, Michael Zephyr, Behrooz
Hashemian, Sachidanand Alle, Mohammad Zalbagi Darestani, Charlie Budd,
Marc Modat, Tom Vercauteren, Guotai Wang, Yiwen Li, Yipeng Hu, Yunguan
Fu, Benjamin Gorman, Hans Johnson, Brad Genereaux, Barbaros S. Erdal,
Vikash Gupta, Andres Diaz-Pinto, Andre Dourson, Lena Maier-Hein, Paul F.
Jaeger, Michael Baumgartner, Jayashree Kalpathy-Cramer, Mona Flores,
Justin Kirby, Lee A. D. Cooper, Holger R. Roth, Daguang Xu, David
Bericat, Ralf Floca, S. Kevin Zhou, Haris Shuaib, Keyvan Farahani, Klaus
H. Maier-Hein, Stephen Aylward, Prerna Dogra, Sebastien Ourselin, and
Andrew Feng. 2022. MONAI: An open-source framework for deep learning in
healthcare. (November 2022).
DOI:[https://doi.org/https://doi.org/10.48550/arXiv.2211.02701
](https://doi.org/https://doi.org/10.48550/arXiv.2211.02701%20%20%20%20%20%20%20%20%20%20)

</div>

<div id="ref-carithers2015novel">

\[7\] Latarsha J Carithers, Kristin Ardlie, Mary Barcus, Philip A
Branton, Angela Britton, Stephen A Buia, Carolyn C Compton, David S
DeLuca, Joanne Peter-Demchok, Ellen T Gelfand, and others. 2015. A novel
approach to high-quality postmortem tissue procurement: The gtex
project. *Biopreservation and biobanking* 13, 5 (2015), 311–319.

</div>

<div id="ref-Chen2020_CausalML_Uber">

\[8\] Huigang Chen, Totte Harinen, Jeong-Yoon Lee, Mike Yung, and Zhenyu
Zhao. 2020. Causalml: Python package for causal machine learning. *arXiv
preprint arXiv:2002.11631* (2020).

</div>

<div id="ref-Cochrane2022">

\[9\] Cochrane. 2022. Cochrane library. Retrieved from
<https://www.cochranelibrary.com/central>

</div>

<div id="ref-gordon2020sars">

\[10\] David E Gordon, Gwendolyn M Jang, Mehdi Bouhaddou, Jiewei Xu,
Kirsten Obernier, Kris M White, Matthew J O’Meara, Veronica V Rezelj,
Jeffrey Z Guo, Danielle L Swaney, and others. 2020. A sars-cov-2 protein
interaction map reveals targets for drug repurposing. *Nature* 583, 7816
(2020), 459–468.

</div>

<div id="ref-li2017reference">

\[11\] Huipeng Li, Elise T Courtois, Debarka Sengupta, Yuliana Tan, Kok
Hao Chen, Jolene Jie Lin Goh, Say Li Kong, Clarinda Chua, Lim Kiat Hon,
Wah Siew Tan, and others. 2017. Reference component analysis of
single-cell transcriptomes elucidates cellular heterogeneity in human
colorectal tumors. *Nature genetics* 49, 5 (2017), 708–718.

</div>

<div id="ref-medline2023">

\[12\] U. S. National Library of Medicine. 2023. MEDLINE. Retrieved from
<https://www.nlm.nih.gov/medline/medline_overview.html>

</div>

<div id="ref-miller2020whynot">

\[13\] John Miller, Chloe Hsu, Jordan Troutman, Juan Perdomo, Tijana
Zrnic, Lydia Liu, Yu Sun, Ludwig Schmidt, and Moritz Hardt. 2020.
*WhyNot*. Zenodo. DOI:[https://doi.org/10.5281/zenodo.3875775
](https://doi.org/10.5281/zenodo.3875775%20)

</div>

<div id="ref-Awesome_Medical_List_2016">

\[14\] Karri Niemelä and others. 2016. Awesome Health. 

</div>

<div id="ref-petersen2010alzheimer">

\[15\] Ronald Carl Petersen, PS Aisen, Laurel A Beckett, MC Donohue, AC
Gamst, Danielle J Harvey, CR Jack, WJ Jagust, LM Shaw, AW Toga, and
others. 2010. Alzheimer’s disease neuroimaging initiative (adni):
Clinical characterization. *Neurology* 74, 3 (2010), 201–209.

</div>

<div id="ref-Ramsundar-et-al-2019">

\[16\] Bharath Ramsundar, Peter Eastman, Patrick Walters, Vijay Pande,
Karl Leswing, and Zhenqin Wu. 2019. *Deep learning for the life
sciences*. O’Reilly Media.

</div>

<div id="ref-razick2008irefindex">

\[17\] Sabry Razick, George Magklaras, and Ian M Donaldson. 2008.
IRefIndex: A consolidated protein interaction database with provenance.
*BMC bioinformatics* 9, 1 (2008), 1–19.

</div>

<div id="ref-reyfman2019single">

\[18\] Paul A Reyfman, James M Walter, Nikita Joshi, Kishore R Anekalla,
Alexandra C McQuattie-Pimentel, Stephen Chiu, Ramiro Fernandez, Mahzad
Akbarpour, Ching-I Chen, Ziyou Ren, and others. 2019. Single-cell
transcriptomic analysis of human lung provides insights into the
pathobiology of pulmonary fibrosis. *American journal of respiratory and
critical care medicine* 199, 12 (2019), 1517–1536.

</div>

<div id="ref-rios2021benchpress">

\[19\] Felix L. Rios, Giusi Moffa, and Jack Kuipers. 2021. Benchpress: A
scalable and platform-independent workflow for benchmarking structure
learning algorithms for graphical models. Retrieved from
<http://arxiv.org/abs/2107.03863>

</div>

<div id="ref-Sanchez2022">

\[20\] Pedro Sanchez, Jeremy P. Voisey, Tian Xia, Hannah I. Watson,
Alison Q. O’Neil, and Sotirios A. Tsaftaris. 2022. Causal machine
learning for healthcare and precision medicine. *Royal Society Open
Science* 9, (2022).

</div>

<div id="ref-shen2020">

\[21\] Xinpeng Shen, Sisi Ma, Prashanthi Vemuri, Gyorgy Simon, Michael
W. Weiner, and the Alzheimer’s Disease Neuroimaging Initiative. 2020.
Challenges and opportunities with causal discovery algorithms:
Application to alzheimer’s pathophysiology. *Scientific Reports* 10, 1
(2020).

</div>

<div id="ref-shimoni2019evaluation">

\[22\] Yishai Shimoni, Ehud Karavani, Sivan Ravid, Peter Bak, Tan Hung
Ng, Sharon Hensley Alford, Denise Meade, and Yaara Goldschmidt. 2019. An
evaluation toolkit to guide model selection and cohort definition in
causal inference. *arXiv preprint arXiv:1906.00442* (2019).

</div>

<div id="ref-subramanian2017next">

\[23\] Aravind Subramanian, Rajiv Narayan, Steven M Corsello, David D
Peck, Ted E Natoli, Xiaodong Lu, Joshua Gould, John F Davis, Andrew A
Tubelli, Jacob K Asiedu, and others. 2017. A next generation
connectivity map: L1000 platform and the first 1,000,000 profiles.
*Cell* 171, 6 (2017), 1437–1452.

</div>

<div id="ref-tsitsiridis2023corum">

\[24\] George Tsitsiridis, Ralph Steinkamp, Madalina Giurgiu, Barbara
Brauner, Gisela Fobo, Goar Frishman, Corinna Montrone, and Andreas
Ruepp. 2023. CORUM: The comprehensive resource of mammalian protein
complexes–2022. *Nucleic Acids Research* 51, D1 (2023), D539–D545.

</div>

<div id="ref-vanAmsterdam2019">

\[25\] W. A. C. Van Amsterdam, J. J. C. Verhoeff, P. A. de Jong, T.
Leiner, and M. J. C. Eijkemans. 2019. Eliminating biasing signals in
lung cancer images for prognosis predictions with deep learning. *npj
Digital Medicine* 2, 1 (2019).

</div>

<div id="ref-zhang2021gcastle">

\[26\] Keli Zhang, Shengyu Zhu, Marcus Kalander, Ignavier Ng, Junjian
Ye, Zhitang Chen, and Lujia Pan. 2021. GCastle: A python toolbox for
causal discovery. *arXiv preprint arXiv:2111.15155* (2021).

</div>

<div id="ref-ziff2015">

\[27\] Oliver J Ziff, Deirdre A Lane, Monica Samra, Michael Griffith,
Paulus Kirchhof, Gregory Y H Lip, Richard P Steeds, Jonathan Townend,
and Dipak Kotecha. 2015. Safety and efficacy of digoxin: Systematic
review and meta-analysis of observational and controlled trial data.
*BMJ* 351, (2015).

</div>

</div>

[^1]:  See <https://clue.io/connectopedia/perturbagen_types_and_controls>
    for the definition of this term
