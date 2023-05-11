# Auditing (Safety and Accountability)

## Aspects of Trustworthy AI and Application Domain

### [Interpretability](../Interpretability/README.md)

### [Fairness](../Fairness/README.md)

### [Robustness](../Robustness/README.md)

### [Privacy](../Privacy/README.md)

### Auditing (Safety and Accountability)

### [Healthcare](../Healthcare/README.md)

<details>
<summary><h2>Datasets Used by Cited Publications (Click to expand)</h2></summary>

  - **[ScienceDirect](https://www.sciencedirect.com/)** \[[6](#ref-ScienceDirect_2023)\]:
    A bibliographic database that hosts over 18 million publications
    from more than 4,000 journals and more than 30,000 e-books from the
    publisher Elsevier. Launched back in 1997, ScienceDirect includes
    papers from engineering and medical research areas and social
    sciences and humanities. <span>&#10230;</span> **Used
    by**: \[[21](#ref-voegeli2019sustainability)\]

  - **[World
    Bank](https://data.worldbank.org/indicator)** \[[9](#ref-WDI)\]: A
    publicly available collection of datasets that facilitate the
    analysis of global development. Researchers can use this data to
    compare countries under different developmental aspects, including
    agricultural progress, poverty, population dynamics, and economic
    growth. <span>&#10230;</span> **Used
    by**: \[[24](#ref-wu2015causality)\]

  - **World Economic Forum (WEF)** \[[22](#ref-WEF_2023)\]: The WEF is
    an international non-governmental based in Switzerland that
    publishes economic reports such as the Global Competitiveness
    Report. The reports are available online, with some of the data
    being easily accessible through websites like
    [Knoema](https://knoema.com/atlas/sources/WEF).
    <span>&#10230;</span> **Used by**: \[[10](#ref-haseeb2019economic)\]

  - **[OECD.Stat](https://stats.oecd.org/)** \[[14](#ref-OECD_stats_2023)\]:
    This webpage includes data and metadata for OECD countries and
    selected non-member economies. The online platform allows
    researchers to traverse the collected data through given data themes
    or via search-engine queries. <span>&#10230;</span> **Used
    by**: \[[10](#ref-haseeb2019economic)\]

  - **[Global Brand
    Database](https://branddb.wipo.int/en/)** \[[23](#ref-WIPO_2023)\]:
    An online database hosted by the World Intellectual Property
    Organization (WIPO) that contains information about Trademark
    applications (e.g., owner of the trademark, its status, or the
    designation country). It currently contains almost 53 million
    records from 73 data sources. <span>&#10230;</span> **Used
    by**: \[[10](#ref-haseeb2019economic)\]

  - **[PubMed](https://pubmed.ncbi.nlm.nih.gov/)** \[[2](#ref-PubMed2022)\]:
    A widely-known, free-to-access search engine for biomedical and life
    science literature developed and maintained by the National Center
    for Biotechnology Information (NCBI). Researchers can find more than
    34 million citations and abstracts of articles. PubMed does not host
    the articles themselves but frequently provides a link to the
    full-text articles. <span>&#10230;</span> **Used
    by**: \[[8](#ref-gillespie2021impact)\]

  - **[ProQuest
    Central](https://www.proquest.com/)** \[[15](#ref-ProQuest2022)\]:
    A database containing dissertations and theses in a multitude of
    disciplines. It currently contains more than 5 million graduate
    works. <span>&#10230;</span> **Used
    by**: \[[8](#ref-gillespie2021impact)\]

  - **[Cochrane Central Register of Controlled Trials
    (CENTRAL)](https://www.cochranelibrary.com/central)** \[[5](#ref-Cochrane2022)\]:
    A database of reports for randomized and quasi-randomized controlled
    trials collected from different online databases. Although it does
    not contain full-text articles, the CENTRAL includes bibliographic
    details and often an abstract of the report.
    <span>&#10230;</span> **Used by**: \[[8](#ref-gillespie2021impact)\]

  - **[PsycINFO](https://www.apa.org/pubs/databases/psycinfo/index)** \[[1](#ref-PsycINFO2022)\]:
    A database hosted and developed by American Psychological
    Association containing abstracts for more than five million articles
    in the field of psychology. <span>&#10230;</span> **Used
    by**: \[[8](#ref-gillespie2021impact)\]

  - **[Lending
    Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)** \[[7](#ref-Lending_Club_2018)\]:
    A dataset that contains information about all accepted and rejected
    peer-to-peer loan applications of LendingClub. Currently, the data
    are only available through the referenced Kaggle entry, as the
    company no longer provides peer-to-peer loan services[^1]:.
    <span>&#10230;</span> **Used
    by**: \[[19](#ref-tsirtsis2020decisions)\]

  - **[Taiwanese Credit
    Data](https://github.com/ustunb/actionable-recourse/tree/master/examples/paper/data)** \[[20](#ref-ustun2019actionable),[25](#ref-yeh2009comparisons)\]:
    A real-world dataset containing payment data collected in October
    2005 from a Taiwanese bank. The commonly used pre-processed
    version[^2]: \[[20](#ref-ustun2019actionable)\] contains data from
    30,000 individuals described through 16 features (e.g., marital
    status, age, or payment history). <span>&#10230;</span> **Used
    by**: \[[19](#ref-tsirtsis2020decisions)\]

</details>

<details>
<summary><h2>Interesting Causal Tools (Click to expand)</h2></summary>

  - **[CausalImpact](https://google.github.io/CausalImpact/CausalImpact.html)** \[[3](#ref-brodersen2015inferring)\]:
    This R package allows users to conduct causal impact assessment for
    planned interventions on serial data given a response time series
    and an assortment of control time series. For this purpose,
    CausalImpact enables the construction of a Bayesian structural
    time-series model that can be used to predict the resulting
    counterfactual of an intervention.

  - **[Causal
    Inference 360](https://github.com/BiomedSciAI/causallib)** \[[18](#ref-shimoni2019evaluation)\]:
    A Python package developed by IBM to infer causal effects from given
    data. Causal Inference 360 includes multiple estimation methods, a
    medical dataset, and multiple simulation sets. The provided methods
    can be used for any complex ML model through a scikit-learn-inspired
    API.

  - **[gCastle](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle)** \[[26](#ref-zhang2021gcastle)\]:
    An end-to-end causal structure learning toolbox that is equipped
    with 19 techniques for Causal Discovery. It also assists users in
    data generation and evaluating learned structures. Having a firm
    understanding of the causal structure is crucial for safety-related
    research.

  - **[Benchpress](https://github.com/felixleopoldo/benchpress)** \[[16](#ref-rios2021benchpress)\]:
    A benchmark for causal structure learning allowing users to compare
    their causal discovery methods with over 40 variations of
    state-of-the-art algorithms. The plethora of available techniques in
    this single tool could facilitate research into safety and
    accountability of ML systems through causality.

  - **[CauseEffectPairs](https://webdav.tuebingen.mpg.de/cause-effect/)** \[[13](#ref-mooij2016distinguishing)\]:
    A collection of more than 100 databases, each annotated with a
    two-variable cause-effect relationship (e.g., access to drinking
    water affects infant mortality). Given a database, models need to
    distinguish between the cause and effect variables.

</details>

<details>
<summary><h2>Prominent Non-Causal Tools (Click to expand)</h2></summary>

  - **[Government of Canada’s AIA
    tool](https://github.com/canada-ca/aia-eia-js)** \[[4](#ref-CanadaAIA)\]:
    The Algorithmic Impact Assessment (AIA) tool is a questionnaire
    developed in the wake of Canada’s Directive on Automated Decision
    Making[^3]:. Employees of the Canadian Government wishing to employ
    automatic decision-making systems in their projects first need to
    assess the impact of such systems via this tool. Based on answers
    given to ca. 80 questions revolving around different aspects of the
    projects, AIA will output two scores: one indicating the risks that
    automation would bring and one that quantifies the quality of the
    risk management.

  - **[Aequitas](https://github.com/dssg/aequitas)** \[[17](#ref-2018aequitas)\]:
    An open-source auditing tool designed to assess the bias of
    algorithmic decision-making systems. It provides utility for
    evaluating the bias of decision-making outcomes and enables users to
    assess the bias of actions taken directly.

  - **[Error Analysis (*Responsible
    AI*)](https://github.com/microsoft/responsible-ai-toolbox/blob/main/docs/erroranalysis-dashboard-README.md)** \[[12](#ref-ResponsibleAI)\]:
    As part of the Responsible AI toolbox, Error Analysis is a model
    assessment tool capable of identifying subsets of data in which the
    model performs poorly (e.g., black citizens being more frequently
    misclassified as potential re-offenders). It also enables users to
    diagnose the root cause of such poor performance.

  - **[ML-Doctor](https://github.com/liuyugeng/ML-Doctor)** \[[11](#ref-mldoctor_2022)\]:
    A codebase initially used to compare and evaluate different
    inference attacks (membership inference, model stealing, model
    inversion, and attribute inference). Due to its modular structure,
    it can also be used as a Risk Assessment tool for analyzing the
    susceptibility against SOTA privacy attacks.
  
  </details>

<div id="refs" class="references">

## References

<div id="ref-PsycINFO2022">

\[1\] American Psychological Association. 2022. PsycINFO. Retrieved from
<https://www.apa.org/pubs/databases/psycinfo/index>

</div>

<div id="ref-PubMed2022">

\[2\] National Center for Biotechnology Information. 2022. PubMed.
Retrieved from <https://pubmed.ncbi.nlm.nih.gov/>

</div>

<div id="ref-brodersen2015inferring">

\[3\] Kay H Brodersen, Fabian Gallusser, Jim Koehler, Nicolas Remy, and
Steven L Scott. 2015. Inferring causal impact using bayesian structural
time-series models. *The Annals of Applied Statistics* (2015), 247–274.

</div>

<div id="ref-CanadaAIA">

\[4\] Government of Canada. Algorithmic Impact Assessment tool.
Retrieved from <https://open.canada.ca/aia-eia-js/?lang=en>

</div>

<div id="ref-Cochrane2022">

\[5\] Cochrane. 2022. Cochrane library. Retrieved from
<https://www.cochranelibrary.com/central>

</div>

<div id="ref-ScienceDirect_2023">

\[6\] Elsevier. 2023. Science direct. Retrieved from
<https://www.sciencedirect.com>

</div>

<div id="ref-Lending_Club_2018">

\[7\] Nathan George. 2018. All lending club loan data. Retrieved from
<https://www.kaggle.com/datasets/wordsforthewise/lending-club>

</div>

<div id="ref-gillespie2021impact">

\[8\] Brigid M Gillespie, Joseph Gillespie, Rhonda J Boorman, Karin
Granqvist, Johan Stranne, and Annette Erichsen-Andersson. 2021. The
impact of robotic-assisted surgery on team performance: A systematic
mixed studies review. *Human factors* 63, 8 (2021), 1352–1379.

</div>

<div id="ref-WDI">

\[9\] The World Bank Group. 2022. World development indicators.
Retrieved from <https://data.worldbank.org/indicator>

</div>

<div id="ref-haseeb2019economic">

\[10\] Muhammad Haseeb, Leonardus WW Mihardjo, Abid Rashid Gill,
Kittisak Jermsittiparsert, and others. 2019. Economic impact of
artificial intelligence: New look for the macroeconomic assessment in
asia-pacific region. *International Journal of Computational
Intelligence Systems* 12, 2 (2019), 1295.

</div>

<div id="ref-mldoctor_2022">

\[11\] Yugeng Liu, Rui Wen, Xinlei He, Ahmed Salem, Zhikun Zhang,
Michael Backes, Emiliano De Cristofaro, Mario Fritz, and Yang Zhang.
2022. ML-Doctor: Holistic risk assessment of inference attacks against
machine learning models. In *31st usenix security symposium (usenix
security 22)*, USENIX Association, Boston, MA, 4525–4542. Retrieved from
<https://www.usenix.org/conference/usenixsecurity22/presentation/liu-yugeng>

</div>

<div id="ref-ResponsibleAI">

\[12\] Microsoft. 2022. Responsible AI Toolbox. 

</div>

<div id="ref-mooij2016distinguishing">

\[13\] Joris M Mooij, Jonas Peters, Dominik Janzing, Jakob Zscheischler,
and Bernhard Schölkopf. 2016. Distinguishing cause from effect using
observational data: Methods and benchmarks. *The Journal of Machine
Learning Research* 17, 1 (2016), 1103–1204.

</div>

<div id="ref-OECD_stats_2023">

\[14\] Organisation for Economic Co-operation and Development. 2023.
OECD statistics. Retrieved from <https://stats.oecd.org/>

</div>

<div id="ref-ProQuest2022">

\[15\] ProQuest. 2022. ProQuest. Retrieved from
<https://www.proquest.com/>

</div>

<div id="ref-rios2021benchpress">

\[16\] Felix L. Rios, Giusi Moffa, and Jack Kuipers. 2021. Benchpress: A
scalable and platform-independent workflow for benchmarking structure
learning algorithms for graphical models. Retrieved from
<http://arxiv.org/abs/2107.03863>

</div>

<div id="ref-2018aequitas">

\[17\] Pedro Saleiro, Benedict Kuester, Abby Stevens, Ari Anisfeld,
Loren Hinkson, Jesse London, and Rayid Ghani. 2018. Aequitas: A bias and
fairness audit toolkit. *arXiv preprint arXiv:1811.05577* (2018).

</div>

<div id="ref-shimoni2019evaluation">

\[18\] Yishai Shimoni, Ehud Karavani, Sivan Ravid, Peter Bak, Tan Hung
Ng, Sharon Hensley Alford, Denise Meade, and Yaara Goldschmidt. 2019. An
evaluation toolkit to guide model selection and cohort definition in
causal inference. *arXiv preprint arXiv:1906.00442* (2019).

</div>

<div id="ref-tsirtsis2020decisions">

\[19\] Stratis Tsirtsis and Manuel Gomez Rodriguez. 2020. Decisions,
counterfactual explanations and strategic behavior. *Advances in Neural
Information Processing Systems* 33, (2020), 16749–16760.

</div>

<div id="ref-ustun2019actionable">

\[20\] Berk Ustun, Alexander Spangher, and Yang Liu. 2019. Actionable
recourse in linear classification. In *Proceedings of the conference on
fairness, accountability, and transparency*, 10–19.

</div>

<div id="ref-voegeli2019sustainability">

\[21\] Guillaume Voegeli, Werner Hediger, and Franco Romerio. 2019.
Sustainability assessment of hydropower: Using causal diagram to seize
the importance of impact pathways. *Environmental Impact Assessment
Review* 77, (2019), 69–84.

</div>

<div id="ref-WEF_2023">

\[22\] World Economic Forum. 2023. World economic forum. Retrieved from
<https://www.weforum.org/reports/>

</div>

<div id="ref-WIPO_2023">

\[23\] World Intellectual Property Organization. 2023. Global brand
database. Retrieved from <https://branddb.wipo.int/en/>

</div>

<div id="ref-wu2015causality">

\[24\] Susie R Wu, Jiquan Chen, Defne Apul, Peilei Fan, Yanfa Yan, Yi
Fan, and Peiling Zhou. 2015. Causality in social life cycle impact
assessment (slcia). *The International Journal of Life Cycle Assessment*
20, 9 (2015), 1312–1323.

</div>

<div id="ref-yeh2009comparisons">

\[25\] I-Cheng Yeh and Che-hui Lien. 2009. The comparisons of data
mining techniques for the predictive accuracy of probability of default
of credit card clients. *Expert systems with applications* 36, 2 (2009),
2473–2480.

</div>

<div id="ref-zhang2021gcastle">

\[26\] Keli Zhang, Shengyu Zhu, Marcus Kalander, Ignavier Ng, Junjian
Ye, Zhitang Chen, and Lujia Pan. 2021. GCastle: A python toolbox for
causal discovery. *arXiv preprint arXiv:2111.15155* (2021).

</div>

</div>

[^1]:  <https://www.lendingclub.com/investing/peer-to-peer>

[^2]:  Available at
    <https://github.com/ustunb/actionable-recourse/tree/master/examples/paper/data>
    under the name "credit\_processed.csv"

[^3]:  <http://www.tbs-sct.gc.ca/pol/doc-eng.aspx?id=32592>
