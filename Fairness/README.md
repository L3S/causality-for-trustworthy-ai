# Fairness

## Aspects of Trustworthy AI and Application Domain

### [Interpretability](../Interpretability/README.md)

### Fairness

### [Robustness](../Robustness/README.md)

### [Privacy](../Privacy/README.md)

### [Auditing (Safety and Accountability)](../Auditing/README.md)

### [Healthcare](../Healthcare/README.md)

<details>
<summary><h2>Datasets Used by Cited Publications (Click to expand)</h2></summary>

  - **[Adult (Census
    Income)](https://archive.ics.uci.edu/ml/datasets/adult)** \[[11](#ref-dheeru2017uci),[17](#ref-kohavi1996ADULTS)\]:
    A tabular dataset containing anonymized data from the 1994 Census
    bureau database.[^1] Classifiers try to predict whether a given
    person will earn over or under 50,000 USD worth of salary. Each
    person is described via 15 features (including their id), e.g.,
    gender, education, and occupation. <span>&#10230;</span> **Used
    by**: \[[13](#ref-galhotra2022causal),[24](#ref-nabi2018fair),[25](#ref-pan2021explaining),[28](#ref-salimi2019interventional),[30](#ref-wu2019counterfactual)–[32](#ref-yan2020silva),[34](#ref-zhang2017causal),[35](#ref-zhang2018causal)\]

  - **[COMPAS Recidivism
    Risk](https://github.com/propublica/compas-analysis/)** \[[1](#ref-angwin2016COMPAS)\]:
    A set of criminological datasets published by ProPublica to evaluate
    the bias of COMPAS - an algorithm used to assess the likelihood of
    criminal defendants reoffending. All COMPAS-related datasets include
    data from over 10,000 defendants, each being described via 52
    features (e.g., age, sex, race) and with a label indicating whether
    they were rearrested within two years. <span>&#10230;</span> **Used
    by**: \[[8](#ref-chiappa2018causal),[13](#ref-galhotra2022causal),[22](#ref-mishler2021fairness)–[24](#ref-nabi2018fair),[28](#ref-salimi2019interventional)\]

  - **[FICO Credit
    Risk](https://community.fico.com/s/explainable-machine-learning-challenge?tabset-3158a=2&tabset-158d9=3)** \[[14](#ref-board2007FICO)\]:
    In this dataset, ML models have to predict whether or not credit
    applicants will at least once be more than 90 days due with their
    payment within a two-year timespan. It includes anonymized
    information about HELOC applicants described through 23 features
    (e.g., months since the most recent delinquency or number of
    inquiries in last 6 months) \[[6](#ref-chen2018interpretable)\]
    <span>&#10230;</span> **Used by**: \[[9](#ref-creager2020causal)\]

  - **[German Credit
    Risk](https://archive.ics.uci.edu/ml/datasets/statlog+\(german+credit+data\))** \[[11](#ref-dheeru2017uci)\]:
    A collection of data from 1,000 anonymized German bank account
    holders that applied for a credit. Based on the 20 features of the
    applicant and their application (e.g., credit history, purpose of
    credit, or employment status), models need to estimate the risk of
    giving the person a credit and categorize them as either good or bad
    credit recipients.. <span>&#10230;</span> **Used
    by**: \[[13](#ref-galhotra2022causal)\]

  - **[Medical Expenditure
    (MEPS](https://meps.ahrq.gov/mepsweb/data_stats/data_overview.jsp))** \[[5](#ref-MEPS)\]:
    A collection of large-scale surveys of US citizens, their medical
    providers, and employers. It includes information like race, gender,
    and the ICD-10 code of the diagnosis of a patient. The given
    information can be used to predict the total number of patients’
    hospital visits. <span>&#10230;</span> **Used
    by**: \[[13](#ref-galhotra2022causal)\]

  - **[MIMIC
    III](https://mimic.mit.edu/docs/gettingstarted/)** \[[16](#ref-johnson2016mimic)\]:
    A dataset of anonymized clinical records of the Beth Israel
    Deaconess Medical Center in Boston, Massachusetts. Records contain
    information like ICD-9 codes for diagnoses and medical procedures,
    vital signs, medication, or even imaging data. The dataset includes
    records from 38,597 distinct adult patients.
    <span>&#10230;</span> **Used by**: \[[29](#ref-singh2021fairness)\]

  - **[MovieLens](https://grouplens.org/datasets/movielens/)** \[[15](#ref-MovieLens2015)\]:
    A group of datasets containing movie ratings between 0 and 5 (with
    0.5 increments) collected from the MovieLens website. Movies are
    described through their title, genre, and relevance scores of tags
    (e.g., romantic or funny). GroupLens Research constantly releases
    new up-to-date MovieLens databases in different sizes.
    <span>&#10230;</span> **Used by**: \[[20](#ref-li2021towards)\]

  - **[Zimnat Insurance
    Recommendation](https://www.kaggle.com/mrmorj/insurance-recommendation)** \[[36](#ref-Insurance2020)\]:
    A data collection of almost 40,000 Zimnat (a Zimbabwean insurance
    provider) customers. The data contain personal information (e.g.,
    marital status or occupation) and the insurance products that the
    customers own. In inference time, models must predict which product
    was artificially removed based on customer information.
    <span>&#10230;</span> **Used by**: \[[20](#ref-li2021towards)\]

  - **[Civil Rights Data Collection
    (CRDC)](https://ocrdata.ed.gov/resources/downloaddatafile)** \[[12](#ref-CRDC_2023)\]:
    This is an online collection of education-related data. Since 1968,
    the U.S. Department of Education’s Office for Civil Rights (OCR)
    biennially collects data from U.S. public primary and secondary
    schools. The dataset includes information such as race distribution,
    the percentage of students who take college entrance exams, or
    whether specific courses (e.g., Calculus) are offered.
    <span>&#10230;</span> **Used by**: \[[18](#ref-kusner2019making)\]

  - **[Berkeley](https://discovery.cs.illinois.edu/dataset/berkeley/)** \[[3](#ref-bickel1975sex)\]:
    A simple gender bias dataset published back in 1975 containing
    information on all 12,763 applicants to the University of
    California, Berkeley graduate programs in Fall 1973. Each candidate
    entry consists of the candidate’s major, gender, year of application
    (always 1973), and whether they were accepted.
    <span>&#10230;</span> **Used by**: \[[32](#ref-yan2020silva)\]

</details>

<details>
<summary><h2>Interesting Causal Tools (Click to expand)</h2></summary>

  - **Collection of Annotated Datasets** \[[19](#ref-le2022survey)\]: As
    part of a survey that provides a thorough overview of commonly used
    datasets for evaluating the fairness of
    ML, \[[19](#ref-le2022survey)\] generated Bayesian Networks
    encompassing the relationships of attributes for each dataset. This
    information could be used as a reference point for potential causal
    annotations of fairness-related datasets.

  - **[WhyNot](https://github.com/mrtzh/whynot)** \[[21](#ref-miller2020whynot)\]:
    A Python package that provides researchers with many simulation
    environments for analyzing causal inference and decision-making in a
    dynamic setting. It allows benchmarking of multiple decision-making
    systems on 13 different simulators. Crucially for this section,
    WhyNot also enables comparisons based on other evaluation criteria,
    such as the fairness of the decision-making.

  - **[gCastle](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle)** \[[33](#ref-zhang2021gcastle)\]:
    An end-to-end causal structure learning toolbox that is equipped
    with 19 techniques for Causal Discovery. It also assists users in
    data generation and evaluating learned structures. Having a firm
    understanding of the causal structure is crucial for
    fairness-related research.

  - **[Benchpress](https://github.com/felixleopoldo/benchpress)** \[[26](#ref-rios2021benchpress)\]:
    A benchmark for causal structure learning allowing users to compare
    their causal discovery methods with over 40 variations of
    state-of-the-art algorithms. The plethora of available techniques in
    this single tool could facilitate research into fair ML through
    causality.

  - **[CausalML](https://github.com/uber/causalml)** \[[7](#ref-Chen2020_CausalML_Uber)\]:
    The Python package enables users to analyze the Conditional Average
    Treatment Effect (CATE) or Individual Treatment Effect (ITE)
    observable in experimental data. The package includes tree-based
    algorithms, meta-learner algorithms, instrumental variable
    algorithms, and neural-network-based algorithms. Fair-ML researchers
    could use the provided methods to investigate the causal effect of
    sensitive attributes on the predicted outcome.

</details>

<details>
<summary><h2>Prominent Non-Causal Tools (Click to expand)</h2></summary>

  - **[AI
    Fairness 360](https://github.com/Trusted-AI/AIF360)** \[[2](#ref-aif360-oct-2018)\]:
    An open-source library (compatible with both Python and R) that
    allows researchers to measure and mitigate possible bias within
    their models/algorithms. It includes six real-world datasets, five
    fairness metrics, and 15 bias mitigation algorithms.

  - **[Fairlearn](https://github.com/fairlearn/fairlearn)** \[[4](#ref-bird2020fairlearn)\]:
    A Python package developed by Microsoft, which is part of the
    Responsible AI toolbox[^2]. It contains various fairness metrics,
    six unfairness-mitigating algorithms, and five datasets.

  - **[Aequitas](https://github.com/dssg/aequitas)** \[[27](#ref-2018aequitas)\]:
    An open-source auditing tool designed to assess the bias of
    algorithmic decision-making systems. It provides utility for
    evaluating the bias of decision-making outcomes and enables users to
    assess the bias of actions taken directly.

  - **[ML-Fairness-Gym](https://github.com/google/ml-fairness-gym)** \[[10](#ref-fairness_gym)\]:
    A third-party extension of the OpenAI gym designed to analyze bias
    within RL agents. Although not built upon real-world data, the
    simulations developed for this benchmark can lead to insights
    applicable to the real world. It comes with four simulation
    environments.

</details>

<div id="refs" class="references">

## References

<div id="ref-angwin2016COMPAS">

\[1\] Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner. 2016.
Machine bias. In *Ethics of data and analytics*. Auerbach Publications,
254–264.

</div>

<div id="ref-aif360-oct-2018">

\[2\] Rachel K. E. Bellamy, Kuntal Dey, Michael Hind, Samuel C. Hoffman,
Stephanie Houde, Kalapriya Kannan, Pranay Lohia, Jacquelyn Martino,
Sameep Mehta, Aleksandra Mojsilovic, Seema Nagar, Karthikeyan Natesan
Ramamurthy, John Richards, Diptikalyan Saha, Prasanna Sattigeri,
Moninder Singh, Kush R. Varshney, and Yunfeng Zhang. 2018. AI Fairness
360: An extensible toolkit for detecting, understanding, and mitigating
unwanted algorithmic bias. Retrieved from
<https://arxiv.org/abs/1810.01943>

</div>

<div id="ref-bickel1975sex">

\[3\] Peter J Bickel, Eugene A Hammel, and J William O’Connell. 1975.
Sex bias in graduate admissions: Data from berkeley: Measuring bias is
harder than is usually assumed, and the evidence is sometimes contrary
to expectation. *Science* 187, 4175 (1975), 398–404.

</div>

<div id="ref-bird2020fairlearn">

\[4\] Sarah Bird, Miro Dudík, Richard Edgar, Brandon Horn, Roman Lutz,
Vanessa Milan, Mehrnoosh Sameki, Hanna Wallach, and Kathleen Walker.
2020. *Fairlearn: A toolkit for assessing and improving fairness in AI*.
Microsoft. Retrieved from
<https://www.microsoft.com/en-us/research/publication/fairlearn-a-toolkit-for-assessing-and-improving-fairness-in-ai/>

</div>

<div id="ref-MEPS">

\[5\] AHRQ Data Center. 2016. The Medical Expenditure Panel Survey
(MEPS). Retrieved from <https://meps.ahrq.gov/mepsweb/>

</div>

<div id="ref-chen2018interpretable">

\[6\] Chaofan Chen, Kangcheng Lin, Cynthia Rudin, Yaron Shaposhnik,
Sijia Wang, and Tong Wang. 2018. An interpretable model with globally
consistent explanations for credit risk. *arXiv preprint
arXiv:1811.12615* (2018).

</div>

<div id="ref-Chen2020_CausalML_Uber">

\[7\] Huigang Chen, Totte Harinen, Jeong-Yoon Lee, Mike Yung, and Zhenyu
Zhao. 2020. Causalml: Python package for causal machine learning. *arXiv
preprint arXiv:2002.11631* (2020).

</div>

<div id="ref-chiappa2018causal">

\[8\] Silvia Chiappa and William S Isaac. 2018. A causal bayesian
networks viewpoint on fairness. In *IFIP international summer school on
privacy and identity management*, Springer, 3–20.

</div>

<div id="ref-creager2020causal">

\[9\] Elliot Creager, David Madras, Toniann Pitassi, and Richard Zemel.
2020. Causal modeling for fairness in dynamical systems. In *ICML*,
PMLR, 2185–2195.

</div>

<div id="ref-fairness_gym">

\[10\] Alexander D’Amour, Hansa Srinivasan, James Atwood, Pallavi
Baljekar, D. Sculley, and Yoni Halpern. 2020. Fairness is not static:
Deeper understanding of long term fairness via simulation studies. In
*Proceedings of the 2020 conference on fairness, accountability, and
transparency* (FAccT ’20), Association for Computing Machinery, New
York, NY, USA, 525–534. DOI:[https://doi.org/10.1145/3351095.3372878
](https://doi.org/10.1145/3351095.3372878%20)

</div>

<div id="ref-dheeru2017uci">

\[11\] Dua Dheeru and E Karra Taniskidou. 2017. UCI machine learning
repository. (2017).

</div>

<div id="ref-CRDC_2023">

\[12\] U. S. Department of Education’s Office for Civil Rights (OCR).
2023. Civil rights data collection. Retrieved from
<https://ocrdata.ed.gov/>

</div>

<div id="ref-galhotra2022causal">

\[13\] Sainyam Galhotra, Karthikeyan Shanmugam, Prasanna Sattigeri, Kush
R Varshney, Rachel Bellamy, Kuntal Dey, and others. 2022. Causal feature
selection for algorithmic fairness. (2022).

</div>

<div id="ref-board2007FICO">

\[14\] Board of Governors of the Federal Reserve System (US). 2007.
*Report to the congress on credit scoring and its effects on the
availability and affordability of credit*. Board of Governors of the
Federal Reserve System.

</div>

<div id="ref-MovieLens2015">

\[15\] F. Maxwell Harper and Joseph A. Konstan. 2015. The movielens
datasets: History and context. *ACM Trans. Interact. Intell. Syst.* 5, 4
(December 2015). DOI:[https://doi.org/10.1145/2827872
](https://doi.org/10.1145/2827872%20)

</div>

<div id="ref-johnson2016mimic">

\[16\] Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman,
Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo
Anthony Celi, and Roger G Mark. 2016. MIMIC-iii, a freely accessible
critical care database. *Scientific data* 3, 1 (2016), 1–9.

</div>

<div id="ref-kohavi1996ADULTS">

\[17\] Ron Kohavi and others. 1996. Scaling up the accuracy of
naive-bayes classifiers: A decision-tree hybrid. In *Kdd*, 202–207.

</div>

<div id="ref-kusner2019making">

\[18\] Matt Kusner, Chris Russell, Joshua Loftus, and Ricardo Silva.
2019. Making decisions that reduce discriminatory impacts. In
*International conference on machine learning*, PMLR, 3591–3600.

</div>

<div id="ref-le2022survey">

\[19\] Tai Le Quy, Arjun Roy, Vasileios Iosifidis, Wenbin Zhang, and
Eirini Ntoutsi. 2022. A survey on datasets for fairness-aware machine
learning. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge
Discovery* (2022), e1452.

</div>

<div id="ref-li2021towards">

\[20\] Yunqi Li, Hanxiong Chen, Shuyuan Xu, Yingqiang Ge, and Yongfeng
Zhang. 2021. Towards personalized fairness based on causal notion. In
*Proceedings of the 44th international acm sigir conference on research
and development in information retrieval*, 1054–1063.

</div>

<div id="ref-miller2020whynot">

\[21\] John Miller, Chloe Hsu, Jordan Troutman, Juan Perdomo, Tijana
Zrnic, Lydia Liu, Yu Sun, Ludwig Schmidt, and Moritz Hardt. 2020.
*WhyNot*. Zenodo. DOI:[https://doi.org/10.5281/zenodo.3875775
](https://doi.org/10.5281/zenodo.3875775%20)

</div>

<div id="ref-mishler2021fairness">

\[22\] Alan Mishler, Edward H Kennedy, and Alexandra Chouldechova. 2021.
Fairness in risk assessment instruments: Post-processing to achieve
counterfactual equalized odds. In *FAccT*, 386–400.

</div>

<div id="ref-nabi2019learning">

\[23\] Razieh Nabi, Daniel Malinsky, and Ilya Shpitser. 2019. Learning
optimal fair policies. In *ICML*, PMLR, 4674–4682.

</div>

<div id="ref-nabi2018fair">

\[24\] Razieh Nabi and Ilya Shpitser. 2018. Fair inference on outcomes.
In *AAAI*.

</div>

<div id="ref-pan2021explaining">

\[25\] Weishen Pan, Sen Cui, Jiang Bian, Changshui Zhang, and Fei Wang.
2021. Explaining algorithmic fairness through fairness-aware causal path
decomposition. In *SIGKDD*, 1287–1297.

</div>

<div id="ref-rios2021benchpress">

\[26\] Felix L. Rios, Giusi Moffa, and Jack Kuipers. 2021. Benchpress: A
scalable and platform-independent workflow for benchmarking structure
learning algorithms for graphical models. Retrieved from
<http://arxiv.org/abs/2107.03863>

</div>

<div id="ref-2018aequitas">

\[27\] Pedro Saleiro, Benedict Kuester, Abby Stevens, Ari Anisfeld,
Loren Hinkson, Jesse London, and Rayid Ghani. 2018. Aequitas: A bias and
fairness audit toolkit. *arXiv preprint arXiv:1811.05577* (2018).

</div>

<div id="ref-salimi2019interventional">

\[28\] Babak Salimi, Luke Rodriguez, Bill Howe, and Dan Suciu. 2019.
Interventional fairness: Causal database repair for algorithmic
fairness. In *MOD*, 793–810.

</div>

<div id="ref-singh2021fairness">

\[29\] Harvineet Singh, Rina Singh, Vishwali Mhasawade, and Rumi
Chunara. 2021. Fairness violations and mitigation under covariate shift.
In *Proceedings of the 2021 acm conference on fairness, accountability,
and transparency*, 3–13.

</div>

<div id="ref-wu2019counterfactual">

\[30\] Yongkai Wu, Lu Zhang, and Xintao Wu. 2019. Counterfactual
fairness: Unidentification, bound and algorithm. In *Proceedings of the
twenty-eighth international joint conference on artificial
intelligence*.

</div>

<div id="ref-xu2019achieving">

\[31\] Depeng Xu, Yongkai Wu, Shuhan Yuan, Lu Zhang, and Xintao Wu.
2019. Achieving causal fairness through generative adversarial networks.
In *Proceedings of the twenty-eighth international joint conference on
artificial intelligence*.

</div>

<div id="ref-yan2020silva">

\[32\] Jing Nathan Yan, Ziwei Gu, Hubert Lin, and Jeffrey M
Rzeszotarski. 2020. Silva: Interactively assessing machine learning
fairness using causality. In *CHI*, 1–13.

</div>

<div id="ref-zhang2021gcastle">

\[33\] Keli Zhang, Shengyu Zhu, Marcus Kalander, Ignavier Ng, Junjian
Ye, Zhitang Chen, and Lujia Pan. 2021. GCastle: A python toolbox for
causal discovery. *arXiv preprint arXiv:2111.15155* (2021).

</div>

<div id="ref-zhang2017causal">

\[34\] Lu Zhang, Yongkai Wu, and Xintao Wu. 2017. A causal framework for
discovering and removing direct and indirect discrimination. In
*Proceedings of the twenty-sixth international joint conference on
artificial intelligence*.

</div>

<div id="ref-zhang2018causal">

\[35\] Lu Zhang, Yongkai Wu, and Xintao Wu. 2018. Causal modeling-based
discrimination discovery and removal: Criteria, bounds, and algorithms.
*IEEE Transactions on Knowledge and Data Engineering* 31, 11 (2018),
2035–2050.

</div>

<div id="ref-Insurance2020">

\[36\] Zimnat. 2020. Zimnat insurance recommendation challenge.
Retrieved from
<https://zindi.africa/competitions/zimnat-insurance-recommendation-challenge>

</div>

</div>

[^1]:  <http://www.census.gov/en.html>

[^2]:  <https://github.com/microsoft/responsible-ai-toolbox>
