# Interpretability

## Aspects of Trustworthy AI and Application Domain

### Interpretability

### [Fairness](../Fairness/README.md)

### [Robustness](../Robustness/README.md)

### [Privacy](../Privacy/README.md)

### [Auditing (Safety and Accountability)](../Auditing/README.md)

### [Healthcare](../Healthcare/README.md)

<details>
<summary><h2>Datasets Used by Cited Publications (Click to expand)</h2></summary>

  - **[CANDLE](https://github.com/causal-disentanglement/CANDLE)** \[[42](#ref-reddy_2022_AAAI_candle_disentangled_dataset)\]:
    A dataset of realistic images of objects in a specific scene
    generated based on observed and unobserved confounders (object,
    size, color, rotation, light, and scene). As each of the 12546
    images is annotated with the ground-truth information of the six
    generating factors, it is possible to emulate interventions on image
    features. <span>&#10230;</span> **Used
    by**: \[[42](#ref-reddy_2022_AAAI_candle_disentangled_dataset)\]

  - **[MIND](https://msnews.github.io/)** \[[53](#ref-wu2020mind)\]: A
    news recommendation dataset built upon user click logs of Microsoft
    News. It contains 15 million impression logs describing the click
    behavior of more than 1 Million users across over 160k English news
    articles. Each news article entry contains its title, category,
    abstract, and body. Each log entry is made up of the users’ click
    events, non-clicked events, and historical news click behaviors
    prior to this impression. <span>&#10230;</span> **Used
    by**: \[[48](#ref-si2022wwwcausalRecSearch)\]

  - **[MovieLens](https://grouplens.org/datasets/movielens/)** \[[18](#ref-MovieLens2015)\]:
    A group of datasets containing movie ratings between 0 and 5 (with
    0.5 increments) collected from the MovieLens website. Movies are
    described through their title, genre, and relevance scores of tags
    (e.g., romantic or funny). GroupLens Research constantly releases
    new up-to-date MovieLens databases in different sizes.
    <span>&#10230;</span> **Used
    by**: \[[58](#ref-zheng2021wwwdisentangeluserinterestConformity)\]

  - **[Netflix
    Prize](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)** \[[4](#ref-bennett2007netflix)\]:
    A movie rating dataset consisting of about 100 Million ratings for
    17,770 movies given by 480,189 users. Ratings consists of four
    entries: user, movie title, date of grade, and a grade ranging from
    1 to 5. Users and movies are represented with integer IDs.
    <span>&#10230;</span> **Used
    by**: \[[58](#ref-zheng2021wwwdisentangeluserinterestConformity)\]

  - **[WMT 14](https://www.statmt.org/wmt14/)** \[[6](#ref-bojar2014findings)\]:
    WMT[^1] is a yearly workshop in which researchers develop machine
    translation models for several different tasks. WMT14 was created
    for the event in 2014 and included a translation, a quality
    estimation, a metrics, and a medical translation task. Each category
    comprises different subtasks (e.g., translating between two specific
    languages). <span>&#10230;</span> **Used
    by**: \[[2](#ref-alvarez2017causal)\]

  - **[OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles-v2018.php)** \[[28](#ref-lison2016opensubtitles2016)\]:
    A text corpus comprising over 2.6 billion sentences from movie
    dialogues. The data stem from pre-processing 3,735,070 files from
    the online database *OpenSubtitles.org*[^2]. This corpus covers
    dialogues from ca. 2.8 million movies in 62 languages.
    <span>&#10230;</span> **Used by**: \[[2](#ref-alvarez2017causal)\]

  - **[LAMA](https://github.com/facebookresearch/LAMA)** \[[39](#ref-petroni2019language)\]:
    A probe designed to examine the factual and commonsense knowledge in
    pretrained language models. It is built upon four different,
    prominent corpora of facts that cover a wide range of knowledge
    types. <span>&#10230;</span> **Used
    by**: \[[8](#ref-cao2022aclpromtLMBiasCausal)\]

  - **[Comma.ai Driving
    Dataset](https://github.com/commaai/research)** \[[45](#ref-santana2016learning)\]:
    A video dataset made up of 11 video clips of variable size capturing
    the windshield view of an Acura ILX 2016. The driving data contains
    7.25 hours of footage, which was mostly recorded on highways. Each
    video is accompanied by measurements such as the car’s speed,
    acceleration, or steering angle. <span>&#10230;</span> **Used
    by**: \[[24](#ref-Kim_2017_ICCV)\]

  - **[Udacity Driving
    Dataset](https://github.com/udacity/self-driving-car)** \[[50](#ref-UdacityDriving)\]:
    A driving video dataset developed for the Udacity *Self-Driving Car
    Nanodegree Program*[^3]. The GitHub repository contains two
    annotated datasets in which computer vision systems have to label
    objects, such as cars or pedestrians, within driving footage.
    <span>&#10230;</span> **Used by**: \[[24](#ref-Kim_2017_ICCV)\]

  - **[T-REx](https://github.com/hadyelsahar/RE-NLG-Dataset)** \[[15](#ref-elsahar2018trex)\]:
    A dataset of large-scale alignments between Wikipedia abstracts and
    Wikidata triples. Such triples encode semantic information in the
    form of subject-predicate-object relationships. T-REx consists of 11
    million triples with 3.09 million Wikipedia abstracts (6.2 million
    sentences). <span>&#10230;</span> **Used
    by**: \[[27](#ref-li2022aclPTCaptureFactsCausal)\]

  - **[MNIST](http://yann.lecun.com/exdb/mnist/)** \[[26](#ref-lecun1998gradient)\]:
    An extraordinarily well-known and widely used image dataset
    comprising 28 \(\times\) 28 grayscale images of handwritten digits.
    It contains 60,000 training and 10,000 test samples.
    <span>&#10230;</span> **Used by**: \[[46](#ref-Schwab2019)\]

  - **[ImageNet](https://www.image-net.org/about.php)** \[[12](#ref-ImageNet2009)\]:
    Another well-known, more sophisticated image dataset containing more
    than 14 million images. The images depict more than 20,000 *synsets*
    (i.e., concepts "possibly described by multiple words or word
    phrases"[^4]). <span>&#10230;</span> **Used
    by**: \[[46](#ref-Schwab2019)\]

  - **[Adult (Census
    Income)](https://archive.ics.uci.edu/ml/datasets/adult)** \[[13](#ref-dheeru2017uci),[25](#ref-kohavi1996ADULTS)\]:
    A tabular dataset containing anonymized data from the 1994 Census
    bureau database.[^5] Classifiers try to predict whether a given
    person will earn over or under 50,000 USD worth of salary. Each
    person is described via 15 features (including their id), e.g.,
    gender, education, and occupation. <span>&#10230;</span> **Used
    by**: \[[17](#ref-NEURIPS2020_0d770c49),[33](#ref-Mahajan2019)\]

  - **[Human Activity
    Recognition](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)** \[[3](#ref-anguita2013public)\]:
    This dataset contains smartphone-recorded sensor data from 30
    subjects performing *Activities of Daily Living*. The database
    differentiates between the activities walking (upstairs, downstairs,
    or on the same level), sitting, standing, and laying.
    <span>&#10230;</span> **Used by**: \[[20](#ref-Janzing2020)\]

  - **[Yelp](https://www.yelp.com/dataset)** \[[54](#ref-YelpDataset)\]:
    A dataset of almost 7 million Yelp user reviews of around 150k
    businesses across 11 cities in the US and Canada. Review entries
    contain not only their associated text and an integer star rating
    between 1 and 5 but also additional information like the amount of
    *useful*, *funny*, and *cool* votes for the review.
    <span>&#10230;</span> **Used by**: \[[49](#ref-Tan2021)\]

  - **[Amazon (Product)
    Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)** \[[37](#ref-ni2019justifying)\]:
    An extensive dataset of 233.1 million Amazon reviews between May
    1996 and October 2018. The data include not only information about
    the review itself and product metadata (e.g., descriptions, price,
    product size, or package type) but also *also bought* and *also
    viewed* links. <span>&#10230;</span> **Used
    by**: \[[49](#ref-Tan2021)\]

  - **[Sangiovese
    Grapes](https://github.com/divyat09/cf-feasibility)** \[[32](#ref-magrini2017conditional)\]:
    A conditional linear Bayesian network that captures the effects of
    different canopy management techniques on the quality of Sangiovese
    grapes. Based on a two-year study of Tuscan Sangiovese grapes, the
    authors created a network with 14 features (13 of which are
    continuous variables). The data used for experiments in
    \[[33](#ref-Mahajan2019)\] are linked in their repository (see the
    link behind the term “Sangiovese Grapes”).
    <span>&#10230;</span> **Used by**: \[[33](#ref-Mahajan2019)\]

  - **[WikiText-2](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/)** \[[35](#ref-merity2016pointer)\]:
    An NLP benchmark containing over 100 million tokens extracted from
    verified Good and Featured articles on Wikipedia. Contrary to
    previous token collections, however, WikiText-2 is more extensive
    and comprises more realistic tokens (e.g., lower-case tokens).
    <span>&#10230;</span> **Used
    by**: \[[21](#ref-jeoung2022arxivDebiasCausalMediation)\]

  - **[Jigsaw Toxicity
    Detection](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)** \[[22](#ref-Jigsaw2019)\]:
    A dataset of comments made across around 50 English-language news
    sites built to analyze unintended bias in toxicity classification
    within a Kaggle competition organized by Jigsaw and Google. Each
    comment in the training set comes with a human-annotated toxicity
    label (e.g., obscene or threat) and labels for mentioned identities
    (e.g., gender, ethnicity, sexuality, or religion) in the comment.
    <span>&#10230;</span> **Used
    by**: \[[21](#ref-jeoung2022arxivDebiasCausalMediation)\]

  - **[RTGender](https://nlp.stanford.edu/robvoigt/rtgender/)** \[[52](#ref-voigt2018rtgender)\]:
    A collection of comments made on online content across different
    platforms such as Facebook or Reddit. Each post and comment is
    annotated with the gender of the author in order to analyze gender
    bias in social media. <span>&#10230;</span> **Used
    by**: \[[21](#ref-jeoung2022arxivDebiasCausalMediation)\]

  - **[CrowS-Pairs](https://github.com/nyu-mll/crows-pairs)** \[[36](#ref-nangia2020crows)\]:
    A benchmark designed to investigate the social bias of NLP models.
    Each entry consists of two sentences: one representing a
    stereotypical statement for a given bias type (e.g., religion or
    nationality) and an anti-stereotypical version of the statement,
    where the described group/identity was substituted.
    <span>&#10230;</span> **Used
    by**: \[[21](#ref-jeoung2022arxivDebiasCausalMediation)\]

  - **[Professions](https://github.com/sebastianGehrmann/CausalMediationAnalysis)** \[[51](#ref-DBLPconf/nips/VigGBQNSS20)\]:
    A set of templates (originating from \[[30](#ref-lu2020gender)\])
    that were augmented with professions from
    \[[7](#ref-bolukbasi2016man)\]. Each sentence template follows the
    pattern “The \[occupation\] \[verb\] because”, and each profession
    has a crowdsourced rating that describes its definitionality and
    stereotypicality. <span>&#10230;</span> **Used
    by**: \[[51](#ref-DBLPconf/nips/VigGBQNSS20)\]

  - **[WinoBias](https://uclanlp.github.io/corefBias/overview)** \[[56](#ref-WinoBias2018)\]:
    A collection of 3,160 WinoCoRef style sentences created to estimate
    gender bias within NLP models. Sentences come in pairs that only
    differ by the gender of one pronoun, with each sentence describing
    an interaction between two people with different occupations.
    <span>&#10230;</span> **Used
    by**: \[[51](#ref-DBLPconf/nips/VigGBQNSS20)\]

  - **[Winogender
    Schemas](https://github.com/rudinger/winogender-schemas)** \[[44](#ref-Winogender2018)\]:
    A Winograd-style collection of templates that generate pairs of
    sentences that only differ by the gender of one pronoun. Researchers
    can generate 720 different sentences by defining the building blocks
    *occupation*, *participant*, and *pronoun* as a benchmark for gender
    bias detection. <span>&#10230;</span> **Used
    by**: \[[51](#ref-DBLPconf/nips/VigGBQNSS20)\]

  - **[English UD
    Treebank](https://universaldependencies.org/en/)** \[[34](#ref-mcdonald2013universal)\]:
    The English UD Treebanks represents a subset of a data collection
    containing uniformly analyzed sentences across six different
    languages. The English treebank consists of 43,948 sentences and
    1,046,829 tokens. <span>&#10230;</span> **Used
    by**: \[[14](#ref-DBLPjournals/tacl/ElazarRJG21)\]

  - **[Gender-Neutral GloVe Word
    Embeddings](https://github.com/uclanlp/gn_glove)** \[[57](#ref-zhao2018learning)\]:
    This variant of GloVe produces gender-neutral word embeddings by
    maintaining all gender-related information exclusively in specific
    dimensions of word vectors. The resulting word embeddings can be a
    starting point for more unbiased NLP. <span>&#10230;</span> **Used
    by**: \[[40](#ref-ravfogel2020aclINLP),[41](#ref-ravfogel2022aclRLACE)\]

  - **[Biographies](https://github.com/Microsoft/biosbias)** \[[11](#ref-de2019bias)\]:
    A collection of 397,340 online biographies covering 28 occupations
    (e.g., professors, physicians, or rappers). Each biography is stored
    as a dictionary containing the title, the (binary) gender, the
    length of the first sentence, and the entire text of the biography.
    <span>&#10230;</span> **Used
    by**: \[[40](#ref-ravfogel2020aclINLP),[41](#ref-ravfogel2022aclRLACE)\]

  - **[TwitterAAE
    corpus](http://slanglab.cs.umass.edu/TwitterAAE/)** \[[5](#ref-blodgett2016demographic)\]:
    A collection of 59.2 million tweets sent out by 2.8 million users
    from the US in 2013. Each tweet is annotated with a vector
    describing the “likely demographics of the author and the
    neighborhood they live in.” \[[5](#ref-blodgett2016demographic)\]
    These demographic approximations of users were built upon US census
    data. <span>&#10230;</span> **Used
    by**: \[[40](#ref-ravfogel2020aclINLP)\]

  - **[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)** \[[29](#ref-liu2015faceattributes)\]:
    A face image dataset containing 202,599 images of size 178
    \(\times\) 218 from 10,177 unique celebrities. Each image is
    annotated with 40 binary facial attributes (e.g., *Is this person
    smiling?*) and five landmark positions describing the 2D position of
    the eyes, the nose, and the mouth (split into *left* and *right*
    side of the mouth). <span>&#10230;</span> **Used
    by**: \[[41](#ref-ravfogel2022aclRLACE)\]

</details>

<details>
<summary><h2>Interesting Causal Tools (Click to expand)</h2></summary>

  - **[CausalFS](https://github.com/kuiy/CausalFS)** \[[55](#ref-Yu2020)\]:
    An open-source package for C++ that contains 28 local causal
    structure learning methods for feature selection. It is specifically
    designed to facilitate the development and benchmarking of new
    causal feature selection techniques.

  - **[CEBaB](https://github.com/CEBaBing/CEBaB)** \[[1](#ref-abraham2022cebab)\]:
    A recently designed benchmark to estimate and compare the quality of
    concept-based explanation for NLP. CEBaB includes a set of
    restaurant reviews accompanied by human-generated counterfactuals,
    which enables researchers to investigate the model’s ability to
    assess causal concept effects.

  - **[CausaLM
    Datasets](https://github.com/amirfeder/CausaLM)** \[[16](#ref-feder2021coliCausaLM)\]:
    As part of the analysis of *CausaLM*, the authors developed four NLP
    datasets for evaluating causal explanations. These datasets
    represent real-world applications of ML that come with ground-truth
    information.

  - **Competing with Causal Toolboxes**: Several causal tools like
    *[YLearn](https://github.com/DataCanvasIO/YLearn)* \[[10](#ref-YLearn2022)\],
    *[DoWhy](https://github.com/py-why/dowhy)* \[[47](#ref-DoWhy_2019_Microsoft)\],
    *[CausalML](https://github.com/uber/causalml)* \[[9](#ref-Chen2020_CausalML_Uber)\],
    or
    *[EconML](https://github.com/microsoft/EconML)* \[[23](#ref-EconML_2019_Microsoft)\]
    introduce an entire causal inference pipeline with their own
    interpreter module. Comparing newly developed interpretation
    techniques with such packages could be very insightful.

</details>

<details>
<summary><h2>Prominent Non-Causal Tools (Click to expand)</h2></summary>

  - **[LIME](https://github.com/marcotcr/lime)** \[[43](#ref-ribeiro2016LIME)\]:
    A very prominent Python package that allows researchers to explain
    individual predictions of image, text, and tabular data classifiers.
    Applicable to any black-box classifier that implements a function
    that outputs class probabilities given raw text or a NumPy array.

  - **[ROAR](https://github.com/google-research/google-research/tree/master/interpretability_benchmark)** \[[19](#ref-hooker2019ROAR)\]:
    A benchmark method that evaluates interpretability approaches based
    on how well they quantify feature importance. The technique was used
    to assess model explanations of image classifiers over multiple
    datasets.

  - **[SHAP](https://github.com/slundberg/shap)** \[[31](#ref-lundberg2017unified)\]:
    Another well-known interpretability package which is based on game
    theory. Although compatible with any ML model, SHAP comes with a
    C++-based algorithm for tree ensemble algorithms such as XGBoost.

  - **[InterpretML](https://github.com/interpretml/interpret)** \[[38](#ref-nori2019interpretml)\]:
    An open-source package developed by Microsoft that includes multiple
    state-of-the-art methods for model interpretability. It also allows
    users to train an *Explainable Boosting Machine* (EBM) - a model
    that provides exact explanations and performs as well as random
    forests and gradient-boosted trees.

</details>

<div id="refs" class="references">

## References

<div id="ref-abraham2022cebab">

\[1\] Eldar David Abraham, Karel D’Oosterlinck, Amir Feder, Yair Ori
Gat, Atticus Geiger, Christopher Potts, Roi Reichart, and Zhengxuan Wu.
2022. CEBaB: Estimating the causal effects of real-world concepts on nlp
model behavior. *arXiv preprint arXiv:2205.14140* (2022).

</div>

<div id="ref-alvarez2017causal">

\[2\] David Alvarez-Melis and Tommi S Jaakkola. 2017. A causal framework
for explaining the predictions of black-box sequence-to-sequence models.
*arXiv preprint* (2017).

</div>

<div id="ref-anguita2013public">

\[3\] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra Perez,
and Jorge Luis Reyes Ortiz. 2013. A public domain dataset for human
activity recognition using smartphones. In *Proceedings of the 21th
international european symposium on artificial neural networks,
computational intelligence and machine learning*, 437–442.

</div>

<div id="ref-bennett2007netflix">

\[4\] James Bennett, Stan Lanning, and others. 2007. The netflix prize.
In *Proceedings of kdd cup and workshop*, New York, NY, USA., 35.

</div>

<div id="ref-blodgett2016demographic">

\[5\] Su Lin Blodgett, Lisa Green, and Brendan O’Connor. 2016.
Demographic dialectal variation in social media: A case study of
african-american english. *arXiv preprint arXiv:1608.08868* (2016).

</div>

<div id="ref-bojar2014findings">

\[6\] Ondřej Bojar, Christian Buck, Christian Federmann, Barry Haddow,
Philipp Koehn, Johannes Leveling, Christof Monz, Pavel Pecina, Matt
Post, Herve Saint-Amand, and others. 2014. Findings of the 2014 workshop
on statistical machine translation. In *Proceedings of the ninth
workshop on statistical machine translation*, 12–58.

</div>

<div id="ref-bolukbasi2016man">

\[7\] Tolga Bolukbasi, Kai-Wei Chang, James Y Zou, Venkatesh Saligrama,
and Adam T Kalai. 2016. Man is to computer programmer as woman is to
homemaker? Debiasing word embeddings. *Advances in neural information
processing systems* 29, (2016).

</div>

<div id="ref-cao2022aclpromtLMBiasCausal">

\[8\] Boxi Cao, Hongyu Lin, Xianpei Han, Fangchao Liu, and Le Sun. 2022.
Can prompt probe pretrained language models? Understanding the invisible
risks from a causal view. In *ACL*.

</div>

<div id="ref-Chen2020_CausalML_Uber">

\[9\] Huigang Chen, Totte Harinen, Jeong-Yoon Lee, Mike Yung, and Zhenyu
Zhao. 2020. Causalml: Python package for causal machine learning. *arXiv
preprint arXiv:2002.11631* (2020).

</div>

<div id="ref-YLearn2022">

\[10\] DataCanvas. 2022. YLearn. *GitHub repository*.

</div>

<div id="ref-de2019bias">

\[11\] Maria De-Arteaga, Alexey Romanov, Hanna Wallach, Jennifer Chayes,
Christian Borgs, Alexandra Chouldechova, Sahin Geyik, Krishnaram
Kenthapadi, and Adam Tauman Kalai. 2019. Bias in bios: A case study of
semantic representation bias in a high-stakes setting. In *Proceedings
of the conference on fairness, accountability, and transparency*,
120–128.

</div>

<div id="ref-ImageNet2009">

\[12\] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li
Fei-Fei. 2009. ImageNet: A large-scale hierarchical image database. In
*2009 ieee conference on computer vision and pattern recognition*,
248–255. DOI:[https://doi.org/10.1109/CVPR.2009.5206848
](https://doi.org/10.1109/CVPR.2009.5206848%20%20%20)

</div>

<div id="ref-dheeru2017uci">

\[13\] Dua Dheeru and E Karra Taniskidou. 2017. UCI machine learning
repository. (2017).

</div>

<div id="ref-DBLPjournals/tacl/ElazarRJG21">

\[14\] Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg.
2021. Amnesic probing: Behavioral explanation with amnesic
counterfactuals. *Trans. Assoc. Comput. Linguistics* (2021).

</div>

<div id="ref-elsahar2018trex">

\[15\] Hady Elsahar, Pavlos Vougiouklis, Arslen Remaci, Christophe
Gravier, Jonathon Hare, Frederique Laforest, and Elena Simperl. 2018.
T-rex: A large scale alignment of natural language with knowledge base
triples. In *Proceedings of the eleventh international conference on
language resources and evaluation (lrec 2018)*.

</div>

<div id="ref-feder2021coliCausaLM">

\[16\] Amir Feder, Nadav Oved, Uri Shalit, and Roi Reichart. 2021.
CausaLM: Causal model explanation through counterfactual language
models. *Comput. Linguistics* (2021).

</div>

<div id="ref-NEURIPS2020_0d770c49">

\[17\] Christopher Frye, Colin Rowat, and Ilya Feige. 2020. Asymmetric
shapley values: Incorporating causal knowledge into model-agnostic
explainability. In *NeurIPS*.

</div>

<div id="ref-MovieLens2015">

\[18\] F. Maxwell Harper and Joseph A. Konstan. 2015. The movielens
datasets: History and context. *ACM Trans. Interact. Intell. Syst.* 5, 4
(December 2015). DOI:[https://doi.org/10.1145/2827872
](https://doi.org/10.1145/2827872%20)

</div>

<div id="ref-hooker2019ROAR">

\[19\] Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, and Been Kim.
2019. A benchmark for interpretability methods in deep neural networks.
In *Advances in neural information processing systems 32*, H. Wallach,
H. Larochelle, A. Beygelzimer, F. dAlché-Buc, E. Fox and R. Garnett
(eds.). Curran Associates, Inc., 9737–9748. Retrieved from
<http://papers.nips.cc/paper/9167-a-benchmark-for-interpretability-methods-in-deep-neural-networks.pdf>

</div>

<div id="ref-Janzing2020">

\[20\] Dominik Janzing, Lenon Minorics, and Patrick Blöbaum. 2020.
Feature relevance quantification in explainable ai: A causal problem.
*AISTATS* (2020).

</div>

<div id="ref-jeoung2022arxivDebiasCausalMediation">

\[21\] Sullam Jeoung and Jana Diesner. 2022. What changed? Investigating
debiasing methods using causal mediation analysis. *CoRR* (2022).

</div>

<div id="ref-Jigsaw2019">

\[22\] Jigsaw. 2019. Jigsaw Unintended Bias in Toxicity Classification.
Retrieved from
<https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification>

</div>

<div id="ref-EconML_2019_Microsoft">

\[23\] Maggie Hei Keith Battocchi Eleanor Dillon. 2019. EconML: A Python
Package for ML-Based Heterogeneous Treatment Effects Estimation. 

</div>

<div id="ref-Kim_2017_ICCV">

\[24\] Jinkyu Kim and John Canny. 2017. Interpretable learning for
self-driving cars by visualizing causal attention. In *ICCV*.

</div>

<div id="ref-kohavi1996ADULTS">

\[25\] Ron Kohavi and others. 1996. Scaling up the accuracy of
naive-bayes classifiers: A decision-tree hybrid. In *Kdd*, 202–207.

</div>

<div id="ref-lecun1998gradient">

\[26\] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner.
1998. Gradient-based learning applied to document recognition.
*Proceedings of the IEEE* 86, 11 (1998), 2278–2324.

</div>

<div id="ref-li2022aclPTCaptureFactsCausal">

\[27\] Shaobo Li, Xiaoguang Li, Lifeng Shang, Zhenhua Dong, Chengjie
Sun, Bingquan Liu, Zhenzhou Ji, Xin Jiang, and Qun Liu. 2022. How
pre-trained language models capture factual knowledge? A causal-inspired
analysis. In *ACL*.

</div>

<div id="ref-lison2016opensubtitles2016">

\[28\] Pierre Lison and Jörg Tiedemann. 2016. Opensubtitles2016:
Extracting large parallel corpora from movie and tv subtitles. (2016).

</div>

<div id="ref-liu2015faceattributes">

\[29\] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. 2015. Deep
learning face attributes in the wild. In *Proceedings of international
conference on computer vision (iccv)*.

</div>

<div id="ref-lu2020gender">

\[30\] Kaiji Lu, Piotr Mardziel, Fangjing Wu, Preetam Amancharla, and
Anupam Datta. 2020. Gender bias in neural natural language processing.
In *Logic, language, and security*. Springer, 189–202.

</div>

<div id="ref-lundberg2017unified">

\[31\] Scott M Lundberg and Su-In Lee. 2017. A unified approach to
interpreting model predictions. *Advances in neural information
processing systems* 30, (2017).

</div>

<div id="ref-magrini2017conditional">

\[32\] Alessandro Magrini, Stefano Di Blasi, Federico Mattia Stefanini,
and others. 2017. A conditional linear gaussian network to assess the
impact of several agronomic settings on the quality of tuscan sangiovese
grapes. *Biometrical Letters* 54, 1 (2017), 25–42.

</div>

<div id="ref-Mahajan2019">

\[33\] Divyat Mahajan, Chenhao Tan, and Amit Sharma. 2019. Preserving
causal constraints in counterfactual explanations for machine learning
classifiers. (December 2019). Retrieved from
<http://arxiv.org/abs/1912.03277>

</div>

<div id="ref-mcdonald2013universal">

\[34\] Ryan McDonald, Joakim Nivre, Yvonne Quirmbach-Brundage, Yoav
Goldberg, Dipanjan Das, Kuzman Ganchev, Keith Hall, Slav Petrov, Hao
Zhang, Oscar Täckström, and others. 2013. Universal dependency
annotation for multilingual parsing. In *Proceedings of the 51st annual
meeting of the association for computational linguistics (volume 2:
Short papers)*, 92–97.

</div>

<div id="ref-merity2016pointer">

\[35\] Stephen Merity, Caiming Xiong, James Bradbury, and Richard
Socher. 2016. Pointer sentinel mixture models. *arXiv preprint
arXiv:1609.07843* (2016).

</div>

<div id="ref-nangia2020crows">

\[36\] Nikita Nangia, Clara Vania, Rasika Bhalerao, and Samuel R Bowman.
2020. CrowS-pairs: A challenge dataset for measuring social biases in
masked language models. *arXiv preprint arXiv:2010.00133* (2020).

</div>

<div id="ref-ni2019justifying">

\[37\] Jianmo Ni, Jiacheng Li, and Julian McAuley. 2019. Justifying
recommendations using distantly-labeled reviews and fine-grained
aspects. In *Proceedings of the 2019 conference on empirical methods in
natural language processing and the 9th international joint conference
on natural language processing (emnlp-ijcnlp)*, 188–197.

</div>

<div id="ref-nori2019interpretml">

\[38\] Harsha Nori, Samuel Jenkins, Paul Koch, and Rich Caruana. 2019.
InterpretML: A unified framework for machine learning interpretability.
*arXiv preprint arXiv:1909.09223* (2019).

</div>

<div id="ref-petroni2019language">

\[39\] Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin,
Yuxiang Wu, Alexander H Miller, and Sebastian Riedel. 2019. Language
models as knowledge bases? *arXiv preprint arXiv:1909.01066* (2019).

</div>

<div id="ref-ravfogel2020aclINLP">

\[40\] Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and
Yoav Goldberg. 2020. Null it out: Guarding protected attributes by
iterative nullspace projection. In *ACL*.

</div>

<div id="ref-ravfogel2022aclRLACE">

\[41\] Shauli Ravfogel, Michael Twiton, Yoav Goldberg, and Ryan
Cotterell. 2022. Linear adversarial concept erasure. In *ICML*.

</div>

<div id="ref-reddy_2022_AAAI_candle_disentangled_dataset">

\[42\] Abbavaram Gowtham Reddy, Benin Godfrey L, and Vineeth N.
Balasubramanian. 2022. On causally disentangled representations. In
*AAAI*.

</div>

<div id="ref-ribeiro2016LIME">

\[43\] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. 2016. "
Why should i trust you?" Explaining the predictions of any classifier.
In *Proceedings of the 22nd acm sigkdd international conference on
knowledge discovery and data mining*, 1135–1144.

</div>

<div id="ref-Winogender2018">

\[44\] Rachel Rudinger, Jason Naradowsky, Brian Leonard, and Benjamin
Van Durme. 2018. Gender bias in coreference resolution. *arXiv preprint
arXiv:1804.09301* (2018).

</div>

<div id="ref-santana2016learning">

\[45\] Eder Santana and George Hotz. 2016. Learning a driving simulator.
*arXiv preprint arXiv:1608.01230* (2016).

</div>

<div id="ref-Schwab2019">

\[46\] Patrick Schwab and Walter Karlen. 2019. CXPlain: Causal
explanations for model interpretation under uncertainty. *NeurIPS*
(2019).

</div>

<div id="ref-DoWhy_2019_Microsoft">

\[47\] Amit Sharma, Emre Kiciman, and others. 2019. DoWhy: A Python
package for causal inference. 

</div>

<div id="ref-si2022wwwcausalRecSearch">

\[48\] Zihua Si, Xueran Han, Xiao Zhang, Jun Xu, Yue Yin, Yang Song, and
Ji-Rong Wen. 2022. A model-agnostic causal learning framework for
recommendation using search data. In *WWW*.

</div>

<div id="ref-Tan2021">

\[49\] Juntao Tan, Shuyuan Xu, Yingqiang Ge, Yunqi Li, Xu Chen, and
Yongfeng Zhang. 2021. Counterfactual explainable recommendation. In
*International Conference on Information and Knowledge Management,
Proceedings*.

</div>

<div id="ref-UdacityDriving">

\[50\] Udacity. 2016. Self-driving car. *GitHub repository*.

</div>

<div id="ref-DBLPconf/nips/VigGBQNSS20">

\[51\] Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian,
Daniel Nevo, Yaron Singer, and Stuart M. Shieber. 2020. Investigating
gender bias in language models using causal mediation analysis. In
*NeurIPS*.

</div>

<div id="ref-voigt2018rtgender">

\[52\] Rob Voigt, David Jurgens, Vinodkumar Prabhakaran, Dan Jurafsky,
and Yulia Tsvetkov. 2018. RtGender: A corpus for studying differential
responses to gender. In *Proceedings of the eleventh international
conference on language resources and evaluation (lrec 2018)*.

</div>

<div id="ref-wu2020mind">

\[53\] Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi,
Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu, and
others. 2020. Mind: A large-scale dataset for news recommendation. In
*Proceedings of the 58th annual meeting of the association for
computational linguistics*, 3597–3606.

</div>

<div id="ref-YelpDataset">

\[54\] Yelp. Yelp Open Dataset. 

</div>

<div id="ref-Yu2020">

\[55\] Kui Yu, Xianjie Guo, Lin Liu, Jiuyong Li, Hao Wang, Zhaolong
Ling, and Xindong Wu. 2020. Causality-based feature selection: Methods
and evaluations. *ACM Computing Surveys* (2020).

</div>

<div id="ref-WinoBias2018">

\[56\] Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and
Kai-Wei Chang. 2018. Gender bias in coreference resolution: Evaluation
and debiasing methods. *arXiv preprint arXiv:1804.06876* (2018).

</div>

<div id="ref-zhao2018learning">

\[57\] Jieyu Zhao, Yichao Zhou, Zeyu Li, Wei Wang, and Kai-Wei Chang.
2018. Learning gender-neutral word embeddings. *arXiv preprint
arXiv:1809.01496* (2018).

</div>

<div id="ref-zheng2021wwwdisentangeluserinterestConformity">

\[58\] Yu Zheng, Chen Gao, Xiang Li, Xiangnan He, Yong Li, and Depeng
Jin. 2021. Disentangling user interest and conformity for recommendation
with causal embedding. In *WWW*.

</div>

</div>

[^1]:  <https://machinetranslate.org/wmt>

[^2]:  <https://www.opensubtitles.org/>

[^3]:  <https://udacity.com/self-driving-car>

[^4]:  <https://www.image-net.org/about.php>

[^5]:  <http://www.census.gov/en.html>
