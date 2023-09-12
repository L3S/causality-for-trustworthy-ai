# Robustness

## Aspects of Trustworthy AI and Application Domain


### [Interpretability](../Interpretability/README.md)

### [Fairness](../Fairness/README.md)

### Robustness

### [Privacy](../Privacy/README.md)

### [Auditing (Safety and Accountability)](../Auditing/README.md)

### [Healthcare](../Healthcare/README.md)

<details>
<summary><h2>Datasets Used by Cited Publications (Click to expand)</h2></summary>

  - **[Rotated
    MNIST](https://github.com/ghif/mtae)** \[[18](#ref-ghifary2015domain)\]:
    The dataset consists of MNIST images with each domain containing
    images rotated by a particular angle
    \(0°, 15°, 30°, 45°, 60°, 75°\)
    <span>&#10230;</span> **Used
    by**: \[[24](#ref-Ilse_2021_simulatinginterventions)\]

  - **[ColoredMNIST](https://github.com/facebookresearch/InvariantRiskMinimization)** \[[5](#ref-arjovsky2019invariant)\]:
    The dataset consists of input images with digits 0-4 colored red and
    labelled 0 while digits 5-9 are colored green representing the two
    domains. <span>&#10230;</span> **Used
    by**: \[[5](#ref-arjovsky2019invariant),[24](#ref-Ilse_2021_simulatinginterventions),[41](#ref-lu2021invariant)\]

  - **[PACS](https://github.com/facebookresearch/DomainBed)** \[[35](#ref-li2017deeper)\]:
    An image classification dataset categorized into 10 classes that are
    scattered across four different domains, each having a distinct
    trait: photograph, art, cartoon and sketch.
    <span>&#10230;</span> **Used
    by**: \[[24](#ref-Ilse_2021_simulatinginterventions)\]

  - **[Amazon (Product)
    Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)** \[[44](#ref-ni2019justifying)\]:
    An extensive dataset of 233.1 million Amazon reviews between May
    1996 and October 2018. The data include not only information about
    the review itself and product metadata (e.g., descriptions, price,
    product size, or package type) but also *also bought* and *also
    viewed* links. <span>&#10230;</span> **Used
    by**[^1]: \[[27](#ref-kaushik2020explaining),[62](#ref-wang2021enhancing)\]

  - **[SemEval-2017
    Task 4](https://alt.qcri.org/semeval2017/task4/)** \[[54](#ref-rosenthal-etal-2017-semeval)\]:
    SemEval[^2] is a yearly NLP workshop where participants compete on
    different sentiment analysis tasks. Each workshop comes with its own
    set of tasks to solve. In SemEval-2017 Task 4, NLP models compete on
    sentiment analysis tasks on English and Arabic Twitter data.
    <span>&#10230;</span> **Used
    by**: \[[27](#ref-kaushik2020explaining)\]

  - **[Yelp](https://www.yelp.com/dataset)** \[[66](#ref-YelpDataset)\]:
    A dataset of almost 7 million Yelp user reviews of around 150k
    businesses across 11 cities in the US and Canada. Review entries
    contain not only their associated text and an integer star rating
    between 1 and 5 but also additional information like the amount of
    *useful*, *funny*, and *cool* votes for the review.
    <span>&#10230;</span> **Used
    by**: \[[27](#ref-kaushik2020explaining)\]

  - **[IMDb
    extension](https://github.com/acmi-lab/counterfactually-augmented-data)** \[[26](#ref-kaushik2019learning)\]:
    A set of 2440 IMDb reviews, where a human-annotated counterfactual
    example accompanies each review. The human annotators were found
    through Amazon’s crowdsourcing platform *Mechanical Turk*[^3]. The
    dataset is designed to assess the performance of sentiment analysis
    and natural language inference models. <span>&#10230;</span> **Used
    by**: \[[27](#ref-kaushik2020explaining),[58](#ref-teney2020learning),[62](#ref-wang2021enhancing)\]

  - **[SNLI
    extension](https://github.com/acmi-lab/counterfactually-augmented-data)** \[[26](#ref-kaushik2019learning)\]:
    The original SNLI dataset \[[8](#ref-bowman2015large)\] is a text
    dataset developed to evaluate natural language inference (NLI)
    models. Models must decide whether a given hypothesis is
    contradictory to, entailed by, or neutral to the given premise.
     \[[26](#ref-kaushik2019learning)\] extended this dataset via
    humanly-manufactured counterfactual examples.
    <span>&#10230;</span> **Used
    by**: \[[27](#ref-kaushik2020explaining),[58](#ref-teney2020learning)\]

  - **Parkinson’s voice data** \[[38](#ref-little2019causal)\]: A set of
    extracted features from audio samples (i.e., sustained phonations)
    of patients with Parkinson’s disease and people from healthy control
    groups. This dataset combines data from three different and
    independent labs from the
    [US](https://archive.ics.uci.edu/ml/datasets/Parkinsons),
    [Turkey](http://archive.ics.uci.edu/ml/datasets/Parkinson's+Disease+Classification),
    and
    [Spain](https://archive.ics.uci.edu/ml/datasets/Parkinson+Dataset+with+replicated+acoustic+features+).
    The classification task is to detect patients with Parkinson’s
    disease. <span>&#10230;</span> **Used
    by**: \[[38](#ref-little2019causal)\]

  - **[DomainNet](http://ai.bu.edu/M3SDA/)** \[[47](#ref-DomainNet_2019)\]:
    An unsupervised domain adaptation image dataset containing six
    domains (referring to the “style” of the image, e.g., sketch, quick
    drawing or real image) and about ca. 600k images distributed among
    345 categories.

  - **[ImageNet](https://www.image-net.org/about.php)** \[[13](#ref-ImageNet2009)\]:
    Another well-known, more sophisticated image dataset containing more
    than 14 million images. The images depict more than 20,000 *synsets*
    (i.e., concepts "possibly described by multiple words or word
    phrases"[^4]). <span>&#10230;</span> **Used
    by**: \[[42](#ref-Mao_2021_generative),[43](#ref-Mitrovic_2020_relic)\]

  - **[ImageNet-C](https://github.com/hendrycks/robustness)** \[[23](#ref-hendrycks2019benchmarking)\]:
    This dataset tests the model’s robustness by applying corruptions to
    validation images of ImageNet. Each of the 15 corruption types
    (e.g., gaussian noise, snow, motion blur, or contrast) comes with
    five levels of corruption intensity. <span>&#10230;</span> **Used
    by**: \[[42](#ref-Mao_2021_generative)\]

  - **[ImageNet-V2](https://github.com/modestyachts/ImageNetV2)** \[[50](#ref-recht2019imagenet)\]:
    A new test set for ImageNet designed to assess the model’s
    generalization ability. Despite closely following the original
    dataset creation process, models trained on the original ImageNet
    demonstrate worse performance on ImageNet-V2. ImageNet models with
    better generalization should perform stably on both variants.
    <span>&#10230;</span> **Used by**: \[[42](#ref-Mao_2021_generative)\]

  - **[ObjectNet](https://objectnet.dev/)** \[[6](#ref-barbu2019objectnet)\]:
    An image dataset designed to demonstrate the transfer learning
    ability of ImageNet models. Due to this, ObjectNet provides no
    training set. Instead, all 50,000 images of ObjectNet combine into a
    single test set. Each image depicts an object with random
    backgrounds, viewpoints, and rotations of the object.
    <span>&#10230;</span> **Used by**: \[[42](#ref-Mao_2021_generative)\]

  - **[ImageNet
    ILSVRC-2012](https://www.tensorflow.org/datasets/catalog/imagenet2012)** \[[55](#ref-russakovsky2015imagenet)\]:
    The dataset used for the ImageNet Large Scale Visual Recognition
    Challenge 2012 (ILSVRC2012). The 1.5 million images depict objects
    from 1,000 different synsets. <span>&#10230;</span> **Used
    by**: \[[43](#ref-Mitrovic_2020_relic)\]

  - **[ImageNet-R](https://github.com/hendrycks/imagenet-r)** \[[22](#ref-hendrycks2021many)\]:
    A variation of ImageNet designed to evaluate the susceptibility to
    spurious correlations of ImageNet models. It includes 30,000
    artistic renditions (e.g., paintings, origami, or sculptures) of 200
    ImageNet object classes. The images were primarily collected from
    *Flickr*[^5]. <span>&#10230;</span> **Used
    by**: \[[43](#ref-Mitrovic_2020_relic)\]

  - **[The Arcade Learning Environment
    (ALE)](https://github.com/mgbellemare/Arcade-Learning-Environment)** \[[7](#ref-bellemare13arcade)\]:
    A suite of Atari 2600 games that allows researchers to develop AI
    agents (mostly RL agents) for more than 100 games. ALE supports
    OpenAI gym, Python, and C++ and provides researchers with a plethora
    of features to evaluate different agents.
    <span>&#10230;</span> **Used
    by**: \[[19](#ref-goyal2021recurrent),[43](#ref-Mitrovic_2020_relic)\]

  - **[CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)** \[[30](#ref-krizhevsky2009learning)\]:
    The two CIFAR datasets, CIFAR-10 and CIFAR-100, are labeled images
    stemming from the now withdrawn Tiny Images dataset[^6]. The more
    prominent set, CIFAR-10, contains 60000 32 &times; 32 color
    images separated into ten mutually exclusive classes, with 6000
    images per class. CIFAR-100 is simply a 100-class version of
    CIFAR-10. <span>&#10230;</span> **Used
    by**: \[[25](#ref-ilyas2022datamodels),[69](#ref-zhang2021causaladv)\]

  - **[Functional Map of the World
    (FMoW)](https://github.com/fMoW/dataset)** \[[11](#ref-christie2018functional)\]:
    A collection of over 1 million satellite images depicting more than
    200 countries. Each satellite contains at least one of 63 box
    annotations categorizing visible landmarks, such as *flooded road*
    or *airport*. <span>&#10230;</span> **Used
    by**: \[[25](#ref-ilyas2022datamodels)\]

  - **[Chemical
    Environment](https://github.com/dido1998/CausalMBRL)** \[[28](#ref-ke2021systematic)\]:
    This synthetic environment was designed to evaluate causal
    reinforcement learning (RL) agents exhaustively. In this task,
    agents must change the colors of a given set of objects. However,
    altering one object influences the color of other objects. The
    causal dynamics are set by either a user-defined causal graph or a
    randomly generated DAG. <span>&#10230;</span> **Used
    by**: \[[63](#ref-Wang_2022_CDL)\]

  - **[robosuite](https://github.com/ARISE-Initiative/robosuite)** \[[70](#ref-zhu2020robosuite)\]:
    A simulation framework built upon the MuJoCo physics engine allowing
    researchers to simulate contact dynamics for robot learning tasks.
    Given a set of cubes, RL agents must maneuver a robotic arm to solve
    different tasks (e.g., stacking the cubes or lifting one to a
    specified height). <span>&#10230;</span> **Used
    by**: \[[63](#ref-Wang_2022_CDL)\]

  - **[COMPAS Recidivism
    Risk](https://github.com/propublica/compas-analysis/)** \[[4](#ref-angwin2016COMPAS)\]:
    A set of criminological datasets published by ProPublica to evaluate
    the bias of COMPAS - an algorithm used to assess the likelihood of
    criminal defendants reoffending. All COMPAS-related datasets include
    data from over 10,000 defendants, each being described via 52
    features (e.g., age, sex, race) and with a label indicating whether
    they were rearrested within two years. <span>&#10230;</span> **Used
    by**: \[[15](#ref-dominguez2022adversarial)\]

  - **[Adult (Census
    Income)](https://archive.ics.uci.edu/ml/datasets/adult)** \[[14](#ref-dheeru2017uci),[29](#ref-kohavi1996ADULTS)\]:
    A tabular dataset containing anonymized data from the 1994 Census
    bureau database.[^7] Classifiers try to predict whether a given
    person will earn over or under 50,000 USD worth of salary. Each
    person is described via 15 features (including their id), e.g.,
    gender, education, and occupation. <span>&#10230;</span> **Used
    by**: \[[15](#ref-dominguez2022adversarial)\]

  - **[South German
    Credit](https://archive.ics.uci.edu/ml/datasets/South+German+Credit+%28UPDATE%29)** \[[20](#ref-groemping2019south)\]:
    Designed as a successor to the German Credit dataset, this dataset
    contains 1000 credit scoring entries from a south german bank
    between 1973 and 1975. Each row contains 20 columns (e.g., savings,
    job, and credit history) based on which models must assess the risk
    of granting credit. <span>&#10230;</span> **Used
    by**: \[[15](#ref-dominguez2022adversarial)\]

  - **[Bail
    (DATA 1978)](https://github.com/RicardoDominguez/AdversariallyRobustRecourse/tree/main/data)** \[[57](#ref-schmidt1988predicting)\]:
    A collection of criminal records from 9,327 individuals that were
    released from a North Carolina prison between 1977 and 1978. This
    dataset was created to investigate factors that influence the
    likelihood of recidivism. Each record contains 19 variables,
    including a binary ethnicity variable (black or not black) and a
    variable indicating previous use of hard drugs.
    <span>&#10230;</span> **Used
    by**: \[[15](#ref-dominguez2022adversarial)\]

  - **[Colored
    FashionMNIST](https://github.com/IBM/OoD/tree/master/IRM_games)** \[[3](#ref-ahuja2020invariant)\]:
    This dataset was inspired by \[[5](#ref-arjovsky2019invariant)\]
    Colored MNIST dataset. \[[3](#ref-ahuja2020invariant)\] use the same
    coloring approach to induce spurious correlations into FashionMNIST
    data (greyscaled Zalando articles). <span>&#10230;</span> **Used
    by**: \[[41](#ref-lu2021invariant)\]

  - **[VLCS](https://github.com/belaalb/G2DM#download-vlcs)** \[[59](#ref-torralba2011unbiased)\]:
    A collection of 10,729 images from four standard datasets designed
    to evaluate the OOD performance of image classifiers. Each image
    depicts a bird, car, chair, dog, or person.
    <span>&#10230;</span> **Used by**: \[[41](#ref-lu2021invariant)\]

  - **[VQA-CP](https://www.iro.umontreal.ca/~agrawal/vqa-cp/)** \[[1](#ref-agrawal2018don)\]:
    A dataset for Visual Question Answering (VQA) models that actively
    punishes the use of spurious correlations. This is achieved by
    rearranging the VQA v1 and VQA v2 data splits. The resulting
    training and test data differ in the "distribution of answers per
    question type". <span>&#10230;</span> **Used
    by**: \[[58](#ref-teney2020learning)\]

  - **[COCO](https://cocodataset.org/#download)** \[[37](#ref-lin2014microsoft)\]:
    An object detection dataset containing 328k images that depict 91
    different types of objects. Each object within an image has its
    unique annotation, leading to more than 2.5 million labels across
    the entire dataset. <span>&#10230;</span> **Used
    by**: \[[58](#ref-teney2020learning)\]

  - **[Law School Admission
    Data](http://www.seaphe.org/databases.php)** \[[48](#ref-LawSchoolAdmission)\]:
    A tabular dataset of admission data from 25 US law schools between
    2005 and 2007. This dataset contains information from more than
    100,000 applicants (e.g., gender, ethnic group, LSAT score), with
    each entry having a binary admission status variable.
    <span>&#10230;</span> **Used by**: \[[62](#ref-wang2021enhancing)\]

  - **[MNIST](http://yann.lecun.com/exdb/mnist/)** \[[33](#ref-lecun1998gradient)\]:
    An extraordinarily well-known and widely used image dataset
    comprising 28 &times; 28 grayscale images of handwritten digits.
    It contains 60,000 training and 10,000 test samples.
    <span>&#10230;</span> **Used
    by**: \[[38](#ref-little2019causal),[67](#ref-zhang2020causal),[69](#ref-zhang2021causaladv)\]

  - **[Sequential MNIST Resolution
    Task](https://github.com/teganmaharaj/zoneout)** \[[31](#ref-krueger2016zoneout)\]:
    A sequential version of MNIST, where pixels of an handwritten digit
    are shown one at a time. <span>&#10230;</span> **Used
    by**: \[[19](#ref-goyal2021recurrent)\]

  - **[Bouncing
    Ball](https://github.com/sjoerdvansteenkiste/Relational-NEM)** \[[60](#ref-van2018relational)\]:
    A simulation environment where multiple balls of different sizes and
    weights independently move according to Newtonian physics. This
    environment is used to assess the model’s physical reasoning
    capabilities under different conditions (e.g., different amounts of
    balls). <span>&#10230;</span> **Used
    by**: \[[19](#ref-goyal2021recurrent)\]

  - **[BabyAI](https://github.com/mila-iqia/babyai)** \[[10](#ref-babyai_iclr19)\]:
    A RL framework that supports the development of agents that can
    understand language instructions. For this purpose, the authors
    developed agents that simulate human experts capable of
    communicating with task-solving agents using synthetic natural
    language. The platform provides 19 levels to alter the difficulty of
    the task. <span>&#10230;</span> **Used
    by**: \[[19](#ref-goyal2021recurrent)\]

  - **ETH and
    UCY** \[[16](#ref-ess2007eth),[34](#ref-lerner2007crowds)\]: Both
    *[ETH](https://data.vision.ee.ethz.ch/cvl/aess/dataset/)* \[[16](#ref-ess2007eth)\]
    and
    *[UCY](https://github.com/CHENGY12/CausalHTP)* \[[34](#ref-lerner2007crowds)\]
    are datasets containing real-world pedestrian trajectories. More
    novel papers combine both datasets to simulate multiple training and
    testing environments. Together, they contain trajectories of 1536
    detected pedestrians collected from five locations.
    <span>&#10230;</span> **Used
    by**: \[[9](#ref-Chen_2021_posthoc),[39](#ref-liu2022towards)\]

  - **[Stanford Drone
    dataset](https://cvgl.stanford.edu/projects/uav_data/)** \[[53](#ref-robicquet2016learning)\]:
    A video dataset containing over 100 top-view scenes of the Stanford
    University campus that were shot with a quadcopter. The videos
    depict 20,000 manually annotated targets (e.g., pedestrians,
    bicyclists, or cars). <span>&#10230;</span> **Used
    by**: \[[9](#ref-Chen_2021_posthoc),[39](#ref-liu2022towards)\]

  - **[Waterbirds](https://github.com/kohpangwei/group_DRO)** \[[56](#ref-sagawa2019distributionally)\]:
    A binary image classification task where models must decide whether
    the depicted bird is a waterbird or a landbird. Good-performing
    models must rely on something other than the intrinsic spurious
    correlation between the background and the label (e.g., only 56 out
    of 4795 training images depict a waterbird with a land background).
    <span>&#10230;</span> **Used by**: \[[61](#ref-wang2022ISR)\]

  - **[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)** \[[40](#ref-liu2015faceattributes)\]:
    A face image dataset containing 202,599 images of size 178×218 from
    10,177 unique celebrities. Each image is annotated with 40 binary
    facial attributes (e.g., *Is this person smiling?*) and five
    landmark positions describing the 2D position of the eyes, the nose,
    and the mouth (split into *left* and *right* side of the mouth).
    <span>&#10230;</span> **Used by**: \[[61](#ref-wang2022ISR)\]

  - **[MultiNLI](https://cims.nyu.edu/~sbowman/multinli/)** \[[64](#ref-williams2017broad)\]:
    A text dataset developed to evaluate natural language inference
    (NLI) models. Models must decide whether a given hypothesis is
    contradictory to, entailed by, or neutral to the given premise.
    Contrary to other NLI datasets, MultiNLI includes text from 10
    written and spoken English domains. <span>&#10230;</span> **Used
    by**: \[[61](#ref-wang2022ISR)\]

  - **[Abalone](https://archive.ics.uci.edu/ml/datasets/abalone)** \[[14](#ref-dheeru2017uci)\]:
    In this task, ML models need to predict the number of rings an
    *abalone* (a shellfish) has based on the given features *sex*,
    *width*, *height*, and *shell diameter*. The dataset contains 4177
    entries. <span>&#10230;</span> **Used
    by**: \[[32](#ref-kyono2019improving)\]

  - **[Bike Sharing in Washington
    D.C.](https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset)** \[[17](#ref-fanaee2014event)\]:
    This dataset contains the hourly and daily count of rental bikes
    used in Washington D.C. between 2011 and 2012 (17,379 entries).
    Given weather and seasonal information, models need to predict the
    count of total rental bikes. <span>&#10230;</span> **Used
    by**: \[[32](#ref-kyono2019improving)\]

  - **[OpenPowerlifting](https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database)** \[[46](#ref-OpenPowerlifting2019)\]:
    This powerlifting competition dataset includes more than 22,000
    competitions and more than 412,000 competitors as of April 2019. The
    data stem from OpenPowerlifting[^8], with each entry containing
    information about the lifter, the equipment used, weight class, and
    their performance across different powerlifting disciplines.
    <span>&#10230;</span> **Used by**: \[[32](#ref-kyono2019improving)\]

</details>

<details>
<summary><h2>Interesting Causal Tools (Click to expand)</h2></summary>

  - **[CANDLE](https://github.com/causal-disentanglement/CANDLE)** \[[51](#ref-reddy2022AAAIcandle_disentangled_dataset)\]:
    A dataset of realistic images of objects in a specific scene
    generated based on observed and unobserved confounders (object,
    size, color, rotation, light, and scene). As each of the 12546
    images is annotated with the ground-truth information of the six
    generating factors, it is possible to emulate interventions on image
    features.

  - **[CausalWorld](https://github.com/rr-learning/CausalWorld)** \[[2](#ref-ahmed2021causalworld)\]:
    A simulation framework and benchmark that provides RL agents
    different learning tasks in a robotic manipulation environment. The
    environment comes with a causal structure on which users and agents
    can intervene on variables such as object masses, colors or sizes.

  - **[gCastle](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle)** \[[68](#ref-zhang2021gcastle)\]:
    An end-to-end causal structure learning toolbox that is equipped
    with 19 techniques for Causal Discovery. It also assists users in
    data generation and evaluating learned structures. Having a firm
    understanding of the causal structure allows models to deduce the
    content and style variables of the domain.

  - **[Benchpress](https://github.com/felixleopoldo/benchpress)** \[[52](#ref-rios2021benchpress)\]:
    A benchmark for causal structure learning allowing users to compare
    their causal discovery methods with over 40 variations of
    state-of-the-art algorithms. The plethora of available techniques in
    this single tool could facilitate research into robustness of ML
    systems through causality.

</details>

<details>
<summary><h2>Prominent Non-Causal Tools (Click to expand)</h2></summary>

  - **DomainBed and
    OOD-Bench** \[[21](#ref-Gulrajani_2020_DomainBed),[65](#ref-Ye2021ood)\]:
    [DomainBed](https://github.com/facebookresearch/DomainBed) is a
    benchmark for OOD-learning that enables performance comparisons with
    more than 20 OOD-algorithms on 10 different, popular OOD-datasets.
    [OOD-Bench](https://github.com/m-Just/OoD-Bench) is built upon
    DomainBed and introduces a measurement to quantify the Diversity
    shift and Correlation shift inherit to OOD-datasets. The resulting
    categorization allows researchers to pinpoint strengths and
    weaknesses of OOD-learning algorithms.

  - **[RobustBench](https://github.com/RobustBench/robustbench)** \[[12](#ref-croce2020robustbench)\]:
    A standardized adversarial robustness benchmark capable of emulating
    a variety of adversarial attacks for image classification through
    *AutoAttack*. It also provides multiple continuously updated
    leaderboards of the most robust models, which allows for direct
    comparisons between causal and non-causal methods.

  - **[Foolbox](https://github.com/bethgelab/foolbox)** \[[49](#ref-rauber2017foolboxnative)\]:
    A popular Python library that allows researchers to test their
    adversarial defenses against state-of-the-art adversarial attacks.
    Foolbox is very compatible, natively supporting Pytorch, Tensorflow
    and JAX models.

  - **[VeriGauge](https://github.com/AI-secure/VeriGauge)** \[[36](#ref-Li_2020_SoK)\]:
    A Python toolbox that allows users to verify the robustness of their
    adversarial defense approach for deep neural networks. It not only
    covers a multitude of verification techniques but also comes with an
    up-to-date leaderboard.

  - **[Adversarial Robustness Toolbox
    (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)** \[[45](#ref-art2018)\]:
    An extensive Python library for Adversarial Machine Learning. It not
    only equips researchers with various attacks and defenses across
    four different attack threats (evasion, extraction, poisoning, and
    inference) but also provides the means to assess the performance of
    such algorithms thoroughly. ART is compatible with many popular
    frameworks and supports various data types and learning tasks.

</details>

<div id="refs" class="references">

## References

<div id="ref-agrawal2018don">

\[1\] Aishwarya Agrawal, Dhruv Batra, Devi Parikh, and Aniruddha
Kembhavi. 2018. Don’t just assume; look and answer: Overcoming priors
for visual question answering. In *Proceedings of the ieee conference on
computer vision and pattern recognition*, 4971–4980.

</div>

<div id="ref-ahmed2021causalworld">

\[2\] Ossama Ahmed, Frederik Träuble, Anirudh Goyal, Alexander Neitz,
Manuel Wüthrich, Yoshua Bengio, Bernhard Schölkopf, and Stefan Bauer.
2021. CausalWorld: A robotic manipulation benchmark for causal structure
and transfer learning. In *International conference on learning
representations*.

</div>

<div id="ref-ahuja2020invariant">

\[3\] Kartik Ahuja, Karthikeyan Shanmugam, Kush Varshney, and Amit
Dhurandhar. 2020. Invariant risk minimization games. In *International
conference on machine learning*, PMLR, 145–155.

</div>

<div id="ref-angwin2016COMPAS">

\[4\] Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner. 2016.
Machine bias. In *Ethics of data and analytics*. Auerbach Publications,
254–264.

</div>

<div id="ref-arjovsky2019invariant">

\[5\] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David
Lopez-Paz. 2019. Invariant risk minimization. *arXiv* (2019).

</div>

<div id="ref-barbu2019objectnet">

\[6\] Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher
Wang, Dan Gutfreund, Josh Tenenbaum, and Boris Katz. 2019. Objectnet: A
large-scale bias-controlled dataset for pushing the limits of object
recognition models. *Advances in neural information processing systems*
32, (2019).

</div>

<div id="ref-bellemare13arcade">

\[7\] M. G. Bellemare, Y. Naddaf, J. Veness, and M. Bowling. 2013. The
arcade learning environment: An evaluation platform for general agents.
*Journal of Artificial Intelligence Research* 47, (June 2013), 253–279.

</div>

<div id="ref-bowman2015large">

\[8\] Samuel R Bowman, Gabor Angeli, Christopher Potts, and Christopher
D Manning. 2015. A large annotated corpus for learning natural language
inference. *arXiv preprint arXiv:1508.05326* (2015).

</div>

<div id="ref-Chen_2021_posthoc">

\[9\] Guangyi Chen, Junlong Li, Jiwen Lu, and Jie Zhou. 2021. Human
trajectory prediction via counterfactual analysis. IEEE/CVF.

</div>

<div id="ref-babyai_iclr19">

\[10\] Maxime Chevalier-Boisvert, Dzmitry Bahdanau, Salem Lahlou, Lucas
Willems, Chitwan Saharia, Thien Huu Nguyen, and Yoshua Bengio. 2019.
BabyAI: First steps towards grounded language learning with a human in
the loop. In *International conference on learning representations*.
Retrieved from <https://openreview.net/forum?id=rJeXCo0cYX>

</div>

<div id="ref-christie2018functional">

\[11\] Gordon Christie, Neil Fendley, James Wilson, and Ryan Mukherjee.
2018. Functional map of the world. In *Proceedings of the ieee
conference on computer vision and pattern recognition*, 6172–6180.

</div>

<div id="ref-croce2020robustbench">

\[12\] Francesco Croce, Maksym Andriushchenko, Vikash Sehwag, Edoardo
Debenedetti, Nicolas Flammarion, Mung Chiang, Prateek Mittal, and
Matthias Hein. 2020. Robustbench: A standardized adversarial robustness
benchmark. *arXiv preprint arXiv:2010.09670* (2020).

</div>

<div id="ref-ImageNet2009">

\[13\] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li
Fei-Fei. 2009. ImageNet: A large-scale hierarchical image database. In
*2009 ieee conference on computer vision and pattern recognition*,
248–255. DOI:[https://doi.org/10.1109/CVPR.2009.5206848
](https://doi.org/10.1109/CVPR.2009.5206848%20%20%20)

</div>

<div id="ref-dheeru2017uci">

\[14\] Dua Dheeru and E Karra Taniskidou. 2017. UCI machine learning
repository. (2017).

</div>

<div id="ref-dominguez2022adversarial">

\[15\] Ricardo Dominguez-Olmedo, Amir H Karimi, and Bernhard Schölkopf.
2022. On the adversarial robustness of causal algorithmic recourse. In
*International conference on machine learning*, PMLR, 5324–5342.

</div>

<div id="ref-ess2007eth">

\[16\] Andreas Ess, Bastian Leibe, and Luc Van Gool. 2007. Depth and
appearance for mobile scene analysis. In *2007 ieee 11th international
conference on computer vision*, IEEE, 1–8.

</div>

<div id="ref-fanaee2014event">

\[17\] Hadi Fanaee-T and Joao Gama. 2014. Event labeling combining
ensemble detectors and background knowledge. *Progress in Artificial
Intelligence* 2, 2 (2014), 113–127.

</div>

<div id="ref-ghifary2015domain">

\[18\] Muhammad Ghifary, W Bastiaan Kleijn, Mengjie Zhang, and David
Balduzzi. 2015. Domain generalization for object recognition with
multi-task autoencoders. In *ICCV*, 2551–2559.

</div>

<div id="ref-goyal2021recurrent">

\[19\] Anirudh Goyal, Alex Lamb, Jordan Hoffmann, Shagun Sodhani, Sergey
Levine, Yoshua Bengio, and Bernhard Schölkopf. 2021. Recurrent
independent mechanisms. ICLR.

</div>

<div id="ref-groemping2019south">

\[20\] Ulrike Groemping. 2019. South german credit data: Correcting a
widely used data set. *Rep. Math., Phys. Chem., Berlin, Germany, Tech.
Rep* 4, (2019), 2019.

</div>

<div id="ref-Gulrajani_2020_DomainBed">

\[21\] Ishaan Gulrajani and David Lopez-Paz. 2020. In search of lost
domain generalization. *arXiv preprint arXiv:2007.01434* (2020).

</div>

<div id="ref-hendrycks2021many">

\[22\] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank
Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo,
and others. 2021. The many faces of robustness: A critical analysis of
out-of-distribution generalization. In *Proceedings of the ieee/cvf
international conference on computer vision*, 8340–8349.

</div>

<div id="ref-hendrycks2019benchmarking">

\[23\] Dan Hendrycks and Thomas Dietterich. 2019. Benchmarking neural
network robustness to common corruptions and perturbations. *arXiv
preprint arXiv:1903.12261* (2019).

</div>

<div id="ref-Ilse_2021_simulatinginterventions">

\[24\] Maximilian Ilse, Jakub M Tomczak, and Patrick Forré. 2021.
Selecting data augmentation for simulating interventions. In
*International conference on machine learning*, PMLR, 4555–4562.

</div>

<div id="ref-ilyas2022datamodels">

\[25\] Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc,
and Aleksander Madry. 2022. Datamodels: Understanding predictions with
data and data with predictions. In *International conference on machine
learning*, PMLR, 9525–9587.

</div>

<div id="ref-kaushik2019learning">

\[26\] Divyansh Kaushik, Eduard Hovy, and Zachary C Lipton. 2019.
Learning the difference that makes a difference with
counterfactually-augmented data. *arXiv preprint arXiv:1909.12434*
(2019).

</div>

<div id="ref-kaushik2020explaining">

\[27\] Divyansh Kaushik, Amrith Setlur, Eduard Hovy, and Zachary C
Lipton. 2020. Explaining the efficacy of counterfactually augmented
data. *arXiv preprint arXiv:2010.02114* (2020).

</div>

<div id="ref-ke2021systematic">

\[28\] Nan Rosemary Ke, Aniket Didolkar, Sarthak Mittal, Anirudh Goyal,
Guillaume Lajoie, Stefan Bauer, Danilo Rezende, Yoshua Bengio, Michael
Mozer, and Christopher Pal. 2021. Systematic evaluation of causal
discovery in visual model based reinforcement learning. *arXiv preprint
arXiv:2107.00848* (2021).

</div>

<div id="ref-kohavi1996ADULTS">

\[29\] Ron Kohavi and others. 1996. Scaling up the accuracy of
naive-bayes classifiers: A decision-tree hybrid. In *Kdd*, 202–207.

</div>

<div id="ref-krizhevsky2009learning">

\[30\] Alex Krizhevsky, Geoffrey Hinton, and others. 2009. Learning
multiple layers of features from tiny images. (2009).

</div>

<div id="ref-krueger2016zoneout">

\[31\] David Krueger, Tegan Maharaj, János Kramár, Mohammad Pezeshki,
Nicolas Ballas, Nan Rosemary Ke, Anirudh Goyal, Yoshua Bengio, Aaron
Courville, and Chris Pal. 2016. Zoneout: Regularizing rnns by randomly
preserving hidden activations. *arXiv preprint arXiv:1606.01305* (2016).

</div>

<div id="ref-kyono2019improving">

\[32\] Trent Kyono and Mihaela van der Schaar. 2019. Improving model
robustness using causal knowledge. *arXiv preprint arXiv:1911.12441*
(2019).

</div>

<div id="ref-lecun1998gradient">

\[33\] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner.
1998. Gradient-based learning applied to document recognition.
*Proceedings of the IEEE* 86, 11 (1998), 2278–2324.

</div>

<div id="ref-lerner2007crowds">

\[34\] Alon Lerner, Yiorgos Chrysanthou, and Dani Lischinski. 2007.
Crowds by example. In *Computer graphics forum*, Wiley Online Library,
655–664.

</div>

<div id="ref-li2017deeper">

\[35\] Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy M Hospedales. 2017.
Deeper, broader and artier domain generalization. In *ICCV*, 5542–5550.

</div>

<div id="ref-Li_2020_SoK">

\[36\] Linyi Li, Xiangyu Qi, Tao Xie, and Bo Li. 2020. Sok: Certified
robustness for deep neural networks. *arXiv preprint arXiv:2009.04131*
(2020).

</div>

<div id="ref-lin2014microsoft">

\[37\] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro
Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014.
Microsoft coco: Common objects in context. In *European conference on
computer vision*, Springer, 740–755.

</div>

<div id="ref-little2019causal">

\[38\] Max A Little and Reham Badawy. 2019. Causal bootstrapping. *arXiv
preprint arXiv:1910.09648* (2019).

</div>

<div id="ref-liu2022towards">

\[39\] Yuejiang Liu, Riccardo Cadei, Jonas Schweizer, Sherwin Bahmani,
and Alexandre Alahi. 2022. Towards robust and adaptive motion
forecasting: A causal representation perspective. In *Proceedings of the
ieee/cvf conference on computer vision and pattern recognition*,
17081–17092.

</div>

<div id="ref-liu2015faceattributes">

\[40\] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. 2015. Deep
learning face attributes in the wild. In *Proceedings of international
conference on computer vision (iccv)*.

</div>

<div id="ref-lu2021invariant">

\[41\] Chaochao Lu, Yuhuai Wu, José Miguel Hernández-Lobato, and
Bernhard Schölkopf. 2021. Invariant causal representation learning for
out-of-distribution generalization. In *International conference on
learning representations*.

</div>

<div id="ref-Mao_2021_generative">

\[42\] Chengzhi Mao, Augustine Cha, Amogh Gupta, Hao Wang, Junfeng Yang,
and Carl Vondrick. 2021. Generative interventions for causal learning.
In *Proceedings of the ieee/cvf conference on computer vision and
pattern recognition*, 3947–3956.

</div>

<div id="ref-Mitrovic_2020_relic">

\[43\] Jovana Mitrovic, Brian McWilliams, Jacob Walker, Lars Buesing,
and Charles Blundell. 2020. Representation learning via invariant causal
mechanisms. *arXiv preprint arXiv:2010.07922* (2020).

</div>

<div id="ref-ni2019justifying">

\[44\] Jianmo Ni, Jiacheng Li, and Julian McAuley. 2019. Justifying
recommendations using distantly-labeled reviews and fine-grained
aspects. In *Proceedings of the 2019 conference on empirical methods in
natural language processing and the 9th international joint conference
on natural language processing (emnlp-ijcnlp)*, 188–197.

</div>

<div id="ref-art2018">

\[45\] Maria-Irina Nicolae, Mathieu Sinn, Minh Ngoc Tran, Beat Buesser,
Ambrish Rawat, Martin Wistuba, Valentina Zantedeschi, Nathalie
Baracaldo, Bryant Chen, Heiko Ludwig, Ian Molloy, and Ben Edwards. 2018.
Adversarial robustness toolbox v1.2.0. *CoRR* 1807.01069, (2018).
Retrieved from <https://arxiv.org/pdf/1807.01069>

</div>

<div id="ref-OpenPowerlifting2019">

\[46\] OpenPowerlifting. 2019. Powerlifting Database. Retrieved from
<https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database>

</div>

<div id="ref-DomainNet_2019">

\[47\] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko,
and Bo Wang. 2019. Moment matching for multi-source domain adaptation.
In *Proceedings of the ieee/cvf international conference on computer
vision*, 1406–1415.

</div>

<div id="ref-LawSchoolAdmission">

\[48\] Project SEAPHE. 2007. Project seaphe: Databases. Retrieved from
<http://www.seaphe.org/databases.php>

</div>

<div id="ref-rauber2017foolboxnative">

\[49\] Jonas Rauber, Roland Zimmermann, Matthias Bethge, and Wieland
Brendel. 2020. Foolbox native: Fast adversarial attacks to benchmark the
robustness of machine learning models in pytorch, tensorflow, and jax.
*Journal of Open Source Software* 5, 53 (2020), 2607.
DOI:[https://doi.org/10.21105/joss.02607
](https://doi.org/10.21105/joss.02607%20%20%20)

</div>

<div id="ref-recht2019imagenet">

\[50\] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal
Shankar. 2019. Do imagenet classifiers generalize to imagenet? In
*International conference on machine learning*, PMLR, 5389–5400.

</div>

<div id="ref-reddy2022AAAIcandle_disentangled_dataset">

\[51\] Abbavaram Gowtham Reddy, Benin Godfrey L, and Vineeth N.
Balasubramanian. 2022. On causally disentangled representations. In
*AAAI*.

</div>

<div id="ref-rios2021benchpress">

\[52\] Felix L. Rios, Giusi Moffa, and Jack Kuipers. 2021. Benchpress: A
scalable and platform-independent workflow for benchmarking structure
learning algorithms for graphical models. Retrieved from
<http://arxiv.org/abs/2107.03863>

</div>

<div id="ref-robicquet2016learning">

\[53\] Alexandre Robicquet, Amir Sadeghian, Alexandre Alahi, and Silvio
Savarese. 2016. Learning social etiquette: Human trajectory
understanding in crowded scenes. In *European conference on computer
vision*, Springer, 549–565.

</div>

<div id="ref-rosenthal-etal-2017-semeval">

\[54\] Sara Rosenthal, Noura Farra, and Preslav Nakov. 2017.
SemEval-2017 task 4: Sentiment analysis in Twitter. In *Proceedings of
the 11th international workshop on semantic evaluation (SemEval-2017)*,
Association for Computational Linguistics, Vancouver, Canada, 502–518.
DOI:<https://doi.org/10.18653/v1/S17-2088>

</div>

<div id="ref-russakovsky2015imagenet">

\[55\] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev
Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla,
Michael Bernstein, and others. 2015. Imagenet large scale visual
recognition challenge. *International journal of computer vision* 115, 3
(2015), 211–252.

</div>

<div id="ref-sagawa2019distributionally">

\[56\] Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy
Liang. 2019. Distributionally robust neural networks for group shifts:
On the importance of regularization for worst-case generalization.
*arXiv preprint arXiv:1911.08731* (2019).

</div>

<div id="ref-schmidt1988predicting">

\[57\] Peter Schmidt and Ann D Witte. 1988. *Predicting recidivism in
north carolina, 1978 and 1980*. 

</div>

<div id="ref-teney2020learning">

\[58\] Damien Teney, Ehsan Abbasnedjad, and Anton van den Hengel. 2020.
Learning what makes a difference from counterfactual examples and
gradient supervision. In *European conference on computer vision*,
Springer, 580–599.

</div>

<div id="ref-torralba2011unbiased">

\[59\] Antonio Torralba and Alexei A Efros. 2011. Unbiased look at
dataset bias. In *CVPR 2011*, IEEE, 1521–1528.

</div>

<div id="ref-van2018relational">

\[60\] Sjoerd Van Steenkiste, Michael Chang, Klaus Greff, and Jürgen
Schmidhuber. 2018. Relational neural expectation maximization:
Unsupervised discovery of objects and their interactions. *arXiv
preprint arXiv:1802.10353* (2018).

</div>

<div id="ref-wang2022ISR">

\[61\] Haoxiang Wang, Haozhe Si, Bo Li, and Han Zhao. 2022. Provable
domain generalization via invariant-feature subspace recovery. *arXiv
preprint arXiv:2201.12919* (2022).

</div>

<div id="ref-wang2021enhancing">

\[62\] Zhao Wang, Kai Shu, and Aron Culotta. 2021. Enhancing model
robustness and fairness with causality: A regularization approach.
*arXiv preprint arXiv:2110.00911* (2021).

</div>

<div id="ref-Wang_2022_CDL">

\[63\] Zizhao Wang, Xuesu Xiao, Zifan Xu, Yuke Zhu, and Peter Stone.
2022. Causal dynamics learning for task-independent state abstraction.
*arXiv preprint arXiv:2206.13452* (2022).

</div>

<div id="ref-williams2017broad">

\[64\] Adina Williams, Nikita Nangia, and Samuel R Bowman. 2017. A
broad-coverage challenge corpus for sentence understanding through
inference. *arXiv preprint arXiv:1704.05426* (2017).

</div>

<div id="ref-Ye2021ood">

\[65\] Nanyang Ye, Kaican Li, Lanqing Hong, Haoyue Bai, Yiting Chen,
Fengwei Zhou, and Zhenguo Li. 2021. OoD-bench: Benchmarking and
understanding out-of-distribution generalization datasets and
algorithms. *arXiv preprint arXiv:2106.03721* (2021).

</div>

<div id="ref-YelpDataset">

\[66\] Yelp. Yelp Open Dataset. 

</div>

<div id="ref-zhang2020causal">

\[67\] Cheng Zhang, Kun Zhang, and Yingzhen Li. 2020. A causal view on
robustness of neural networks. *Advances in Neural Information
Processing Systems* 33, (2020), 289–301.

</div>

<div id="ref-zhang2021gcastle">

\[68\] Keli Zhang, Shengyu Zhu, Marcus Kalander, Ignavier Ng, Junjian
Ye, Zhitang Chen, and Lujia Pan. 2021. GCastle: A python toolbox for
causal discovery. *arXiv preprint arXiv:2111.15155* (2021).

</div>

<div id="ref-zhang2021causaladv">

\[69\] Yonggang Zhang, Mingming Gong, Tongliang Liu, Gang Niu, Xinmei
Tian, Bo Han, Bernhard Schölkopf, and Kun Zhang. 2021. CausalAdv:
Adversarial robustness through the lens of causality. *arXiv* (2021).

</div>

<div id="ref-zhu2020robosuite">

\[70\] Yuke Zhu, Josiah Wong, Ajay Mandlekar, and Roberto
Martı́n-Martı́n. 2020. Robosuite: A modular simulation framework and
benchmark for robot learning. *arXiv preprint arXiv:2009.12293* (2020).

</div>

</div>

[^1]:  \[[62](#ref-wang2021enhancing)\] used a subset of Kindle Book
    reviews from an older version of this dataset

[^2]:  <https://semeval.github.io/>

[^3]:  <https://www.mturk.com/>

[^4]:  <https://www.image-net.org/about.php>

[^5]:  <https://www.flickr.com/>

[^6]:  <http://groups.csail.mit.edu/vision/TinyImages/>

[^7]:  <http://www.census.gov/en.html>

[^8]:  <https://www.openpowerlifting.org/>
