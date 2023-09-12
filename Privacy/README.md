# Privacy

## Aspects of Trustworthy AI and Application Domain

### [Interpretability](../Interpretability/README.md)

### [Fairness](../Fairness/README.md)

### [Robustness](../Robustness/README.md)

### Privacy

### [Auditing (Safety and Accountability)](../Auditing/README.md)

### [Healthcare](../Healthcare/README.md)


<details>
<summary><h2>Datasets Used by Cited Publications (Click to expand)</h2></summary>

  - **[Rotated
    MNIST](https://github.com/ghif/mtae)** \[[9](#ref-ghifary2015domain)\]:
    The dataset consists of MNIST images with each domain containing
    images rotated by a particular angle
    \(0°, 15°, 30°, 45°, 60°, 75°\)
    <span>&#10230;</span> **Used
    by**: \[[7](#ref-francis2021towards),[18](#ref-de2022mitigating)\]

  - **[PACS](https://github.com/facebookresearch/DomainBed)** \[[15](#ref-li2017deeper)\]:
    An image classification dataset categorized into 10 classes that are
    scattered across four different domains, each having a distinct
    trait: photograph, art, cartoon and sketch.
    <span>&#10230;</span> **Used by**: \[[18](#ref-de2022mitigating)\]

  - **[Office-Home](https://www.hemanthdv.org/officeHomeDataset.html)** \[[21](#ref-venkateswara2017deep)\]:
    Image classification dataset analogous to PACS, having four distinct
    image domains: Art, ClipArt, Product and Real-World.
    <span>&#10230;</span> **Used by**: \[[18](#ref-de2022mitigating)\]

  - **[ColoredMNIST](https://github.com/facebookresearch/InvariantRiskMinimization)** \[[3](#ref-arjovsky2019invariant)\]:
    The dataset consists of input images with digits 0-4 colored red and
    labelled 0 while digits 5-9 are colored green representing the two
    domains. <span>&#10230;</span> **Used
    by**: \[[7](#ref-francis2021towards),[10](#ref-gupta2022fl)\]

  - **[Colored
    FashionMNIST](https://github.com/IBM/OoD/tree/master/IRM_games)** \[[2](#ref-ahuja2020invariant)\]:
    This dataset was inspired by \[[3](#ref-arjovsky2019invariant)\]
    Colored MNIST dataset. \[[2](#ref-ahuja2020invariant)\] use the same
    coloring approach to induce spurious correlations into FashionMNIST
    data (greyscaled Zalando articles). <span>&#10230;</span> **Used
    by**: \[[10](#ref-gupta2022fl)\]

  - **[CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)** \[[14](#ref-krizhevsky2009learning)\]:
    The two CIFAR datasets, CIFAR-10 and CIFAR-100, are labeled images
    stemming from the now withdrawn Tiny Images dataset[^1]. The more
    prominent set, CIFAR-10, contains 60000 32 &times; 32 color
    images separated into ten mutually exclusive classes, with 6000
    images per class. CIFAR-100 is simply a 100-class version of
    CIFAR-10. <span>&#10230;</span> **Used
    by**: \[[10](#ref-gupta2022fl),[13](#ref-jiang2021tsmobn)\]

  - **[Digits-DG](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#digits-dg)** \[[24](#ref-zhou2020learning)\]:
    An image dataset specifically designed to evaluate the performance
    of models on OOD data. It includes images from four different
    handwritten digits databases. Each dataset represents a unique
    domain as images from different datasets significantly differ in
    terms of, e.g., handwriting style or background color.
    <span>&#10230;</span> **Used by**: \[[13](#ref-jiang2021tsmobn)\]

  - **[Camelyon17](https://camelyon17.grand-challenge.org/Data/)** \[[4](#ref-bandi2018detection)\]:
    A publicly available medical dataset containing 1000 histology
    images from five Dutch hospitals. Given an image, classification
    models need to detect breast cancer metastases.
    <span>&#10230;</span> **Used by**: \[[13](#ref-jiang2021tsmobn)\]

</details>

<details>
<summary><h2>Interesting Causal Tools For Federated Learning (Click to expand)</h2></summary>

The publications reviewed in our survey are largely
causal approaches to Federated Learning (FL). As such, we mainly provide
an overview of causal and non-causal tools for FL.

  - **Federated Causal
    Discovery** \[[1](#ref-abyaneh2022fed),[8](#ref-gao2021federated)\]:
    Until this point, we suggested general causal discovery tools like
    *gCastle* \[[23](#ref-zhang2021gcastle)\] or
    *benchpress* \[[20](#ref-rios2021benchpress)\]. However, the
    provided methods translate poorly into the federated setting due to
    the decentralized data. As such, we would like to refer readers to
    recently developed **Federated Causal Discovery** techniques
    (e.g., \[[1](#ref-abyaneh2022fed),[8](#ref-gao2021federated)\]).
    These methods are specifically designed to conduct causal discovery
    on decentralized data in a privacy-preserving manner.

  - **[CANDLE](https://github.com/causal-disentanglement/CANDLE)** \[[19](#ref-reddy2022AAAIcandle_disentangled_dataset)\]:
    A dataset of realistic images of objects in a specific scene
    generated based on observed and unobserved confounders (object,
    size, color, rotation, light, and scene). As each of the 12546
    images is annotated with the ground-truth information of the six
    generating factors, it is possible to emulate interventions on image
    features. Users/Devices could be simulated by altering the scenery.

  - **[Federated Causal Effect
    Estimation](https://github.com/vothanhvinh/FedCI)** \[[22](#ref-vo2022bayesian)\]:
    Similar to causal discovery, standard causal effect estimation
    methods were not designed for decentralized data. Only very
    recently, \[[22](#ref-vo2022bayesian)\] developed a causal effect
    estimation framework compatible with federated learning. Despite
    this line of work’s infancy, we believe that this publication is
    important for more privacy-preserving causal learning.

</details>

<details>
<summary><h2>Prominent Non-Causal Federated Learning Tools (Click to expand)</h2></summary>

  - **[LEAF](https://github.com/TalwalkarLab/leaf)** \[[5](#ref-caldas2018leaf)\]:
    A benchmark containing datasets explicitly designed to analyze FL
    algorithms. The six datasets include existing re-designed databases
    such as *CelebA* \[[17](#ref-liu2015faceattributes)\] to emulate
    different devices/users and newly created datasets. LEAF also
    provides evaluation methods and baseline reference implementations
    for each dataset.

  - **[FedEval](https://github.com/Di-Chai/FedEval)** \[[6](#ref-chai2020fedeval)\]:
    A publicly available evaluation platform for FL. It allows
    researchers to compare their FL methods with existing
    state-of-the-art algorithms on seven datasets based on five
    FL-relevant metrics (Accuracy, Communication, Time efficiency,
    Privacy, and Robustness). The benchmark utilizes Docker container
    technology to simulate the server and clients and socket IO for
    simulating communication between the two.

  - **[OARF](https://github.com/Xtra-Computing/OARF)** \[[12](#ref-hu2022oarf)\]:
    An extensive benchmark suite designed to assess state-of-the-art FL
    algorithms for both horizontal and vertical FL. It includes 22
    datasets that cover different domains for both FL variants.
    Additionally, OARF provides several metrics to evaluate FL
    algorithms, and its modular design enables researchers to test their
    own methods.

  - **[FedGraphNN](https://github.com/FedML-AI/FedML/tree/master/python/app/fedgraphnn)** \[[11](#ref-he2021fedgraphnn)\]:
    An FL benchmark for Graph Neural Networks (GNN). In order to provide
    a unified platform for the development of graph-based FL solutions,
    FedGraphNN supplies users with 36 graph datasets across seven
    different domains. Researchers can also employ and compare their own
    *PyTorch (Geometric)* models with different GNNs.

  - **[ML-Doctor](https://github.com/liuyugeng/ML-Doctor)** \[[16](#ref-mldoctor_2022)\]:
    A codebase initially used to compare and evaluate different
    inference attacks (membership inference, model stealing, model
    inversion, and attribute inference). Its modular structure enables
    researchers to assess the effectiveness of their privacy-preserving
    algorithms against SOTA privacy attacks.
  
  </details>

<div id="refs" class="references">

## References

<div id="ref-abyaneh2022fed">

\[1\] Amin Abyaneh, Nino Scherrer, Patrick Schwab, Stefan Bauer,
Bernhard Schölkopf, and Arash Mehrjou. 2022. FED-cd: Federated causal
discovery from interventional and observational data. *arXiv preprint
arXiv:2211.03846* (2022).

</div>

<div id="ref-ahuja2020invariant">

\[2\] Kartik Ahuja, Karthikeyan Shanmugam, Kush Varshney, and Amit
Dhurandhar. 2020. Invariant risk minimization games. In *International
conference on machine learning*, PMLR, 145–155.

</div>

<div id="ref-arjovsky2019invariant">

\[3\] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David
Lopez-Paz. 2019. Invariant risk minimization. *arXiv* (2019).

</div>

<div id="ref-bandi2018detection">

\[4\] Peter Bandi, Oscar Geessink, Quirine Manson, Marcory Van Dijk,
Maschenka Balkenhol, Meyke Hermsen, Babak Ehteshami Bejnordi, Byungjae
Lee, Kyunghyun Paeng, Aoxiao Zhong, and others. 2018. From detection of
individual metastases to classification of lymph node status at the
patient level: The camelyon17 challenge. *IEEE transactions on medical
imaging* 38, 2 (2018), 550–560.

</div>

<div id="ref-caldas2018leaf">

\[5\] Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li,
Jakub Konečnỳ, H Brendan McMahan, Virginia Smith, and Ameet Talwalkar.
2018. Leaf: A benchmark for federated settings. *arXiv preprint
arXiv:1812.01097* (2018).

</div>

<div id="ref-chai2020fedeval">

\[6\] Di Chai, Leye Wang, Kai Chen, and Qiang Yang. 2020. Fedeval: A
benchmark system with a comprehensive evaluation model for federated
learning. *arXiv preprint arXiv:2011.09655* (2020).

</div>

<div id="ref-francis2021towards">

\[7\] Sreya Francis, Irene Tenison, and Irina Rish. 2021. Towards causal
federated learning for enhanced robustness and privacy. *arXiv preprint
arXiv:2104.06557* (2021).

</div>

<div id="ref-gao2021federated">

\[8\] Erdun Gao, Junjia Chen, Li Shen, Tongliang Liu, Mingming Gong, and
Howard Bondell. 2021. Federated causal discovery. *arXiv preprint
arXiv:2112.03555* (2021).

</div>

<div id="ref-ghifary2015domain">

\[9\] Muhammad Ghifary, W Bastiaan Kleijn, Mengjie Zhang, and David
Balduzzi. 2015. Domain generalization for object recognition with
multi-task autoencoders. In *ICCV*, 2551–2559.

</div>

<div id="ref-gupta2022fl">

\[10\] Sharut Gupta, Kartik Ahuja, Mohammad Havaei, Niladri Chatterjee,
and Yoshua Bengio. 2022. FL games: A federated learning framework for
distribution shifts. *arXiv preprint arXiv:2205.11101* (2022).

</div>

<div id="ref-he2021fedgraphnn">

\[11\] Chaoyang He, Keshav Balasubramanian, Emir Ceyani, Carl Yang, Han
Xie, Lichao Sun, Lifang He, Liangwei Yang, Philip S Yu, Yu Rong, and
others. 2021. Fedgraphnn: A federated learning system and benchmark for
graph neural networks. *arXiv preprint arXiv:2104.07145* (2021).

</div>

<div id="ref-hu2022oarf">

\[12\] Sixu Hu, Yuan Li, Xu Liu, Qinbin Li, Zhaomin Wu, and Bingsheng
He. 2022. The oarf benchmark suite: Characterization and implications
for federated learning systems. *ACM Transactions on Intelligent Systems
and Technology (TIST)* 13, 4 (2022), 1–32.

</div>

<div id="ref-jiang2021tsmobn">

\[13\] Meirui Jiang, Xiaofei Zhang, Michael Kamp, Xiaoxiao Li, and Qi
Dou. 2021. TsmoBN: Interventional generalization for unseen clients in
federated learning. *arXiv preprint arXiv:2110.09974* (2021).

</div>

<div id="ref-krizhevsky2009learning">

\[14\] Alex Krizhevsky, Geoffrey Hinton, and others. 2009. Learning
multiple layers of features from tiny images. (2009).

</div>

<div id="ref-li2017deeper">

\[15\] Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy M Hospedales. 2017.
Deeper, broader and artier domain generalization. In *ICCV*, 5542–5550.

</div>

<div id="ref-mldoctor_2022">

\[16\] Yugeng Liu, Rui Wen, Xinlei He, Ahmed Salem, Zhikun Zhang,
Michael Backes, Emiliano De Cristofaro, Mario Fritz, and Yang Zhang.
2022. ML-Doctor: Holistic risk assessment of inference attacks against
machine learning models. In *31st usenix security symposium (usenix
security 22)*, USENIX Association, Boston, MA, 4525–4542. Retrieved from
<https://www.usenix.org/conference/usenixsecurity22/presentation/liu-yugeng>

</div>

<div id="ref-liu2015faceattributes">

\[17\] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. 2015. Deep
learning face attributes in the wild. In *Proceedings of international
conference on computer vision (iccv)*.

</div>

<div id="ref-de2022mitigating">

\[18\] Artur Back de Luca, Guojun Zhang, Xi Chen, and Yaoliang Yu. 2022.
Mitigating data heterogeneity in federated learning with data
augmentation. *arXiv preprint arXiv:2206.09979* (2022).

</div>

<div id="ref-reddy2022AAAIcandle_disentangled_dataset">

\[19\] Abbavaram Gowtham Reddy, Benin Godfrey L, and Vineeth N.
Balasubramanian. 2022. On causally disentangled representations. In
*AAAI*.

</div>

<div id="ref-rios2021benchpress">

\[20\] Felix L. Rios, Giusi Moffa, and Jack Kuipers. 2021. Benchpress: A
scalable and platform-independent workflow for benchmarking structure
learning algorithms for graphical models. Retrieved from
<http://arxiv.org/abs/2107.03863>

</div>

<div id="ref-venkateswara2017deep">

\[21\] Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and
Sethuraman Panchanathan. 2017. Deep hashing network for unsupervised
domain adaptation. In *Proceedings of the ieee conference on computer
vision and pattern recognition*, 5018–5027.

</div>

<div id="ref-vo2022bayesian">

\[22\] Thanh Vinh Vo, Young Lee, Trong Nghia Hoang, and Tze-Yun Leong.
2022. Bayesian federated estimation of causal effects from observational
data. In *The 38th conference on uncertainty in artificial
intelligence*.

</div>

<div id="ref-zhang2021gcastle">

\[23\] Keli Zhang, Shengyu Zhu, Marcus Kalander, Ignavier Ng, Junjian
Ye, Zhitang Chen, and Lujia Pan. 2021. GCastle: A python toolbox for
causal discovery. *arXiv preprint arXiv:2111.15155* (2021).

</div>

<div id="ref-zhou2020learning">

\[24\] Kaiyang Zhou, Yongxin Yang, Timothy Hospedales, and Tao Xiang.
2020. Learning to generate novel domains for domain generalization. In
*European conference on computer vision*, Springer, 561–578.

</div>

</div>

[^1]:  <http://groups.csail.mit.edu/vision/TinyImages/>
