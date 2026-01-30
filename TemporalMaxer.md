# TemporalMaxer: Maximize Temporal Context with only Max Poolingfor Temporal Action Localization

Tuan N. Tang, Kwonyoung Kim, Kwanghoon Sohn*School of Electrical and Electronic EngineeringYonsel University

{tuantng, kyk12, khsohn}@yonsei.ac.kr

# Abstract

Temporal Action Localization (TAL) is a challengingtask in video understanding that aims to identify and lo-calize actions within a video sequence. Recent studies haveemphasized the importance of applying long-term temporalcontext modeling (TCM) blocks to the extracted video clipfeatures such as employing complex self-attention mecha-nisms. In this paper, we present the simplest method ever toaddress this task and argue that the extracted video clip fea-tures are already informative to achieve outstanding perfor-mance without sophisticated architectures. To this end, weintroduce TemporalMaxer, which minimizes long-term tem-poral context modeling while maximizing information fromthe extracted video clip features with a basic, parameter-free, and local region operating max-pooling block. Pick-ing out only the most critical information for adjacent andlocal clip embeddings, this block results in a more effi-cient TAL model. We demonstrate that TemporalMaxer out-performs other state-of-the-art methods that utilize long-term TCM such as self-attention on various TAL datasetswhile requiring significantly fewer parameters and com-putational resources. The code for our approach is pub-licly available at https://github.com/TuanTNG/TemporalMaxer.

# 1. Introduction

Temporal action localization (TAL), which aims to lo-calize action instances in time and assign their categori-cal labels, is a challenging but essential task within thefield of video comprehension. Various approaches havebeen proposed to address this task, such as action propos-als [29], anchor windows [36], or dense prediction [28]. Awidely accepted notion for improving the performance ofTAL models is to integrate a component capable of captur-ing long-term temporal dependencies within the extracted

video clip features[40, 16, 60, 19, 35, 39, 46, 53]. Specif-ically, the TAL model first employs pre-extracted featuresfrom a pre-trained 3D-CNN network, such as I3D [7] andTSN [52], as an input. Then, an encoder, called backbone,encodes features to latent space, and the decoder, calledhead, predicts action instances as illustrated in Fig. 1. Tobetter capture long-term temporal dependencies, long-termtemporal context modeling (TCM) blocks are incorporatedinto the backbone. Particularly, prior works [54, 61] haveemployed Graph [25], or more complex modules such asLocal-Global Temporal Encoder [40] which uses a channelgrouping strategy, or Relation-aware pyramid Network [16]which exploits bi-directional long-range relations. Recentlythere has been notable interest in the application of the self-attention [51] for long-term TCM [60, 35, 63, 23], resultingin surprising performance improvements.

Long-term TCM in the backbone can help the model tocapture long-term temporal dependencies between frames,which can be useful in identifying complex actions that mayunfold over longer periods of time or heavily overlappingactions. Despite their performance improvement on bench-marks, the inference speed and the effectiveness of the ex-isting approaches are rarely considered. While [60] intro-duced and pursued the minimalist design, their backbone isstill limited to the transformer architecture requiring expen-sive parameters and computations.

Recently, [58] proposed general architecture abstractedfrom transformer. Motivated by the success of recentapproaches replacing attention module with MLP-likemodules[49] or Fourier Transform[26], they deemed theessential of those modules as token mixer which aggre-gates information among tokens. In turn, they came to pro-pose PoolFormer, equiped with extremely minimized tokenmixer which replaces the exhausting attention module withvery simple pooling layer.

In this paper, motivated by their proposal, we focus onextreme minimization of the backbone network by concen-trating on and maximizing the information from the ex-tracted video clip features in a short-term perspective rather

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-30/6fc311af-54f7-48a1-bafe-93d404668861/84e0d149558a1d8eb261180951c6243029d3f67c1bff7cbfcb97feee3f820ffb.jpg)



Figure 1. Common architecture in Temporal Action Localization (TAL) and different temporal context modeling (TCM) blocks. Existingworks have incorporated extensive parameters, high computational costs, and complex modules in a backbone such as 1D Convolutionallayer in AFSD [28] to capture local temporal context, Graph [25] in G-TAD [54] and Transformers [51] in ActionFormer [60] to modellong-term temporal contexts. Our proposed method, termed as TemporalMaxer, for the first time exploits the potentials of strong featuresfrom pretrained 3D CNN by only utilizing a basic, parameter-free, local operating Max Pooling block. Our proposed method, the simplestbackbone ever for TAL, maintains only the most critical information on adjacent and local clip embeddings. Combined with the largereceptive field of deep networks, the whole model outperforms other works by a large margin on various datasets.


than employing exhausting long-term encoding process.

The transformer architecture leading state-of-the-art per-formance in the machine translation task [51] and com-puter vision areas [12] have inspired recent works in TAL[60, 35, 63, 23]. From the perspective of long-term TCM,the property that calculates attention weights for the longinput sequence has led to recent progress in the TAL. How-ever, such long-term consideration comes at a price of highcomputation costs and the effectiveness of those approacheshave not yet been carefully analyzed.

We argue the essential properties of the video clip fea-tures that have not been fully exploited to date. Firstly,the video clips exhibit a high redundancy which leads toa high similarity of the pre-extracted features as demon-strated in Fig. 2. It raises the question of the effective-ness of employing self-attention or graph methods for long-range TCM. Unlike in other domains such as machine trans-lation task where input tokens exhibit distinctiveness, theinput clip embeddings in TAL frequently exhibit a high de-gree of similarity. Consequently, as depicted in Fig. 2, theself-attention recently employed in TAL tend to average theembeddings of clips within the attention scope, losing tem-porally local minute changes by redundant similar framesunder the long-term temporal context modeling. We arguethat only certain information within clip embeddings is rel-evant to the action context of TAL, whereas the remainderof the information is similar across adjacent clips. There-fore, an optimal TCM must be capable of preserving themost discriminative features of clip embeddings that carrythe essential information.

To this end, we aim to propose simple yet effective TCMin a straightforward manner. We argue that the TCM canretain the simplest architecture while maximizing informa-

tive features extracted from 3D CNN. Max Pooling [5] isdeemed the most fitting block for this purpose.

Our proposed method, TemporalMaxer, presents thesimplest architecture ever for this task gaining much fasterinference speed. Our finding suggests that the pre-trainedfeature extractor already possess great potential and withthose feature, short-term TCM solely can benefit the perfor-mance for this task.

Extensive experiments prove the superiority and effec-tiveness of the proposed method, showing state-of-the-art performance in terms of both accuracy and speed forTAL on various challenging datasets including THUMOS[22], EPIC-Kitchens 100 [11], MultiTHUMOS [57], andMUSES [34].

# 2. Related Work

Temporal Action Localization (TAL). Two-Stage andSingle-Stage methods are used in TAL to detect actions invideos. Two-Stage methods first generate possible actionproposals and classify them into actions. The proposals aregenerated through anchor windows [13, 6, 20], detecting ac-tion boundaries [30, 17, 62], graph representation [4, 54],or Transformers [53, 8, 46]. Single-stage TAL performsboth action proposal generation and classification in a singlepass, without using a separate proposal generation step. Thepioneering work [40] developed anchor-based single-stageTAL using convolutional networks, inspired by a single-stage object detector [42, 33]. Meanwhile, [28] proposedan anchor-free single-stage model with a saliency-based re-finement module.

Long-term Temporal Context Modeling (TCM). Re-cent studies have emphasized the necessity of long-term

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-30/6fc311af-54f7-48a1-bafe-93d404668861/07088e3f5bf26b38611138febf2d94e5dbd7285b80f291faaacac56cd8e7de88.jpg)



(a) Simplified structure of Attention and Max Pooling module


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-30/6fc311af-54f7-48a1-bafe-93d404668861/41c89e958c482a7ad2337a56e785d9189f7de495346b5e719f1fe25fd7f37a20.jpg)



Figure 2. Comparison of effectiveness of Max Pooling overTransformer [51]. (a) When the input clip embeddings are highlysimilar, given that the value features $V$ are highly similar and theattention score is distributed equally, the Transformer tends to av-erage out the clip embeddings as done in Average Pooling. In con-trast, Max Pooling can retain the most crucial information aboutadjacent clip embeddings and further remove redundancy in thevideo. (b) Cosine similarity matrix(bottom-left) of value featuresin self-attention in ActionFormer [60] where the red boxes exhibitthe action intervals. (c) Attention score(bottom-right) after train-ing the ActionFormer[60].


TCM to improve model performance. Long-term TCMhelps the model capture long-term temporal dependenciesbetween frames, which can be useful in identifying intricateactions that span over extended timeframes or heavily over-lapping actions. Prior work has addressed long-term TCMusing Relation-aware pyramid Network [16], Multi-StageCNN [14], Temporal Context Aggregation Network [40].Recent works [54, 61] employ Graph [25] where each videoclip feature represents a node in a graph as long-term TCM.More recently, Transformer [51] demonstrates an outstand-ing capacity to capture long-range dependency of the inputsequence in the machine translation tasks. Thus, it is a nat-ural fit for Temporal Action Localization (TAL) where eachvideo clip embedding represents a token. Therefore, recentstudies [60, 35, 63, 23] have employed Transformers as along-term TCM.

Our approach, TemporalMaxer, belongs to the single-stage TAL model that utilizes a state-of-the-art Action-

Former [60] as the baseline for comparison. Similar to Ac-tionFormer, TemporalMaxer follows a minimalistic designof sequence labeling where every moment is classified, andtheir corresponding action boundaries are regressed. Themain difference is that we avoid exhausting attention be-tween clips from long-term timeframes which can uninten-tionally flatten minute information among the crowd of sim-ilar frames, but keep the minute information in short-termmanner.

# 3. Method

# 3.1. Problem Statement

Temporal Action Localization. Assume that anuntrimmed video $X$ can be represented by a set of featurevectors $X ~ = ~ \{ x _ { 1 } , x _ { 2 } , . . . , x _ { T } \}$ , where the number of dis-crete time steps $t = \{ 1 , 2 , . . . , T \}$ may vary depending onthe length of the video. The feature vector $x _ { t }$ is extractedfrom a pre-trained 3D convolutional network and representsa video clip at a specific moment $t$ . The aim of TAL isto predict a set of action instances $\Psi = \{ \psi _ { 1 } , \psi _ { 2 } , \ldots , \psi _ { N } \}$based on the input video sequence $X$ , where $N$ is the num-ber of action instances in $X$ . Each action instance $\psi _ { n }$ con-sists of $( s _ { n } , e _ { n } , a _ { n } )$ where sn, en, and $a _ { n }$ are starting time,ending time, and associated action label $a _ { n }$ respectively,$s _ { n } ~ \in ~ [ 1 , T ]$ , $e _ { n } \in [ 1 , T ]$ , $s _ { n } ~ < ~ e _ { n }$ , and the action label$a _ { n }$ belongs to the pre-defined set of $C$ categories.

# 3.2. TemporalMaxer

Action Representation. We follow the anchor-freesingle-stage representation [28, 60] for an action instance.Each moment is classified into the background or one of $C$categories, and regressed the onset and offset based on thecurrent time step of that moment. Consequently, the predic-tion in TAL is formulated as a sequence labeling problem.

$$
X = \left\{x _ {1}, x _ {2}, \dots , x _ {T} \right\}\rightarrow \hat {\Psi} = \left\{\hat {\psi} _ {1}, \hat {\psi} _ {2}, \dots , \hat {\psi} _ {T} \right\} \tag {1}
$$

At the time step $t .$ , the output $\hat { \psi } _ { t } = ( o _ { t } ^ { s } , o _ { t } ^ { e } , c _ { t } )$ is defined asfollowing:

• $o _ { t } ^ { s } \ > \ 0$ and $o _ { t } ^ { e } ~ > ~ 0$ represent the temporal intervalsbetween the current time step $t$ and the onset and offsetof a given moment, respectively.

• Given $C$ action categories, the action probability $c _ { t }$ canbe considered as a set of $c _ { t } ^ { i }$ which is a probability foraction $i ^ { t h }$ , where $1 \leq i \leq C$ .

The predicted action instance at time step $t$ can be retrievedfrom $\hat { \psi } _ { t } = ( o _ { t } ^ { s } , o _ { t } ^ { e } , c _ { t } )$ by:

$$
a _ {t} = \arg \max  \left(c _ {t}\right), \quad s _ {t} = t - o _ {t} ^ {s}, \quad e _ {t} = t + o _ {t} ^ {e} \tag {2}
$$

Architecture overview. The overall architecture is de-picted in Fig. 3. Our proposed method aims to learn to

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-30/6fc311af-54f7-48a1-bafe-93d404668861/0bfcbd59e8c288814f220984f7e151f31677af09a1f97b17be00aa90adb6dcab.jpg)



Figure 3. Overview of TemporalMaxer. The proposed method utilizes Max Pooling as a Temporal Context Modeling block applied betweentemporal feature pyramid levels to maximize informative features of high similarity clip embedding. Specifically, it first extracts features ofevery clip using pre-trained 3D CNN. After that, the backbone encodes clip features to form a multi-scale feature pyramid. The backboneconsists of 1D convolutional layers and TemporalMaxer layers. Finally, a lightweight classification and regression head decodes the featurepyramid to action candidates for every input moment.


label every input moment by $f ( \boldsymbol { X } )  \hat { \boldsymbol { \Psi } }$ with $f$ is a deeplearning model. $f$ follows an encoder-decoder design andcan be decomposed as $e \circ d$ . The encoder here is the back-bone, and the decoder is the classification and regressionhead. $e : X \to Z$ learn to encode the input video feature$X$ into latent vector $Z$ , and $d : Z \to { \hat { \Psi } }$ learns to predictlabels for every input moment. To effectively capture ac-tions transpiring at various temporal scales, we also adopta multi-scale feature pyramid representation, which is de-noted as $Z = \{ Z ^ { 1 } , Z ^ { 2 } , . . . , Z ^ { L } \}$ .

Encoder design The input feature $X$ is first en-coded into multi-scale temporal feature pyramid $Z =$$\{ Z ^ { 1 } , Z ^ { 2 } , . . . , Z ^ { L } \}$ using encoder $e$ . The encoder $e$ simplycontains two 1D convolutional neural network layers as fea-ture projection layers, followed by $L - 1$ Temporal Con-text Modeling (TCM) blocks to produce feature pyramid $Z$ .Formally, the feature projection layers are described as:

$$
X _ {p} = E _ {2} \left(E _ {1} (\operatorname {C o n c a t} (X))\right) \tag {3}
$$

The input video feature sequence $X = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { T } \}$ ,where $x _ { i } ~ \in ~ \mathbb { R } ^ { 1 \times D _ { i n } }$ , is first concatenated in the first di-mension and then fed into two feature projection modules$E _ { 1 }$ , and $E _ { 2 }$ in equation 3, resulting in the projected fea-ture $X _ { p } \in \mathbb { R } ^ { T \times D }$ with D-dimensional feature space. Eachprojection module comprises one 1D-convolutional neuralnetwork layer, followed by Layer Normalization [3], andReLU [1]. We simply assign $Z ^ { 1 } = X _ { p }$ as the first featurein $Z$ . Finally, the multi-scale temporal feature pyramid $Z$ is

encoded by TemporalMaxer:

$$
Z ^ {l} = \operatorname {T e m p o r a l M a x e r} \left(Z ^ {l - 1}\right). \tag {4}
$$

Here: TemporalMaxer is Max Pooling and employed withstride 2, $\bar { Z ^ { l } } \in \mathbb R ^ { \frac { T } { 2 ^ { l - 1 } } \times D }$ , $2 < = l < = L$ . It is worth not-ing that ActionFormer [60] employs Transformer [51] as aTCM block where each clip feature at the moment $t$ repre-sents a token, our proposed method adopts only Max Pool-ing [5] as TCM block.

Decoder Design The decoder $d$ learns to predictsequence labeling, $\begin{array} { c c l } { \hat { \Psi } } & { = } & { \left\{ \hat { \psi } _ { 1 } , \hat { \psi } _ { 2 } , \dots , \hat { \psi } _ { T } \right\} } \end{array}$ , for ev-ery moment using multi-scale feature pyramid $Z =$$\{ Z ^ { 1 } , Z ^ { 2 } , . . . , Z ^ { L } \}$ . The decoder adopts a lightweight con-volutional neural network and consists of classification andregression heads. Formally, the two heads are defined as:

$$
C _ {l} = \mathcal {F} _ {c} \left(E _ {4} \left(E _ {3} \left(Z ^ {l}\right)\right)\right) \tag {5}
$$

$$
O _ {l} = \operatorname {R e L U} \left(\mathcal {F} _ {o} \left(E _ {6} \left(E _ {5} \left(Z ^ {l}\right)\right)\right)\right) \tag {6}
$$

Here, $\begin{array} { r l r } { Z ^ { l } } & { { } \in } & { \mathbb { R } ^ { \frac { T } { 2 ^ { l - 1 } } \times C } } \end{array}$ is the latent feature of level$ \mathrm { ~ , ~ } \ C _ { l } = \{ c _ { 0 } , c _ { 2 ^ { l - 1 } } , . . . , c _ { T } \} \in \mathbb { R } ^ { \frac { T } { 2 ^ { l - 1 } } \times C }$ denotes theclassification probability with $c _ { i } \in \mathbb { R } ^ { C }$ , and $\begin{array} { r l } { O _ { l } } & { { } = } \end{array}$$\left\{ \left( o _ { 0 } ^ { s } , o _ { 0 } ^ { e } \right) , \left( o _ { 2 ^ { l - 1 } } ^ { s } , o _ { 2 ^ { l - 1 } } ^ { e } \right) , . . . , \left( o _ { T } ^ { s } , o _ { T } ^ { e } \right) \right\} \in \mathbb { R } ^ { \frac { T } { 2 ^ { l - 1 } } \times 2 } .$ is the on-set and offset prediction of input moment $\{ 0 , 2 ^ { l - 1 } , . . . , T \}$ .$E$ denotes the 1D convolution followed by Layer Normal-ization and ReLU activation function. $\mathcal { F } _ { c }$ and $\mathcal { F } _ { o }$ are both1D convolution. Note that all the weights of the decoderare shared between the different features in the multi-scalefeature pyramid $Z$ .

Learning Objective. The model predicts $\begin{array} { r l } { \hat { \psi } _ { t } } & { { } = } \end{array}$$( o _ { t } ^ { s } , o _ { t } ^ { e } , c _ { t } )$ for every moment of the input $X$ . Following thebaseline [60], the Focal Loss [31] and DIoU loss [64] areemployed to supervise classification and regression outputsrespectively. The overall loss function is defined as:

$$
\mathcal {L} _ {\text {t o t a l}} = \sum_ {t} \left(\mathcal {L} _ {\text {c l s}} + \mathbb {1} _ {c _ {t}} \mathcal {L} _ {\text {r e g}}\right) / T _ {+} \tag {7}
$$

where $\mathcal { L } _ { \boldsymbol { r } e g }$ denotes regression loss and is applied onlywhen the indicator function, $\mathbb { 1 } _ { c _ { t } }$ , indicates that the currenttime step $t$ is a positive sample. $T _ { + }$ is the number of positivesamples. $\mathcal { L } _ { c l s }$ is $C$ way classification loss. The loss function$\mathcal { L } _ { t o t a l }$ is applied to all levels on the output of multi-scalefeature pyramid $Z$ and averaged across all video samplesduring training.

# 4. Experimental Results

In this section, we show that our proposed method, Tem-poralMaxer, demonstrates the outstanding results achievedacross a variety of challenging datasets, namely THUMOS[22], EPIC-Kitchens 100 [11], MultiTHUMOS [57], andMUSES [34]. These datasets are recognized as standardbenchmarks in the Temporal Action Localization task. Ourapproach surpasses the state-of-the-art baseline, Action-Former [60], in each dataset, showcasing its superior per-formance compared to other works.

Evaluation Metric. We employ a widely-used eval-uation metric for TAL known as the mean average pre-cision (mAP) calculated at various temporal intersectionsover union (tIoU). tIoU is the intersection over union be-tween two temporal windows, i.e., the 1D Jaccard index.We report the mAP scores for all action categories basedon the given tIoU thresholds, and further report an averagedmAP value across all tIoU thresholds.

Training Details. To ensure a fair and unbiased com-parison, we employed the experimental setup of the ro-bust baseline model, ActionFormer [60]. This setup in-cluded various components such as decoder design $d$ ,non-maximum suppression (NMS) hyper-parameters in thepost-processing stage, data augmentation, learning rate, fea-ture extraction, and the number of feature pyramid level $L$ .The sole variation in our study was the substitution of theTransformer block in ActionFormer with the Max Poolingblock. All experiments are conducted with a kernel size of3 for all TCM blocks. The subsequent ablation will thor-ough analysis of the effects of varying kernel sizes. Duringtraining, the input feature length is kept constant at 2304,corresponding to approximately 5 minutes of video on bothTHUMOS14 and MultiTHUMOS datasets, roughly 20 min-utes on the EPIC-Kitchens 100 dataset, and approximately45 minutes on MUSES. Additionally, Model EMA [21] andgradient clipping techniques are employed, consistent withthose used in [60], to promote training stability.

# 4.1. Results on THUMOS14

Dataset. THUMOS14 dataset [22] contains 200 valida-tion videos and 213 testing videos with 20 action classes.Following previous work [29, 30, 54, 62, 60], we trainedthe model using validation videos and measured the perfor-mance on testing videos.

Feature Extraction. Following [60, 62], we extract thefeatures of THUMOS14 dataset using two-stream I3D [7]pre-trained on Kinetics [24]. 16 consecutive frames are fedinto I3D pre-trained network with a sliding window of stride4. The extracted feature is collected after the last fully con-nected layer and has 1024-D feature space. After that, thetwo-stream features are further concatenated (2048-D) andutilized as the input of the model.

Results. We compare the performance evaluated onthe THUMOS14 dataset [22] with state-of-the-art meth-ods. TemporalMaxer demonstrates remarkable perfor-mance, achieving an average mAP of $6 7 . 7 \%$ mAP, out-performing all previous approaches, both single-stage, andtwo-stage methods, by a significant margin, with a $1 . 1 \%$ in-crease in mAP at tIoU=0.4. Especially, TemporalMaxer sur-passes all recent methods that utilize long-term TCM blocksincluding self-attention such as TadTR [35], HTNet [23],TAGS [38], GLFormer [19], ActionFormer [60], or Graph-based like G-TAD [54], VSGN [61], or complex moduleincluding Local-Global Temporal Encoder [40].

Moreover, the comparisons between our method andother approaches show that the proposed method not onlyexhibits outstanding performance but also is efficient interms of inference speed. Specifically, our method onlytakes 50 ms on average to fully process an entire video onTHUMOS. It is 1.6x faster than the ActionFormer baselineand $3 . 9 \mathrm { x }$ faster than TadTR. However, in section 4.5, weshow that our model only takes 10.4 ms for forward time. Itmeans that the rest 39.6 ms is NMS time, causing the mosttime-consuming. In the later section 4.5, we show that theforward time and the backbone time of our model are $2 . 9 \mathbf { x }$and $8 . 0 \mathbf { x }$ faster than ActionFormer, respectively.

# 4.2. Results on EPIC-Kitchens 100

Dataset. The EPIC-Kitchens 100 dataset [11] is a com-prehensive collection of egocentric action videos, featuring100 hours of footage from 700 sessions that document cook-ing activities in a variety of kitchens. Additionally, EPIC-Kitchens 100 is three times larger in terms of total videohours and more than ten times larger in terms of action in-stances (averaging 128 per video) when compared to THU-MOS14. These videos are recorded from a first-person per-spective, resulting in significant camera motion, and repre-sent a novel challenge for TAL research.

Feature Extraction. Following previous work [60, 29,54], we extract the videos feature using SlowFast network[15] pre-trained on EPICKitchens [11]. We utilized a 32-

<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td rowspan="2">Feature</td><td colspan="6">tIoU↑</td><td rowspan="2">time(ms)↓</td></tr><tr><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>Avg.</td></tr><tr><td rowspan="16">Two-Stage</td><td>BMN [29]</td><td>TSN [52]</td><td>56.0</td><td>47.4</td><td>38.8</td><td>29.7</td><td>20.5</td><td>38.5</td><td>483*</td></tr><tr><td>DBG [27]</td><td>TSN [52]</td><td>57.8</td><td>49.4</td><td>39.8</td><td>30.2</td><td>21.7</td><td>39.8</td><td>—</td></tr><tr><td>G-TAD [54]</td><td>TSN [52]</td><td>54.5</td><td>47.6</td><td>40.3</td><td>30.8</td><td>23.4</td><td>39.3</td><td>4440*</td></tr><tr><td>BC-GNN [4]</td><td>TSN [52]</td><td>57.1</td><td>49.1</td><td>40.4</td><td>31.2</td><td>23.1</td><td>40.2</td><td>—</td></tr><tr><td>TAL-MR [62]</td><td>I3D [7]</td><td>53.9</td><td>50.7</td><td>45.4</td><td>38.0</td><td>28.5</td><td>43.3</td><td>&gt;644*</td></tr><tr><td>P-GCN [59]</td><td>I3D [7]</td><td>63.6</td><td>57.8</td><td>49.1</td><td>—</td><td>—</td><td>—</td><td>7298*</td></tr><tr><td>P-GCN [59] +TSP [2]</td><td>R(2+1)1 D [50]</td><td>69.1</td><td>63.3</td><td>53.5</td><td>40.4</td><td>26.0</td><td>50.5</td><td>—</td></tr><tr><td>TSA-Net [17]</td><td>P3D [41]</td><td>61.2</td><td>55.9</td><td>46.9</td><td>36.1</td><td>25.2</td><td>45.1</td><td>—</td></tr><tr><td>MUSES [34]</td><td>I3D [7]</td><td>68.9</td><td>64.0</td><td>56.9</td><td>46.3</td><td>31.0</td><td>53.4</td><td>2101*</td></tr><tr><td>TCANet [40]</td><td>TSN [52]</td><td>60.6</td><td>53.2</td><td>44.6</td><td>36.8</td><td>26.7</td><td>44.3</td><td>—</td></tr><tr><td>BMN-CSA [45]</td><td>TSN [52]</td><td>64.4</td><td>58.0</td><td>49.2</td><td>38.2</td><td>27.8</td><td>47.7</td><td>—</td></tr><tr><td>ContextLoc [65]</td><td>I3D [7]</td><td>68.3</td><td>63.8</td><td>54.3</td><td>41.8</td><td>26.2</td><td>50.9</td><td>—</td></tr><tr><td>VSGN [61]</td><td>TSN [52]</td><td>66.7</td><td>60.4</td><td>52.4</td><td>41.0</td><td>30.4</td><td>50.2</td><td>—</td></tr><tr><td>RTD-Net [46]</td><td>I3D [7]</td><td>68.3</td><td>62.3</td><td>51.9</td><td>38.8</td><td>23.7</td><td>49.0</td><td>&gt;211*</td></tr><tr><td>Disentangle [66]</td><td>I3D [7]</td><td>72.1</td><td>65.9</td><td>57.0</td><td>44.2</td><td>28.5</td><td>53.5</td><td>—</td></tr><tr><td>SAC [55]</td><td>I3D [7]</td><td>69.3</td><td>64.8</td><td>57.6</td><td>47.0</td><td>31.5</td><td>54.0</td><td>—</td></tr><tr><td rowspan="11">Single-Stage</td><td>A2Net [56]</td><td>I3D [7]</td><td>58.6</td><td>54.1</td><td>45.5</td><td>32.5</td><td>17.2</td><td>41.6</td><td>1554*</td></tr><tr><td>GTAN [36]</td><td>P3D [41]</td><td>57.8</td><td>47.2</td><td>38.8</td><td>—</td><td>—</td><td>—</td><td>—</td></tr><tr><td>PBRNet [32]</td><td>I3D [7]</td><td>58.5</td><td>54.6</td><td>51.3</td><td>41.8</td><td>29.5</td><td>—</td><td>—</td></tr><tr><td>AFSD [28]</td><td>I3D [7]</td><td>67.3</td><td>62.4</td><td>55.5</td><td>43.7</td><td>31.1</td><td>52.0</td><td>3245*</td></tr><tr><td>TAGS [38]</td><td>I3D [7]</td><td>68.6</td><td>63.8</td><td>57.0</td><td>46.3</td><td>31.8</td><td>52.8</td><td>—</td></tr><tr><td>HTNet [23]</td><td>I3D [7]</td><td>71.2</td><td>67.2</td><td>61.5</td><td>51.0</td><td>39.3</td><td>58.0</td><td>—</td></tr><tr><td>TadTR [35]</td><td>I3D [7]</td><td>74.8</td><td>69.1</td><td>60.1</td><td>46.6</td><td>32.8</td><td>56.7</td><td>195*</td></tr><tr><td>GLFormer [19]</td><td>I3D [7]</td><td>75.9</td><td>72.6</td><td>67.2</td><td>57.2</td><td>41.8</td><td>62.9</td><td>—</td></tr><tr><td>AMNet [35]</td><td>I3D [7]</td><td>76.7</td><td>73.1</td><td>66.8</td><td>57.2</td><td>42.7</td><td>63.3</td><td>—</td></tr><tr><td>ActionFormer [60]</td><td>I3D [7]</td><td>82.1</td><td>77.8</td><td>71.0</td><td>59.4</td><td>43.9</td><td>66.8</td><td>80</td></tr><tr><td>ActionFormer [60] + GAP [37]</td><td>I3D [7]</td><td>82.3</td><td>—</td><td>71.4</td><td>—</td><td>44.2</td><td>66.9</td><td>&gt;80</td></tr><tr><td></td><td>Our (TemporalMaxer)</td><td>I3D [7]</td><td>82.8</td><td>78.9</td><td>71.8</td><td>60.5</td><td>44.7</td><td>67.7</td><td>50</td></tr></table>


Table 1. The results obtained on the THUMOS14 dataset [22] are presented for various tIoU thresholds, with the average mAP calculatedin the range [0.3:0.7:0.1]. The top-performing results are highlighted in bold. The time(ms) is the average inference time for one video,without extracting features from 3D CNN and including the post-processing step, such as NMS. We measure the inference time usinga single GeForce GTX 1080 Ti GPU. Results indicated with * are taken from [35] which are measured using Tesla P100 GPU, a muchmore powerful GPU than the 1080 Ti. In comparison to early works, including both one-stage and two-stage methods, and those utilizinglong-term TCM, TemporalMaxer achieves superior performance in both mAP and inference speed.


frame input sequence with a stride of 16 to generate a set of2304-D features. These features were then fed as input toour model.

Result. Tab. 2 shows our results. TemporalMaxerdemonstrates notable performance on the EPIC-Kitchens100 dataset, achieving an average mAP of $2 4 . 5 \%$ and $2 2 . 8 \%$for verb and noun, respectively. The superiority of our ap-proach is further confirmed by a large margin over a strongand robust baseline, ActionFormer [60], with an averageimprovement of $1 . 0 \%$ mAP for verb and $0 . 9 \%$ mAP fornoun. Again, TemporalMaxer outperforms other methodsthat utilize long-term TCM including self-attention [60] orGraph [54]. These results provide empirical evidence ofthe effectiveness of the simplest backbone in advancing thestate-of-the-art on this challenging task.

# 4.3. Results on MUSES

Dataset. The MUSES dataset [34] is a collection of3,697 videos, with 2,587 for training and 1,110 for testing.MUSES has 31,477 action instances over a duration of 716

video hours with 25 action classes, designed to facilitatemulti-shot analyses making the dataset challenging.

Feature Extraction. We directly employ the pre-extracted feature provided by [34]. The feature is extractedusing a pre-trained I3D network[7] on the Kinetics dataset[24] using only RGB stream, resulting in 1024-D featurespace for a video clip embedding.

Result Tab. 3 shows the results on MUSES dataset. Ourmethod significantly outperforms other works that employlong-term TCM such as G-TAD [54], Ag-Trans [63], andActionFormer [60] at every tIoU threshold. Notably, Tem-poralMaxer improves 1.0 mAP at $\mathrm { \ t I o U { = } } 0 . 7$ . On average,we achieve $2 7 . 2 \ \mathrm { m A P } ,$ which is 1.0 mAP higher than theprevious approaches, demonstrating the robustness of ourmethod. It is worth noting that we implemented the Action-Former [60] on the MUSES dataset using the code providedby the authors.

<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td colspan="6">tIoU</td></tr><tr><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>Avg</td></tr><tr><td rowspan="4">Verb</td><td>BMN [29, 11]</td><td>10.8</td><td>9.8</td><td>8.4</td><td>7.1</td><td>5.6</td><td>8.4</td></tr><tr><td>G-TAD [54]</td><td>12.1</td><td>11.0</td><td>9.4</td><td>8.1</td><td>6.5</td><td>9.4</td></tr><tr><td>ActionFormer [60]</td><td>26.6</td><td>25.4</td><td>24.2</td><td>22.3</td><td>19.1</td><td>23.5</td></tr><tr><td>Our (TemporalMaxer)</td><td>27.8</td><td>26.6</td><td>25.3</td><td>23.1</td><td>19.9</td><td>24.5</td></tr><tr><td rowspan="4">Noun</td><td>BMN [29, 11]</td><td>10.3</td><td>8.3</td><td>6.2</td><td>4.5</td><td>3.4</td><td>6.5</td></tr><tr><td>G-TAD [54]</td><td>11.0</td><td>10.0</td><td>8.6</td><td>7.0</td><td>5.4</td><td>8.4</td></tr><tr><td>ActionFormer [60]</td><td>25.2</td><td>24.1</td><td>22.7</td><td>20.5</td><td>17.0</td><td>21.9</td></tr><tr><td>Our (TemporalMaxer)</td><td>26.3</td><td>25.2</td><td>23.5</td><td>21.3</td><td>17.6</td><td>22.8</td></tr></table>


Table 2. The performance of our proposed method on the EPIC-Kitchens 100 dataset [11] is evaluated using various tIoU thresholds. Theaverage mAP is reported over a range of tIoU thresholds [0.1:0.5:0.1]. The top-performing methods are highlighted in bold. Our proposedmethod outperforms the other methods significantly.


<table><tr><td rowspan="2">Method</td><td colspan="4">tIoU</td></tr><tr><td>0.3</td><td>0.5</td><td>0.7</td><td>Avg</td></tr><tr><td>BU-TAL [62]</td><td>12.9</td><td>9.2</td><td>5.9</td><td>9.4</td></tr><tr><td>G-TAD [54]</td><td>19.1</td><td>11.1</td><td>4.7</td><td>11.4</td></tr><tr><td>P-GCN [59]</td><td>19.9</td><td>13.1</td><td>5.4</td><td>13.0</td></tr><tr><td>MUSES [34]</td><td>25.9</td><td>18.9</td><td>10.6</td><td>18.6</td></tr><tr><td>Ag-Trans [63]</td><td>24.8</td><td>19.4</td><td>10.9</td><td>18.6</td></tr><tr><td>ActionFormer [60]</td><td>35.9</td><td>26.9</td><td>15.2</td><td>26.2</td></tr><tr><td>Our (TemporalMaxer)</td><td>36.7</td><td>27.8</td><td>16.2</td><td>27.2</td></tr></table>


Table 3. We report mAP at different tIoU thresholds [0.3, 0.5, 0.7]and the average mAP in [0.3:0.1:0.7] on MUSES dataset [34]. Allmethods used the same I3D features. Our method outperformsconcurrent works by a large margin.


# 4.4. Results on MultiTHUMOS

Dataset. The MultiTHUMOS dataset [57] is a denselylabeled extension of THUMOS14, consisting of 413 sportsvideos with 65 distinct action classes. The dataset presentsa significant increase in the average number of distinctiveaction categories per video, compared to THUMOS14. Assuch, it poses a greater challenge for TAL than THUMOS.While MultiTHUMOS are being used in action detectionbenchmark [10, 9], a novel approach for action detection,PointTAD [47], utilizes the TAL evaluation metric to assessthe completeness of predicted action instances. Given thatTAL and action detection share the same setting in terms ofinput features and annotations, we evaluate the performanceof our model on MultiTHUMOS and compare it against thestate-of-the-art action detection methods [10, 48, 9, 47], andstrong baseline ActionFormer [60].

Feature Extraction. We only utilize RGB stream as in-put for I3D network [7] pre-trained on Kinetics [24], fol-lowing [47], to extract features for MultiTHUMOS. TheI3D pre-trained network is fed with 16 sequential framesthrough a sliding window with a stride of 4. The feature isextracted from the final fully connected layer, resulting in a1024-D feature space that serves as input for the model.

Result. Tab. 4 provides a comparison of our perfor-

<table><tr><td rowspan="2">Method</td><td colspan="4">tIoU</td></tr><tr><td>0.2</td><td>0.5</td><td>0.7</td><td>Avg</td></tr><tr><td>PDAN [10]</td><td>—</td><td>—</td><td>—</td><td>17.3</td></tr><tr><td>MLAD [48]</td><td>—</td><td>—</td><td>—</td><td>14.2</td></tr><tr><td>MS-TCT [9]</td><td>—</td><td>—</td><td>—</td><td>16.2</td></tr><tr><td>PointTAD [47]</td><td>39.7</td><td>24.9</td><td>12.0</td><td>23.5</td></tr><tr><td>ActionFormer [60]</td><td>46.4</td><td>32.4</td><td>15.0</td><td>28.6</td></tr><tr><td>Our (TemporalMaxer)</td><td>47.5</td><td>33.4</td><td>17.4</td><td>29.9</td></tr></table>


Table 4. Comparison with the state-of-the-art methods on the Mul-tiTHUMOS dataset. We report the results at different tIoU thresh-olds [0.2, 0.5, 0.7] and average mAP in [0.1:0.9:0.1].


mance on the MultiTHUMOS dataset [57] with recent state-of-the-art methods. Specifically, our method surpasses theprior work [47] that utilizes TransFormer as feature en-coding by a large margin, $6 . 4 \%$ mAP on average. More-over, TemporalMaxer improves the robust baseline, Action-Former [60], by $2 . 4 \%$ mAP at tIoU=0.7, and $1 . 3 \%$ mAP onaverage. It should be noted that we utilized the code pro-vided by the authors to implement ActionFormer [60] onthe MultiTHUMOS dataset, and the results [10, 48, 9] aretaken from [47].

# 4.5. Ablation Study

We perform various ablation studies to verify the effec-tiveness of TemporalMaxer. To better understand what isthe effective component for TCM, we gradually replace theMax Pooling with other blocks such as convolution, sub-sampling, and Average Pooling. Furthermore, we evaluatenumerous kernel sizes of Max Pooling. Note that all exper-iments in this section are conducted on the train and valida-tion set of the THUMOS14 dataset.

Effective of TemporalMaxer. Tab. 5 presents the re-sults of other blocks other than Max Pooling. Motivatedby PoolFormer [58] that replaces the computationally in-tensive and highly parameterized attention module with themost basic block in deep learning, the pooling layer. Ourstudies started by first questioning the most straightforwardapproach to leverage the potential of the extracted features

<table><tr><td rowspan="2">TCM</td><td colspan="3">tIoU</td><td rowspan="2">GMACs ↓</td><td rowspan="2">#params (M) ↓</td><td rowspan="2">time (ms) ↓</td><td rowspan="2">backbone time (ms) ↓</td></tr><tr><td>0.5</td><td>0.7</td><td>Avg.</td></tr><tr><td>Conv [28] (Our Impl)</td><td>62.8</td><td>37.1</td><td>59.4</td><td>45.6</td><td>30.5</td><td>16.3</td><td>9.0</td></tr><tr><td>Subsampling</td><td>64.3</td><td>37.7</td><td>61.0</td><td>16.2</td><td>7.1</td><td>10.4</td><td>2.5</td></tr><tr><td>Average Pooling</td><td>66.1</td><td>39.4</td><td>63.2</td><td>16.4</td><td>7.1</td><td>10.4</td><td>2.5</td></tr><tr><td>Transformer [51]</td><td>71.0</td><td>43.9</td><td>66.8</td><td>45.3</td><td>29.3</td><td>30.5</td><td>20.1</td></tr><tr><td>TemporalMaxer</td><td>71.8</td><td>44.7</td><td>67.7</td><td>16.4</td><td>7.1</td><td>10.4</td><td>2.5</td></tr></table>


Table 5. Ablation studies about different TCM blocks on THUMOS14. Inference times are measured using an input video with 2304 clipembeddings, a 5 minutes video, on a GeForce GTX 1080 Ti GPU without post-processing (NMS) and pre-extracted features step.


from 3D-CNN for the TAL task. PoolFormer retains thestructure of Transformer such as FFN [43, 44], residualconnection [18], and Layer Normalization [3] because thetokens have to be encoded by the Poolformer itself. How-ever, in TAL the features from 3D CNN have already beenpre-extracted and contain useful information. Therefore,we posit that there will be a straightforward block that aremore efficient than Transformer/PoolFormer, does not re-quire much computational as well as parameters, and beable to effectively exploit the pre-extracted features.

Our ablation study starts to employ a 1-D convolutionmodule as an alternative to Transformer [51] for TCM blockto see how many mAP drop. The result is reported in thefirst row of Tab. 5. As expected, the convolution layerdecreases by $7 . 4 \%$ mAP compared with the ActionFormerbaseline. This deduction can be explained for two reasons.First, given the video clip features that are informative buthighly similar, and the convolution weight is fixed aftertraining, consequently the convolution operation cannot re-tain the most informative features of local clip embeddings.Secondly, the convolution layer introduces the most param-eters which tend to overparameterize the model. We arguethat given the informative features extracted from the pre-trained 3D CNN, the model should not contain too manyparameters, which may lead to overfitting and thus less gen-eralization. To further concrete our thought, we replace theTransformer with a parameter-free operation, subsamplingtechnique, in which features at even indexes are kept andodd indexes are removed to simulate stride 2 of TCM. Sur-prisingly, this none-parameter operation achieves higher re-sults than the convolution layer, shown in the second row ofTab. 5. This finding proves that TCM block may not need tocontain many parameters, and the features from pretrained3D CNN are informative and is the potential for TAL.

However, subsampling is prone to losing the most cru-cial information as only half of the video clip embeddingsare kept after a TCM block. Thus, we replace Transformerwith Average Pooling with kernel size 3 and stride 2. Asexpected, Average Pooling improves the result by $2 . 2 ~ \%$mAP on average compared with the subsampling. It isbecause this operation does not drop any clip embeddingwhich helps to retain the crucial information. However, theAverage Pooling averages the nearby features, thus the cru-

cial information of adjacent clips is reduced. That is whyAverage Pooling decreases by $3 . 6 \%$ mAP compared withTransformer.

The ablation study with Average Pooling suggests thatthe most important information of clip embeddings shouldbe maintained. For that reason, we employ Max Pooling asTCM. The result is provided in the last row of Tab. 5. MaxPooling achieves the highest results at every tIoU thresholdand significantly outperforms the strong and robust base-line, ActionFormer. To clarify, TemporalMaxer effectivelyhighlights only critical information from nearby clips anddiscards the less important information. This result suggeststhat the feature from pretrained 3D CNN are informativeand can be effectively utilized for the TAL model withoutcomplex modules like prior works.

Our method, TemporalMaxer, results in the simplestmodel ever for TAL task that contains minimalist param-eters and computational cost for the TAL model. Tempo-ralMaxer is effective at modeling temporal contexts, whichoutperforms the robust baseline, ActionFormer, with $2 . 8 \mathbf { x }$fewer GMACs and $3 \mathbf { x }$ faster inference speed. Especially,when comparing only the backbone time, our proposedmethod only takes $2 . 5 ~ \mathrm { m s }$ which is incredibly 8.0x fasterthan ActionFormer backbone [63], 20.1 ms.

Different Values of kernel size. We make an ablationwith the kernel size of TemporalMaxer 3, 4, 5, 6 and theaverage mAPs are 67.7, 67.1, 66.8, 65.7, respectively. Ourmodel achieved the highest performance with a kernel sizeof 3, while the lowest performance was observed with a ker-nel size of 6. This decrease in performance can be attributedto the corresponding loss of information during training.The results obtained for kernel sizes 3, 4, and 5 are not verysensitive to the kernel size. This suggests that these kernelsizes can effectively capture the relevant temporal informa-tion for the task at hand.

# 5. Conclusion

In this paper, we propose an extremely simplified ap-proach, TemporalMaxer for temporal action localizationtask. To minimize the structure, we explore the way tosimply maximize the underlying information in the videoclip features from pre-trained 3D-CNN. To this end, with abasic, non-parametric and temporally local operating max-

pooling block which can effectively and efficiently keepthe local minute changes among the sequential similar in-put images. We achieve competitive performance to otherstate-of-the-art methods with sophisticated, parametric andlong-term temporal context modeling models.

# References



[1] Abien Fred Agarap. Deep learning using rectified linear units(relu). arXiv preprint arXiv:1803.08375, 2018.





[2] Humam Alwassel, Silvio Giancola, and Bernard Ghanem.Tsp: Temporally-sensitive pretraining of video encoders forlocalization tasks. In Proceedings of the IEEE/CVF Inter-national Conference on Computer Vision, pages 3173–3183,2021.





[3] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hin-ton. Layer normalization. arXiv preprint arXiv:1607.06450,2016.





[4] Yueran Bai, Yingying Wang, Yunhai Tong, Yang Yang,Qiyue Liu, and Junhui Liu. Boundary content graph neuralnetwork for temporal action proposal generation. In Com-puter Vision–ECCV 2020: 16th European Conference, Glas-gow, UK, August 23–28, 2020, Proceedings, Part XXVIII 16,pages 121–137. Springer, 2020.





[5] Y-Lan Boureau, Jean Ponce, and Yann LeCun. A theoreticalanalysis of feature pooling in visual recognition. In Proceed-ings of the 27th international conference on machine learn-ing (ICML-10), pages 111–118, 2010.





[6] Shyamal Buch, Victor Escorcia, Chuanqi Shen, BernardGhanem, and Juan Carlos Niebles. Sst: Single-stream tem-poral action proposals. In Proceedings of the IEEE con-ference on Computer Vision and Pattern Recognition, pages2911–2920, 2017.





[7] Joao Carreira and Andrew Zisserman. Quo vadis, actionrecognition? a new model and the kinetics dataset. In pro-ceedings of the IEEE Conference on Computer Vision andPattern Recognition, pages 6299–6308, 2017.





[8] Shuning Chang, Pichao Wang, Fan Wang, Hao Li, and Ji-ashi Feng. Augmented transformer with adaptive graphfor temporal action proposal generation. arXiv preprintarXiv:2103.16024, 2021.





[9] Rui Dai, Srijan Das, Kumara Kahatapitiya, Michael S Ryoo,and Franc¸ois Bremond. Ms-tct: multi-scale temporal con- ´vtransformer for action detection. In Proceedings of theIEEE/CVF Conference on Computer Vision and PatternRecognition, pages 20041–20051, 2022.





[10] Rui Dai, Srijan Das, Luca Minciullo, Lorenzo Garattoni, Gi-anpiero Francesca, and Franc¸ois Bremond. Pdan: Pyramiddilated attention network for action detection. In Proceed-ings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 2970–2979, 2021.





[11] Dima Damen, Hazel Doughty, Giovanni Maria Farinella,Antonino Furnari, Evangelos Kazakos, Jian Ma, DavideMoltisanti, Jonathan Munro, Toby Perrett, Will Price,et al. Rescaling egocentric vision. arXiv preprintarXiv:2006.13256, 2020.





[12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,





Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-vain Gelly, et al. An image is worth 16x16 words: Trans-formers for image recognition at scale. arXiv preprintarXiv:2010.11929, 2020.





[13] Victor Escorcia, Fabian Caba Heilbron, Juan Carlos Niebles,and Bernard Ghanem. Daps: Deep action proposals for ac-tion understanding. In Computer Vision–ECCV 2016: 14thEuropean Conference, Amsterdam, The Netherlands, Octo-ber 11-14, 2016, Proceedings, Part III 14, pages 768–784.Springer, 2016.





[14] Yazan Abu Farha and Jurgen Gall. Ms-tcn: Multi-stage tem-poral convolutional network for action segmentation. In Pro-ceedings of the IEEE/CVF conference on computer visionand pattern recognition, pages 3575–3584, 2019.





[15] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, andKaiming He. Slowfast networks for video recognition. InProceedings of the IEEE/CVF international conference oncomputer vision, pages 6202–6211, 2019.





[16] Jialin Gao, Zhixiang Shi, Guanshuo Wang, Jiani Li, YufengYuan, Shiming Ge, and Xi Zhou. Accurate temporal actionproposal generation with relation-aware pyramid network.In Proceedings of the AAAI conference on artificial intelli-gence, volume 34, pages 10810–10817, 2020.





[17] Guoqiang Gong, Liangfeng Zheng, and Yadong Mu. Scalematters: Temporal scale aggregation network for precise ac-tion localization in untrimmed videos. In 2020 IEEE Inter-national Conference on Multimedia and Expo (ICME), pages1–6. IEEE, 2020.





[18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.Deep residual learning for image recognition. In Proceed-ings of the IEEE conference on computer vision and patternrecognition, pages 770–778, 2016.





[19] Yilong He, Yong Zhong, Lishun Wang, and Jiachen Dang.Glformer: Global and local context aggregation network fortemporal action detection. Applied Sciences, 12(17):8557,2022.





[20] Fabian Caba Heilbron, Juan Carlos Niebles, and BernardGhanem. Fast temporal activity proposals for efficient de-tection of human actions in untrimmed videos. In Proceed-ings of the IEEE conference on computer vision and patternrecognition, pages 1914–1923, 2016.





[21] Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, John EHopcroft, and Kilian Q Weinberger. Snapshot ensembles:Train 1, get m for free. arXiv preprint arXiv:1704.00109,2017.





[22] Haroon Idrees, Amir R Zamir, Yu-Gang Jiang, Alex Gorban,Ivan Laptev, Rahul Sukthankar, and Mubarak Shah. Thethumos challenge on action recognition for videos “in thewild”. Computer Vision and Image Understanding, 155:1–23, 2017.





[23] Tae-Kyung Kang, Gun-Hee Lee, and Seong-Whan Lee. Ht-net: Anchor-free temporal action localization with hierarchi-cal transformers. In 2022 IEEE International Conferenceon Systems, Man, and Cybernetics (SMC), pages 365–370.IEEE, 2022.





[24] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang,Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola,





Tim Green, Trevor Back, Paul Natsev, et al. The kinetics hu-man action video dataset. arXiv preprint arXiv:1705.06950,2017.





[25] Thomas N Kipf and Max Welling. Semi-supervised classi-fication with graph convolutional networks. arXiv preprintarXiv:1609.02907, 2016.





[26] James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santi-ago Ontanon. Fnet: Mixing tokens with fourier transforms.arXiv preprint arXiv:2105.03824, 2021.





[27] Chuming Lin, Jian Li, Yabiao Wang, Ying Tai, DonghaoLuo, Zhipeng Cui, Chengjie Wang, Jilin Li, Feiyue Huang,and Rongrong Ji. Fast learning of temporal action pro-posal via dense boundary generator. In Proceedings of theAAAI conference on artificial intelligence, volume 34, pages11499–11506, 2020.





[28] Chuming Lin, Chengming Xu, Donghao Luo, Yabiao Wang,Ying Tai, Chengjie Wang, Jilin Li, Feiyue Huang, and Yan-wei Fu. Learning salient boundary feature for anchor-free temporal action localization. In Proceedings of theIEEE/CVF Conference on Computer Vision and PatternRecognition, pages 3320–3329, 2021.





[29] Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, and Shilei Wen.Bmn: Boundary-matching network for temporal action pro-posal generation. In Proceedings of the IEEE/CVF inter-national conference on computer vision, pages 3889–3898,2019.





[30] Tianwei Lin, Xu Zhao, Haisheng Su, Chongjing Wang, andMing Yang. Bsn: Boundary sensitive network for temporalaction proposal generation. In Proceedings of the Europeanconference on computer vision (ECCV), pages 3–19, 2018.





[31] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, andPiotr Dollar. Focal loss for dense object detection. In´ Pro-ceedings of the IEEE international conference on computervision, pages 2980–2988, 2017.





[32] Qinying Liu and Zilei Wang. Progressive boundary refine-ment network for temporal action detection. In Proceedingsof the AAAI conference on artificial intelligence, volume 34,pages 11612–11619, 2020.





[33] Wei Liu, Dragomir Anguelov, Dumitru Erhan, ChristianSzegedy, Scott Reed, Cheng-Yang Fu, and Alexander CBerg. Ssd: Single shot multibox detector. In ComputerVision–ECCV 2016: 14th European Conference, Amster-dam, The Netherlands, October 11–14, 2016, Proceedings,Part I 14, pages 21–37. Springer, 2016.





[34] Xiaolong Liu, Yao Hu, Song Bai, Fei Ding, Xiang Bai, andPhilip HS Torr. Multi-shot temporal event localization: abenchmark. In Proceedings of the IEEE/CVF Conferenceon Computer Vision and Pattern Recognition, pages 12596–12606, 2021.





[35] Xiaolong Liu, Qimeng Wang, Yao Hu, Xu Tang, ShiweiZhang, Song Bai, and Xiang Bai. End-to-end temporal ac-tion detection with transformer. IEEE Transactions on ImageProcessing, 31:5427–5441, 2022.





[36] Fuchen Long, Ting Yao, Zhaofan Qiu, Xinmei Tian, JieboLuo, and Tao Mei. Gaussian temporal awareness networksfor action localization. In Proceedings of the IEEE/CVFConference on Computer Vision and Pattern Recognition,pages 344–353, 2019.





[37] Sauradip Nag, Xiatian Zhu, Yi-Zhe Song, and Tao Xiang.Post-processing temporal action detection. arXiv preprintarXiv:2211.14924, 2022.





[38] Sauradip Nag, Xiatian Zhu, Yi-Zhe Song, and Tao Xiang.Proposal-free temporal action detection via global segmen-tation mask learning. In Computer Vision–ECCV 2022: 17thEuropean Conference, Tel Aviv, Israel, October 23–27, 2022,Proceedings, Part III, pages 645–662. Springer, 2022.





[39] Megha Nawhal and Greg Mori. Activity graph trans-former for temporal action localization. arXiv preprintarXiv:2101.08540, 2021.





[40] Zhiwu Qing, Haisheng Su, Weihao Gan, Dongliang Wang,Wei Wu, Xiang Wang, Yu Qiao, Junjie Yan, Changxin Gao,and Nong Sang. Temporal context aggregation networkfor temporal action proposal refinement. In Proceedings ofthe IEEE/CVF conference on computer vision and patternrecognition, pages 485–494, 2021.





[41] Zhaofan Qiu, Ting Yao, and Tao Mei. Learning spatio-temporal representation with pseudo-3d residual networks.In proceedings of the IEEE International Conference onComputer Vision, pages 5533–5541, 2017.





[42] Joseph Redmon, Santosh Divvala, Ross Girshick, and AliFarhadi. You only look once: Unified, real-time object de-tection. In Proceedings of the IEEE conference on computervision and pattern recognition, pages 779–788, 2016.





[43] Frank Rosenblatt. Principles of neurodynamics. perceptronsand the theory of brain mechanisms. Technical report, Cor-nell Aeronautical Lab Inc Buffalo NY, 1961.





[44] David E Rumelhart, Geoffrey E Hinton, and Ronald JWilliams. Learning internal representations by error propa-gation. Technical report, California Univ San Diego La JollaInst for Cognitive Science, 1985.





[45] Deepak Sridhar, Niamul Quader, Srikanth Muralidharan,Yaoxin Li, Peng Dai, and Juwei Lu. Class semantics-based attention for action detection. In Proceedings of theIEEE/CVF International Conference on Computer Vision,pages 13739–13748, 2021.





[46] Jing Tan, Jiaqi Tang, Limin Wang, and Gangshan Wu. Re-laxed transformer decoders for direct action proposal gener-ation. In Proceedings of the IEEE/CVF international confer-ence on computer vision, pages 13526–13535, 2021.





[47] Jing Tan, Xiaotong Zhao, Xintian Shi, Bing Kang, and LiminWang. Pointtad: Multi-label temporal action detection withlearnable query points. arXiv preprint arXiv:2210.11035,2022.





[48] Praveen Tirupattur, Kevin Duarte, Yogesh S Rawat, andMubarak Shah. Modeling multi-label action dependen-cies for temporal action localization. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and PatternRecognition, pages 1460–1470, 2021.





[49] Ilya O Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lu-cas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung,Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, et al.Mlp-mixer: An all-mlp architecture for vision. Advancesin neural information processing systems, 34:24261–24272,2021.





[50] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, YannLeCun, and Manohar Paluri. A closer look at spatiotemporal





convolutions for action recognition. In Proceedings of theIEEE conference on Computer Vision and Pattern Recogni-tion, pages 6450–6459, 2018.





[51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-reit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and IlliaPolosukhin. Attention is all you need. Advances in neuralinformation processing systems, 30, 2017.





[52] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, DahuaLin, Xiaoou Tang, and Luc Van Gool. Temporal segment net-works: Towards good practices for deep action recognition.In European conference on computer vision, pages 20–36.Springer, 2016.





[53] Lining Wang, Haosen Yang, Wenhao Wu, Hongxun Yao,and Hujie Huang. Temporal action proposal generation withtransformers. arXiv preprint arXiv:2105.12043, 2021.





[54] Mengmeng Xu, Chen Zhao, David S Rojas, Ali Thabet, andBernard Ghanem. G-tad: Sub-graph localization for tempo-ral action detection. In Proceedings of the IEEE/CVF Con-ference on Computer Vision and Pattern Recognition, pages10156–10165, 2020.





[55] Le Yang, Junwei Han, Tao Zhao, Nian Liu, and DingwenZhang. Structured attention composition for temporal actionlocalization. IEEE Transactions on Image Processing, 2022.





[56] Le Yang, Houwen Peng, Dingwen Zhang, Jianlong Fu, andJunwei Han. Revisiting anchor mechanisms for temporal ac-tion localization. IEEE Transactions on Image Processing,29:8535–8548, 2020.





[57] Serena Yeung, Olga Russakovsky, Ning Jin, Mykhaylo An-driluka, Greg Mori, and Li Fei-Fei. Every moment counts:Dense detailed labeling of actions in complex videos. Inter-national Journal of Computer Vision, 126:375–389, 2018.





[58] Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou,Xinchao Wang, Jiashi Feng, and Shuicheng Yan. Metaformeris actually what you need for vision. In Proceedings ofthe IEEE/CVF conference on computer vision and patternrecognition, pages 10819–10829, 2022.





[59] Runhao Zeng, Wenbing Huang, Mingkui Tan, Yu Rong,Peilin Zhao, Junzhou Huang, and Chuang Gan. Graph con-volutional networks for temporal action localization. In Pro-ceedings of the IEEE/CVF international conference on com-puter vision, pages 7094–7103, 2019.





[60] Chen-Lin Zhang, Jianxin Wu, and Yin Li. Actionformer: Lo-calizing moments of actions with transformers. In ComputerVision–ECCV 2022: 17th European Conference, Tel Aviv, Is-rael, October 23–27, 2022, Proceedings, Part IV, pages 492–510. Springer, 2022.





[61] Chen Zhao, Ali K Thabet, and Bernard Ghanem. Video self-stitching graph network for temporal action localization. InProceedings of the IEEE/CVF International Conference onComputer Vision, pages 13658–13667, 2021.





[62] Peisen Zhao, Lingxi Xie, Chen Ju, Ya Zhang, Yanfeng Wang,and Qi Tian. Bottom-up temporal action localization withmutual regularization. In Computer Vision–ECCV 2020:16th European Conference, Glasgow, UK, August 23–28,2020, Proceedings, Part VIII 16, pages 539–555. Springer,2020.





[63] Peisen Zhao, Lingxi Xie, Ya Zhang, and Qi Tian.Actionness-guided transformer for anchor-free temporal ac-tion localization. IEEE Signal Processing Letters, 29:194–198, 2021.





[64] Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, RongguangYe, and Dongwei Ren. Distance-iou loss: Faster and betterlearning for bounding box regression. In Proceedings of theAAAI conference on artificial intelligence, volume 34, pages12993–13000, 2020.





[65] Zixin Zhu, Wei Tang, Le Wang, Nanning Zheng, and GangHua. Enriching local and global contexts for temporal actionlocalization. In Proceedings of the IEEE/CVF internationalconference on computer vision, pages 13516–13525, 2021.





[66] Zixin Zhu, Le Wang, Wei Tang, Ziyi Liu, Nanning Zheng,and Gang Hua. Learning disentangled classification and lo-calization representations for temporal action localization.In Proceedings of the AAAI Conference on Artificial Intel-ligence, volume 36, pages 3644–3652, 2022.

