# Bird recognition - review of useful resources
A list of useful resources in the bird sound recognition
* [Datasets](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#Datasets)
* [Papers](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#Papers)
* [Open Source Projects](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#Open-Source-Projects)
* [Competitions](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#Competitions)
* [Articles](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#Articles)


![Singing bird](https://www.sciencemag.org/sites/default/files/styles/article_main_large/public/bird_16x9_3.jpg)

Feel free to make a pull request or to ⭐️ the repository if you like it! 
# Introduction
What are challenges in bird song recognition?
Elias Sprengel, Martin Jaggi, Yannic Kilcher, and Thomas Hofmann in their paper [Audio Based Bird Species Identification using
Deep Learning Techniques](http://ceur-ws.org/Vol-1609/16090547.pdf) point out some very important issues:
* Background noise in the recordings - city noises, churches, cars...
* Very often multiple birds singing at the same time - multi-label classification problem
* Differences between mating calls and songs - mating calls are short, whereas songs are longer 
* Inter-species variance - same bird species singing in different countries might sound completely different
* Variable length of sound recordings
* Large number of different species


# Datasets
![Flying bird](http://www.kuwaitbirds.org/sites/default/files/files-misc/birding-bird-shapes-1.jpg)

- **[xeno-canto.org](https://www.xeno-canto.org/)** is a website dedicated to sharing bird sounds from all over the world (480k, September 2019).
**Scripts that make downloading easier can be found here:**
  - [github.com/AgaMiko/xeno-canto-download](https://github.com/AgaMiko/xeno-canto-download) - Simple and easy scraper to download sound with metadata, written in python
  - [github.com/ntivirikin/xeno-canto-py](https://github.com/ntivirikin/xeno-canto-py) - Python API wrapper designed to help users easily download xeno-canto.org recordings and associated information. Avaiable to install with pip manager.

- **[Macaulay Library](https://search.macaulaylibrary.org/catalog)** is the world's largest archive of animal sounds. It includes more than 175,000 audio recordings covering 75 percent of the world's bird species. There are an ever-increasing numbers of insect, fish, frog, and mammal recordings. The video archive includes over 50,000 clips, representing over 3,500 species.[1] The Library is part of Cornell Lab of Ornithology of the Cornell University. 

- **[tierstimmenarchiv.de](https://www.tierstimmenarchiv.de/)** - Animal sound album at  the  Museum  für  Naturkunde  in  Berlin, with a collection of bird songs and calls.

- **[RMBL-Robin database](http://www.seas.ucla.edu/spapl/projects/Bird.html)** - Database for Noise Robust Bird Song Classification, Recognition, and Detection.A 78 minutes Robin song database collected by using a close-field song meter (www.wildlifeacoustics.com) at the Rocky Mountain Biological Laboratory near Crested Butte, Colorado in the summer of 2009. The recorded Robin songs are naturally corrupted by different kinds of background noises, such as wind, water and other vocal bird species. Non-target songs may overlap with target songs. Each song usually consists of 2-10 syllables. The timing boundaries and noise conditions of the syllables and songs, and human inferred syllable patterns are annotated.

- **[floridamuseum.ufl.edu/bird-sounds](https://www.floridamuseum.ufl.edu/bird-sounds/)** - A collection of bird sound recordings from the Florida Museum Bioacoustic Archives, with 27,500 cataloged recordings representing about 3,000 species, is perhaps third or fourth largest in the world in number of species.

- **Field recordings, worldwide ("freefield1010")** - a collection of 7,690 excerpts from field recordings around the world, gathered by the FreeSound project, and then standardised for research. This collection is very diverse in location and environment, and for the BAD Challenge we have annotated it for the presence/absence of birds.
   - #### Download: [data labels](https://ndownloader.figshare.com/files/10853303) • [audio files (5.8 Gb zip)](https://archive.org/download/ff1010bird/ff1010bird_wav.zip) (or [via bittorrent](https://archive.org/download/ff1010bird/ff1010bird_archive.torrent))
   
- **Crowdsourced dataset, UK ("warblrb10k")** - 8,000 smartphone audio recordings from around the UK, crowdsourced by users of Warblr the bird recognition app. The audio covers a wide distribution of UK locations and environments, and includes weather noise, traffic noise, human speech and even human bird imitations.
  - #### Download: [data labels](https://ndownloader.figshare.com/files/10853306) • [audio files (4.3 Gb zip)](https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip) (or [via bittorrent](https://archive.org/download/warblrb10k_public/warblrb10k_public_archive.torrent))
  
- **Remote monitoring flight calls, USA ("BirdVox-DCASE-20k")** - 20,000 audio clips collected from remote monitoring units placed near Ithaca, NY, USA during the autumn of 2015, by the BirdVox project. More info about BirdVox-DCASE-20k
  - #### Download: [data labels](https://ndownloader.figshare.com/files/10853300) • [audio files (15.4 Gb zip)](https://zenodo.org/record/1208080/files/BirdVox-DCASE-20k.zip)

- **[british-birdsongs.uk](https://www.british-birdsongs.uk/)** - A collection of bird songs, calls and alarms calls from Great Britain
- **[birding2asia.com/W2W/freeBirdSounds](https://www.birding2asia.com/W2W/freeBirdSounds.html)** - Bird recordigns from India, Philippines,   Taiwan and Thailad.

- **[azfo.org/SoundLibrary/sounds_library](http://www.azfo.org/SoundLibrary/sounds_library.html)** - All recordings are copyrighted© by the recordist. Downloading and copying are authorized for noncommercial educational or personal use only. 

#### Feel free to add other datasets to a list if you know any!

# Papers

![Flying bird](http://www.kuwaitbirds.org/sites/default/files/files-misc/birding-bird-shapes-1.jpg)

## 2019
- Stowell, Dan, et al. ["Automatic acoustic detection of birds through deep learning: the first Bird Audio Detection challenge."](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13103) Methods in Ecology and Evolution 10.3 (2019): 368-380.
 &nbsp;&nbsp;&nbsp;&nbsp;  <details><summary> Abstract </summary>
     Assessing the presence and abundance of birds is important for monitoring specific species as well as overall ecosystem health. Many birds are most readily detected by their sounds, and thus, passive acoustic monitoring is highly appropriate. Yet acoustic monitoring is often held back by practical limitations such as the need for manual configuration, reliance on example sound libraries, low accuracy, low robustness, and limited ability to generalise to novel acoustic conditions.
    Here, we report outcomes from a collaborative data challenge. We present new acoustic monitoring datasets, summarise the machine learning techniques proposed by challenge teams, conduct detailed performance evaluation, and discuss how such approaches to detection can be integrated into remote monitoring projects.
    Multiple methods were able to attain performance of around 88% area under the receiver operating characteristic (ROC) curve (AUC), much higher performance than previous general‐purpose methods.
    With modern machine learning, including deep learning, general‐purpose acoustic bird detection can achieve very high retrieval rates in remote monitoring data, with no manual recalibration, and no pretraining of the detector for the target species or the acoustic conditions in the target environment.
</details>


- Koh, Chih-Yuan, et al. ["Bird Sound Classification using Convolutional Neural Networks."](http://www.dei.unipd.it/~ferro/CLEF-WN-Drafts/CLEF2019/paper_68.pdf) (2019).
&nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Abstract </summary>
  Accurate prediction of bird species from audio recordings
is beneficial to bird conservation. Thanks to the rapid advance in deep
learning, the accuracy of bird species identification from audio recordings
has greatly improved in recent years. This year, the BirdCLEF2019[4]
task invited participants to design a system that could recognize 659
bird species from 50,000 audio recordings. The challenges in this competition included memory management, the number of bird species for the
machine to recognize, and the mismatch in signal-to-noise ratio between
the training and the testing sets. To participate in this competition,
we adopted two recently popular convolutional neural network architectures — the ResNet[1] and the inception model[13]. The inception model
achieved 0.16 classification mean average precision (c-mAP) and ranked
the second place among five teams that successfully submitted their predictions.
 </details>

- Kahl, S., et al. ["Overview of BirdCLEF 2019: large-scale bird recognition in Soundscapes."](http://ceur-ws.org/Vol-2380/paper_256.pdf) CLEF working notes (2019).
 &nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Abstract </summary>
  The BirdCLEF challenge—as part of the 2019 LifeCLEF Lab[7]—offers a large-scale proving ground for system-oriented evaluation ofbird species identification based on audio recordings. The challenge usesdata  collected  through  Xeno-canto,  the  worldwide  community  of  birdsound recordists. This ensures that BirdCLEF is close to the conditionsof  real-world  application,  in  particular  with  regard  to  the  number  ofspecies in the training set (659). In 2019, the challenge was focused onthe difficult task of recognizing all birds vocalizing in omni-directionalsoundscape recordings. Therefore, the dataset of the previous year wasextended with more than 350 hours of manually annotated soundscapesthat were recorded using 30 field recorders in Ithaca (NY, USA). Thispaper describes the methodology of the conducted evaluation as well asthe synthesis of the main results and lessons learned.
 </details>

## 2018
- Kojima, Ryosuke, et al. ["HARK-Bird-Box: A Portable Real-time Bird Song Scene Analysis System."](https://ieeexplore.ieee.org/abstract/document/8594070/) 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018.
&nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Abstract </summary>
  This paper addresses real-time bird song scene analysis. Observation of animal behavior such as communication of wild birds would be aided by a portable device implementing a real-time system that can localize sound sources, measure their timing, classify their sources, and visualize these factors of sources. The difficulty of such a system is an integration of these functions considering the real-time requirement. To realize such a system, we propose a cascaded approach, cascading sound source detection, localization, separation, feature extraction, classification, and visualization for bird song analysis. Our system is constructed by combining an open source software for robot audition called HARK and a deep learning library to implement a bird song classifier based on a convolutional neural network (CNN). Considering portability, we implemented this system on a single-board computer, Jetson TX2, with a microphone array and developed a prototype device for bird song scene analysis. A preliminary experiment confirms a computational time for the whole system to realize a real-time system. Also, an additional experiment with a bird song dataset revealed a trade-off relationship between classification accuracy and time consuming and the effectiveness of our classifier.
 </details> 
 
 - Fazeka, Botond, et al. ["A multi-modal deep neural network approach to bird-song identification."](https://arxiv.org/abs/1811.04448) arXiv preprint arXiv:1811.04448 (2018).
 &nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Abstract </summary>
  We present a multi-modal Deep Neural Network (DNN) approach for bird song identification. The presented approach takes both audio samples and metadata as input. The audio is fed into a Convolutional Neural Network (CNN) using four convolutional layers. The additionally provided metadata is processed using fully connected layers. The flattened convolutional layers and the fully connected layer of the metadata are joined and fed into a fully connected layer. The resulting architecture achieved 2., 3. and 4. rank in the BirdCLEF2017 task in various training configurations.
 </details>
 
- Lasseck, Mario. ["Audio-based Bird Species Identification with Deep Convolutional Neural Networks."](http://ceur-ws.org/Vol-2125/paper_140.pdf) CLEF (Working Notes). 2018.
 &nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Abstract </summary>
  This paper presents deep learning techniques for audio-based bird
identification at very large scale. Deep Convolutional Neural Networks
(DCNNs) are fine-tuned to classify 1500 species. Various data augmentation
techniques are applied to prevent overfitting and to further improve model accuracy and generalization. The proposed approach is evaluated in the BirdCLEF
2018 campaign and provides the best system in all subtasks. It surpasses previous state-of-the-art by 15.8 % identifying foreground species and 20.2 % considering also background species achieving a mean reciprocal rank (MRR) of
82.7 % and 74.0 % on the official BirdCLEF Subtask1 test set.
 </details>
 
 - Priyadarshani, Nirosha, Stephen Marsland, and Isabel Castro. ["Automated birdsong recognition in complex acoustic environments: a review."](https://onlinelibrary.wiley.com/doi/full/10.1111/jav.01447) Journal of Avian Biology 49.5 (2018): jav-01447.
  &nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Abstract </summary>
 onservationists are increasingly using autonomous acoustic recorders to determine the presence/absence and the abundance of bird species. Unlike humans, these recorders can be left in the field for extensive periods of time in any habitat. Although data acquisition is automated, manual processing of recordings is labour intensive, tedious, and prone to bias due to observer variations. Hence automated birdsong recognition is an efficient alternative.
However, only few ecologists and conservationists utilise the existing birdsong recognisers to process unattended field recordings because the software calibration time is exceptionally high and requires considerable knowledge in signal processing and underlying systems, making the tools less user‐friendly. Even allowing for these difficulties, getting accurate results is exceedingly hard. In this review we examine the state‐of‐the‐art, summarising and discussing the methods currently available for each of the essential parts of a birdsong recogniser, and also available software. The key reasons behind poor automated recognition are that field recordings are very noisy, calls from birds that are a long way from the recorder can be faint or corrupted, and there are overlapping calls from many different birds. In addition, there can be large numbers of different species calling in one recording, and therefore the method has to scale to large numbers of species, or at least avoid misclassifying another species as one of particular interest. We found that these areas of importance, particularly the question of noise reduction, are amongst the least researched. In cases where accurate recognition of individual species is essential, such as in conservation work, we suggest that specialised (species‐specific) methods of passive acoustic monitoring are required. We also believe that it is important that comparable measures, and datasets, are used to enable methods to be compared.
</details>


 - Goeau, Herve, et al. ["Overview of BirdCLEF 2018: monospecies vs. soundscape bird identification."](http://ceur-ws.org/Vol-2125/invited_paper_9.pdf) 2018.
  &nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Abstract </summary>
 The BirdCLEF challenge offers a large-scale proving groundfor system-oriented evaluation of bird species identification based on au-dio recordings of their sounds. One of its strengths is that it uses datacollected through Xeno-canto, the worldwide community of bird soundrecordists. This ensures that BirdCLEF is close to the conditions of real-world application, in particular with regard to the number of species inthe training set (1500). Two main scenarios are evaluated: (i) the identifi-cation of a particular bird species in a recording, and (ii), the recognitionof all species vocalising in a long sequence (up to one hour) of raw sound-scapes that can contain tens of birds singing more or less simultaneously.This paper reports an overview of the systems developed by the six par-ticipating  research  groups,  the  methodology  of  the  evaluation  of  theirperformance, and an analysis and discussion of the results obtained.
</details>

## 2017

- Zhao, Zhao, et al. ["Automated bird acoustic event detection and robust species classification."](https://www.sciencedirect.com/science/article/pii/S157495411630231X) Ecological Informatics 39 (2017): 99-108.
&nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Abstract </summary>
Non-invasive bioacoustic monitoring is becoming increasingly popular for biodiversity conservation. Two automated methods for acoustic classification of bird species currently used are frame-based methods, a model that uses Hidden Markov Models (HMMs), and event-based methods, a model consisting of descriptive measurements or restricted to tonal or harmonic vocalizations. In this work, we propose a new method for automated field recording analysis with improved automated segmentation and robust bird species classification. We used a Gaussian Mixture Model (GMM)-based frame selection with an event-energy-based sifting procedure that selected representative acoustic events. We employed a Mel, band-pass filter bank on each event's spectrogram. The output in each subband was parameterized by an autoregressive (AR) model, which resulted in a feature consisting of all model coefficients. Finally, a support vector machine (SVM) algorithm was used for classification. The significance of the proposed method lies in the parameterized features depicting the species-specific spectral pattern. This experiment used a control audio dataset and real-world audio dataset comprised of field recordings of eleven bird species from the Xeno-canto Archive, consisting of 2762 bird acoustic events with 339 detected “unknown” events (corresponding to noise or unknown species vocalizations). Compared with other recent approaches, our proposed method provides comparable identification performance with respect to the eleven species of interest. Meanwhile, superior robustness in real-world scenarios is achieved, which is expressed as the considerable improvement from 0.632 to 0.928 for the F-score metric regarding the “unknown” events. The advantage makes the proposed method more suitable for automated field recording analysis.
</details>  
   
-  Hershey, S. et. al., [CNN Architectures for Large-Scale Audio Classification](https://research.google.com/pubs/pub45611.html), ICASSP 2017

-  Gemmeke, J. et. al., [AudioSet: An ontology and human-labelled dataset for audio events](https://research.google.com/pubs/pub45857.html), ICASSP 2017

-  Salamon, Justin, et al. ["Fusing shallow and deep learning for bioacoustic bird species classification."](https://ieeexplore.ieee.org/abstract/document/7952134) 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017.
&nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Abstract </summary> Automated classification of organisms to species based on their vocalizations would contribute tremendously to abilities to monitor biodiversity, with a wide range of applications in the field of ecology. In particular, automated classification of migrating birds' flight calls could yield new biological insights and conservation applications for birds that vocalize during migration. In this paper we explore state-of-the-art classification techniques for large-vocabulary bird species classification from flight calls. In particular, we contrast a “shallow learning” approach based on unsupervised dictionary learning with a deep convolutional neural network combined with data augmentation. We show that the two models perform comparably on a dataset of 5428 flight calls spanning 43 different species, with both significantly outperforming an MFCC baseline. Finally, we show that by combining the models using a simple late-fusion approach we can further improve the results, obtaining a state-of-the-art classification accuracy of 0.96.</details> 

- Narasimhan, Revathy, Xiaoli Z. Fern, and Raviv Raich. ["Simultaneous segmentation and classification of bird song using CNN."](https://ieeexplore.ieee.org/abstract/document/7952135/) 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017.
&nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Abstract </summary> In bioacoustics, automatic animal voice detection and recognition from audio recordings is an emerging topic for animal preservation. Our research focuses on bird bioacoustics, where the goal is to segment bird syllables from the recording and predict the bird species for the syllables. Traditional methods for this task addresses the segmentation and species prediction separately, leading to propagated errors. This work presents a new approach that performs simultaneous segmentation and classification of bird species using a Convolutional Neural Network (CNN) with encoder-decoder architecture. Experimental results on bird recordings show significant improvement compared to recent state-of-the-art methods for both segmentation and species classification.</details> 

- Grill, Thomas, and Jan Schlüter. ["Two convolutional neural networks for bird detection in audio signals."](https://ieeexplore.ieee.org/abstract/document/8081512) 2017 25th European Signal Processing Conference (EUSIPCO). IEEE, 2017.
 &nbsp;&nbsp;&nbsp;&nbsp;  <details><summary> Abstract </summary>
 We present and compare two approaches to detect the presence of bird calls in audio recordings using convolutional neural networks on mel spectrograms. In a signal processing challenge using environmental recordings from three very different sources, only two of them available for supervised training, we obtained an Area Under Curve (AUC) measure of 89% on the hidden test set, higher than any other contestant. By comparing multiple variations of our systems, we find that despite very different architectures, both approaches can be tuned to perform equally well. Further improvements will likely require a radically different approach to dealing with the discrepancy between data sources.
</details>


## 2016

- Tóth, Bálint Pál, and Bálint Czeba. [Convolutional Neural Networks for Large-Scale Bird Song Classification in Noisy Environment](https://www.researchgate.net/profile/Balint_Gyires-Toth/publication/306287320_Convolutional_Neural_Networks_for_Large-Scale_Bird_Song_Classification_in_Noisy_Environment/links/57b6da6608ae2fc031fd6eed/Convolutional-Neural-Networks-for-Large-Scale-Bird-Song-Classification-in-Noisy-Environment.pdf), CLEF (Working Notes). 2016.
&nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Abstract </summary> This paper describes a convolutional neural network based deep learning approach for bird song classification that was used in an audio record-based bird identification challenge, called BirdCLEF 2016. The training and test set contained about 24k and 8.5k recordings, belonging to 999 bird species. The recorded waveforms were very diverse in terms of length and content. We converted
the waveforms into frequency domain and splitted into equal segments. The segments were fed into a convolutional neural network for feature learning, which was followed by fully connected layers for classification. In the official scores our solution reached a MAP score of over 40% for main species, and MAP score of over 33% for main species mixed with background species</details> 

- Nicholson, David. ["Comparison of machine learning methods applied to birdsong element classification."](http://conference.scipy.org/proceedings/scipy2016/pdfs/david_nicholson.pdf) Proceedings of the 15th Python in Science Conference. 2016.
&nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Abstract </summary> Songbirds provide neuroscience with a model system for understanding how the brain learns and produces a motor skill similar to speech.
Much like humans, songbirds learn their vocalizations from social interactions
during a critical period in development. Each bird’s song consists of repeated
elements referred to as “syllables”. To analyze song, scientists label syllables
by hand, but a bird can produce hundreds of songs a day, many more than
can be labeled. Several groups have applied machine learning algorithms to
automate labeling of syllables, but little work has been done comparing these
various algorithms. For example, there are articles that propose using support
vector machines (SVM), K-nearest neighbors (k-NN), and even deep learning
to automate labeling song of the Bengalese Finch (a species whose behavior
has made it the subject of an increasing number of neuroscience studies).
This paper compares algorithms for classifying Bengalese Finch syllables (building on previous work [https://youtu.be/ghgniK4X_Js]). Using a standard crossvalidation approach, classifiers were trained on syllables from a given bird,
and then classifier accuracy was measured with large hand-labeled testing
datasets for that bird. The results suggest that both k-NN and SVM with a
non-linear kernel achieve higher accuracy than a previously published linear
SVM method. Experiments also demonstrate that the accuracy of linear SVM
is impaired by "intro syllables", a low-amplitude high-noise syllable found in
all Bengalese Finch songs. Testing of machine learning algorithms was carried out using Scikit-learn and Numpy/Scipy via Anaconda. Figures from this
paper in Jupyter notebook form, as well as code and links to data, are here:
https://github.com/NickleDave/ML-comparison-birdsong</details> 

- Sprengel, Elias, et al. [Audio based bird species identification using deep learning techniques](http://ceur-ws.org/Vol-1609/16090547.pdf). No. CONF. 2016.
 &nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Abstract </summary>
 In this paper we present a new audio classification methodfor bird species identification. Whereas most approaches apply nearestneighbour matching [6] or decision trees [8] using extracted templates foreach bird species, ours draws upon techniques from speech recognitionand recent advances in the domain of deep learning. With novel prepro-cessing and data augmentation methods, we train a convolutional neuralnetwork on the biggest publicly available dataset [5]. Our network archi-tecture achieves a mean average precision score of 0.686 when predictingthe main species of each sound file and scores 0.555 when backgroundspecies  are  used  as  additional  prediction  targets.  As  this  performancesurpasses current state of the art results, our approach won this yearsinternational BirdCLEF 2016 Recognition Challenge [3,4,1].
</details>

- Stowell, Dan, et al. ["Bird detection in audio: a survey and a challenge."](https://arxiv.org/abs/1608.03417) 2016 IEEE 26th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2016.
 &nbsp;&nbsp;&nbsp;&nbsp;  <details><summary> Abstract </summary>
 Many biological monitoring projects rely on acoustic detection of birds. Despite increasingly large datasets, this detection is often manual or semi-automatic, requiring manual tuning/postprocessing. We review the state of the art in automatic bird sound detection, and identify a widespread need for tuning-free and species-agnostic approaches. We introduce new datasets and an IEEE research challenge to address this need, to make possible the development of fully automatic algorithms for bird sound detection. 
</details>

## 2015

- Tan, Lee N., et al. ["Dynamic time warping and sparse representation classification for birdsong phrase classification using limited training data."](https://asa.scitation.org/doi/abs/10.1121/1.4906168) The Journal of the Acoustical Society of America 137.3 (2015): 1069-1080.
&nbsp;&nbsp;&nbsp;&nbsp; <details><summary> Abstract </summary> Annotation of phrases in birdsongs can be helpful to behavioral and population studies. To reduce the need for manual annotation, an automated birdsong phrase classification algorithm for limited data is developed. Limited data occur because of limited recordings or the existence of rare phrases. In this paper, classification of up to 81 phrase classes of Cassin's Vireo is performed using one to five training samples per class. The algorithm involves dynamic time warping (DTW) and two passes of sparse representation (SR) classification. DTW improves the similarity between training and test phrases from the same class in the presence of individual bird differences and phrase segmentation inconsistencies. The SR classifier works by finding a sparse linear combination of training feature vectors from all classes that best approximates the test feature vector. When the class decisions from DTW and the first pass SR classification are different, SR classification is repeated using training samples from these two conflicting classes. Compared to DTW, support vector machines, and an SR classifier without DTW, the proposed classifier achieves the highest classification accuracies of 94% and 89% on manually segmented and automatically segmented phrases, respectively, from unseen Cassin's Vireo individuals, using five training samples per class.</details> 

# Competitions
![Flying bird](http://www.kuwaitbirds.org/sites/default/files/files-misc/birding-bird-shapes-1.jpg)
- [kaggle - Cornell Birdcall Identification](https://www.kaggle.com/c/birdsong-recognition/overview) - Build tools for bird population monitoring. Identify a wide variety of bird vocalizations in soundscape recordings. Due to the complexity of the recordings, they contain weak labels. There might be anthropogenic sounds (e.g., airplane overflights) or other bird and non-bird (e.g., chipmunk) calls in the background, with a particular labeled bird species in the foreground. Bring your new ideas to build effective detectors and classifiers for analyzing complex soundscape recordings!
- [LifeCLEF 2020 - BirdCLEF](https://www.imageclef.org/BirdCLEF2020) - Two scenarios will be evaluated: (i) the recognition of all specimens singing in a long sequence (up to one hour) of raw soundscapes that can contain tens of birds singing simultaneously, and (ii) chorus source separation in complex soundscapes that were recorded in stereo at very high sampling rate (250 kHz SR).
The training set used for the challenge will be a version of the 2019 training set enriched by new contributions from the Xeno-canto network and a geographic extension. It will contain approximately 80K recordings covering between 1500 and 2000 species from North, Central and South America, as well as Europe. This will be the largest bioacoustic dataset used in the literature. 
- [LifeCLEF 2019 Bird Recognition](https://www.crowdai.org/challenges/lifeclef-2019-bird-recognition) - The goal of the challenge is to detect and classify all audible bird vocalizations within the provided soundscape recordings. Each soundscape is divided into segments of 5 seconds. Participants should submit a list of species associated with probability scores for each segment.
- [LifeCLEF 2018 Bird - Monophone](https://www.crowdai.org/challenges/lifeclef-2018-bird-monophone) - The goal of the task is to identify the species of the most audible bird (i.e. the one that was intended to be recorded) in each of the provided test recordings. Therefore, the evaluated systems have to return a ranked list of possible species for each of the 12,347 test recordings. 
- [LifeCLEF 2018 Bird - Soundscape](https://www.crowdai.org/challenges/lifeclef-2018-bird-soundscape) - The goal of the task is to localize and identify all audible birds within the provided soundscape recordings. Each soundscape is divided into segments of 5 seconds, and a list of species associated to probability scores will have to be returned for each segment. 
- [Bird audio detection DCASE2018](http://dcase.community/challenge2018/task-bird-audio-detection) - The task is to design a system that, given a short audio recording, returns a binary decision for the presence/absence of bird sound (bird sound of any kind). The output can be just "0" or "1", but we encourage weighted/probability outputs in the continuous range [0,1] for the purposes of evaluation. For the main assessment we will use the well-known "Area Under the ROC Curve" (AUC) measure of classification performance.
- [Bird Audio Detection Challenge 2016–2017](http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/) - contest organized in collaboration with the IEEE Signal Processing Society. They propose a research data challenge to create a robust and scalable bird detection algorithm. Organizers offer new datasets collected in real live bioacoustics monitoring projects, and an objective, standardised evaluation framework – and prizes for the strongest submissions. Results and summary of the best submissions can be found [here](http://c4dm.eecs.qmul.ac.uk/events/badchallenge_results/).

# Open Source Projects

![Flying bird](http://www.kuwaitbirds.org/sites/default/files/files-misc/birding-bird-shapes-1.jpg)
- [Polish bird species recognition - 19 class recognition, 2019](https://github.com/wimlds-trojmiasto/birds) - Repositorium with code to download, and cut files into melspectrograms with librosa library. Later, files are classified with deep neural networks in Keras.
- [Large-Scale Bird Sound Classification using Convolutional Neural Networks, 2017](https://github.com/kahst/BirdCLEF2017) - Code repo for our submission to the LifeCLEF bird identification task BirdCLEF2017.
- [Automatic recognition of element classes and boundaries in the birdsong with variable sequences](https://github.com/cycentum/birdsong-recognition) - This is a source code for the manuscript “Automatic recognition of element classes and boundaries in the birdsong with variable sequences” by Takuya Koumura and Kazuo Okanoya (http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0159188).
- [Trainig scripts for deep convolutional neural network based audio classification in Keras](https://github.com/bapalto/birdsong-keras) - The following scripts were created for the BirdCLEF 2016 competition by Bálint Czeba and Bálint Pál Tóth.
- [BirdSong Recognition](https://github.com/AmyangXYZ/BirdSong_Recognition) - Classification system based on the classic HMM+MFCC method. It has 22 kinds of birds now, the correct rate is 81.8% (with only 1.7G data trained).
- [A model for bird sound classification](https://github.com/gojibjib/jibjib-model) - pretrained VGGish/ Audioset model by Google and finetune it by letting it iterate during training on more than 80,000 audio samples of 10 second length (195 bird classes)
- [Bird brain](https://github.com/davipatti/birdbrain) - This repo contains code to search the Xeno-canto bird sound database, and train a machine learning model to classify birds according to those sounds.
- [Bird-Species-Classification](https://github.com/zahan97/Bird-Species-Classification) - The project uses a neural-net in tensorflow to classify the species to which a bird belongs to based on the features it has. There are total 312 features and 11787 examples.
- [Bird Species Classification by song](https://github.com/johnmartinsson/bird-species-classification) - This repository is not actively maintained. It is the result of a master's thesis and the code has been made available as a reference if anyone would like to reproduce the results of the thesis.
- [Recognizing Birds from Sound - The 2018 BirdCLEF Baseline System](https://github.com/kahst/BirdCLEF-Baseline) -  a baseline system for the LifeCLEF bird identification task BirdCLEF2018. Authors encourage participants to build upon the code base and share their results for future reference. They promise to keep the repository updated and add improvements and submission boilerplate in the future.

# Articles
![Flying bird](http://www.kuwaitbirds.org/sites/default/files/files-misc/birding-bird-shapes-1.jpg)

- [SOUND-BASED BIRD CLASSIFICATION](https://towardsdatascience.com/sound-based-bird-classification-965d0ecacb2b) - How a group of Polish women used deep learning, acoustics and ornithology to classify birds


