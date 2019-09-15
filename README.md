# Bird recognition - review of useful resources
A list of useful resources in the bird sound recognition
* [Datasets](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#Datasets)
* [Papers](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#Papers)
* [Open Source Projects](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#Open-Source-Projects)
* [Competitions](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#Competitions)

![Singing bird](https://www.sciencemag.org/sites/default/files/styles/article_main_large/public/bird_16x9_3.jpg)

# Datasets

- **[xeno-canto.org](https://www.xeno-canto.org/)** is a website dedicated to sharing bird sounds from all over the world (around 480k	Recordings in September 2019).
**Scripts that make downloading easier can be found here:**
  - [github.com/AgaMiko/xeno-canto-download](https://github.com/AgaMiko/xeno-canto-download) - Simple and easy scraper to download sound with metadata, written in python
  - [github.com/ntivirikin/xeno-canto-py](https://github.com/ntivirikin/xeno-canto-py) - Python API wrapper designed to help users easily download xeno-canto.org recordings and associated information. Avaiable to install with pip manager.
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
- Koh, Chih-Yuan, et al. ["Bird Sound Classification using Convolutional Neural Networks."](http://www.dei.unipd.it/~ferro/CLEF-WN-Drafts/CLEF2019/paper_68.pdf) (2019).
 <details><summary> Abstract </summary>
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
 
## 2018
- Kojima, Ryosuke, et al. ["HARK-Bird-Box: A Portable Real-time Bird Song Scene Analysis System."](https://ieeexplore.ieee.org/abstract/document/8594070/) 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018.
<details><summary> Abstract </summary>
  This paper addresses real-time bird song scene analysis. Observation of animal behavior such as communication of wild birds would be aided by a portable device implementing a real-time system that can localize sound sources, measure their timing, classify their sources, and visualize these factors of sources. The difficulty of such a system is an integration of these functions considering the real-time requirement. To realize such a system, we propose a cascaded approach, cascading sound source detection, localization, separation, feature extraction, classification, and visualization for bird song analysis. Our system is constructed by combining an open source software for robot audition called HARK and a deep learning library to implement a bird song classifier based on a convolutional neural network (CNN). Considering portability, we implemented this system on a single-board computer, Jetson TX2, with a microphone array and developed a prototype device for bird song scene analysis. A preliminary experiment confirms a computational time for the whole system to realize a real-time system. Also, an additional experiment with a bird song dataset revealed a trade-off relationship between classification accuracy and time consuming and the effectiveness of our classifier.
 </details> 
 
 - Fazeka, Botond, et al. ["A multi-modal deep neural network approach to bird-song identification."](https://arxiv.org/abs/1811.04448) arXiv preprint arXiv:1811.04448 (2018).
 <details><summary> Abstract </summary>
  We present a multi-modal Deep Neural Network (DNN) approach for bird song identification. The presented approach takes both audio samples and metadata as input. The audio is fed into a Convolutional Neural Network (CNN) using four convolutional layers. The additionally provided metadata is processed using fully connected layers. The flattened convolutional layers and the fully connected layer of the metadata are joined and fed into a fully connected layer. The resulting architecture achieved 2., 3. and 4. rank in the BirdCLEF2017 task in various training configurations.
 </details>
 
- Lasseck, Mario. ["Audio-based Bird Species Identification with Deep Convolutional Neural Networks."](http://ceur-ws.org/Vol-2125/paper_140.pdf) CLEF (Working Notes). 2018.
 <details><summary> Abstract </summary>
  This paper presents deep learning techniques for audio-based bird
identification at very large scale. Deep Convolutional Neural Networks
(DCNNs) are fine-tuned to classify 1500 species. Various data augmentation
techniques are applied to prevent overfitting and to further improve model accuracy and generalization. The proposed approach is evaluated in the BirdCLEF
2018 campaign and provides the best system in all subtasks. It surpasses previous state-of-the-art by 15.8 % identifying foreground species and 20.2 % considering also background species achieving a mean reciprocal rank (MRR) of
82.7 % and 74.0 % on the official BirdCLEF Subtask1 test set.
 </details>

## 2017

- Zhao, Zhao, et al. ["Automated bird acoustic event detection and robust species classification."](https://www.sciencedirect.com/science/article/pii/S157495411630231X) Ecological Informatics 39 (2017): 99-108.
<details><summary> Abstract </summary>
Non-invasive bioacoustic monitoring is becoming increasingly popular for biodiversity conservation. Two automated methods for acoustic classification of bird species currently used are frame-based methods, a model that uses Hidden Markov Models (HMMs), and event-based methods, a model consisting of descriptive measurements or restricted to tonal or harmonic vocalizations. In this work, we propose a new method for automated field recording analysis with improved automated segmentation and robust bird species classification. We used a Gaussian Mixture Model (GMM)-based frame selection with an event-energy-based sifting procedure that selected representative acoustic events. We employed a Mel, band-pass filter bank on each event's spectrogram. The output in each subband was parameterized by an autoregressive (AR) model, which resulted in a feature consisting of all model coefficients. Finally, a support vector machine (SVM) algorithm was used for classification. The significance of the proposed method lies in the parameterized features depicting the species-specific spectral pattern. This experiment used a control audio dataset and real-world audio dataset comprised of field recordings of eleven bird species from the Xeno-canto Archive, consisting of 2762 bird acoustic events with 339 detected “unknown” events (corresponding to noise or unknown species vocalizations). Compared with other recent approaches, our proposed method provides comparable identification performance with respect to the eleven species of interest. Meanwhile, superior robustness in real-world scenarios is achieved, which is expressed as the considerable improvement from 0.632 to 0.928 for the F-score metric regarding the “unknown” events. The advantage makes the proposed method more suitable for automated field recording analysis.
</details>  
   
-  Hershey, S. et. al., [CNN Architectures for Large-Scale Audio Classification](https://research.google.com/pubs/pub45611.html), ICASSP 2017

-  Gemmeke, J. et. al., [AudioSet: An ontology and human-labelled dataset for audio events](https://research.google.com/pubs/pub45857.html), ICASSP 2017

-  Salamon, Justin, et al. ["Fusing shallow and deep learning for bioacoustic bird species classification."](https://ieeexplore.ieee.org/abstract/document/7952134) 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017.
<details><summary> Abstract </summary> Automated classification of organisms to species based on their vocalizations would contribute tremendously to abilities to monitor biodiversity, with a wide range of applications in the field of ecology. In particular, automated classification of migrating birds' flight calls could yield new biological insights and conservation applications for birds that vocalize during migration. In this paper we explore state-of-the-art classification techniques for large-vocabulary bird species classification from flight calls. In particular, we contrast a “shallow learning” approach based on unsupervised dictionary learning with a deep convolutional neural network combined with data augmentation. We show that the two models perform comparably on a dataset of 5428 flight calls spanning 43 different species, with both significantly outperforming an MFCC baseline. Finally, we show that by combining the models using a simple late-fusion approach we can further improve the results, obtaining a state-of-the-art classification accuracy of 0.96.</details> 

- Narasimhan, Revathy, Xiaoli Z. Fern, and Raviv Raich. ["Simultaneous segmentation and classification of bird song using CNN."](https://ieeexplore.ieee.org/abstract/document/7952135/) 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017.
<details><summary> Abstract </summary> In bioacoustics, automatic animal voice detection and recognition from audio recordings is an emerging topic for animal preservation. Our research focuses on bird bioacoustics, where the goal is to segment bird syllables from the recording and predict the bird species for the syllables. Traditional methods for this task addresses the segmentation and species prediction separately, leading to propagated errors. This work presents a new approach that performs simultaneous segmentation and classification of bird species using a Convolutional Neural Network (CNN) with encoder-decoder architecture. Experimental results on bird recordings show significant improvement compared to recent state-of-the-art methods for both segmentation and species classification.</details> 

## 2016

- Tóth, Bálint Pál, and Bálint Czeba. [Convolutional Neural Networks for Large-Scale Bird Song Classification in Noisy Environment](https://www.researchgate.net/profile/Balint_Gyires-Toth/publication/306287320_Convolutional_Neural_Networks_for_Large-Scale_Bird_Song_Classification_in_Noisy_Environment/links/57b6da6608ae2fc031fd6eed/Convolutional-Neural-Networks-for-Large-Scale-Bird-Song-Classification-in-Noisy-Environment.pdf), CLEF (Working Notes). 2016.
<details><summary> Abstract </summary> This paper describes a convolutional neural network based deep learning approach for bird song classification that was used in an audio record-based bird identification challenge, called BirdCLEF 2016. The training and test set contained about 24k and 8.5k recordings, belonging to 999 bird species. The recorded waveforms were very diverse in terms of length and content. We converted
the waveforms into frequency domain and splitted into equal segments. The segments were fed into a convolutional neural network for feature learning, which was followed by fully connected layers for classification. In the official scores our solution reached a MAP score of over 40% for main species, and MAP score of over 33% for main species mixed with background species</details> 

- Nicholson, David. ["Comparison of machine learning methods applied to birdsong element classification."](http://conference.scipy.org/proceedings/scipy2016/pdfs/david_nicholson.pdf) Proceedings of the 15th Python in Science Conference. 2016.
<details><summary> Abstract </summary> Songbirds provide neuroscience with a model system for understanding how the brain learns and produces a motor skill similar to speech.
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


## 2015

- Tan, Lee N., et al. ["Dynamic time warping and sparse representation classification for birdsong phrase classification using limited training data."](https://asa.scitation.org/doi/abs/10.1121/1.4906168) The Journal of the Acoustical Society of America 137.3 (2015): 1069-1080.
<details><summary> Abstract </summary> Annotation of phrases in birdsongs can be helpful to behavioral and population studies. To reduce the need for manual annotation, an automated birdsong phrase classification algorithm for limited data is developed. Limited data occur because of limited recordings or the existence of rare phrases. In this paper, classification of up to 81 phrase classes of Cassin's Vireo is performed using one to five training samples per class. The algorithm involves dynamic time warping (DTW) and two passes of sparse representation (SR) classification. DTW improves the similarity between training and test phrases from the same class in the presence of individual bird differences and phrase segmentation inconsistencies. The SR classifier works by finding a sparse linear combination of training feature vectors from all classes that best approximates the test feature vector. When the class decisions from DTW and the first pass SR classification are different, SR classification is repeated using training samples from these two conflicting classes. Compared to DTW, support vector machines, and an SR classifier without DTW, the proposed classifier achieves the highest classification accuracies of 94% and 89% on manually segmented and automatically segmented phrases, respectively, from unseen Cassin's Vireo individuals, using five training samples per class.</details> 

# Competitions
![Flying bird](http://www.kuwaitbirds.org/sites/default/files/files-misc/birding-bird-shapes-1.jpg)

- [LifeCLEF 2019 Bird Recognition](https://www.crowdai.org/challenges/lifeclef-2019-bird-recognition) - The goal of the challenge is to detect and classify all audible bird vocalizations within the provided soundscape recordings. Each soundscape is divided into segments of 5 seconds. Participants should submit a list of species associated with probability scores for each segment.
- [LifeCLEF 2018 Bird - Monophone](https://www.crowdai.org/challenges/lifeclef-2018-bird-monophone) - The goal of the task is to identify the species of the most audible bird (i.e. the one that was intended to be recorded) in each of the provided test recordings. Therefore, the evaluated systems have to return a ranked list of possible species for each of the 12,347 test recordings. 
- [LifeCLEF 2018 Bird - Soundscape](https://www.crowdai.org/challenges/lifeclef-2018-bird-soundscape) - The goal of the task is to localize and identify all audible birds within the provided soundscape recordings. Each soundscape is divided into segments of 5 seconds, and a list of species associated to probability scores will have to be returned for each segment. 
- [Bird audio detection DCASE2018](http://dcase.community/challenge2018/task-bird-audio-detection) - The task is to design a system that, given a short audio recording, returns a binary decision for the presence/absence of bird sound (bird sound of any kind). The output can be just "0" or "1", but we encourage weighted/probability outputs in the continuous range [0,1] for the purposes of evaluation. For the main assessment we will use the well-known "Area Under the ROC Curve" (AUC) measure of classification performance.

# Open Source Projects

![Flying bird](http://www.kuwaitbirds.org/sites/default/files/files-misc/birding-bird-shapes-1.jpg)

- [Large-Scale Bird Sound Classification using Convolutional Neural Networks, 2017](https://github.com/kahst/BirdCLEF2017) - Code repo for our submission to the LifeCLEF bird identification task BirdCLEF2017.
- [Automatic recognition of element classes and boundaries in the birdsong with variable sequences](https://github.com/cycentum/birdsong-recognition) - This is a source code for the manuscript “Automatic recognition of element classes and boundaries in the birdsong with variable sequences” by Takuya Koumura and Kazuo Okanoya (http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0159188).
- [Trainig scripts for deep convolutional neural network based audio classification in Keras](https://github.com/bapalto/birdsong-keras) - The following scripts were created for the BirdCLEF 2016 competition by Bálint Czeba and Bálint Pál Tóth.
- [BirdSong Recognition](https://github.com/AmyangXYZ/BirdSong_Recognition) - Classification system based on the classic HMM+MFCC method. It has 22 kinds of birds now, the correct rate is 81.8% (with only 1.7G data trained).
- [A model for bird sound classification](https://github.com/gojibjib/jibjib-model) - pretrained VGGish/ Audioset model by Google and finetune it by letting it iterate during training on more than 80,000 audio samples of 10 second length (195 bird classes)
- [Bird brain](https://github.com/davipatti/birdbrain) - This repo contains code to search the Xeno-canto bird sound database, and train a machine learning model to classify birds according to those sounds.
- [Bird-Species-Classification](https://github.com/zahan97/Bird-Species-Classification) - The project uses a neural-net in tensorflow to classify the species to which a bird belongs to based on the features it has. There are total 312 features and 11787 examples.
- [Bird Species Classification by song](https://github.com/johnmartinsson/bird-species-classification) - This repository is not actively maintained. It is the result of a master's thesis and the code has been made available as a reference if anyone would like to reproduce the results of the thesis.
- [Recognizing Birds from Sound - The 2018 BirdCLEF Baseline System](https://github.com/kahst/BirdCLEF-Baseline) -  a baseline system for the LifeCLEF bird identification task BirdCLEF2018. Authors encourage participants to build upon the code base and share their results for future reference. They promise to keep the repository updated and add improvements and submission boilerplate in the future.

