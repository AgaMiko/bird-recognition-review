# Bird recognition - review of useful resources
A list of useful resources in the bird sound recognition
* [Datasets](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#Datasets)
* [Papers](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#Papers)
* [Open Source Projects](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#Open%20Source%20Projects)
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

## 2019

## 2018

## 2017
- Hershey, S. et. al., [CNN Architectures for Large-Scale Audio Classification](https://research.google.com/pubs/pub45611.html), ICASSP 2017
- Gemmeke, J. et. al., [AudioSet: An ontology and human-labelled dataset for audio events](https://research.google.com/pubs/pub45857.html), ICASSP 2017
- Salamon, Justin, et al. ["Fusing shallow and deep learning for bioacoustic bird species classification."](https://ieeexplore.ieee.org/abstract/document/7952134) 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017.
    - **Abstract.** Automated classification of organisms to species based on their vocalizations would contribute tremendously to abilities to monitor biodiversity, with a wide range of applications in the field of ecology. In particular, automated classification of migrating birds' flight calls could yield new biological insights and conservation applications for birds that vocalize during migration. In this paper we explore state-of-the-art classification techniques for large-vocabulary bird species classification from flight calls. In particular, we contrast a “shallow learning” approach based on unsupervised dictionary learning with a deep convolutional neural network combined with data augmentation. We show that the two models perform comparably on a dataset of 5428 flight calls spanning 43 different species, with both significantly outperforming an MFCC baseline. Finally, we show that by combining the models using a simple late-fusion approach we can further improve the results, obtaining a state-of-the-art classification accuracy of 0.96.

## 2016
- Tóth, Bálint Pál, and Bálint Czeba. [Convolutional Neural Networks for Large-Scale Bird Song Classification in Noisy Environment](https://www.researchgate.net/profile/Balint_Gyires-Toth/publication/306287320_Convolutional_Neural_Networks_for_Large-Scale_Bird_Song_Classification_in_Noisy_Environment/links/57b6da6608ae2fc031fd6eed/Convolutional-Neural-Networks-for-Large-Scale-Bird-Song-Classification-in-Noisy-Environment.pdf), CLEF (Working Notes). 2016.
    - **Abstract.** This paper describes a convolutional neural network based deep learning approach for bird song classification that was used in an audio record-based bird identification challenge, called BirdCLEF 2016. The training and test set contained about 24k and 8.5k recordings, belonging to 999 bird species. The recorded waveforms were very diverse in terms of length and content. We converted
the waveforms into frequency domain and splitted into equal segments. The segments were fed into a convolutional neural network for feature learning, which was followed by fully connected layers for classification. In the official scores our solution reached a MAP score of over 40% for main species, and MAP score of over 33% for main species mixed with background species

## 2015
- Tan, Lee N., et al. ["Dynamic time warping and sparse representation classification for birdsong phrase classification using limited training data."](https://asa.scitation.org/doi/abs/10.1121/1.4906168) The Journal of the Acoustical Society of America 137.3 (2015): 1069-1080.
    - **Abstract.** Annotation of phrases in birdsongs can be helpful to behavioral and population studies. To reduce the need for manual annotation, an automated birdsong phrase classification algorithm for limited data is developed. Limited data occur because of limited recordings or the existence of rare phrases. In this paper, classification of up to 81 phrase classes of Cassin's Vireo is performed using one to five training samples per class. The algorithm involves dynamic time warping (DTW) and two passes of sparse representation (SR) classification. DTW improves the similarity between training and test phrases from the same class in the presence of individual bird differences and phrase segmentation inconsistencies. The SR classifier works by finding a sparse linear combination of training feature vectors from all classes that best approximates the test feature vector. When the class decisions from DTW and the first pass SR classification are different, SR classification is repeated using training samples from these two conflicting classes. Compared to DTW, support vector machines, and an SR classifier without DTW, the proposed classifier achieves the highest classification accuracies of 94% and 89% on manually segmented and automatically segmented phrases, respectively, from unseen Cassin's Vireo individuals, using five training samples per class.
    
# Competitions
- [LifeCLEF 2019 Bird Recognition](https://www.crowdai.org/challenges/lifeclef-2019-bird-recognition) - The goal of the challenge is to detect and classify all audible bird vocalizations within the provided soundscape recordings. Each soundscape is divided into segments of 5 seconds. Participants should submit a list of species associated with probability scores for each segment.
- [LifeCLEF 2018 Bird - Monophone](https://www.crowdai.org/challenges/lifeclef-2018-bird-monophone) - The goal of the task is to identify the species of the most audible bird (i.e. the one that was intended to be recorded) in each of the provided test recordings. Therefore, the evaluated systems have to return a ranked list of possible species for each of the 12,347 test recordings. 
- [LifeCLEF 2018 Bird - Soundscape](https://www.crowdai.org/challenges/lifeclef-2018-bird-soundscape) - The goal of the task is to localize and identify all audible birds within the provided soundscape recordings. Each soundscape is divided into segments of 5 seconds, and a list of species associated to probability scores will have to be returned for each segment. 
- [Bird audio detection DCASE2018](http://dcase.community/challenge2018/task-bird-audio-detection) - The task is to design a system that, given a short audio recording, returns a binary decision for the presence/absence of bird sound (bird sound of any kind). The output can be just "0" or "1", but we encourage weighted/probability outputs in the continuous range [0,1] for the purposes of evaluation. For the main assessment we will use the well-known "Area Under the ROC Curve" (AUC) measure of classification performance.

# Open Source Projects
- [Large-Scale Bird Sound Classification using Convolutional Neural Networks, 2017](https://github.com/kahst/BirdCLEF2017) - Code repo for our submission to the LifeCLEF bird identification task BirdCLEF2017.
- [Automatic recognition of element classes and boundaries in the birdsong with variable sequences](https://github.com/cycentum/birdsong-recognition) - This is a source code for the manuscript “Automatic recognition of element classes and boundaries in the birdsong with variable sequences” by Takuya Koumura and Kazuo Okanoya (http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0159188).
- [Trainig scripts for deep convolutional neural network based audio classification in Keras](https://github.com/bapalto/birdsong-keras) - The following scripts were created for the BirdCLEF 2016 competition by Bálint Czeba and Bálint Pál Tóth.
- [BirdSong Recognition](https://github.com/AmyangXYZ/BirdSong_Recognition) - Classification system based on the classic HMM+MFCC method. It has 22 kinds of birds now, the correct rate is 81.8% (with only 1.7G data trained).
- [A model for bird sound classification](https://github.com/gojibjib/jibjib-model) - pretrained VGGish/ Audioset model by Google and finetune it by letting it iterate during training on more than 80,000 audio samples of 10 second length (195 bird classes)
- [Bird brain](https://github.com/davipatti/birdbrain) - This repo contains code to search the Xeno-canto bird sound database, and train a machine learning model to classify birds according to those sounds.
- [Bird-Species-Classification](https://github.com/zahan97/Bird-Species-Classification) - The project uses a neural-net in tensorflow to classify the species to which a bird belongs to based on the features it has. There are total 312 features and 11787 examples.
- [Bird Species Classification by song](https://github.com/johnmartinsson/bird-species-classification) - This repository is not actively maintained. It is the result of a master's thesis and the code has been made available as a reference if anyone would like to reproduce the results of the thesis.
- [Recognizing Birds from Sound - The 2018 BirdCLEF Baseline System](https://github.com/kahst/BirdCLEF-Baseline) -  a baseline system for the LifeCLEF bird identification task BirdCLEF2018. Authors encourage participants to build upon the code base and share their results for future reference. They promise to keep the repository updated and add improvements and submission boilerplate in the future.

