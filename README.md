# Bird recognition - review of useful resources
A list of useful resources in the bird sound recognition
* [Datasets](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#Datasets)
* [Papers](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#Papers)
* [Open Source Projects](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#OpenSourceProjects)
* [Competitions](https://github.com/AgaMiko/Bird-recognition-review/blob/master/README.md#Competitions)


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



# Competitions
* [LifeCLEF 2019 Bird Recognition](https://www.crowdai.org/challenges/lifeclef-2019-bird-recognition) - The goal of the challenge is to detect and classify all audible bird vocalizations within the provided soundscape recordings. Each soundscape is divided into segments of 5 seconds. Participants should submit a list of species associated with probability scores for each segment.
* [LifeCLEF 2018 Bird - Monophone](https://www.crowdai.org/challenges/lifeclef-2018-bird-monophone) - The goal of the task is to identify the species of the most audible bird (i.e. the one that was intended to be recorded) in each of the provided test recordings. Therefore, the evaluated systems have to return a ranked list of possible species for each of the 12,347 test recordings. 
* [LifeCLEF 2018 Bird - Soundscape](https://www.crowdai.org/challenges/lifeclef-2018-bird-soundscape) - The goal of the task is to localize and identify all audible birds within the provided soundscape recordings. Each soundscape is divided into segments of 5 seconds, and a list of species associated to probability scores will have to be returned for each segment. 
* [Bird audio detection DCASE2018](http://dcase.community/challenge2018/task-bird-audio-detection) - The task is to design a system that, given a short audio recording, returns a binary decision for the presence/absence of bird sound (bird sound of any kind). The output can be just "0" or "1", but we encourage weighted/probability outputs in the continuous range [0,1] for the purposes of evaluation. For the main assessment we will use the well-known "Area Under the ROC Curve" (AUC) measure of classification performance.

# Open Source Projects
