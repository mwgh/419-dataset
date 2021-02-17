# Project Group 13: Dear User, Are You Flirting With Me?

This is our 419-dataset containing flirting videos, and the following gives a instruction of what we did for our dataset.

[GitHub repository](https://github.com/mwgh/419-dataset)

## The structure of your code/folders

Our project can be executed by running the code in `FinalModel.ipynb`. Please skip the cell that calculates the similarity matrix, which is used to train our model, because calculating the similarity matrix takes a long time. We saved the matrix to `Similarity_Matrix_total`.

### Notebooks
* `Clips&Annotation.ipynb`: textual descriptions of the videos in `clipped_videos/`.
* `FinalModel.ipynb`: k-NN and k-means models. Creates the file `aggregated_openface_data.pickle` if it does not exist.
* `419_Project_Data_PreProcessing.ipynb`: runs OpenFace on the clipped_videos. Saves the features to `openface_processed/multi/`.
* `Semi_Supervised.ipynb`: semi-supervised learning on the OpenFace data using TensorFlow.
* `419_Project_Analysis.ipynb`: cluster the OpenFace data using tslearn.
* `CMPT419_project_sound.ipynb`: get the audio features. Saves the features to `audio-features.csv` and the audio files to `audio_output/`.

### Collected dataset
* `clipped_videos/`: flirting and non-flirting videos.

### Computed features and model
* `aggregated_openface_data.pickle`: Python 3 'pickle' of a Pandas Dataframe containing the aggregated output of OpenFace.
* `audio-features.csv`: contains the audio features. Output of the notebook `CMPT419_project_sound.ipynb`.
* `Similarity_Matrix_total`: Python 3 'pickle' of a numpy array used to train the models. Represents the similarity between every pair of clips.

### Output of OpenFace and FFmpeg
* `openface_processed/multi/`: OpenFace's output on the clipped_videos. (Not included because it was too large.)
* `audio_output/`: audio files extracted from the clipped_videos with FFmpeg. (Not included because it was too large.)

## A self-evaluation of your project with respect to your proposal.
Initially, we planned to do semi-supervised learning with only 20 videos in order to get more data. Then train with a model that deals with sequence data. However, we did not understand the semi-supervised learning well. We spend a lot of time working on learning the TensorFlow which seems to provide a solution for the shortage of data. This link provides the historical work we tried with Tensorflow: https://colab.research.google.com/drive/13H3Q38XdNJAncF0MxcCofa6RkWlb3w6S?usp=sharing. The method is called Neural Structured Learning provided in the link: https://www.tensorflow.org/neural_structured_learning. However, we seem to misunderstand the Tensorflow that we were told Tensorflow only works well with a large dataset. With limited time, we decided to annotate as much data as possible and train with a K-NN model with the independently multidimensional dynamic time warping. In the end, we had 105 clips in total, 55 for flirting action, 50 for non-flirting action. We then train our model with the final similarity matrix computed with the independently multidimensional DTW. Overall, I think our group is trying as hard as we can, and we gain a lot more new knowledge on the process of doing this project. Good Job Team!
