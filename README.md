NTHU eggnet Auditory-eeg-challenge-2024
=======================================
### Group 7, Introduction to Machine Learning

This repository documents the methodologies and implementation strategies utilized by Team eggnet from National Tsing Hua University to address Task 1 of the Auditory EEG Challenge at ICASSP 2024. The main goal of this task was to precisely identify the correct auditory stimulus corresponding to a given EEG segment from a set of five candidates.

<div align="center">
    <a href="./">
        <img src="./figures/task1.png" width="70%"/>
    </a>
  <p>Schematic overview of the task</p>
</div>
<br></br>
Our approach leverages Long Short-Term Memory (LSTM) networks within the model, leading to Team eggnet securing the 24th position out of 60 teams in the competition. 

# Getting Started

1. Check the installed version of Python using the following command:
```bash
python -v
```
Ensure that you have Python 3.6 or a later version installed on your system, as Python 3.6+ is required to run the model successfully. If the displayed version is below Python 3.6, consider updating your Python installation to meet the specified requirements.

2. Clone the repository to your local machine, then navigate to the root folder of the repository. In the root folder, execute the following command in your terminal to install the necessary dependencies specified in the requirements.txt file:
```bash
python3 -m install requirements.txt
```

3. To acquire the dataset for our model, please proceed to [download the data provided](https://homes.esat.kuleuven.be/~lbollens)
using the password provided when you registered.

Specifically, the `split_data` folder houses the preprocessed data that is ready for use in our model.

4. To ensure proper configuration, please make adjustments in the `config.json` file.

Specifically, locate the `dataset_folder` parameter within the file and update its value from `null` to the absolute path leading to the directory housing all pertinent data. In our model, change its value to the path of `split_data`
<br></br>

# Run the Task

Navigate to the "experiments" folder within the "task1_match_mismatch" directory and execute the following command:
```bash
python dilated_convolutional_model.py
```
This command is designed for both training and testing the model. Simply run it to initiate the desired operation.

# Model Training Output Example

If you run the code successfully, the model will start training and print the training statistics.

```text
Epoch 1/100
7082/7082 [==============================] - 1226s 172ms/step - loss: 1.0915 - accuracy: 0.5653 - val_loss: 1.4165 - val_accuracy: 0.4882
Epoch 2/100
7082/7082 [==============================] - 1204s 170ms/step - loss: 0.9989 - accuracy: 0.6137 - val_loss: 1.1126 - val_accuracy: 0.5592
Epoch 3/100
7082/7082 [==============================] - 1217s 172ms/step - loss: 0.9386 - accuracy: 0.6389 - val_loss: 1.0767 - val_accuracy: 0.5804
Epoch 4/100
7082/7082 [==============================] - 1225s 173ms/step - loss: 0.8983 - accuracy: 0.6563 - val_loss: 1.1335 - val_accuracy: 0.5502
Epoch 5/100
7082/7082 [==============================] - 1211s 171ms/step - loss: 0.8687 - accuracy: 0.6676 - val_loss: 0.9795 - val_accuracy: 0.6186
Epoch 6/100
7082/7082 [==============================] - 1209s 171ms/step - loss: 0.8466 - accuracy: 0.6765 - val_loss: 0.9942 - val_accuracy: 0.6120
Epoch 7/100
7082/7082 [==============================] - 1210s 171ms/step - loss: 0.8262 - accuracy: 0.6860 - val_loss: 1.1198 - val_accuracy: 0.5572
Epoch 8/100
7082/7082 [==============================] - 1209s 171ms/step - loss: 0.8105 - accuracy: 0.6912 - val_loss: 0.9590 - val_accuracy: 0.6243
Epoch 9/100
7082/7082 [==============================] - 1209s 171ms/step - loss: 0.7968 - accuracy: 0.6963 - val_loss: 0.9493 - val_accuracy: 0.6341
...
```

```text
639/639 - 18s - loss: 1.2018 - accuracy: 0.5196
558/558 - 12s - loss: 1.1811 - accuracy: 0.5333
749/749 - 16s - loss: 0.7837 - accuracy: 0.7095
613/613 - 13s - loss: 0.5439 - accuracy: 0.7993
639/639 - 14s - loss: 1.4661 - accuracy: 0.4122
826/826 - 18s - loss: 0.9582 - accuracy: 0.6259
298/298 - 6s - loss: 1.6356 - accuracy: 0.3705
613/613 - 13s - loss: 0.9958 - accuracy: 0.6189
549/549 - 12s - loss: 0.7300 - accuracy: 0.7173
815/815 - 17s - loss: 0.8568 - accuracy: 0.6616
826/826 - 17s - loss: 1.3676 - accuracy: 0.4906
826/826 - 17s - loss: 1.0536 - accuracy: 0.5976
535/535 - 11s - loss: 1.2430 - accuracy: 0.5077
535/535 - 11s - loss: 1.2065 - accuracy: 0.5364
639/639 - 13s - loss: 1.1520 - accuracy: 0.5612
303/303 - 6s - loss: 2.0578 - accuracy: 0.2561
536/536 - 11s - loss: 0.8785 - accuracy: 0.6634
558/558 - 12s - loss: 1.2344 - accuracy: 0.4792
558/558 - 11s - loss: 0.7200 - accuracy: 0.7409
826/826 - 17s - loss: 0.8283 - accuracy: 0.6712
...
```

# Performance

We've tried multiple methodologies to enhance the performance of our model. The graphical representation below encapsulates our journey, presenting both the accuracy and loss metrics:
<div align="center">
    <a href="./">
        <img src="./figures/plots.png" width="70%"/>
    </a>
  <p>Training and validation accuracy/loss plots</p>
</div>
<br></br>
Here is our ultimate score in the challenge, captured in the image below, extracted directly from the leaderboard snapshot:
<div align="center">
    <a href="./">
        <img src="./figures/submission2score.png" width="50%"/>
    </a>
    <p>Final score of our submission</p>
</div>
<br></br>

We secured the **24th** position out of **60** teams. Below, you'll find a comparative analysis of scores, encompassing the baseline, Team MLG (the counterpart focusing on the same topic), and our performance.

<div align="center">
  
| Team | Submission Number | Subjects Mean | Subjects Std | Final Score |
| :-: | :-: | :-: | :-: | :-: |
| **eggnet** | **2** | **53.11** | **12.81** | **53.11** |
| MLG | 2 | 52.18 | 13.02 | 52.18 |
| MLG | 1 | 51.59 | 13.86 | 51.59 |
| eggnet | 1 | 51.38 | 12.79 | 51.38 |
| baseline | . | 50.74 | 12.71 | 50.74 |

</div>
<br></br>

