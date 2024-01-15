NTHU eegnet Auditory-eeg-challenge-2024
=======================================
### Group: 7
### Team Leader: 林立上
### Members: 蔡承翰 林榮翼 黃諺霖 林諺瓏
### Mentor: 林家合

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

3. To acquire the dataset for our model, please proceed to [download the data provided](https://drive.google.com/file/d/19oTqfCzovGbtwqHMtIEZL_8XFRGmmwU1/view?usp=sharing)  
Specifically, this folder houses the preprocessed data that is ready for use in our model.

4. To ensure proper configuration, please make adjustments in the `config.json` file. 
Specifically, locate the `dataset_folder` parameter within the file and update its value from `null` to the absolute path leading to the directory housing all pertinent data. In our model, change its value to the path of `split_data`

# Run the Task
