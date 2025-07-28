You are provided with the code base of the research: Prompt-CAM. The research main text is in PromptCAM.md.

### Purpose:
- Adapt the idea of PromptCAM to a similar task: identical twin faces verification (given two highly similar face image, tell whether they are same person or not). Twin faces look highy similar, except for small details - which can be consider as an fine-grained image analysis task.
- Modify the codebase to match that idea.

### Dataset for training:
- Dataset for training is the ND TWIN 2009 - 2010 dataset. Faces images were pre-processed through: face landmarks detection, face alignment, face detection, cropped out face-only images. The images are square, and can be resize to any size to fit the input size of any models.
- The file id_to_images.json shows the structure of the whole dataset: under a dictionary, with keys are person ids and corresponding value of each key is a list of image paths. Since the dataset is an identical twin faces dataset, all ids has their own twin person, formed into twin pairs. The train_twin_pairs.json contains the information on twin pairs under a list of twin person id pairs. The same with test set: test_twin_pairs.json. train_twin_pairs for training and test_twin_pairs for testing. Take a look at them to know the format.
- There are 175 train twin pairs, the number of images per person is minimum of 4, maximum is 68, total image are 6639. Test set has 29 twin pairs, 602 images. Take a look at some of their first lines for more information.

### Training resources:
- Kaggle (Nvidia Tesla P100 - 16GB VRAM).

### Evaluation methods:
- EER, AUC, ROC, Verification Accuracy, TAR FAR. Proposed some more metrics if necessary. Calculate metrics focus on twin only and same person, we do not care non twin.

### Tracking method:
- For Kaggle, API key is in Kaggle kaggle secrets, in WANDB_API_KEY variable. The entity name is hunchoquavodb-hanoi-university-of-science-and-technology. Let WanDB set the name for each run.

### More information:
- Model checkpoint should be performed every 2 epoch.
- There should be two option: fresh training or resume training from checkpoint.
- This repo will be push to GitHub, with all .json file containing dataset information. On Kaggle, we clone the repo from Github:

```Kaggle
!git clone "https://github.com/github_name/this_repo.git"
```

and training:

```python
import os
# Change working dir to our repo
os.chdir('/kaggle/working/repo_name')

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")
print("WandB configured")

# Run training
!python etc
```

### Question:
- How can I modify PromptCAM to adapt the new task?
- What is the input size images I should resize to?
- Read the research paper and the codebase carefully, if there are anything that you do not clear in both, tell me.
- If the strategy include sampling negative pairs, just sample hard-negatives (negative image from twin person). And positive pairs are from same person.
- If you still have any question, tell me?

### Things to do:
- Write an implementation plan markdown file, for YOU to revisit while modify the code base to match the idea. This file's purpose is to prevent you from forget what you are doing. It should be clear, fully in detail but short, concise. The content can be the project structure, what you are writing in this files, etc...
- Break down implementation into phase, to do list. We will go step by step.
- I want a code base that efficient, clear, just write code that needed. The code base should fully utilized both training datset and training resources. Another time, be minimal, just write scripts/code that necessary. I do not need notebooks, utils that never used.
