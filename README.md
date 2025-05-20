# DA6401_A3

## Note on running for evaluation

- The experiments were run using a Kaggle Notebook to be able to access GPUs. Hence the scripts in the script folder are only fragments of the notebook essentilaly.
- It is hence recommended that the .ipynb file be used to run and not the scripts

## Instructions to run the Notebook & View Results

- Ensure that all required libraries are installed. Use the reuiremnts.txt file 
- Run the following  in the directory location
```pip install -r requirements.txt``` to do that
- **To make life (evaluation in this case) easier, just run the notebook (``` a3-implementation.ipynb```) in a Kaggle instance with a GPU enabled.**
- **I used the T4-GPUs so those are recommended**
- But if you have decided to use scripts, below is a possible procedure:
- ### Data Setup
  - Run the Data Loading Section in the notebook. (If you really must run the scripts, despite warnings, run ```load_data.py```)
- ### Training and testing
- - After correct setup, run ```train_non_attn.py``` or ```train_with_attn.py``` to define train functions for models. Add wandb sweep if sweeping desired.
- ### Evaluation
  - The ```test_non_attn.py``` and ```test_with_attn.py``` scripts should return the test-set accuracies for the best models.
  - The best models are saved as  **model.pth** and **attn_model.pth**
  - Attention visualization can be done by the so-named script. Furhter the Connectivity Visualization script can also be use
  - The Connectivity script generates an HTML file. This may not run in terminal. A case-based HTML file is attached here as ```transliteration_connectivity.html```
