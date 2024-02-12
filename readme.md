# Prompted Contextual Vectors for Spear Phishing Detection
This repo is the implementation of the paper titled 'Prompted Contextual Vectors for Spear Phishing Detection'.
### Vectors Generation Steps:
1. Configure your LLM deployments in 'src/llms.py', see example functions in this file.
2. Store your relevant API keys in a .env file (see example.env).
3. Run 'src/pipelines/generate_prompted_contextual_vectors.py'
4. The output will be saved in the 'output' folder. 
It is recommended to run 'src/pipelines/verify_output_df_pipeline.py' on the generated 'results.pkl' file in order to impute invalid LLM output.

Acknowledgement: This work was done in collaboration with Accenture Labs, Israel.

### Collaborators
#### Accenture Cyber Research Lab
Daniel Nahmias  
Gal Engelberg  
Dan Klein  

#### Ben-Gurion University of the Negev
Asaf Shabtai
