## New Patient Models

Code: `new_patient_model.py`, `new_patient_additional_model_descs.py`, `simplified_models.py`

Data: `patient_data_venex/` (currently private)

### Model 1

References: Jost, F., Schalk, E., Rinke, K., Fischer, T. & Sager, S. Mathematical models for cytarabine-derived myelosuppression in acute myeloid leukaemia. PLoS One 14, e0204540 (2019). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6602180/

Jost, F. et al. Model-Based Optimal AML Consolidation Treatment. IEEE Trans. Biomed. Eng. 67, 3296–3306 (2020). https://ieeexplore.ieee.org/document/9093109/

Code: `new_patient_model.py`


### Model 4a

References: Stiehl, T., Wang, W., Lutz, C. & Marciniak-Czochra, A. Mathematical Modeling Provides Evidence for Niche Competition in Human AML and Serves as a Tool to Improve Risk Stratification. Cancer Research 80, 3983–3992 (2020). https://doi.org/10.1158/0008-5472.CAN-20-0283

(note: these references are for components of the model, not the model as a whole)

The main new component of this model is the "carrying capacity" of total blast + proliferating cell count.

Code: `new_patient_additional_model_descs.py`

### Model 4b

Ref: Miraki-Moud, F. et al. Acute myeloid leukemia does not deplete normal hematopoietic stem cells but induces cytopenias by impeding their differentiation. Proceedings of the National Academy of Sciences 110, 13576–13581 (2013). https://www.pnas.org/doi/full/10.1073/pnas.1301891110

New model component: inhibitory effect of blasts on healthy stem cell differentiation and proliferation

Code: `new_patient_additional_model_descs.py`

### Model 2a, 2b, 2c

Code: `simplified_models.py`


### Analysis of patient data

Data: `patient_data_venex/` (currently private)


## Helper code

`tellurium_model_fitting.py` - this contains code for constructing objective functions.

`find_map.py` - this is a hack of PyMC's `find_MAP` function that allows for additional gradient-free optimization methods, such as PyBOBY-QA.

`systematic_model_comparisons_multiprocessing.py` - this is a script for running the models on the dataset.
