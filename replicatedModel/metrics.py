import pandas as pd
import numpy as np

def collect_metrics(y_score_test,y_true_test,y_pred_test,sample_indices_test):
    MASTER_DATASET_INDEX = pd.read_csv("/home/tomeu/spektral/data/master_metadata_index.csv")
    rows=[]
    for i in range(len(sample_indices_test)):    
        temp={} # Definicion de un diccionario
        idx = sample_indices_test[i]
        temp["patient_ID"] = str(MASTER_DATASET_INDEX.loc[idx, "patient_ID"])
        temp["sample_idx"] = idx;
        temp["y_true"] = y_true_test[sample_indices_test[i]]
        temp["y_probs"] = y_score_test[i]
        temp["y_pred"] = y_pred_test[i]
        rows.append(temp) 

    test_patient_df = pd.DataFrame(rows)
    y_probs_patient,y_true_patient, y_pred_patient = get_patient_prediction(test_patient_df)

    return y_probs_patient,y_true_patient, y_pred_patient

    
def get_patient_prediction(test_p_df):
    ls_uniq_patients =  list(test_p_df["patient_ID"].unique())
    grouped_df = test_p_df.groupby("patient_ID") # Todas las muestras agrupadas por sujeto
    rows= []
    for patient in ls_uniq_patients:
        patient_df = grouped_df.get_group(patient) # Data frame de un solo sujeto
        temp = { }
        temp["patient_ID"] = patient
        temp["y_true"] = list(patient_df["y_true"].unique())[0]
        assert len(list(patient_df["y_true"].unique())) == 1
        temp["y_pred"] = patient_df["y_pred"].mode()[0]
        temp["y_probs"] = patient_df["y_probs"].mean()
        rows.append(temp)
        
    return_df = pd.DataFrame(rows)
    y_probs_patient = np.array(list(return_df["y_probs"]))
    y_true_patient= np.array(list(return_df["y_true"]))
    y_pred_patient= np.array(list(return_df["y_pred"]))
    
    return y_probs_patient,y_true_patient, y_pred_patient