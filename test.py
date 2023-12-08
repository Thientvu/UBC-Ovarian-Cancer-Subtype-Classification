import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import numpy as np

def test(model, data_loader):
    y_pred_list = np.array([])
    y_target_list=np.array([])
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            image, label = data
            output = model(image)
            y_pred_tag = torch.argmax(output,1)

            #print(y_pred_tag, label)
            
            y_pred_list= np.append(y_pred_list, y_pred_tag.detach().numpy())
            y_target_list = np.append(y_target_list, label.detach().numpy())

    accuracy = accuracy_score(y_target_list,y_pred_list)    

    precision = precision_score(y_target_list,y_pred_list,
                        average='weighted',
                        zero_division=0)
    
    confusion = confusion_matrix(y_target_list,y_pred_list)

    print(f"Accuracy:\t{accuracy}")

    print(f"Precision score:\t{precision}")

    print(f"Confusion matrix:\t{confusion}")
    
    return accuracy, precision, confusion