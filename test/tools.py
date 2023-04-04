
def confusion_matrix_cm():
    np.set_printoptions(precision=2)
    cm = confusion_matrix(y_trues, y_hat_probs)
    print(labels)
    print(cm)
    # Define a labels array for future use
    labels = [ 
            'Air Conditioner',
            'Car Horn',
            'Children Playing',
            'Dog bark',
            'Drilling',
            'Engine Idling',
            'Gun Shot',
            'Jackhammer',
            'Siren',
            'Street Music',
            ]   
    # Build classification report
    re = classification_report(y_trues, y_hat_probs, labels=[0,1,2,3,4,5,6,7,8,9], target_names=labels)
    print(re)

