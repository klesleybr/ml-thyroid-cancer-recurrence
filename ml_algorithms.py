from sklearn.model_selection  import train_test_split

class ml_algorithms:

    accuracy_value = None
    recall_value = None
    f1_value = None

    def __init__(self, X_train=None, X_test=None, y_train=None, y_test=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    
    def prepared_data(self, X, y, size=0.3, number_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=size, random_state=number_state)
        
        
    
    