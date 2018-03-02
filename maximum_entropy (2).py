
import os
import sys
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy import optimize
from collections import defaultdict
from util import evaluate, load_data


class MaximumEntropyModel():
    
    def __init__(self):
        self.weight = None
        self.data_len = None
        
        self.dict_char = None
        self.char_index = None
        self.char_list_len = None
        
        self.label = None
        self.label_list = None
        self.label_list_len = None
        self.y_index = None
        self.phi_y_index = None
        self.vec = None
              
    def label_process(self,training_label):
        self.label = training_label
        self.data_len = len(training_label)
        self.label_list = list(set(self.label))
        self.label_list_len = len(self.label_list)
        self.y_index = [self.label_list.index(self.label[i]) for i in range(self.data_len)]
        self.phi_y_index = [self.y_index[i]+i*(self.label_list_len) for i in range(len(self.y_index))]
        vector = np.zeros(self.data_len*self.label_list_len)
        for i in self.phi_y_index:
            vector[i] = 1
        self.vec = vector
        
    def gen_dict(self,training_data):
        char_dict = defaultdict(int)
        char_index = {}
        count=0
        for i in range(self.data_len):
            for temp_char in training_data[i]:
                char_dict[temp_char] += 1
        self.dict_char = {k:v for k,v in char_dict.items() if v > 1}
        self.char_index = {k:index for index, k in enumerate(self.dict_char.keys())}
        self.char_list_len = len(self.char_index)
        
        
    def phi_x(self,train_data):
        sparse_mat = sp.dok_matrix((self.data_len,self.char_list_len), dtype=np.int8)
        for i in range(self.data_len):
            for item in train_data[i]:
                 if item in self.char_index:
                    sparse_mat[i,self.char_index.get(item)] = 1     
        sparse_mat = sparse_mat.tocsr()
        return sparse_mat
    
    def phi_matrix(self,training_data):
        print(len(training_data))
        mat = sp.dok_matrix((len(training_data)*self.label_list_len,self.char_list_len*self.label_list_len),dtype=np.int8)
        for i in range(len(training_data)):
            addon = i*self.label_list_len
            for temp_char in training_data[i]:
                if temp_char in self.char_index:
                    temp_index = self.char_index[temp_char]
                    for k in range(self.label_list_len):
                        mat[addon+k,k*self.char_list_len+temp_index] = 1
        print(mat.shape)
        phi = sp.csr_matrix(mat)
        return phi
   
    def prob_cal(self,weight,phi):
        w_phi = phi.dot(weight)
        max_5 = [np.max(w_phi[i:i+self.label_list_len]) for i in range(0, len(w_phi), self.label_list_len)]
        full_e_vec = [w_phi[i]-max_5[int(i/self.label_list_len)] for i in range(len(w_phi))]
        e_vec= np.exp(full_e_vec)
        # sum over every five elements
        sum_p = [np.sum(e_vec[i:i+self.label_list_len]) for i in range(0,len(e_vec),self.label_list_len)] 
        prob_vec = [e_vec[i]/sum_p[int(i/self.label_list_len)] for i in range(len(e_vec))]
        return prob_vec
    
    def lw(self,weight,phi):
        prob = self.prob_cal(weight,phi)
        log_lw = np.sum(np.log([prob[i] for i in self.phi_y_index]))
        deriv = prob*phi - self.vec*phi
        return -1*log_lw/self.data_len,np.array(deriv)/self.data_len
    
    def run_opt(self,weight,phi):
        weight, min_val,info = optimize.fmin_l_bfgs_b(self.lw, weight,args=(phi,))
        return weight,min_val,info
        
    def train(self, training_data, training_label):
        self.label_process(training_label)
        self.gen_dict(training_data)
        self.phi_x(training_data)
        phi = self.phi_matrix(training_data)
        weight_0 = np.random.rand(self.char_list_len*self.label_list_len, 1)
        self.weight, min_val,info = self.run_opt(weight_0,phi)
        return info

    def predict(self, test):
        test_phi = self.phi_matrix(test)
        prob_res = self.prob_cal(self.weight,test_phi)
        predict = [np.argmax(prob_res[i:i+self.label_list_len]) for i in range(0, len(prob_res), self.label_list_len)]
        predict_final = [self.label_list[i] for i in predict]
        return predict_final

if __name__ == "__main__":
    train_data, train_label, dev_data, dev_label, test_data, data_type = load_data(sys.argv)

    # Train the model using the training data.
    model = MaximumEntropyModel()
    model.train(train_data,train_label)
    
    # report accuracy on train data
    train_accuracy = evaluate(model,
                            train_data,train_label,
                            os.path.join("results", "perceptron_" + data_type + "_train_predictions.csv"))
    print(train_accuracy)
    # Predict on the development set. 
    print("before dev_acc ",len(dev_data))
    dev_accuracy = evaluate(model,
                            dev_data,dev_label,
                            os.path.join("results", "maxent_" + data_type + "_dev_predictions.csv"))
    print(dev_accuracy)

    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    seudo_label = np.zeros(len(test_data))
    evaluate(model,
             test_data,seudo_label,
             os.path.join("results", "maxent_" + data_type + "_test_predictions.csv"))
