import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

class AdversarialAgent:

    def __init__(self):
        self.X = []
        self.y = []
        self.X2 = []
        self.y2 = []
        self.C = 2.0

    def generate_data(self):
        np.random.seed(1337)
        n_samples = 100
        n_samples_val = 100
        n_features = 2
        slope = 1
        bias = 30

        y = np.random.binomial(n=1,p=0.5,size=n_samples)
        self.y = np.subtract( np.multiply(y,2), 1 )

        # randomly generate X in 2D
        self.X = np.empty( (n_samples,2) )
        self.X[:,0] = np.random.uniform(-20,20,n_samples)
        self.X[:,1] = slope * X[:,0] + np.random.normal(0,10,n_samples)
        self.X[:,1][y > 0] += bias

        # randomly generate validation data as well
        y2 = np.random.binomial(n=1,p=0.5,size=n_samples_val)
        self.y2 = np.subtract( np.multiply(y2,2), 1 )

        self.X2 = np.empty( (n_samples_val,2) )
        self.X2[:,0] = np.random.uniform(-20,20,n_samples_val)
        self.X2[:,1] = slope * X2[:,0] + np.random.normal(0,10,n_samples_val)
        self.X2[:,1][y2 > 0] += bias

    def load_classifier(self):
        self.clf = svm.SVC(C = self.C, kernel = 'linear')

    def fit_classifier(self, X, y):
        self.clf.fit(X, y)

    def find_margin_support_vectors(self,support_vecs,support_inds,
                                    labels,dual_vars,C):
        mask = np.abs(dual_vars) < C
        margin_support_vectors = support_vecs[mask]
        margin_indices = support_inds[mask]
        margin_labels = labels[mask]
        return margin_support_vectors, margin_indices, margin_labels

    def calculate_hinge_loss(self,clf,X2,y2):
        m = y2.shape[0]
        # Calculate raw hinge losses
        loss_array = 1 - y2 * clf.decision_function(X2)
        # Only keep positive losses
        loss_array[loss_array < 0] = 0
        loss = np.sum(loss_array)
        loss_per_data = loss / m
        return loss, loss_per_data

    def loss_function_SVM(self,Xtrain,ytrain,xc,yc,Xvalid,yvalid,C = 2.0):
        X = np.concatenate((Xtrain, xc), axis = 0)
        y = np.concatenate((ytrain, np.array([yc])), axis = 0)
        clf = svm.SVC(C = C, kernel = 'linear')
        clf.fit(X,y)
        loss, loss_per_data = calculate_hinge_loss(clf, Xvalid, yvalid)
        return loss

    def finite_differences(self,f, x, h):
        n = x.shape[0]
        e = np.eye(n)
        fDiff = lambda x, i: (f(x + (1/2)*h*e[i]) - f(x - (1/2)*h*e[i])) / h
        fDiffArray = np.array([fDiff(x, i) for i in range(n)])
        return fDiffArray

    def calculate_Qss(  self,margin_support_vectors,margin_labels,
                        poisoned_is_margin = False):
        x = margin_support_vectors
        y = np.array(margin_labels)[:, np.newaxis]
        Qss = (y * x) @ (y * x).T
        return Qss

    def calculate_A(self,margin_support_vectors,margin_labels):
        # First row = [0, y^T]
        # Bottom =    [y, Qss]
        y = margin_labels
        first_row = np.concatenate([[0], y])
        first_row = np.expand_dims(first_row, axis = 0)
        Qss = calculate_Qss(margin_support_vectors,margin_labels)
        bottom_rows = np.concatenate([y[:, np.newaxis], Qss], axis = 1)
        A = np.concatenate([first_row, bottom_rows], axis = 0)
        return A

    def calculate_b(self,alpha_c,yc,margin_support_vectors,margin_labels,l):
        dQsc = - yc * margin_labels * margin_support_vectors[:, l] * alpha_c
        b = np.concatenate([[0], dQsc])
        b = np.expand_dims(b, 1)
        return b

    def calculate_coef_gradient(self,margin_support_vectors,margin_labels,
                                alpha_c,yc,poisoned_is_margin = False):
        A = calculate_A(margin_support_vectors,margin_labels)
        # Calculate b matrix
        b_list = []
        for l in range(margin_support_vectors.shape[1]):
            b_l = calculate_b(alpha_c,yc,margin_support_vectors,margin_labels,l)
            b_list.append(b_l)
        b = np.concatenate(b_list, axis = 1)
        partial = np.linalg.inv(A) @ b
        # if poisoned is in margin, d_alphac will be at bottom of vector, else 0
        partial_b = np.expand_dims(partial[0], axis = 0)
        if poisoned_is_margin:
            partial_alpha = partial[1:-1]
            partial_alpha_c = partial[-1]
        else:
            partial_alpha = partial[1:]
            partial_alpha_c = np.zeros(shape = (1, partial.shape[1]))
        return partial_b, partial_alpha, partial_alpha_c


    def decision_function_gradient( self,partial_alpha,partial_b,partial_alpha_c,
                                    alpha_c,margin_support_vectors,margin_labels,
                                    xc,yc,xk):
        partial_b, partial_alpha, partial_alpha_c = \
            calculate_coef_gradient(margin_support_vectors, margin_labels,
                                    alpha_c, yc, poisoned_is_margin)
        terms = [0] * 4
        # First term
        xk = np.expand_dims(xk, axis = 0)
        y_s = np.expand_dims(margin_labels, axis = 1)
        x_s = margin_support_vectors
        terms[0] = np.sum(partial_alpha * (y_s * (x_s @ xk.T)), axis = 0)
        # Second term
        terms[1] = np.sum(partial_alpha_c * (yc * (xc @ xk.T)), axis = 0)
        # Third term
        terms[2] = alpha_c * yc * xk
        # Fourth term
        terms[3] = partial_b
        f_grad = sum(terms)
        return f_grad

    def loss_gradient_approximation(self,X,y,xc,yc,X2,y2, C = 2.0):
        # Append adversy to training data
        Xpoison = np.empty((X.shape[0] + 1, X.shape[1]))
        Xpoison[0:-1] = X
        Xpoison[-1] = xc
        ypoison = np.empty(y.shape[0] + 1)
        ypoison[0:-1] = y
        ypoison[-1] = yc
        
        # Train classifier
        clf = svm.SVC(C = C, kernel = 'linear')
        clf.fit(Xpoison,ypoison)
        
        # Get supports and duals
        support_indices = clf.support_
        support_vectors = clf.support_vectors_
        dual_vars = clf.dual_coef_[0]
        alpha_c = dual_vars[-1]
        poisoned_is_margin = (0 < alpha_c < C)
        
        # Get partial derivatives
        partial_b, partial_alpha, partial_alpha_c = \
            calculate_coef_gradient(margin_support_vectors,margin_labels,
                                    alpha_c,yc,poisoned_is_margin)
        
        # Calculate loss over validation set
        m = X2.shape[0]
        n = X2.shape[1]
        
        loss_gradient = np.zeros((1, n))
        for i in range(m):
            loss_gradient += decision_function_gradient(partial_alpha, \
                partial_b,partial_alpha_c,alpha_c,margin_support_vectors,
                margin_labels,xc,yc,X2[i]) / m    
        return loss_gradient

    def gradient_ascent_with_gradient(self,f, fdot, t, x0, eps = 1e-3, \
        h = 1., t_div = 25., MAX_ITER = 100, verbose = True ):
        
        x0 = np.expand_dims(x0, axis = 0)
        if verbose:
            tic = time.time()
        # Initialize losses
        x_history = []
        x_history.append(x0.copy())
        losses = [0]
        losses.append(f(x0))
        
        for i in range(MAX_ITER):
            if i >= 10:
                trailing = losses[i-10:i]
                if (max(trailing) - min(trailing) < eps):
                    print('Loss increased by less than epsilon.')
                    break
            # Calculate gradient and rescale to unit norm
            grad = fdot(x0)
            if np.linalg.norm(grad) < 1e-1:
                grad = np.random.randn(x0.shape[0], x0.shape[1])
            grad = grad / np.linalg.norm(grad)
            
            # Find best step size and update
            best_step_size = np.argmax([f(x0 + i*t/t_div*grad) \
                                for i in range(int(t_div+1))])
            if best_step_size == 0:
                grad = np.random.randn(len(x0))
                grad = grad / np.linalg.norm(grad)
                best_step_size = t_div
            
            x0 += best_step_size*(t/t_div)*grad
            # Append to history
            x_history.append(x0.copy())
            losses.append(f(x0))    
        xopt = x0
        losses = losses[1:]
        if verbose:
            toc = time.time()
            print('total running time is {} seconds.'.format(toc - tic))
        return xopt, x_history, losses
