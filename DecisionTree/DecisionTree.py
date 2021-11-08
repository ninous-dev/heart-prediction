from .Node import Node

import numpy as np
import os
import seaborn as sns
import ast
import sys
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix

"""
DecisionTree Class

Decision tree containing decision and leaf node to predict a label.

"""

class DecisionTree:
    def __init__(self, min_participant=2, max_depth=2):
        """
        Atributes:
            x_col_names (list(str)) : List of input column names.
            root = Node of the root of the tree.

            min_participant (int): Minimum number of variables to compare individuals.
            max_depth (int): Maximum depth of the tree.
        """

        self.x_col_names = None

        self.root = None

        self.min_participant = min_participant
        self.max_depth = max_depth

    def split_input_label(self, dataset):
        """
        Split the dataset into inputs and labels.

        Parameters:
            dataset (array) : The dataset array.

        Returns:
            inputs (array), labels (array)
        """
        return dataset[:, :len(self.x_col_names)], dataset[:, len(self.x_col_names):]

    def build_tree(self, dataset, depth=0):
        """
        Builder of the tree based on the dataset

        Parameters:
            dataset (array) : The dataset array.
            depth (int) : The current depth of the tree.

        Returns:
            subtree (Node)
        """

        count_participant, _ = np.shape(dataset)

        if count_participant >= self.min_participant and depth <= self.max_depth:
            split = self.get_split_value(dataset)

            if split['information_gain'] > 0 :
                left = self.build_tree(split['left'], depth=depth + 1)
                right = self.build_tree(split['right'], depth=depth + 1)

                #Case where the evaluation of nodes are the same.
                if left.value != None and right.value != None and left.value == right.value:
                    return Node(value=left.value)

                #Case of a decision node.
                return Node(left, right, split['information_gain'],
                            split['threshold'], split['column_name'],
                            split['column_index'], split['operator'])

        #Case of a leaf node.
        leaf_value = self.compute_leaf_value(dataset)
        return Node(value=leaf_value)

    def get_split_value(self, dataset):
        """
        Find the split with the best information gain in the current dataset.

        Parameters:
            dataset (array) : The dataset array.

        Returns:
            split (dict)
        """

        max_gain = float('-inf')
        split = {}
        inputs, labels = self.split_input_label(dataset)

        for i, column in enumerate(self.x_col_names):

            binary_col = False
            col = dataset[:, i]
            possible_thresholds = np.unique(col)

            if len(possible_thresholds) == 2:
                binary_col = True
                possible_thresholds = possible_thresholds[:1]
            else:
                possible_thresholds = possible_thresholds[1:-1]

            for threshold in possible_thresholds:
                left, right = self.split(dataset, threshold, i, binary_col)
                gain = self.compute_information_gain(dataset, left, right)

                if gain > max_gain:
                    max_gain = gain
                    split['information_gain'] = gain
                    split['threshold'] = threshold
                    split['left'] = left
                    split['right'] = right
                    split['column_name'] = column
                    split['column_index'] = i
                    split['operator'] = "=" if binary_col else "<="

        return split

    def split(self, dataset, threshold, column_index, binary_col):
        """
        Split the dataset based on the condition for the node construction.

        Parameters:
            dataset (array) : The dataset array.
            threshold (int) : Condition to split.
            column_index (int) : Column index to use for the condition.
            binary_col (bool) : True if the column contains binary values.

        Returns:
            left (array), right (array)
        """

        if binary_col:
            left = dataset[np.where(dataset[:, column_index] == threshold)]
            right = dataset[np.where(dataset[:, column_index] != threshold)]
        else:
            left = dataset[np.where(dataset[:, column_index] <= threshold)]
            right = dataset[np.where(dataset[:, column_index] > threshold)]

        return left, right

    def compute_information_gain(self, dataset, left, right):
        """
        Compute information gain with the entropy for the current split.

        Parameters:
            dataset (array) : The dataset array.
            left (array) : The left array.
            right (array) : The right array.

        Returns:
            information_gain (float)
        """

        _, labels = self.split_input_label(dataset)
        _, left_labels = self.split_input_label(left)
        _, right_labels = self.split_input_label(right)

        weight_left = len(left) / len(dataset)
        weight_right = len(right) / len(dataset)

        childs_entropy = self.entropy(left_labels) * weight_left + self.entropy(right_labels) * weight_right

        information_gain = self.entropy(labels) - childs_entropy

        return information_gain

    def entropy(self, labels):
        """
        Entopy function.
        Sum of the entropy of each label.
        First compute the proportion of the value in the current set.
        Then add this proportion mutliply by (-1) and the log2 of this proportion.

        Parameters:
            labels (array) : The labels array to compute.

        Returns:
            entropy_res (float)
        """

        entropy_res = 0
        for value in [0, 1]: #possible values
            value_proportion = len(labels[labels == value]) / len(labels)
            if value_proportion != 0:
                entropy_res += -value_proportion * np.log2(value_proportion)
        return entropy_res

    def compute_leaf_value(self, dataset):
        """
        Get the most represented label in this leaf node.

        Parameters:
            dataset (array) : The dataset array.

        Returns:
            value (int)
        """

        inputs, labels = self.split_input_label(dataset)
        values, counts = np.unique(labels, return_counts=True)
        index = np.argmax(counts)

        return values[index]

    def fit(self, X, Y, x_col_names):
        """
        Train the decision tree with X and Y.

        Parameters:
            X (array) : Inputs dataset.
            Y (array) : Labels dataset.
            x_col_names : List of column names of the inputs.

        Returns:
            None
        """

        dataset = np.concatenate((X, Y), axis=1)
        self.x_col_names = x_col_names
        self.root = self.build_tree(dataset)

    def evaluate(self, x, decision_tree):
        """
        Evaluation of the tree for an individual for the current subtree

        Parameters:
            x (list): List of inputs for an individual.
            decision_tree : Current subtree

        Returns:
            evaluation (int)
        """

        #leaf
        if decision_tree.value != None:
            return decision_tree.value

        #decision node
        x_column_value = x[decision_tree.column_index]

        if x_column_value <= decision_tree.threshold:
            return self.evaluate(x, decision_tree.left)
        return self.evaluate(x, decision_tree.right)

    def predict(self, X):
        """
        Prediction of an array of individual.

        Parameters:
            X (array) : Array of individual.

        Returns:
            evaluate (int)
        """

        return [self.evaluate(x, self.root) for x in X]

    def evaluate_with_list(self, x, decision_tree, display_evaluation_flow=True, evaluation_list=[]):
        """
        Evaluation of the tree with a dispay of the evaluation flow.

        Parameters:
            x (list) : Individual values.
            decision_tree (Node) : Decision tree to evaluate.
            evaluation_list (list) : List of all informations about the evaluation of the tree.
            display_evaluation_flow (bool) : If True display the evaluation flow.

        Returns:
            Evaluate (int)
        """

        #leaf node
        if decision_tree.value != None:
            if display_evaluation_flow:
                print(*evaluation_list, sep='\n')

            print("\x1b[6;30;42mValue :\x1b[0m", decision_tree.value)
            return evaluation_list, decision_tree.value

        #decision node
        x_column_value = x[decision_tree.column_index]

        #left evaluation
        if x_column_value <= decision_tree.threshold:
            evaluation_list.append(decision_tree.column_name + " : " + str(x_column_value) + " " + decision_tree.operator + " " + str(decision_tree.threshold))
            return self.evaluate_with_list(x, decision_tree.left, display_evaluation_flow, evaluation_list)

        #right evaluation
        if decision_tree.operator == "=":
            operator = ' != '
        else:
            operator = ' > '

        evaluation_list.append(decision_tree.column_name + " : " + str(x_column_value) + operator + str(decision_tree.threshold))
        return self.evaluate_with_list(x, decision_tree.right, display_evaluation_flow, evaluation_list)

    def predict_one_element(self, x, display_evaluation_flow=True):
        """
        Evaluation of one individual by calling evaluate_with_list

        Parameters:
            x (list) : Individual values.
            diplay_evaluation_flow (bool) : If True display the evaluation flow.

        Returns:
            Evaluate (int)
        """

        return self.evaluate_with_list(x, self.root, display_evaluation_flow=display_evaluation_flow)

    def pretty_print(self, information_gain=True):
        """
        Pretty print the decision tree

        Parameters:
            information_gain (bool) : If True display the information gain for each node.

        Returns:
            None
        """

        if self.root == None:
            print("No tree, please use the fit method")
            return

        self.pretty_print_rec(self.root)

    def pretty_print_rec(self, node, tiret="|---"):
        """
        Pretty print the subtree

        Parameters:
            node (Node) : The current Node object to print.
            information_gain (bool) : If True display the information gain for each node.
            tiret (str) : String to print before each new depth

        Returns:
            #TODO
        """

        #leaf node
        if node.value != None:
            print(tiret, "value", node.value)
            return

        #decision node
        print(tiret, node.column_name, node.operator, node.threshold, end=' ')

        #two leaf nodes as child
        if node.left.value != None and node.right.value != None:
            print("value", node.left.value , "else", node.right.value)
            return

        #left leaf node
        if node.left.value != None:
            print("value", node.left.value)
            self.pretty_print_rec(node.right, tiret=tiret + "|---")
            return

        #right leaf node
        if node.right.value != None:
            print("else ", node.right.value)
            self.pretty_print_rec(node.left, tiret=tiret + "|---")
            return

        #two decision nodes as child
        print()
        self.pretty_print_rec(node.left, tiret=tiret + "|---")
        self.pretty_print_rec(node.right, tiret=tiret + "|---")

    def prediction_analyse(self, X_test, Y_test, confusion_matrix_display=True, proportion_informations=True):
        """
        Predict and display tools to analyse the result.

        Parameters:
            X_test (array) : Inputs array.
            Y_test (array) : Labels array.
            confusion_matrix_display (bool) : If True display the confusion matrix
            proportion_informations (bool) : If True display proportion information (sensitivity, specificity, PPV, NPV)

        Returns:
            accuracy (float)
        """
        #Predict values for each individuals
        Y_pred = self.predict(X_test)

        #Compute accuracy score
        accuracy = accuracy_score(Y_test.flatten(), Y_pred)
        print("accuracy ==>", accuracy)

        #Confusion matrix
        if confusion_matrix_display:
            cm = confusion_matrix(Y_test.flatten(), Y_pred)

            group_names = ['True Neg','False Pos','False Neg','True Pos']
            group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
            labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                     zip(group_names,group_counts,group_percentages)]
            labels = np.asarray(labels).reshape(2,2)
            sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
            plt.show()

        #proportion informations
        if proportion_informations:
            TP = cm[1,1] # true positive 
            TN = cm[0,0] # true negatives
            FP = cm[0,1] # false positives
            FN = cm[1,0] # false negatives

            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)

            PPV = TP / (TP + FP) if (TP + FP) != 0 else 'no positives values'
            NPV = TN / (TN + FN) if (TN + FN) != 0 else 'no negatives values'

            print(f'sensitivity : {sensitivity}, specificity : {specificity}, PPV : {PPV}, NPV : {NPV}')

        return accuracy

    def save_tree(self, path):
        tree_info = []
        tree_info.append(self.x_col_names)
        tree_info.append(self.min_participant)
        tree_info.append(self.max_depth)

        tree_info.append(self.__get_tree_str(self.root))

        tree_str = str(tree_info)

        f = open(path, "w")
        f.write(tree_str)
        f.close()

    def __get_tree_str(self, node):
        if node == None:
            return []

        if node.left == None and node.right == None:
            return [str(node.value)]

        node_info = []
        node_info.append(str(node.information_gain))
        node_info.append(str(node.threshold))
        node_info.append(node.column_name)
        node_info.append(str(node.column_index))
        node_info.append(node.operator)

        node_info.append(self.__get_tree_str(node.left))
        node_info.append(self.__get_tree_str(node.right))

        return node_info

    def import_tree(self, path):
        if not os.path.isfile(path):
            print("File doesn't exist :", path, file=sys.stderr)
            return

        with open(path) as f:
            tree_str = f.read()
        tree_list = ast.literal_eval(tree_str)

        self.x_col_names = tree_list[0]
        self.min_participant = tree_list[1]
        self.max_depth = tree_list[2]

        tree_list = tree_list[3]

        self.root = self.__build_tree_from_file(tree_list)


    def __build_tree_from_file(self, tree_list):
        if len(tree_list) == 0 :
            return None

        if len(tree_list) == 1:
            return Node(value=float(tree_list[0]))

        return Node(self.__build_tree_from_file(tree_list[5]),
                    self.__build_tree_from_file(tree_list[6]),
                    float(tree_list[0]), float(tree_list[1]),
                    tree_list[2], int(tree_list[3]), tree_list[4])
