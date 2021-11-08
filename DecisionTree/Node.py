"""
Node Class

Node used in the decision tree
"""

class Node:
    def __init__(self, left=None, right=None, information_gain=None,
                 threshold=None, column_name=None, column_index=None,
                 operator=None, value=None):
        """
        Attributes:
            left (Node) : #TODO
            right (Node) : #TODO

            information_gain (float) : The information gain with this decision.
            threshold (float) : The value to compare during the decision.
            column_name (str) : The index of the decision column in the original data.
            column_index (int) : Index of the decision column in the original data.
            operator (str) : '=' if the decision is made on a binary value else '<='

            value (float) : The most represented class by this node.
        """


        #children
        self.left = left
        self.right = right

        #decision node
        self.information_gain = information_gain
        self.threshold = threshold
        self.column_name = column_name
        self.column_index = column_index
        self.operator = operator

        #leaf node
        self.value = value
