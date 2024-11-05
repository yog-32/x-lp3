# Huffman Coding in Python

string = 'BCAADDDCCACACAC'  # The input string for which Huffman coding is to be generated

# Creating tree nodes
class NodeTree(object):  # Definition of the node structure for Huffman Tree

    def __init__(self, left=None, right=None):
        self.left = left  # Left child
        self.right = right  # Right child

    def children(self):  # Return the children of the node
        return (self.left, self.right)

    def nodes(self):  # Return the nodes (same as children method)
        return (self.left, self.right)

    def __str__(self):  # String representation of the node
        return '%s_%s' % (self.left, self.right)

# Main function implementing Huffman coding
def huffman_code_tree(node, left=True, binString=''):
    if type(node) is str:  # If the node is a string, return its binary string
        return {node: binString}
    (l, r) = node.children()  # Get left and right children
    d = dict()  # Initialize dictionary to hold the codes
    d.update(huffman_code_tree(l, True, binString + '0'))  # Recursive call for left child with '0' added to binary string
    d.update(huffman_code_tree(r, False, binString + '1'))  # Recursive call for right child with '1' added to binary string
    return d

# Calculating frequency of characters in the string
freq = {}
for c in string:
    if c in freq:  # If character is already in frequency dictionary, increment its count
        freq[c] += 1
    else:  # Otherwise, initialize its count to 1
        freq[c] = 1

# Greedy step: sorting frequencies to combine smallest ones first
freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)  # Sort the frequency dictionary by frequency in descending order

nodes = freq  # Initialize nodes with the frequency items

while len(nodes) > 1:  # Iterate until there is only one node left (the root of the Huffman Tree)
    # Greedy step: taking two nodes with the smallest frequencies
    (key1, c1) = nodes[-1]  # Take the two nodes with the smallest frequency
    (key2, c2) = nodes[-2]
    nodes = nodes[:-2]  # Remove these two nodes from the list
    node = NodeTree(key1, key2)  # Create a new node with these two nodes as children
    nodes.append((node, c1 + c2))  # Add the new node to the list with combined frequency

    # Greedy step: sorting nodes by frequency again
    nodes = sorted(nodes, key=lambda x: x[1], reverse=True)  # Sort the nodes by frequency again

# Generate the Huffman codes for the characters
huffmanCode = huffman_code_tree(nodes[0][0])  

print(' Char | Frequency | Huffman code ')
print('---------------------------------')
for (char, frequency) in freq:  # Print the Huffman codes for each character along with their frequencies
    print(' %-4r | %9d | %12s' % (char, frequency, huffmanCode[char]))