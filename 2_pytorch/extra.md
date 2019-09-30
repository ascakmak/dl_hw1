Name: Yousef Emam
Email: yemam3@gatech.edu
Email ID: yemam3
Accuracy: 0.53

I tried various architectures starting from:
conv - relu - conv - relu - pool - fc - relu - fc - softmax

This was one of the architectures proposed in the README of the assignment and one I found online. As explained in class, the conv layers increase the depth while decreasing the height and width of the input. The fc layers serve to extract the high level features to be fed to the softmax for classification.

I found that Adam performs better than vanilla SGD so I used it with learning rate = 0.0005 (chosen through trial and error).

Moreover, I found that batch normalization also helps, so I included 2 bn layers and increased the depth of the architecture. The final architecture is:
2x{conv - relu - bn} - {conv - relu} - pool - 3x{fc - relu} - {fc - softmax}
My validation accuracy with 15 epochs was roughly 0.74. The discrepancy between the validation accuracy and the test accuracy indicates that the proposed net overfitted the data during training.
