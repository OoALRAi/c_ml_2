## to implement:

### nn lib
* sigmoid (done)
* softmax (done)
* backward (done)
* test softamx (done)
* mnist is implemented
* solve a bug cause the one hot encoding to have multiple ones at once. (done)
* mnist problem solved with 3 layers deep network where first 2 layers have relu as act function and last one softmax to compute probabilities. loss function is cross entropy loss.
* derivative of softmax in combination with cross entropy loss (done)
* test network  (done)
* free memory to prevent memory leaks in forward and backward. (done)
* add more activation functions (done enough)
* compute statistics to present in github page (done)
* precision and recall for each class (done)
-----------

* 2d-cross-correlation product (2d conv)
* implement it in matrix.c
* loop over input image to do convolution in nn.c
* (2, 10, 10), w=(5, 2, 3, 3) -> (5, 10, 10)



### Very important Observation:
* deeper netowrk needs smaller learning rate in order to converge correctly!
* 3 layers to solve mnist classification problem with no memory leaks consumes only 3.6MB