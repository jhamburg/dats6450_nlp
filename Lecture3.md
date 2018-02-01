# Linear Regression Review 

A linear model makes a prediction by simply computing a weighted
sum of the input features, plus a constant called the bias term (also called the intercept
term):

```
y=θ 0 +θ 1 x 1 +θ 2 x 2 +⋯+θnxn
```
- ŷ is the predicted value.
- n is the number of features.
- xi is the ith feature value.
- θj is the jth model parameter (including the bias term θ 0 and the feature weights
    θ 1 , θ 2 , ⋯, θn).

This can be written much more concisely using a vectorized form:
```
y=hθ x =θT· x
```
- θ is the model’s parameter vector, containing the bias term θ 0 and the feature
    weights θ 1 to θn.
- θT is the transpose of θ (a row vector instead of a column vector).
- **x** is the instance’s feature vector, containing x 0 to xn, with x 0 always equal to 1.
- θT · **x** is the dot product of θT and **x**.
- hθ is the hypothesis function, using the model parameters θ.

Okay, that’s the Linear Regression model, so now how do we train it? 

The most common performance measure
of a regression model is the Root Mean Square Error (RMSE). Therefore, to train a Linear Regression model, you need to find the value of θ that minimizes the RMSE. In practice, it is simpler to minimize the Mean Square Error (MSE)
than the RMSE, and it leads to the same result (because the value that minimizes a
function also minimizes its square root).^1

# Gradient Descent

Gradient Descent is a very generic optimization algorithm capable of finding optimal
solutions to a wide range of problems. The general idea of Gradient Descent is to
tweak parameters iteratively in order to minimize a cost function.


###

Suppose you are lost in the mountains in a dense fog; you can only feel the slope of
the ground below your feet. A good strategy to get to the bottom of the valley quickly
is to go downhill in the direction of the steepest slope. This is exactly what Gradient
Descent does: it measures the local gradient of the error function with regards to the
parameter vector θ, and it goes in the direction of descending gradient. Once the gra‐
dient is zero, you have reached a minimum!

Concretely, you start by filling θ with random values (this is called random initializa‐
tion), and then you improve it gradually, taking one baby step at a time, each step
attempting to decrease the cost function (e.g., the MSE), until the algorithm converges
to a minimum (see Figure 4-3).

An important parameter in Gradient Descent is the size of the steps, determined by
the learning rate hyperparameter. If the learning rate is too small, then the algorithm
will have to go through many iterations to converge, which will take a long time.

On the other hand, if the learning rate is too high, you might jump across the valley
and end up on the other side, possibly even higher up than you were before. This
might make the algorithm diverge, with larger and larger values, failing to find a good
solution.

```
When using Gradient Descent, you should ensure that all features
have a similar scale (e.g., using Scikit-Learn’s StandardScaler
class), or else it will take much longer to converge.
```

This diagram also illustrates the fact that training a model means searching for a
combination of model parameters that minimizes a cost function (over the training
set). It is a search in the model’s parameter space: the more parameters a model has,
the more dimensions this space has, and the harder the search is: searching for a nee‐
dle in a 300-dimensional haystack is much trickier than in three dimensions. Fortu‐
nately, since the cost function is convex in the case of Linear Regression, the needle is
simply at the bottom of the bowl.

## Batch Gradient Descent

To implement Gradient Descent, you need to compute the gradient of the cost func‐
tion with regards to each model parameter θj. 

Notice that this formula involves calculations over the full training
set X , at each Gradient Descent step! This is why the algorithm is
called Batch Gradient Descent: it uses the whole batch of training
data at every step. As a result it is terribly slow on very large train‐
ing sets (but we will see much faster Gradient Descent algorithms
shortly). However, Gradient Descent scales well with the number of
features; training a Linear Regression model when there are hun‐
dreds of thousands of features is much faster using Gradient
Descent than using the Normal Equation.

```
see python notebook
```

On the left, the learning rate is too low: the algorithm will eventually reach the solu‐
tion, but it will take a long time. In the middle, the learning rate looks pretty good: in
just a few iterations, it has already converged to the solution. On the right, the learn‐
ing rate is too high: the algorithm diverges, jumping all over the place and actually
getting further and further away from the solution at every step.

To find a good learning rate, you can use grid search. However, you
may want to limit the number of iterations so that grid search can eliminate models
that take too long to converge.

You may wonder how to set the number of iterations. If it is too low, you will still be
far away from the optimal solution when the algorithm stops, but if it is too high, you
will waste time while the model parameters do not change anymore. A simple solu‐
tion is to set a very large number of iterations but to interrupt the algorithm when the
gradient vector becomes tiny—that is, when its norm becomes smaller than a tiny
number ε (called the tolerance)—because this happens when Gradient Descent has
(almost) reached the minimum.

## Stochastic Gradient Descent

The main problem with Batch Gradient Descent is the fact that it uses the whole
training set to compute the gradients at every step, which makes it very slow when
the training set is large. At the opposite extreme, Stochastic Gradient Descent just
picks a random instance in the training set at every step and computes the gradients
based only on that single instance. Obviously this makes the algorithm much faster
since it has very little data to manipulate at every iteration. It also makes it possible to
train on huge training sets, since only one instance needs to be in memory at each
iteration.

This code implements Stochastic Gradient Descent using a simple learning schedule:

```
n_epochs = 50
t0, t1 = 5, 50 # learning schedule hyperparameters
```
```
def learning_schedule(t):
return t0 / (t + t1)
```
```
theta = np.random.randn(2,1) # random initialization
```
```
for epoch in range(n_epochs):
for i in range(m):
random_index = np.random.randint(m)
xi = X_b[random_index:random_index+1]
yi = y[random_index:random_index+1]
gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
eta = learning_schedule(epoch * m + i)
theta = theta - eta * gradients
```
By convention we iterate by rounds of m iterations; *each round is called an epoch.*

## Mini-batch Gradient Descent

The last Gradient Descent algorithm we will look at is called Mini-batch Gradient
Descent. It is quite simple to understand once you know Batch and Stochastic Gradi‐
ent Descent: at each step, instead of computing the gradients based on the full train‐
ing set (as in Batch GD) or based on just one instance (as in Stochastic GD), Mini-

batch GD computes the gradients on small random sets of instances called mini-
batches. The main advantage of Mini-batch GD over Stochastic GD is that you can
get a performance boost from hardware optimization of matrix operations, especially
when using GPUs.

The algorithm’s progress in parameter space is less erratic than with SGD, especially
with fairly large mini-batches. As a result, Mini-batch GD will end up walking
around a bit closer to the minimum than SGD. But, on the other hand, it may be
harder for it to escape from local minima (in the case of problems that suffer from
local minima, unlike Linear Regression as we saw earlier). Figure 4-11 shows the
paths taken by the three Gradient Descent algorithms in parameter space during
training. They all end up near the minimum, but Batch GD’s path actually stops at the
minimum, while both Stochastic GD and Mini-batch GD continue to walk around.
However, don’t forget that Batch GD takes a lot of time to take each step, and Stochas‐
tic GD and Mini-batch GD would also reach the minimum if you used a good learn‐
ing schedule.

# Tensorflow!


 TensorFlow is a powerful open source software library for numerical computation,
articularly well suited and fine-tuned for large-scale Machine Learning. Its basic
rinciple is simple: you first define in Python a graph of computations to perform
for example, the one in Figure 9-1), and then TensorFlow takes that graph and runs
t efficiently using optimized C++ code.
ost importantly, it is possible to break up the graph into several chunks and run
hem in parallel across multiple CPUs or GPUs, TensorFlow
lso supports distributed computing, so you can train colossal neural networks on
umongous training sets in a reasonable amount of time by splitting the computa‐
ions across hundreds of servers. TensorFlow can train a network
ith millions of parameters on a training set composed of billions of instances with
illions of features each. This should come as no surprise, since TensorFlow was
eveloped by the Google Brain team and it powers many of Google’s large-scale serv‐
ces, such as Google Cloud Speech, Google Photos, and Google Search.
hen TensorFlow was open-sourced in November 2015, there were already many
opular open source libraries for Deep Learning, and to be fair
ost of TensorFlow’s features already existed in one library or another. Nevertheless,
ensorFlow’s clean design, scalability, flexibility,^1 and great documentation (not to
ention Google’s name) quickly boosted it to the top of the list. 


## Installation

Let’s get started! Assuming you installed Jupyter and Scikit-Learn, you can simply use pip to install TensorFlow. 
```
$ pip install tensorflow
```

## Creating Your First Graph and Running It in a Session

The following code creates the graph:

```
import tensorflow as tf
```
```
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2
```
That’s all there is to it! The most important thing to understand is that this code does
not actually perform any computation, even though it looks like it does (especially the
last line). It just creates a computation graph. In fact, even the variables are not ini‐
tialized yet. To evaluate this graph, you need to open a TensorFlow session and use it
to initialize the variables and evaluate f. A TensorFlow session takes care of placing
the operations onto devices such as CPUs and GPUs and running them, and it holds
all the variable values. The following code creates a session, initializes the variables,
and evaluates, and f then closes the session (which frees up resources):

```
>>> sess = tf.Session()
>>> sess.run(x.initializer)
>>> sess.run(y.initializer)
>>> result = sess.run(f)
>>> print (result)
42
>>> sess.close()
```
 Having to repeat sess.run() all the time is a bit cumbersome, but fortunately there is

 a better way:

```
with tf.Session() as sess:
x.initializer.run()
y.initializer.run()
result = f.eval()
```
 Inside the with block, the session is set as the default session. Calling x.initial izer.run() is equivalent to calling tf.get_default_session().run(x.initial izer), and similarly f.eval() is equivalent to calling tf.get_default_session().run(f). This makes the code easier to read. Moreover, the session is automatically closed at the end of the block. Instead of manually running the initializer for every single variable, you can use the global_variables_initializer() function. Note that it does not actually perform the initialization immediately, but rather creates a node in the graph that will initialize all variables when it is run:

```
init = tf.global_variables_initializer() # prepare an init node
```
```
with tf.Session() as sess:
init.run() # actually initialize all the variables
result = f.eval()
```
Inside Jupyter or within a Python shell you may prefer to create an InteractiveSes
sion. The only difference from a regular Session is that when an InteractiveSes
sion is created it automatically sets itself as the default session, so you don’t need a
with block (but you do need to close the session manually when you are done with
it):

```
>>> sess = tf.InteractiveSession()
>>> init.run()
>>> result = f.eval()
>>> print (result)
42
>>> sess.close()
```
 A TensorFlow program is typically split into two parts: the first part builds a compu‐ tation graph (this is called the construction phase), and the second part runs it (this is the execution phase). The construction phase typically builds a computation graph representing the ML model and the computations required to train it. The execution phase generally runs a loop that evaluates a training step repeatedly (for example, one step per mini-batch), gradually improving the model parameters. We will go through an example shortly.

## Managing Graphs

#### Any node you create is automatically added to the default graph:

```
>>> x1 = tf.Variable(1)
>>> x1.graph is tf.get_default_graph()
True
```
 In most cases this is fine, but sometimes you may want to manage multiple independ‐ ent graphs. You can do this by creating a new Graph and temporarily making it the default graph inside a with block, like so:

```
>>> graph = tf.Graph()
>>> with graph.as_default():
... x2 = tf.Variable(2)
...
>>> x2.graph is graph
True
>>> x2.graph is tf.get_default_graph()
False
```
In Jupyter (or in a Python shell), it is common to run the same
 commands more than once while you are experimenting. As a
 result, you may end up with a default graph containing many
 duplicate nodes. One solution is to restart the Jupyter kernel (or
 the Python shell), but a more convenient solution is to just reset the
 default graph by running tf.reset_default_graph().

## Lifecycle of a Node Value
 When you evaluate a node, TensorFlow automatically determines the set of nodes
 that it depends on and it evaluates these nodes first. For example, consider the following code:

```
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3
```
```
with tf.Session() as sess:
print (y.eval()) # 10
print (z.eval()) # 15
```
First, this code defines a very simple graph. Then it starts a session and runs the
 graph to evaluate y: TensorFlow automatically detects that y depends on x, which
 depends on w, so it first evaluates w, then x, then y, and returns the value of y. Finally,
 the code runs the graph to evaluate z. Once again, TensorFlow detects that it must
 first evaluate w and x. It is important to note that it will not reuse the result of the
 previous evaluation of w and x. In short, the preceding code evaluates w and x twice.
 All node values are dropped between graph runs, except variable values, which are
 maintained by the session across graph runs. A variable starts its life when its initializer is run,
 and it ends when the session is closed.
 If you want to evaluate y and z efficiently, without evaluating w and x twice as in the
 previous code, you must ask TensorFlow to evaluate both y and z in just one graph
 run, as shown in the following code:
```
with tf.Session() as sess:
y_val, z_val = sess.run([y, z])
print (y_val) # 10
print (z_val) # 15
```
 In single-process TensorFlow, multiple sessions do not share any
 state, even if they reuse the same graph (each session would have its
 own copy of every variable). In distributed TensorFlow (see Chap‐
 ter 12), variable state is stored on the servers, not in the sessions, so
 multiple sessions can share the same variables.

##Linear Regression with TensorFlow

 TensorFlow operations (also called ops for short) can take any number of inputs and produce any number of outputs. For example, the addition and multiplication ops each take two inputs and produce one output. Constants and variables take no input (they are called source ops). The inputs and outputs are multidimensional arrays, called tensors (hence the name “tensor flow”). Just like NumPy arrays, tensors have a type and a shape. In fact, in the Python API tensors are simply represented by NumPy ndarrays. They typically contain floats, but you can also use them to carry strings (arbitrary byte arrays). In the examples so far, the tensors just contained a single scalar value, but you can of course perform computations on arrays of any shape. For example, the following code manipulates 2D arrays to perform Linear Regression on the California housing data‐ set (introduced in Chapter 2). It starts by fetching the dataset; then it adds an extra bias input feature (x 0 = 1) to all training instances (it does so using NumPy so it runs immediately); then it creates two TensorFlow constant nodes, X and y, to hold this data and the targets,^4 and it uses some of the matrix operations provided by Tensor‐ Flow to define theta. These matrix functions—transpose(), matmul(), and matrix_inverse()—are self-explanatory, but as usual they do not perform any com‐ putations immediately; instead, they create nodes in the graph that will perform them when the graph is run. You may recognize that the definition of theta corresponds to the Normal Equation (θ = ( X T · X )–1 · X T · y ; see Chapter 4). Finally, the code creates a session and uses it to evaluate theta.

```
import numpy as np
from sklearn.datasets import fetch_california_housing
```
```
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
```
```
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
```
```
with tf.Session() as sess:
theta_value = theta.eval()
```
 The main benefit of this code versus computing the Normal Equation directly using
 NumPy is that TensorFlow will automatically run this on your GPU card if you have
 one (provided you installed TensorFlow with GPU support, of course; see Chapter 12
 for more details).


## Implementing Gradient Descent

Let’s try using Batch Gradient Descent (introduced in Chapter 4) instead of the Nor‐
mal Equation. First we will do this by manually computing the gradients, then we will
use TensorFlow’s autodiff feature to let TensorFlow compute the gradients automati‐
cally, and finally we will use a couple of TensorFlow’s out-of-the-box optimizers.
 When using Gradient Descent, remember that it is important to
 first normalize the input feature vectors, or else training may be
 much slower. You can do this using TensorFlow, NumPy, Scikit-
 Learn’s StandardScaler, or any other solution you prefer. The fol‐
 lowing code assumes that this normalization has already been
 done.

### Manually Computing the Gradients

 The following code should be fairly self-explanatory, except for a few new elements:

- The random_uniform() function creates a node in the graph that will generate a

 tensor containing random values, given its shape and value range, much like
NumPy’s rand() function.

- The assign() function creates a node that will assign a new value to a variable.

 In this case, it implements the Batch Gradient Descent step θ(next step) = θ –

η∇θMSE(θ).

- The main loop executes the training step over and over again (n_epochs times),

#### and every 100 iterations it prints out the current Mean Squared Error (mse). You

#### should see the MSE go down at every iteration.

```
n_epochs = 1000
learning_rate = 0.
```
```
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)
```
```
init = tf.global_variables_initializer()
```
```
with tf.Session() as sess:
sess.run(init)
```
```
for epoch in range(n_epochs):

if epoch % 100 == 0:
print ("Epoch", epoch, "MSE =", mse.eval())
sess.run(training_op)
```
```
best_theta = theta.eval()
```
### Using autodiff

 The preceding code works fine, but it requires mathematically deriving the gradients
 from the cost function (MSE). In the case of Linear Regression, it is reasonably easy,
 but if you had to do this with deep neural networks you would get quite a headache:
 it would be tedious and error-prone. You could use symbolic differentiation to auto‐
 matically find the equations for the partial derivatives for you, but the resulting code
 would not necessarily be very efficient.
 To understand why, consider the function f(x)= exp(exp(exp(x))). If you know calcu‐
 lus, you can figure out its derivative f′(x) = exp(x) × exp(exp(x)) × exp(exp(exp(x))).
 If you code f(x) and f′(x) separately and exactly as they appear, your code will not be
 as efficient as it could be. A more efficient solution would be to write a function that
 first computes exp(x), then exp(exp(x)), then exp(exp(exp(x))), and returns all three.
 This gives you f(x) directly (the third term), and if you need the derivative you can
 just multiply all three terms and you are done. With the naïve approach you would
 have had to call the exp function nine times to compute both f(x) and f′(x). With this
 approach you just need to call it three times.
 It gets worse when your function is defined by some arbitrary code. Can you find the
 equation (or the code) to compute the partial derivatives of the following function?
 Hint: don’t even try.

```
def my_func(a, b):
z = 0
for i in range(100):
z = a * np.cos(z + i) + z * np.sin(b - i)
return z
```
 Fortunately, TensorFlow’s autodiff feature comes to the rescue: it can automatically
 and efficiently compute the gradients for you. Simply replace the gradients = ...
 line in the Gradient Descent code in the previous section with the following line, and
 the code will continue to work just fine:

```
gradients = tf.gradients(mse, [theta])[0]
```
 The gradients() function takes an op (in this case mse) and a list of variables (in this
 case just theta), and it creates a list of ops (one per variable) to compute the gradi‐
 ents of the op with regards to each variable. So the gradients node will compute the
 gradient vector of the MSE with regards to theta.
Using an Optimizer
 So TensorFlow computes the gradients for you. But it gets even easier: it also provides
 a number of optimizers out of the box, including a Gradient Descent optimizer. You
 can simply replace the preceding gradients = ... and training_op = ... lines
 with the following code, and once again everything will just work fine:

```
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
```
If you want to use a different type of optimizer, you just need to change one line. For
example, you can use a momentum optimizer (which often converges much faster
than Gradient Descent; see Chapter 11) by defining the optimizer like this:

```
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
momentum=0.9)
```
## Feeding Data to the Training Algorithm
 Let’s try to modify the previous code to implement Mini-batch Gradient Descent. For
 this, we need a way to replace X and y at every iteration with the next mini-batch. The
 simplest way to do this is to use placeholder nodes. These nodes are special because
 they don’t actually perform any computation, they just output the data you tell them
 to output at runtime. They are typically used to pass the training data to TensorFlow
 during training. If you don’t specify a value at runtime for a placeholder, you get an
 exception.
 To create a placeholder node, you must call the placeholder() function and specify
 the output tensor’s data type. Optionally, you can also specify its shape, if you want to
 enforce it. If you specify None for a dimension, it means “any size.” For example, the
 following code creates a placeholder node A, and also a node B = A + 5. When we
 evaluate B, we pass a feed_dict to the eval() method that specifies the value of A.
 Note that A must have rank 2 (i.e., it must be two-dimensional) and there must be
# three columns (or else an exception is raised), but it can have any number of rows.

```
>>> A = tf.placeholder(tf.float32, shape=(None, 3))
>>> B = A + 5
>>> with tf.Session() as sess:
... B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
... B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
...
>>> print (B_val_1)
[[ 6. 7. 8.]]
>>> print (B_val_2)
[[ 9. 10. 11.]
[ 12. 13. 14.]]
```
 You can actually feed the output of any operations, not just place‐
 holders. In this case TensorFlow does not try to evaluate these
 operations; it uses the values you feed it.
To implement Mini-batch Gradient Descent, we only need to tweak the existing code
slightly. First change the definition of X and y in the construction phase to make them

placeholder nodes:

```
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
```
Then define the batch size and compute the total number of batches:

```
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
```
Finally, in the execution phase, fetch the mini-batches one by one, then provide the

value of X and y via the feed_dict parameter when evaluating a node that depends

on either of them.

# Tensorflow tutorial on logistic regression
(https://www.tensorflow.org/tutorials/wide#how_logistic_regression_works)
