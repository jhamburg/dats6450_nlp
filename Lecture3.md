# Building up tensorflow

Quick review of logistic regression: https://www.tensorflow.org/tutorials/wide#how_logistic_regression_works

Tensorflow API tutorial: https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf

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

# In class assignment: Tensorflow tutorial on logistic regression
(https://www.tensorflow.org/tutorials/wide#how_logistic_regression_works)
