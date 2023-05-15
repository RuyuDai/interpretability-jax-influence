# Adaption for the seminar project

In Interpretability.pdf and the `seminar project` folder, there's an application of jax influence function on the Natural Language Inference(NLI) task.

In this report, we applied influence functions using the latest scaling-up techniques to identify label error in the NLI task. As a result, our approach managed to identify 80% of the label error. Notice that the result,in our case, is highly susceptible to the value of `weight_decay`.


# Jax Influence

Scalable implementation of Influence Functions in JaX.

Implementation of the algorithms in
[Scaling Up Influence Functions (AAAI 2022)](https://arxiv.org/abs/2112.03052)
for efficient calculation of Influence Functions.

## Installation

### manual installation

Download the repo and set up a Python environment:

```sh
git clone https://github.com/google-research/jax-influence ~/jax-influence


cd ~/jax-influence
conda env create -f environment.yml
conda activate jax-influence
```

### pip installation

```sh
pip install jax-influence
```

The pip installation will install all necessary prerequisite packages, however
you might want to install the most appropriate version of `jax` and `jaxlib`
in case you use GPUs/TPUs.

## Documentation

An end-to-end example of using the library can be found in
`examples/colab/mnist_tutorial.ipynb`. We plan to add more examples in the
future.

## Disclaimer

This is not an official Google product.

Jax Influence is a research project, and under active development by a
small team; we'd love your suggestions and feedback - drop us a
line in the [issues](https://github.com/google-research/jax-influence).

