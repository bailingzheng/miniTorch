# Language Models

It takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an autoregressive character-level language model.

Note the code is based on [makemore](https://github.com/karpathy/makemore).

## Usage

The included names.txt dataset, as an example, has the most common 32K names takes from [ssa.gov](https://www.ssa.gov/oact/babynames/) for the year 2018. 

Run the script from the project root directory:

```bash
$ PYTHONPATH=. python examples/language_models/main.py
```
