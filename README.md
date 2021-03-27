# CS539-HW-6
### Learning Goal
Learn how to do inference in a higher order probabilistic programming language in a way that is very near to being compatible with tight-bounds model-learning via gradient-based evidence lower-bounding. In particular you are going to learn how to do sequential Monte-Carlo-based inference in the probabilistic programming context. You should note in doing this homework the increased efficiency of inference in the HMM model in particular, the model in which partial "reward" is accumulated.

## Setup
**Note:** This code base was developed on Python3.7

Clone Daphne directly into this repo:
```bash
git clone git@github.com:plai-group/daphne.git
```
(To use Daphne you will need to have both a JVM installed and Leiningen installed)

```bash
pip3 install -r requirements.txt
```

## Usage
Change the daphne path in `src/smc.py` and `src/daphne.py` run:
```bash
cd src
python3 smc.py
```
