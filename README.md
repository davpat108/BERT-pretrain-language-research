# Master's Thesis Project Documentation

---
## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Methodology](#methodology)
- [Data](#data)
- [Testing](#test)
- [Results](#results)

---

### Overview <a name = "overview"></a>
This repository contains the documented code for my master's thesis, which delves into the efficacy of BERT (Bidirectional Encoder Representations from Transformers) in multiple natural language processing tasks. The focus of this research includes Named Entity Recognition (NER), Part of Speech (POS) tagging, Sentiment Analysis, and Morphological Analysis across nine diverse languages: English, German, French, Hindi, Japanese, Korean, Turkish, Chinese, and Hungarian.

---

### Objectives <a name = "objectives"></a>
The primary objective of this project was to evaluate the performance of BERT in understanding and processing different linguistic features and structures inherent in a variety of languages. By utilizing the pre-trained models available through Hugging Face, I aimed to fine-tune these models to suit the specific requirements and nuances of each language task.


---
### Data <a name = "data"></a>

| Task               | Resource                                                                                                                  |
|--------------------|---------------------------------------------------------------------------------------------------------------------------|
| NER                | [WikiAnn](https://huggingface.co/datasets/wikiann)                                                                        |
| NER Hungarian      | [NerKor](https://huggingface.co/NYTK/named-entity-recognition-nerkor-hubert-hungarian)                                    |
| Sentiment Analysis | [The Multilingual Amazon Reviews Corpus](https://aclanthology.org/2020.emnlp-main.369/)                                   |
| Morphology         | [Morphology Probes](https://github.com/juditacs/morphology-probes)                                                        |
| PoS Tagging        | [Universal Dependencies](https://universaldependencies.org/)                                                              |
---
### Testing <a name = "test"></a>
Assuming python 3.9 and cu117 compatibility:

```bash
pip install -r requirements.txt
```
---
### Results <a name = "results"></a>
In a nutshell, for every use case, the monolingual model performs better as expected, with the exception of Named Entity Recognition, where the multilingual sometimes performs better. The likely reason for this is that named entities are often in a foreign language in the WikiAnn dataset. This means that the monolingual tokenizer has a hard time tokenizing the foreign words into meaningful tokens, while the multilingual one is still quite good at it.
