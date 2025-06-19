# Document Level Paraphraser WebApp
Paraphrase generator


## INTRODUCTION
The goal of our project was to develop a paraphrasing tool web app that can generate paraphrases at the document level using a cutting-edge model. The purpose of this study is to describe the approach we took to achieve this goal and compare the performance of our chosen model to that of two other well-known models, Pegasus and Parrot.

## REQUIREMENTS

streamlit

Torch 

Torchvision

transformers

parrot

nltk

## RESULTS
In our project, we created a paraphrasing tool that uses a cutting-edge paraphrasing model to produce document-level paraphrases. We employ an approach called "Towards Document-Level Paraphrase Generation with Sentence Rewriting and Reordering," which seeks to identify chances for paraphrasing at the sentence level before rewriting and rearranging the sentences to produce a paraphrased document.
Pegasus and Parrot, the other two models we studied, are statement-level paraphrasers. The cutting-edge text summarising technique Pegasus can be used to condense the input text when paraphrasing. On the other hand, the Parrot model, which was just recently put forth, employs a self-supervised learning methodology to construct paraphrases at the sentence level.

We saw several <unk> in the result from Model 1 which is due to the lack of vocabulary available for model predictions.
Before our model could identify the possibility of paraphrasing at the sentence level, we preprocessed the training data by segmenting the input documents into sentences. The paraphrased papers were then created using a combination of sentence rewriting and sentence reordering. A large corpus of documents was used to train the model, and same corpus was also used to make adjustments.

For Model 3 and Model 2. Each sentence from the source text was manually entered into the model to produce paraphrases, and the output was a paraphrased version of each sentence. The text was created by combining the paraphrased sentences.
We used the human evaluation standards to assess the performance of the three models. The quality of the paraphrases produced by each model was compared 
And we found that the best paraphrase was produced in Model 1 as it was able to generate new sentences and change the number of sentences. The second one was model 3 which was crisp and retained all essential keywords and the worst of all three was the Pegasus model which is a summarizer and that was what it did. It summarized the paragraph and a lot of keywords were lost.


## CONCLUSION
In conclusion, we created a tool for document-level paraphrasing and evaluated it against statement-level paraphrasers. We used a comparable training corpus and ensured that the input papers were of a similar length in order to conduct a fair comparison. We compared the three models' performances using the BLEU score and human evaluation standards.
  
  
