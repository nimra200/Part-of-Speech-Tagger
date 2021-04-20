# Part-of-Speech-Tagger

### Background 
Speech recognition is used in [Natural Language Processing (NLP)](https://en.wikipedia.org/wiki/Natural_language_processing), a rapidly growing subfield of AI. A [Part of Speech (POS) Tagger](https://en.wikipedia.org/wiki/Part-of-speech_tagging#:~:text=In%20corpus%20linguistics%2C%20part%2Dof,its%20definition%20and%20its%20context) is able to process sentences, and assign each word a tag (ex: noun, verb, adjective, etc.) based on the word's definition and context. The image below is an example of what the tagger does. 

![image](https://user-images.githubusercontent.com/56455442/115433368-79e3a480-a1d5-11eb-90b1-bb496bfbfa58.png)  
[_Image from https://towardsdatascience.com/part-of-speech-tagging-for-beginners-3a0754b2ebba_]

### About the POS Tagger
The tagger is able to make guesses about POS tags based on the training files. The tagger looks at previous examples of marked up text with POS tags and uses that information to mark up a brand new sentence. Of course, this tagger will not always be a 100 % accurate, and its ability to guess is only as good as the examples it receives. The more examples of marked up sentences the program sees, the better it becomes at guessing POS tags. 

### Using a Hidden Markov Model
This tagger uses Hidden Markov Models to improve tagging accuracy. Since the sentences are given to us in the testing file, we call each word an **observable** state. We have to figure out whether each word is a noun, verb, etc. so the tag for each word is the **hidden** state. The likelihood of one tag following another, say a noun following a verb, is called the **transition probability**. The probability of a word, given a POS tag, is the **emission probability**. So for instance, if we know that a word must be a verb, the likelihood of the word being "dog" is very low since "dog" is not a verb. 

### Challenges with Accuracy
Is it possible to create a tagger that is 100 % accurate? Figuring out whether a word is a noun or verb is intuitive for a human, but surprising difficult for a machine. The difficulty arises when the same word can have different tags, depending on the context of the sentence. For example, consider the sentence "Will plays with Spot" and " I spot Mary". In these sentences, "spot" occurs as both as a noun (the person named Spot) _and_ as a verb.  
The best way to deal with ambiguities is to show the tagger lots of examples, but a 100 % accuracy is not guaranteed. 
