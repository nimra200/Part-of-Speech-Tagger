# Part-of-Speech-Tagger
A [Part of Speech (POS) Tagger](https://en.wikipedia.org/wiki/Part-of-speech_tagging#:~:text=In%20corpus%20linguistics%2C%20part%2Dof,its%20definition%20and%20its%20context) is able to process sentences, and assign each word a tag (ex: noun, verb, adjective, etc.) based on the word's definition and context. The image below is an example of what the tagger does. 

![image](https://user-images.githubusercontent.com/56455442/115433368-79e3a480-a1d5-11eb-90b1-bb496bfbfa58.png) 
Image from https://towardsdatascience.com/part-of-speech-tagging-for-beginners-3a0754b2ebba

The tagger is able to make guesses about POS tags based on the training files. In other words, it looks at previous examples of text marked up with POS tags and uses information learned from those examples to to mark up a brand new sentence. Of course, this tagger will not always be a 100 % accurate, because its ability to guess is only as good as the examples it receives. I.e, the more examples of marked up sentences the program sees, the better it becomes at guessing POS tags. 

This area of computer science is called [Natural Language Processing (NLP)](https://en.wikipedia.org/wiki/Natural_language_processing), and POS taggers are used in speech recognition. 
