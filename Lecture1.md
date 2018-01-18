## Natural Language Processing - Introduction - Spring 2018
##### Material cribbed from Johnathan Berant, Chris Manning, Michael Collins, and Yoav Artzi 

### What is NLP?

**Goal:** Develop methods for processing, analyzing and understanding the structure and meaning of language.

**Application:** Build systems that help people do stuff (with text):
- question answering
- virtual assistants
- translation
- tagging
- topic modeling
- search
- sentiment analysis
- ...

### Levels to NLP

*Phonology:* sounds that make up language

*Morphology:* internal structure of words

*Semantics:* meaning of language in the world

*Syntax:* structure of phrases, how words modify one another

*Discourse:* relations between clauses and sentences

### Why is NLP hard?
***Ambiguity***

“Finally, a computer that understands you like your mother”
(Ad , 1985)
- The computer understands you as well as your mother understands you.
- The computer understands that you like your mother.
- The computer understands you as well as it understands your mother.
- “Finally, a computer that understands your lie cured mother”
----
- Enraged Cow Injures Farmer with Ax
- Ban on Nude Dancing on Governor’s Desk
- Teacher Strikes Idle Kids
- Hospitals Are Sued by 7 Foot Doctors
- Iraqi Head Seeks Arms
- Stolen Painting Found by Tree
- Kids Make Nutritious Snacks
- Local HS Dropouts Cut in Half

***Variability***

- Dow ends up 255 points
- All major stock markets surged
- Dow climbs 255
- Dow gains 255 points
- Stock market hits a high record

***Sparsity***

![word sparsity](http://www.inf.ed.ac.uk/teaching/courses/inf1-cg/lectures/23/word_unigram.png)

Frequency of the top 50 words in Moby Dick

Words are infrequent, leading to difficulty in building strong features. Word sparsity makes tasks like translation and interpretation extremely difficult.

***Grounding***

Understanding of language is not solely elicited from looking at the presence of words - interpretation is a difficult task that often isn't represented with the text at hand

## Text processing
Regular expressions are a powerful language for matching text patterns. This page gives a basic introduction to regular expressions themselves sufficient for our Python exercises and shows how regular expressions work in Python. The Python "re" module provides regular expression support.

In Python a regular expression search is typically written as:

  match = re.search(pat, str)

The re.search() method takes a regular expression pattern and a string and searches for that pattern within the string. If the search is successful, search() returns a match object or None otherwise. Therefore, the search is usually immediately followed by an if-statement to test if the search succeeded, as shown in the following example which searches for the pattern 'word:' followed by a 3 letter word (details below):

str = 'an example word:cat!!'
match = re.search(r'word:\w\w\w', str)
# If-statement after search() tests if it succeeded
  if match:                      
    print 'found', match.group() ## 'found word:cat'
  else:
    print 'did not find'

The code match = re.search(pat, str) stores the search result in a variable named "match". Then the if-statement tests the match -- if true the search succeeded and match.group() is the matching text (e.g. 'word:cat'). Otherwise if the match is false (None to be more specific), then the search did not succeed, and there is no matching text.

The 'r' at the start of the pattern string designates a python "raw" string which passes through backslashes without change which is very handy for regular expressions (Java needs this feature badly!). I recommend that you always write pattern strings with the 'r' just as a habit.



