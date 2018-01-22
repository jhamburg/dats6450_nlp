# Natural Language Processing - Introduction - Spring 2018
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

# Text processing
## Regular expressions
##### Material cribbed from Google's intro to Python

Regular expressions are a powerful language for matching text patterns. The Python "re" module provides regular expression support.

In Python a regular expression search is typically written as:
```
  match = re.search(pattern, str)
```

The re.search() method takes a regular expression pattern and a string and searches for that pattern within the string. If the search is successful, search() returns a match object or None otherwise. 

Therefore, the search is usually immediately followed by an if-statement to test if the search succeeded, as shown in the following example which searches for the pattern 'word:' followed by a 3 letter word (details below):
```
str = 'an example word:cat!!'
match = re.search(r'word:\w\w\w', str)
# If-statement after search() tests if it succeeded
  if match:                      
    print 'found', match.group() ## 'found word:cat'
  else:
    print 'did not find'
```

The code match = re.search(pat, str) stores the search result in a variable named "match". 

Then the if-statement tests the match -- if true the search succeeded and match.group() is the matching text (e.g. 'word:cat'). Otherwise if the match is false (None to be more specific), then the search did not succeed, and there is no matching text.

The 'r' at the start of the pattern string designates a python "raw" string which passes through backslashes without change which is very handy for regular expressions.

### Basic Patterns

The power of regular expressions is that they can specify patterns, not just fixed characters. Here are the most basic patterns which match single chars:

   a, X, 9 < -- ordinary characters just match themselves exactly. The meta-characters which do not match themselves because they have special meanings are: . ^ $ * + ? { [ ] \ | ( )
   
   . (a period) -- matches any single character except newline '\n'
    
   \w -- (lowercase w) matches a "word" character: a letter or digit or underbar [a-zA-Z0-9_]. Note that although "word" is the mnemonic for this, it only matches a single word char, not a whole word. \W (upper case W) matches any non-word character.
    
   \b -- boundary between word and non-word
    
   \s -- (lowercase s) matches a single whitespace character -- space, newline, return, tab, form [ \n\r\t\f]. \S (upper case S) matches any non-whitespace character.
    
   \t, \n, \r -- tab, newline, return
    
   \d -- decimal digit [0-9] (some older regex utilities do not support but \d, but they all support \w and \s)
    
   ^ = start, $ = end -- match the start or end of the string
    
   \ -- inhibit the "specialness" of a character. So, for example, use \. to match a period or \\ to match a slash. If you are unsure if a character has special meaning, such as '@', you can put a slash in front of it, \@, to make sure it is treated just as a character. 

*Regular expression python notebook*

**Assignment:** Baby names! https://developers.google.com/edu/python/exercises/baby-names
Starter code is provided in the zip file on the website.

## Lemmatization and stemming

### Stemming:

In linguistic morphology and information retrieval, stemming is the process for reducing inflected (or sometimes derived) words to their stem, base or root form—generally a written word form. The stem need not be identical to the morphological root of the word; it is usually sufficient that related words map to the same stem, even if this stem is not in itself a valid root. Algorithms for stemming have been studied in computer science since the 1960s. Many search engines treat words with the same stem as synonyms as a kind of query expansion, a process called conflation.

### Lemmatization

Lemmatisation (or lemmatization) in linguistics, is the process of grouping together the different inflected forms of a word so they can be analysed as a single item.

In computational linguistics, lemmatisation is the algorithmic process of determining the lemma for a given word. Since the process may involve complex tasks such as understanding context and determining the part of speech of a word in a sentence (requiring, for example, knowledge of the grammar of a language) it can be a hard task to implement a lemmatiser for a new language.

### Lemmatisation vs stemming

The difference is that a stemmer operates on a single word without knowledge of the context, and therefore cannot discriminate between words which have different meanings depending on part of speech. However, stemmers are typically easier to implement and run faster, and the reduced accuracy may not matter for some applications.

For instance:

- The word "better" has "good" as its lemma. This link is missed by stemming, as it requires a dictionary look-up.

- The word "walk" is the base form for word "walking", and hence this is matched in both stemming and lemmatisation.

- The word "meeting" can be either the base form of a noun or a form of a verb ("to meet") depending on the context, e.g., "in our last meeting" or "We are meeting again tomorrow". Unlike stemming, lemmatisation can in principle select the appropriate lemma depending on the context.



## Starting to turn words into numbers - introducing tf-idf
See the accompanying python notebook!
