# CS-552: Modern Natural Language Processing


### Course Description
Natural language processing is ubiquitous in modern intelligent technologies, serving as a foundation for language translators, virtual assistants, search engines, and many more. In this course, we cover the foundations of modern methods for natural language processing, such as word embeddings, recurrent neural networks, transformers, and pretraining, and how they can be applied to important tasks in the field, such as machine translation and text classification. We also cover issues with these state-of-the-art approaches (such as robustness, interpretability, sensitivity), identify their failure modes in different NLP applications, and discuss analysis and mitigation techniques for these issues. 

#### Quick access links:
- [Platforms](#class)
- [Lecture Schedule](#lectures)
- [Exercise Schedule](#exercises)
- [Contact](#contact)

<a name="class"></a>
## Class

| Platform           				| Where & when                                              																								   |
|:------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Lectures           						| **Wednesdays: 11:15-13:00** [[STCC - Cloud C]()] & **Thursdays: 13:15-14:00** [[CE16]()]  |
| Exercises Session  						| **Thursdays: 14:15-16:00** [[CE11]()] 											     	   |
| Project Assistance <br />(not every week) | **Wednesdays: 13:15-14:00** [[STCC - Cloud C]()] 					   						   |
| QA Forum & Annoucements   			    | Ed Forum [[link](https://edstem.org/eu/courses/3119/discussion)]                             | 
| Grades             						| Moodle [[link](https://moodle.epfl.ch/course/view.php?id=17143)]     
| Lecture Recordings						| Mediaspace [[link](https://mediaspace.epfl.ch/channel/CS-552%2BModern%2Bnatural%2Blanguage%2Bprocessing/31346)]			|

All lectures will be given in person and live streamed on Zoom. The link to the Zoom is available on the Ed Forum (pinned post). Beware that, in the event of a technical failure during the lecture, continuing to accompany the lecture live via zoom might not be possible.

Recording of the lectures will be made available on Mediaspace. We will reuse some of last year's recordings and we may record a few new lectures in case of different lecture contents.


<a name="lectures"></a>
## Lecture Schedule


<table>
    <tr>
        <td>Week</td>
        <td>Date</td>
        <td>Topic</td>
        <td>Suggested Reading</td>
        <td>Instructor</td>
    </tr>
    <tr>
        <td><strong>Week 1</strong></td>
        <td>18 Feb <br />19 Feb</td>
        <td>Introduction &#124; Building a simple neural classifier [<a href="https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Lectures/Week_1">slides</a>]<br />Word embeddings [<a href="https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Lectures/Week_1">slides</a>]</td>
        <td><ul><li><a href="https://mitpress.mit.edu/9780262042840/introduction-to-natural-language-processing">Introduction to natural language processing, chapter 3.1 - 3.3 & chapter 14.5 - 14.6</a></li><li><a href="https://arxiv.org/abs/1301.3781">Efficient Estimation of Word Representations in Vector Space</a></li><li><a href="https://aclanthology.org/D14-1162">GloVe: Global Vectors for Word Representation</a></li><li><a href="https://aclanthology.org/Q17-1010">Enriching word vectors with subword information</a></li><li><a href="https://aclanthology.org/L18-1008">Advances in pre-training distributed word representations</a></li></ul></td>
        <td>Antoine Bosselut</td>
    </tr>
    <tr>
        <td><strong>Week 2</strong></td>
        <td>25 Feb <br />26 Feb</td>
        <td>Classical LMs &#124; Neural LMs: Fixed Context Models  [<a href="https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Lectures/Week_2">slides</a>] <br />Neural LMs: RNNs [<a href="https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Lectures/Week_2">slides</a>] </td>
        <td>Suggested reading: <ul><li><a href="https://mitpress.mit.edu/9780262042840/introduction-to-natural-language-processing">Introduction to natural language processing, chapter 6.1-6.4</a></li><li><a href="http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf">A Neural Probabilistic Language Model</a></li><li><a href="https://proceedings.mlr.press/v28/pascanu13.html">On the difficulty of training recurrent neural networks</a></li><li><a href="https://mitpress.mit.edu/9780262042840/introduction-to-natural-language-processing">Introduction to natural language processing, chapter 3.1 - 3.3 & chapter 18.3, 18.4</a></li></ul></td>
        <td>Antoine Bosselut</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>

</table>


<a name="exercises"></a>
## Exercise Schedule

| Week        | Release Date    |  Session Date Topic                                                                                    |  Instructor                                                         |
|:------------|:--------|:------------------------------------------------------------------------------------------|:-------------------------------------------------------------------:|
| **Week 1**  | 19 Feb  | 26 Feb | Intro + Setup   |  Madhur Panwar |
| **Week 2**  | 26 Feb  |  5 Mar |  LMs + Neural LMs: fixed-context models <br/>  Language and Sequence-to-sequence models | Badr AlKhamissi |
| **Week 3**  | 5 Mar  | 12 Mar | Attention + Transformers + Tokenization   |  Badr AlKhamissi |
| **Week 4**  | 12 Mar  | 19 Mar | Pretrained LLMs   |  Badr AlKhamissi |
| **Week 5**  | 19 Mar  | 26 Mar | Transfer Learning   |  Madhur Panwar |
| **Week 6**  | 26 Mar  | 2 Apr | Text Generation   |  Madhur Panwar |
| **Week 7**  | 1 Apr   | 2 Apr | In-context Learning + Post-training   |  TBD |



<a name="contact"></a>
## Contacts

Please email us at **nlp-cs552-spring2026-ta-team [at] groupes [dot] epfl [dot] ch** for any administrative questions, rather than emailing TAs individually. All course content questions need to be asked via [Ed](https://edstem.org/eu/courses/3119/discussion).

**Lecturer**: [Antoine Bosselut](https://people.epfl.ch/antoine.bosselut)

**Teaching assistants**: [Madhur Panwar](https://people.epfl.ch/madhur.panwar), [Badr AlKhamissi](https://people.epfl.ch/badr.alkhamissi), [Zeming (Eric) Chen](https://people.epfl.ch/zeming.chen?lang=en), [Sepideh Mamooler](https://people.epfl.ch/sepideh.mamooler), [Ayush Tarun](https://people.epfl.ch/ayush.tarun), [Lazar Milikic](https://people.epfl.ch/lazar.milikic)



[0e]:https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Exercises/Setup
[1e]:https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Exercises/
