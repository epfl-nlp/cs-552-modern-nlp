# CS-552: Modern Natural Language Processing


### Course Description
Natural language processing is ubiquitous in modern intelligent technologies, serving as a foundation for language translators, virtual assistants, search engines, and many more. In this course, we cover the foundations of modern methods for natural language processing, such as word embeddings, recurrent neural networks, transformers, and pretraining, and how they can be applied to important tasks in the field, such as machine translation and text classification. We also cover issues with these state-of-the-art approaches (such as robustness, interpretability, sensitivity), identify their failure modes in different NLP applications, and discuss analysis and mitigation techniques for these issues. 

#### Quick access links:
- [Platforms](#class)
- [Lecture Schedule](#lectures)
- [Exercise Schedule](#exercises)
- [Grading](#evaluation)
- [Contact](#contact)

<a name="class"></a>
## Class

| Platform           				| Where & when                                              																								   |
|:------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Lectures           						| **Wednesdays: 11:15-13:00** [[STCC - Cloud C](https://plan.epfl.ch/?room=%3DSTCC%20-%20Cloud%20C&dim_floor=0&lang=en&dim_lang=en&tree_groups=centres_nevralgiques%2Cmobilite_acces_grp%2Censeignement%2Ccommerces_et_services&tree_group_layers_centres_nevralgiques=information_epfl%2Cguichet_etudiants&tree_group_layers_mobilite_acces_grp=metro&tree_group_layers_enseignement=&tree_group_layers_commerces_et_services=&baselayer_ref=grp_backgrounds&map_x=2532938&map_y=1152803&map_zoom=11)] & **Thursdays: 13:15-14:00** [[CE16](https://plan.epfl.ch/?room=%3DCE%201%206&dim_floor=1&lang=en&dim_lang=en&tree_groups=centres_nevralgiques%2Cmobilite_acces_grp%2Censeignement%2Ccommerces_et_services&tree_group_layers_centres_nevralgiques=information_epfl%2Cguichet_etudiants&tree_group_layers_mobilite_acces_grp=metro&tree_group_layers_enseignement=&tree_group_layers_commerces_et_services=&baselayer_ref=grp_backgrounds&map_x=2533400&map_y=1152502&map_zoom=13)]  |
| Exercises Session  						| **Thursdays: 14:15-16:00** [[CE11](https://plan.epfl.ch/?room=%3DCE%201%201&dim_floor=1&lang=en&dim_lang=en&tree_groups=centres_nevralgiques%2Cmobilite_acces_grp%2Censeignement%2Ccommerces_et_services&tree_group_layers_centres_nevralgiques=information_epfl%2Cguichet_etudiants&tree_group_layers_mobilite_acces_grp=metro&tree_group_layers_enseignement=&tree_group_layers_commerces_et_services=&baselayer_ref=grp_backgrounds&map_x=2533297&map_y=1152521&map_zoom=13)] 																				   |
| Project Assistance <br />(not every week) | **Wednesdays: 13:15-14:00** [[STCC - Cloud C](https://plan.epfl.ch/?room=%3DSTCC%20-%20Cloud%20C&dim_floor=0&lang=en&dim_lang=en&tree_groups=centres_nevralgiques%2Cmobilite_acces_grp%2Censeignement%2Ccommerces_et_services&tree_group_layers_centres_nevralgiques=information_epfl%2Cguichet_etudiants&tree_group_layers_mobilite_acces_grp=metro&tree_group_layers_enseignement=&tree_group_layers_commerces_et_services=&baselayer_ref=grp_backgrounds&map_x=2532938&map_y=1152803&map_zoom=11)] 													   						   |
| QA Forum & Annoucements   			    | Ed Forum [[link](https://edstem.org/eu/courses/2071/discussion)]                                       													   | 
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
        <td>19 Feb <br />20 Feb</td>
        <td>Introduction &#124; Building a simple neural classifier [<a href="https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Lectures/Week%201">slides</a>, <a href="https://mediaspace.epfl.ch/media/%5B2025%5D+1-2A+Introduction+%26+Simple+neural+classifier/0_ztiealbw/31346">video</a>]<br />Word embeddings [<a href="https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Lectures/Week%201">slides</a>, <a href="https://mediaspace.epfl.ch/media/%5B2025%5D+3A+Word+Embeddings/0_tqc5u2u8/31346">video</a>]</td>
        <td>Suggested reading: <ul><li><a href="https://mitpress.mit.edu/9780262042840/introduction-to-natural-language-processing">Introduction to natural language processing, chapter 3.1 - 3.3 & chapter 14.5 - 14.6</a></li><li><a href="https://arxiv.org/abs/1301.3781">Efficient Estimation of Word Representations in Vector Space</a></li><li><a href="https://aclanthology.org/D14-1162">GloVe: Global Vectors for Word Representation</a></li><li><a href="https://aclanthology.org/Q17-1010">Enriching word vectors with subword information</a></li><li><a href="https://aclanthology.org/L18-1008">Advances in pre-training distributed word representations</a></li></ul></td>
        <td>Antoine Bosselut</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><strong>Week 2</strong></td>
        <td>26 Feb <br />27 Feb</td>
        <td>Classical LMs &#124; Neural LMs: Fixed Context Models  [<a href="https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Lectures/Week%202">slides</a>, <a href="https://mediaspace.epfl.ch/media/%5B2024%5D+-+04.+Classical+Language+Models/0_ufdusu0v/31346">video p1</a>, <a href="https://mediaspace.epfl.ch/media/%5B2024%5D+-+05.+Language+ModelsA+Fixed-context+Neural+Models/0_cq4qpuii/31346">video p2</a> (Last year recordings)] <br />Neural LMs: RNNs [<a href="https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Lectures/Week%202">slides</a>, <a href="https://mediaspace.epfl.ch/media/%5B2025%5D+6A+Recurrent+Neural+Networks/0_em4f6i76/31346"> video</a>] </td>
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
    <tr>
        <td><strong>Week 3</strong></td>
        <td>5 Mar <br />6 Mar</td>
        <td>Sequence-to-sequence Models [<a href="https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Lectures/Week%203">slides</a>, <a href="https://mediaspace.epfl.ch/media/%5B2025%5D+7-8A+Sequence+to+Sequence+Models+%26+Transformers/0_71gviryj/31346">video</a>] &#124; Transformers <br />Pretraining: GPT [<a href="https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Lectures/Week%203">slides</a>, <a href="https://mediaspace.epfl.ch/media/%5B2025%5D+9A+Transformers+Pretraining/0_zb407gx4/31346">video</a>]</td>
        <td>Suggested reading: <ul><li><a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a></li><li><a href="https://aclanthology.org/W18-2509">The Annotated Transformer</a></li><li><a href="https://jalammar.github.io/illustrated-transformer/">The illustrated transformer</a></li><li>GPT: <a href="https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf">Improving language understanding by generative pre-training</a></li><li>GPT2: <a href="https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf">Language Models are Unsupervised Multitask Learners</a></li></ul></td>
        <td>Antoine Bosselut</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><strong>Week 4</strong></td>
        <td>12 Mar <br />13 Mar</td>
        <td><strong>[Online only]</strong> Pretraining: BERT &#124; Transfer Learning <br /> <strong>[Recorded]</strong> Pretraining: T5 </td>
        <td>Suggested reading: <ul><li>Elmo: <a href="https://aclanthology.org/N18-1202">Deep Contextualized Word Representations</a></li><li>BERT: <a href="https://aclanthology.org/N19-1423">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a></li><li>RoBERTa: <a href="https://arxiv.org/abs/1907.11692">RoBERTa: A Robustly Optimized BERT Pretraining Approach</a></li><li><a href="https://www.ruder.io/state-of-transfer-learning-in-nlp/">Transfer Learning in Natural Language Processing</a></li><li>T5: <a href="https://arxiv.org/abs/1910.10683">Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</a></li><li>BART: <a href="https://arxiv.org/abs/1910.13461">BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension</a></li></ul></td>
        <td>Antoine Bosselut</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><strong>Week 5</strong></td>
        <td>19 Mar <br />20 Mar</td>
        <td>Transfer Learning: Dataset Biases <br />Generation: Task </td>
        <td>Suggested reading: -</td>
        <td>Antoine Bosselut</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><strong>Week 6</strong></td>
        <td>26 Mar <br />27 Mar</td>
        <td>Text Generation: Decoding  & Training  <br />Text Generation: Evaluation </td>
        <td>Suggested reading: <ul><li>Decoding: <a href="https://arxiv.org/abs/1503.03535">On Using Monolingual Corpora in Neural Machine Translation</a></li><li>Decoding: <a href="https://arxiv.org/abs/1609.08144">Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation</a></li><li>Decoding: <a href="https://arxiv.org/abs/1604.01729">Improving LSTM-based Video Description with Linguistic Knowledge Mined from Text</a></li><li>Decoding: <a href="https://arxiv.org/abs/1510.03055">A Diversity-Promoting Objective Function for Neural Conversation Models</a></li><li>Decoding: <a href="https://arxiv.org/abs/1705.04304">A Deep Reinforced Model for Abstractive Summarization</a></li><li>Decoding: <a href="https://arxiv.org/abs/1803.10357">Deep Communicating Agents for Abstractive Summarization</a></li><li>Decoding: <a href="https://arxiv.org/abs/1805.06087">Learning to Write with Cooperative Discriminators</a></li><li>Decoding: <a href="https://arxiv.org/abs/1805.04833">Hierarchical Neural Story Generation</a></li><li>Decoding: <a href="https://arxiv.org/abs/1907.01272">Discourse Understanding and Factual Consistency in Abstractive Summarization</a></li><li>Decoding: <a href="https://arxiv.org/abs/1912.02164">Plug and Play Language Models: A Simple Approach to Controlled Text Generation</a></li><li>Decoding: <a href="https://arxiv.org/abs/1904.09751">The Curious Case of Neural Text Degeneration</a></li><li>Decoding: <a href="https://arxiv.org/abs/1911.00172">Generalization through Memorization: Nearest Neighbor Language Models</a></li><li>Training: <a href="https://arxiv.org/abs/1506.03099">Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks</a></li><li>Training: <a href="https://arxiv.org/abs/1511.06732">Sequence Level Training with Recurrent Neural Networks</a></li><li>Training: <a href="https://arxiv.org/abs/1609.08144"> Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation</a></li><li>Training: <a href="https://arxiv.org/abs/1704.03899">Deep Reinforcement Learning-based Image Captioning with Embedding Reward</a></li><li>Training: <a href="https://arxiv.org/abs/1612.00563">Self-critical Sequence Training for Image Captioning</a></li><li>Training: <a href="https://arxiv.org/abs/1612.00370">Improved Image Captioning via Policy Gradient Optimization of SPIDEr</a></li><li>Training: <a href="https://arxiv.org/abs/1703.10931">Sentence Simplification with Deep Reinforcement Learning</a></li><li>Training: <a href="https://arxiv.org/abs/1705.04304">A Deep Reinforced Model for Abstractive Summarization</a></li><li>Training: <a href="https://arxiv.org/abs/1803.10357">Deep Communicating Agents for Abstractive Summarization</a></li><li>Training: <a href="https://arxiv.org/abs/1805.03766">Discourse-Aware Neural Rewards for Coherent Text Generation</a></li><li>Training: <a href="https://arxiv.org/abs/1805.03162">Polite Dialogue Generation Without Parallel Data</a></li><li>Training: <a href="https://arxiv.org/abs/1711.00279">Paraphrase Generation with Deep Reinforcement Learning</a></li><li>Training: <a href="https://arxiv.org/abs/1904.09751">The Curious Case of Neural Text Degeneration</a></li></ul></td>
        <td>Antoine Bosselut</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><strong>Week 7</strong></td>
        <td>2 Apr <br />3 Apr</td>
        <td>LLMs: In-context Learning & Instruction Tuning <br/> <strong>No Class</strong></td>
        <td></td>
        <td>Antoine Bosselut</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
        <tr>
        <td><strong>Week 8</strong></td>
        <td>9 Apr <br />10 Apr</td>
        <td><strong>Midterm</strong> <br/> Project Description</td>
        <td></td>
        <td>Antoine Bosselut</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
        <tr>
        <td><strong>Week 9</strong></td>
        <td>16 Apr May <br />17 Apr</td>
        <td>Retrieval-Augmented LLMs &#124; Agents <br /><strong>No class</strong> (Work on your project)</td>
        <td>Suggested reading: <ul><li><a href="https://arxiv.org/abs/1606.05250">Squad: 100,000+ questions for machine comprehension of text</a></li><li><a href="https://aclanthology.org/Q19-1026/">Natural questions: a benchmark for question answering research</a></li><li><a href="https://arxiv.org/abs/2004.04906">Dense passage retrieval for open-domain question answering</a></li><li><a href="https://proceedings.mlr.press/v119/guu20a.html">Retrieval augmented language model pre-training</a></li><li><a href="https://arxiv.org/abs/2005.11401">Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</a></li><li><a href="https://arxiv.org/abs/2302.04761">Toolformer: Language models can teach themselves to use tools</a></li><li><a href="https://arxiv.org/abs/2210.03629">React: Synergizing reasoning and acting in language models</a></li><li><a href="https://arxiv.org/abs/2112.04426">Improving language models by retrieving from trillions of tokens</a></li><li><a href="https://arxiv.org/abs/2302.07842">Augmented language models: a survey</a></li></ul></td>
        <td>Antoine Bosselut</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td><strong>EASTER BREAK</strong></td>
        <td></td>
        <td></td>
    </tr>
        <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><strong>Week 10</strong></td>
        <td>30 Apr <br />1 May</td>
        <td>Multimodality <br /><strong>No class</strong> (Work on your project)</td>
        <td>Suggested reading: -</td>
        <td>Syrielle Montariol <br />Antoine Bosselut</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
        <tr>
        <td><strong>Week 11</strong></td>
        <td>7 May <br />8 May</td>
        <td>Scaling laws &#124; LLM Efficiency <br /><strong>Guest Lecture:</strong> Rémi Delacourt (Mistral)</td>
        <td>Suggested reading: <ul><li><a href="https://arxiv.org/abs/2001.08361">Scaling laws for neural language models</a></li><li><a href="https://arxiv.org/abs/2203.15556">Training compute-optimal large language models</a></li></ul></td>
        <td>Antoine Bosselut</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><strong>Week 12</strong></td>
        <td>14 May <br />15 May</td>
        <td>Ethics: Bias & Fairness  &#124; Toxicity & Disinformation <br /><strong>No class</strong> (Work on your project)</td>
        <td>Suggested reading: <ul><li><a href="https://faculty.washington.edu/ebender/2017_575/#phil">Ethics in NLP</a></li><li><a href="https://www.ohchr.org/sites/default/files/documents/issues/business/b-tech/overview-human-rights-and-responsible-AI-company-practice.pdf">United Nations recommendations/overview on responsible AI practice</a></li></ul></td>
        <td>Anna Sotnikova</td>
    </tr>
        <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><strong>Week 13</strong></td>
        <td>21 May <br />22 May</td>
        <td>Interpretability <br /><strong>No class</strong> (Work on your project)</td>
        <td>Suggested reading: -</td>
        <td>Gail Weiss</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><strong>Week 14</strong></td>
        <td>28 May <br />29 May</td>
        <td>Tokenization &#124; Multilingual LMs + Looking forward <br /><strong>Holiday</strong></td>
        <td>Suggested reading: <ul><li><a href="https://arxiv.org/abs/2112.10508">Between words and characters: A brief history of open-vocabulary modeling and tokenization in NLP</a></li><li><a href="https://arxiv.org/abs/2105.13626">Byt5: Towards a token-free future with pre-trained byte-to-byte models</a></li><li><a href="https://arxiv.org/abs/1508.07909">Neural machine translation of rare words with subword units</a></li><li><a href="https://arxiv.org/abs/1911.02116">Unsupervised cross-lingual representation learning at scale</a></li><li><a href="https://arxiv.org/abs/1911.01464">Emerging cross-lingual structure in pretrained language models</a></li><li><a href="https://arxiv.org/abs/2005.00052">Mad-x: An adapter-based framework for multi-task cross-lingual transfer</a></li><li><a href="https://arxiv.org/abs/2110.07560">Composable sparse fine-tuning for cross-lingual transfer</a></li><li><a href="https://www.ruder.io/state-of-multilingual-ai/">The State of Multilingual AI</a></li></ul></td>
        <td>Antoine Bosselut</td>
    </tr>

</table>


<a name="exercises"></a>
## Exercise Schedule

| Week        | Date    |  Topic                                                                                    |  Instructor                                                         |
|:------------|:--------|:------------------------------------------------------------------------------------------|:-------------------------------------------------------------------:|
| **Week 1**  |  20 Feb |  Intro + Setup  [[code][1e]]                                                              |  Beatriz Borges                                                     |
|             |         |                                                                                           |                                                                     |
| **Week 2**  |  27 Feb |  LMs + Neural LMs: fixed-context models <br/>  Language and Sequence-to-sequence models   |  Simin Fan                                                          |
|             |         |                                                                                           |                                                                     |
| **Week 3**  |  6 Mar  |  Attention + Transformers, GPT                                                            |  Badr AlKhamissi                                                    |
|             |         |                                                                                           |                                                                     |
| **Week 4**  | 13 Mar  |  Pretraining and Transfer Learning                                                        |  Badr AlKhamissi                                                    |
|             |         |                                                                                           |                                                                     |
| **Week 5**  | 20 Mar  |  Transfer Learning                                                                        |  Simin Fan                                                          |
|             |         |                                                                                           |                                                                     |
| **Week 6**  | 27 Mar  |  Generation                                                                               |  Madhur Panwar                                                      |
|             |         |                                                                                           |                                                                     |
| **Week 7/8**|         |  ***MIDTERM***  /  In-context Learning - GPT-3                                           |  Mete Ismayilzad                                                    |  



### Exercises Session format:
- TAs will provide a small discussion over the **last week's exercises**, answering any questions and explaining the solutions. _(10-15mins)_
- TAs will present **this week's exercise**. _(5mins)_ 
- Students will be solving this week's exercises and TAs will provide answers and clarification if needed.

_**Note**: Please make sure you have already done the setup prerequisites to run the coding parts of the exercises. You can find the instructions [here][0e]._



<a name="evaluation"></a>
## Grading:
Your grade in the course will be computed according to the following guidelines.

### Submission Format
We will be using the [Huggingface Hub](https://huggingface.co/docs/hub/en/index) as a centralized platform for submitting project artifacts, including [model checkpoints](https://huggingface.co/docs/hub/en/models-uploading) and [datasets](https://huggingface.co/docs/hub/en/datasets-adding). Please take some time to familiarize yourself with the functionalities of Huggingface Hub in Python to ensure a smooth workflow.


### Late Days Policy
All milestones are due by 23:59 on their respective due dates. However, we understand that meeting deadlines can sometimes be challenging. 
To accommodate this, you will be given 2 late days for the semester to use at your discretion for group project milestones. 
Additionally, you will have 3 individual late days that can be applied to the individual project milestone.
For group projects, if all members still have late days remaining, those days can be pooled and converted to group late days at a rate of one group late day per four individual late days.
No extensions will be granted beyond the due dates, except for the final report, code, and data, which have a strict final deadline of June 8th.
Late days will be automatically tracked based on your latest commit, so there is no need to notify us. Once all your late days are used, any further late submissions will incur a 25% grade deduction per day.


### Midterm (30%):

More details will be announced in the next weeks.


### Project (70%):
The project will be divided into the project proposal (1%), 2 milestones (19%) and a final submission (50%). Each team will be supervised by one of the course TAs or AEs. 

More details on the content of the project and the deliverables of each milestone will be released at a later date.
<!-- Registration details can be found in the announcement [here][1p]. -->

#### Milestone 1:
- Due: 4 May 2025

#### Milestone 2:
<!-- - Milestone 2 parameters can be found in the [project description][2p]. -->
- Due: 18 May 2025

#### Final Deliverable:
- The final report, code, and date will be due on June 8th. Students are welcome to turn in their materials ahead of time, as soon as the semester ends.
<!-- - More details can be found in the [project description][2p]. -->
- Due: 8 June 2025


<a name="contact"></a>
## Contacts

Please email us at **nlp-cs552-spring2025-ta-team [at] groupes [dot] epfl [dot] ch** for any administrative questions, rather than emailing TAs individually. All course content questions need to be asked via [Ed](https://edstem.org/eu/courses/1159/discussion/).

**Lecturer**: [Antoine Bosselut](https://people.epfl.ch/antoine.bosselut)

**Teaching assistants**: [Angelika Romanou](https://people.epfl.ch/angelika.romanou/), [Badr AlKhamissi](https://people.epfl.ch/badr.alkhamissi), [Beatriz Borges](https://people.epfl.ch/beatriz.borges), [Zeming (Eric) Chen](https://people.epfl.ch/zeming.chen?lang=en), [Simin Fan](https://people.epfl.ch/simin.fan?lang=en), [Silin Gao](https://people.epfl.ch/silin.gao?lang=en), [Mete Ismayilzada](https://people.epfl.ch/mahammad.ismayilzada), [Sepideh Mamooler](https://people.epfl.ch/sepideh.mamooler), [Madhur Panwar](https://people.epfl.ch/madhur.panwar), 
[Auguste Poiroux](https://people.epfl.ch/auguste.poiroux), [Ayush Tarun](https://people.epfl.ch/ayush.tarun)



[0e]:https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Exercises/Setup
[1e]:https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Exercises/
