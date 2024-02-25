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

| Platform           						| Where & when                                              																								   |
|:------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Lectures           						| **Wednesdays: 11:15-13:00** [[STCC - Cloud C](https://plan.epfl.ch/?room=%3DSTCC%20-%20Cloud%20C&dim_floor=0&lang=en&dim_lang=en&tree_groups=centres_nevralgiques%2Cmobilite_acces_grp%2Censeignement%2Ccommerces_et_services&tree_group_layers_centres_nevralgiques=information_epfl%2Cguichet_etudiants&tree_group_layers_mobilite_acces_grp=metro&tree_group_layers_enseignement=&tree_group_layers_commerces_et_services=&baselayer_ref=grp_backgrounds&map_x=2532938&map_y=1152803&map_zoom=11)] & **Thursdays: 13:15-14:00** [[CE16](https://plan.epfl.ch/?room=%3DCE%201%206&dim_floor=1&lang=en&dim_lang=en&tree_groups=centres_nevralgiques%2Cmobilite_acces_grp%2Censeignement%2Ccommerces_et_services&tree_group_layers_centres_nevralgiques=information_epfl%2Cguichet_etudiants&tree_group_layers_mobilite_acces_grp=metro&tree_group_layers_enseignement=&tree_group_layers_commerces_et_services=&baselayer_ref=grp_backgrounds&map_x=2533400&map_y=1152502&map_zoom=13)]  |
| Exercises Session  						| **Thursdays: 14:15-16:00** [[CE11](https://plan.epfl.ch/?room=%3DCE%201%201&dim_floor=1&lang=en&dim_lang=en&tree_groups=centres_nevralgiques%2Cmobilite_acces_grp%2Censeignement%2Ccommerces_et_services&tree_group_layers_centres_nevralgiques=information_epfl%2Cguichet_etudiants&tree_group_layers_mobilite_acces_grp=metro&tree_group_layers_enseignement=&tree_group_layers_commerces_et_services=&baselayer_ref=grp_backgrounds&map_x=2533297&map_y=1152521&map_zoom=13)] 																				   |
| Project Assistance <br />(not every week) | **Wednesdays: 13:15-14:00** [[STCC - Cloud C](https://plan.epfl.ch/?room=%3DSTCC%20-%20Cloud%20C&dim_floor=0&lang=en&dim_lang=en&tree_groups=centres_nevralgiques%2Cmobilite_acces_grp%2Censeignement%2Ccommerces_et_services&tree_group_layers_centres_nevralgiques=information_epfl%2Cguichet_etudiants&tree_group_layers_mobilite_acces_grp=metro&tree_group_layers_enseignement=&tree_group_layers_commerces_et_services=&baselayer_ref=grp_backgrounds&map_x=2532938&map_y=1152803&map_zoom=11)] 													   						   |
| QA Forum & Annoucements       | Ed Forum [[link](https://edstem.org/eu/courses/1159/discussion/)]                                       													   | 
| Grades             						| Moodle [[link](https://moodle.epfl.ch/course/view.php?id=17143)]              																		   |

All lectures will be given in person and live streamed on Zoom. The link to the Zoom is available on the Ed Forum (pinned post). Beware that, in the event of a technical failure during the lecture, continuing to accompany the lecture live via zoom might not be possible.

Recording of the lectures will be made available on SwitchTube. We will reuse some of last year's recordings and we may record a few new lectures in case of different lecture contents.

<a name="lectures"></a>
## Lecture Schedule

| Week        | Date                 |  Topic                                                                                                                      |  Instructor                                |
|:------------|:---------------------|:----------------------------------------------------------------------------------------------------------------------------|:------------------------------------------:|
| **Week 1**  | 21 Feb <br />22 Feb  |  Introduction &#124; Building a simple neural classifier  <br />Neural LMs: word embeddings [[slides][1s]]  |  Antoine Bosselut                   	    |
|             |                      |                                                                                                                             |                                      	    |
| **Week 2**  |  28 Feb <br />29 Feb   |  LM basics &#124; Neural LMs: Fixed Context Models <br />Neural LMs: RNNs, Backpropagation, Vanishing Gradients; LSTMs     |  Antoine Bosselut                   		|
|             |                      |                                                                                                                             |                                      	    |
| **Week 3**  |  6 Mar <br />7 Mar   |  Seq2seq + decoding + attention &#124; Transformers<br />Transformers + Greedy Decoding; GPT |  Antoine Bosselut  		|
|             |                      |                                                                                                                             |                                      	    |
| **Week 4**  | 13 Mar <br />14 Mar  |  Pretraining: ELMo, BERT, MLM, task generality &#124; Transfer Learning: Introduction <br />Pretraining S2S: BART, T5    |  Antoine Bosselut                   		|
|             |                      |                                                                                                                             |                                      	    |
| **Week 5**  | 20 Mar <br />21 Mar  |  Transfer Learning: Dataset Biases  <br />Generation: Task    |  Antoine Bosselut                   		|
|             |                      |                                                                                                                             |                                      	    |
| **Week 6**  | 27 Mar <br />28 Mar  |  Generation: Decoding and Training  <br />Generation: Evaluation  |  Antoine Bosselut                   		|
|             |                      |                                                                                                                             |                                      	    |
|             |                      |  ***EASTER BREAK***                                                                                                       |                                     		|
|             |                      |                                                                                                                             |                                      	    |
| **Week 7**  |  10 Apr  <br />11 Apr  |  In-context Learning - GPT-3 + Prompts &#124; Instruction Tuning<br />Project Description    |  Antoine Bosselut                   		|
|             |                      |                                                                                                                             |                                      	    |
| **Week 8**  | 17 Apr <br />18 Apr  |  Scaling laws &#124; Model Compression <br />**No class** (Project work; A1 Grade Review Session)    |  Antoine Bosselut <br /> 		|
|             |                      |                                                                                                                             |                                      	    |
| **Week 9** | 24 Apr <br />25 Apr  |  Ethics in NLP: Bias / Fairness and Toxicity, Privacy, Disinformation <br />**No class** (Project work; A1 Grade Review Session)    |  Anna Sotnikova                   		|
|             |                      |                                                                                                                             |                                      	    |
| **Week 10** |  1 May <br />2 May   |  Tokenization: BPE, WP, Char-based &#124; Multilingual LMs <br />Guest Lecture: Kayo Yin  |  Negar Foroutan <br /> Kayo Yin                 		|
|             |                      |                                                                                                                             |                                      	    |
| **Week 11** | 8 May <br />9 May  |  Syntactic and Semantic Tasks (NER) &#124; Interpretability: BERTology <br />**No class** (Project work; A2 Grade Review Session) |  Gail Weiss                   		|
|             |                      |                                                                                                                             |                                      	    |
| **Week 12** | 15 May <br />16 May  |  Reading Comprehension &#124; Retrieval-augmented LMs <br />**No class** (Project work; A2 Grade Review Session) |  Antoine Bosselut   |
|             |                      |                                                                                                                             |                                      	    |
| **Week 13** | 22 May <br />23 May  |  Multimodality: L & V <br />Looking forward   |  Syrielle Montariol <br />Antoine Bosselut                       		|
|             |                      |                                                                                                                             |                                      	    |
| **Week 14** | 29 May <br />30 May   |  **No class** (Project work; A3 Grade Review Session) |    |


<a name="exercises"></a>
## Exercise Schedule

| Week        | Date    |  Topic                                                                                |  Instructor                                                         |
|:------------|:--------|:--------------------------------------------------------------------------------------|:-------------------------------------------------------------------:|
| **Week 1**  | 22 Feb  |  Setup + Word embeddings  [[code][1e]]                                                |  Mete Ismayilzada            |
|             |         |                                                                                       |                                                                     |
| **Week 2**  |  29 Feb  |  Word embeddings review <br /> Language and Sequence-to-sequence models  |  Mete Ismayilzada <br />Badr AlKhamissi  |
|             |         |                                                                                       |                                                                     |
| **Week 3**  |  7 Mar  | Language and Sequence-to-sequence models review <br /> Attention + Transformers <br /> Assignment 1 Q&A    |  Badr AlKhamissi <br />Simin Fan         |
|             |         |                                                                                       |                                                                     |
| **Week 4**  | 14 Mar  |  Attention + Transformers review <br />Pretraining and Transfer Learning Pt. 1 <br /> Assignment 1 Q&A  |  Simin Fan <br /> Badr AlKhamissi     |
|             |         |                                                                                       |                                                                     |
| **Week 5**  | 21 Mar  |  Pretraining and Transfer Learning Pt. 1 review <br />Transfer Learning Pt. 2 <br /> Assignment 2 Q&A        |  Simin Fan              |
|             |         |                                                                                       |                                                                     |
| **Week 6**  | 28 Mar  |  Transfer Learning Pt. 2 review <br />Text Generation <br /> Assignment 2 Q&A    |  Simin Fan <br />Deniz Bayazit           |
|             |         |                                                                                       |                                                                     |
| **Week 8**  | 4 Apr  |  ***EASTER BREAK***                                                                   |                                                                     |  
|             |         |                                                                                       |                                                                     |
| **Week 7**  |  11 Apr  |  Text Generation review <br />In-context Learning   <br /> Assignment 3 Q&A                   |  Badr AlKhamissi <br /> Deniz Bayazit <br /> Mete Ismayilzada  |
|             |         |                                                                                       |                                                                     |
| **Week 9**  | 18 Apr  |  In-context Learning review <br /> Assignment 3 Q&A                   |  Badr AlKhamissi <br /> Deniz Bayazit <br /> Mete Ismayilzada |
|             |         |                                                                                       |                                                                     |
| **Week 10** | 25 Apr  |  Project                                                               |  TA meetings on-demand                              |
|             |         |                                                                                       |                                                                     |
| **Week 11** |  2 May  |  Project                                                                             |  TA meetings on-demand                                              |
|             |         |                                                                                       |                                                                     |
| **Week 12** | 9 May  |  Project  <br /> Milestone 1 Feedback                                                |  TA meetings on-demand                              |
|             |         |                                                                                       |                                                                     |
| **Week 13** | 16 May  |  Project                                                                              |  TA meetings on-demand                                              |
|             |         |                                                                                       |                                                                     |
| **Week 14** | 23 May  |  Project                                                 |  TA meetings on-demand                          |
|             |         |                                                                                       |                                                                     |
| **Week 15** | 30 May   |  Project <br /> Milestone 2 Feedback                                                                             |  TA meetings on-demand                                              |


### Exercises Session format:
- TAs will provide a small discussion over the **last week's exercises**, answering any questions and explaining the solutions. _(10-15mins)_
- TAs will present **this week's exercise**. _(5mins)_ 
- Students will be solving this week's exercises and TAs will provide answers and clarification if needed.

_**Note**: Please make sure you have already done the setup prerequisites to run the coding parts of the exercises. You can find the instructions [here][0e]._

<a name="evaluation"></a>
## Grading:
Your grade in the course will be computed according to the following guidelines:

### Assignments (40%):
There will be three assignments throughout the course. They will be released and due according to the following schedule:

#### Assignment 1 (10%)
<!-- Link for the assignment [here][1a]. -->
- Released: 26 February 2024
- Due: 17 March 2024
- Grade released: 14 April 2024
- Grade review sessions: 18 and 25 April 2024

#### Assignment 2 (15%)
<!-- Link for the assignment [here][2a]. -->
- Released: 18 March 2024
- Due: 7 April 2024
- Grade released: 5 May 2024
- Grade review sessions: 9 and 16 May 2024

#### Assignment 3 (15%)
<!-- Link for the assignment [here][3a]. -->
- Released: 1 April 2024
- Due: 21 April 2024
- Grade released: 19 May 2024
- Grade review sessions: 29 and 30 May 2024

Assignment releases will be announced on Ed.

### Project (60%):
The project will be divided into 2 milestones and a final submission. Each milestone will be worth 15% of the final grade with the remaining 30% being allocated to the final report. Each team will be supervised by one of the course TAs or AEs. 

More details on the content of the project and the deliverables of each milestone will be released at a later date.
<!-- Registration details can be found in the announcement [here][1p]. -->

#### Milestone 1 (15%):
<!-- - Milestone 1 parameters can be found in the [project description][2p]. -->
- Due: 5 May 2024

#### Milestone 2 (15%):
<!-- - Milestone 2 parameters can be found in the [project description][2p]. -->
- Due: 26 May 2024

#### Final Deliverable (30%):
- The final report, code, and date will be due on June 14th. Students are welcome to turn in their materials ahead of time, as soon as the semester ends.
<!-- - More details can be found in the [project description][2p]. -->
- Due: 14 June 2024

### Late Days Policy
All assignments and milestones are due at 23:59 on their due date. As we understand that circumstances can make it challenging to abide by these due dates, you will receive 6 late days over the course of the semester to be allocated to the assignments and project milestones as you see fit. No further extensions will be granted. The only exception to this rule is for the final report, code, and data. No extensions will be granted beyond June 14th.


<a name="contact"></a>
## Contacts

**Lecturer**: [Antoine Bosselut](https://people.epfl.ch/antoine.bosselut)

**Teaching assistants**: [Badr AlKhamissi](https://people.epfl.ch/badr.alkhamissi), [Deniz Bayazit](https://people.epfl.ch/deniz.bayazit?lang=en), [Beatriz Borges](https://people.epfl.ch/beatriz.borges), [Zeming (Eric) Chen](https://people.epfl.ch/zeming.chen?lang=en), [Simin Fan](https://people.epfl.ch/simin.fan?lang=en), [Negar Foroutan Eghlidi](https://people.epfl.ch/negar.foroutan), [Silin Gao](https://people.epfl.ch/silin.gao?lang=en), [Mete Ismayilzada](https://people.epfl.ch/mahammad.ismayilzada)

Please contact us for any organizational questions or questions related to the course content.


[1s]:https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Lectures/Week%201
<!-- [2s]: -->
<!-- [3s]: -->
<!-- [4s]: -->
<!-- [5s]: -->
<!-- [6s]: -->
<!-- [7s]: -->
<!-- [8s]: -->
<!-- [9s]: -->
<!-- [10s]: -->
<!-- [11s]: -->
<!-- [12s]: -->
<!-- [13s]: -->
<!-- [14s]: -->

[0e]:https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Exercises/Setup
[1e]:https://github.com/epfl-nlp/cs-552-modern-nlp/tree/main/Exercises/Week%201%20-%20Word%20Embeddings
<!-- [2e]: -->
<!-- [3e]: -->
<!-- [4e]: -->
<!-- [5e]: -->
<!-- [6e]: -->
<!-- [7e]: -->
<!-- [8e]: -->

<!-- [1a]: -->
<!-- [2a]: -->
<!-- [3a]: -->

<!-- [1p]: -->
<!-- [2p]: -->

<!-- [1r]: -->
<!-- [2r]: -->
<!-- [3r]: -->
<!-- [4r]: -->
<!-- [5r]: -->
<!-- [6r]: -->
<!-- [7r]: -->
<!-- [8r]: -->
<!-- [9r]: -->
<!-- [10r]: -->
<!-- [11r]: -->
<!-- [12r]: -->
<!-- [13r]: -->
<!-- [14r]: -->

