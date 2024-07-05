# NLP Project: Emotion-Cause Pair Extraction

## Introduction

### Motivation

Have you ever wondered why someone feels a certain way during a conversation? Understanding the reasons behind emotions expressed in dialogue is not just fascinating—it’s essential. Whether it's enhancing dialogue systems or offering better mental health support, grasping these emotional triggers can make a significant difference.

Our project dives into the heart of this challenge: emotion-cause pair extraction. This means identifying the exact words or phrases that spark specific emotions in conversations. Imagine a system that can automatically pinpoint why someone feels happy, sad, or frustrated based on their words. This capability can transform how we analyze emotional expressions, leading to more intuitive dialogue systems and emotionally aware AI.

By developing a robust system to extract these emotion-cause pairs from conversational data, we aim to deepen our understanding of human interactions. This not only helps in academic research but also holds immense practical value. From improving natural language processing to enhancing human-computer interactions, the potential applications are vast. Our work could pave the way for more empathetic and responsive AI, making our interactions with technology feel more human and understanding.

## Related Work

### Literature Review of Emotion-Cause Pair Extraction in Conversations: A Two-Step Multi-Task Approach

This paper explores the methodology for identifying emotions and their causes in textual conversations. The typical approach involves two stages: first, independently identifying the emotions and their causes, and then pairing them together. Researchers have used advanced techniques like attention networks, dual-questioning mechanisms, and context awareness to improve the accuracy and efficiency of extracting these emotion-cause pairs. By breaking down the process into distinct stages, this multi-task approach aims to precisely pinpoint emotional cues and their triggers within conversations, offering a deeper understanding of emotional dynamics in text.

### Literature Review of ECPEC: Emotion-Cause Pair Extraction in Conversation

The literature review on Emotion-Cause Pair Extraction in Conversations (ECPEC) delves into how emotion-cause pairs are extracted from conversational text. This research area has gained significant attention due to its importance in understanding emotional dynamics across various fields. Scholars have introduced innovative methods, including neural networks, emotion-aware word embeddings, and Bi-LSTM layers, to enhance the accuracy of identifying emotional cues and their causes. By integrating previous research with novel techniques, the ECPEC literature review aims to advance Natural Language Processing by shedding light on the intricate relationship between emotions and their triggers in text-based conversations.

## Methodology

We've implemented a two-phase approach for our emotion-cause pair extraction task. Our model leverages contextual learning from the subsequent utterance to predict the emotion associated with a given sentence. Using a dataset derived from the TV show “Friends,” which features a wide range of emotional contexts, our model analyzes these fluctuations. It employs a specialized sub-model to identify parts of the utterances that trigger emotional causes or shifts.

### First Phase

The first phase generates emotions for each utterance in the entire conversation. We start by generating a sentence embedding using the BERT model. These embeddings encapsulate all the information in the sentence within a 768-dimensional vector. Given that each utterance depends on the preceding sentence, contextual learning and memory are crucial for accurately determining the sentence's emotion. To achieve this, we further process the word embeddings by feeding them into an LSTM model. This model identifies dependencies between the sentences and ultimately outputs the emotion for each utterance.

### Second Phase

The second phase builds on the emotions generated in the first phase. This phase involves identifying the shift in emotion from neutral to another emotion, pinpointing the utterance responsible for this change. We fine-tune the BertForTokenClassification approach to identify the specific part of the sentence causing the emotional shift. We create six different models, one for each emotion, to detect the triggers for each respective emotion. We use the token classifier, treating the span prediction task as a modified NER task, where the label 0 marks non-triggers and the label 1 marks triggers for a certain emotion.

## Observations

One critical aspect confirming the validity of existing theories is the influence of bias within the training data. This is evident through the varying F1 scores obtained for different emotions, where a higher volume of training data correlates with better F1 scores overall. For example, the lower F1 score for the "fear" emotion highlights the impact of this bias.

An intriguing observation post phase 2 training is the model's struggle to describe emotions like "sadness" and "anger" effectively, despite having more data points than "disgust." There are two possible explanations for this phenomenon:

Firstly, the "Friends" TV show dataset is notably biased towards emotions like "joy" and "surprise," a bias apparent when watching the show. This bias challenges the generic BERT models, leading to poorer performance for "sadness" and "anger."

Alternatively, the textual expression of emotions like "sadness" and "anger" may lack the nuanced portrayal facilitated by multimodal data. Unlike "joy" and "surprise," which can be conveyed through punctuation marks like exclamation points and a varied vocabulary, the textual medium may inherently limit the effective communication of "sadness" and "anger."

## Results & Findings

### Emotion Scores
| Emotion   | Score |
|-----------|-------|
| Overall   | 0.42  |
| Anger     | 0.33  |
| Fear      | 0.12  |
| Disgust   | 0.28  |
| Sadness   | 0.39  |
| Surprise  | 0.40  |
| Joy       | 0.45  |
| Neutral   | 0.97  |

### Phase 1 Scores
| Model     | Score |
|-----------|-------|
| Overall   | 0.65  |
| Fear      | 0.51  |
| Sadness   | 0.62  |
| Joy       | 0.58  |
| Anger     | 0.60  |
| Disgust   | 0.52  |
| Surprise  | 0.59  |

### Phase 2 Scores
We proposed a novel method for emotion-cause extraction in conversation using an approach similar to NER. We modified the training data to form binary encoded labels with 0 for a trigger and 1 for non-trigger, which we fed into a BertForTokenClassification model, fine-tuning it without freezing it.

## Conclusion

This project aimed to extract emotion-cause pairs from conversations for diverse applications. We utilized a two-phase approach for our emotion-cause pair extraction task. Our model, based on contextual learning from the next utterance, predicts the emotion associated with the sentence. Using a dataset derived from the TV show “Friends,” which features emotional ups and downs, our two-phase approach addressed context and specialized models. Despite challenges like biased data and the difficulty of expressing certain emotions through text, our results highlighted the importance of data distribution and innovation in advancing emotional intelligence and NLP.

## Future Scope

The low F1 scores for the entire task indicate the need to improve existing models and architecture. Utilizing multi-modal inputs and more efficient inference times are vital for real-time applications in enhancing human-computer interaction and mental health support. The following tasks are worthy of further study:

1. How to effectively model the speaker's relevance for both emotion recognition and cause extraction in conversations?
2. How to utilize external commonsense knowledge to bridge the gap between emotion and cause that are not explicitly reflected in the conversation?

## References

- Literature Review of Emotion-Cause Pair Extraction in Conversations: A Two-Step Multi-Task Approach : Lee, Jaehyeok & Jeong, DongJin & Bak, JinYeong. (2023).
- Literature Review of ECPEC: Emotion-Cause Pair Extraction in Conversation: Wei, Li & Li, Yang & Pandelea, Vlad & Ge, Mengshi & Zhu, Luyao & Cambria, Erik. (2022).
