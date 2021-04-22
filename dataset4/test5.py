import time

import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import os
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Flatten, Dropout, BatchNormalization, Embedding, Input, \
    TimeDistributed
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
'''
model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

article = """(CNN)James Holmes made his introduction to the world in a Colorado cinema filled with spectators watching a midnight showing of the new Batman movie, "The Dark Knight Rises," in June 2012. The moment became one of the deadliest shootings in U.S. history. Holmes is accused of opening fire on the crowd, killing 12 people and injuring or maiming 70 others in Aurora, a suburb of Denver. Holmes appeared like a comic book character: He resembled the Joker, with red-orange hair, similar to the late actor Heath Ledger\'s portrayal of the villain in an earlier Batman movie, authorities said. But Holmes was hardly a cartoon. Authorities said he wore body armor and carried several guns, including an AR-15 rifle, with lots of ammo. He also wore a gas mask. Holmes says he was insane at the time of the shootings, and that is his legal defense and court plea: not guilty by reason of insanity. Prosecutors aren\'t swayed and will seek the death penalty. Opening statements in his trial are scheduled to begin Monday. Holmes admits to the shootings but says he was suffering "a psychotic episode" at the time,  according to court papers filed in July 2013 by the state public defenders, Daniel King and Tamara A. Brady. Evidence "revealed thus far in the case supports the defense\'s position that Mr. Holmes suffers from a severe mental illness and was in the throes of a psychotic episode when he committed the acts that resulted in the tragic loss of life and injuries sustained by moviegoers on July 20, 2012," the public defenders wrote. Holmes no longer looks like a dazed Joker, as he did in his first appearance before a judge in 2012. He appeared dramatically different in January when jury selection began for his trial: 9,000 potential jurors were summoned for duty, described as one of the nation\'s largest jury calls. Holmes now has a cleaner look, with a mustache, button-down shirt and khaki pants. In January, he had a beard and eyeglasses. If this new image sounds like one of an academician, it may be because Holmes, now 27, once was one. Just before the shooting, Holmes was a doctoral student in neuroscience, and he was studying how the brain works, with his schooling funded by a U.S. government grant. Yet for all his learning, Holmes apparently lacked the capacity to command his own mind, according to the case against him. A jury will ultimately decide Holmes\' fate. That panel is made up of 12 jurors and 12 alternates. They are 19 women and five men, and almost all are white and middle-aged. The trial could last until autumn. When jury summonses were issued in January, each potential juror stood a 0.2% chance of being selected, District Attorney George Brauchler told the final jury this month. He described the approaching trial as "four to five months of a horrible roller coaster through the worst haunted house you can imagine." The jury will have to render verdicts on each of the 165 counts against Holmes, including murder and attempted murder charges. Meanwhile, victims and their relatives are challenging all media outlets "to stop the gratuitous use of the name and likeness of mass killers, thereby depriving violent individuals the media celebrity and media spotlight they so crave," the No Notoriety group says. They are joined by victims from eight other mass shootings in recent U.S. history. Raised in central coastal California and in San Diego, James Eagan Holmes is the son of a mathematician father noted for his work at the FICO firm that provides credit scores and a registered nurse mother, according to the U-T San Diego newspaper. Holmes also has a sister, Chris, a musician, who\'s five years younger, the newspaper said. His childhood classmates remember him as a clean-cut, bespectacled boy with an "exemplary" character who "never gave any trouble, and never got in trouble himself," The Salinas Californian reported. His family then moved down the California coast, where Holmes grew up in the San Diego-area neighborhood of Rancho Peñasquitos, which a neighbor described as "kind of like Mayberry," the San Diego newspaper said. Holmes attended Westview High School, which says its school district sits in "a primarily middle- to upper-middle-income residential community." There, Holmes ran cross-country, played soccer and later worked at a biotechnology internship at the Salk Institute and Miramar College, which attracts academically talented students. By then, his peers described him as standoffish and a bit of a wiseacre, the San Diego newspaper said. Holmes attended college fairly close to home, in a neighboring area known as Southern California\'s "inland empire" because it\'s more than an hour\'s drive from the coast, in a warm, low-desert climate. He entered the University of California, Riverside, in 2006 as a scholarship student. In 2008 he was a summer camp counselor for disadvantaged children, age 7 to 14, at Camp Max Straus, run by Jewish Big Brothers Big Sisters of Los Angeles. He graduated from UC Riverside in 2010 with the highest honors and a bachelor\'s degree in neuroscience. "Academically, he was at the top of the top," Chancellor Timothy P. White said. He seemed destined for even higher achievement. By 2011, he had enrolled as a doctoral student in the neuroscience program at the University of Colorado Anschutz Medical Campus in Aurora, the largest academic health center in the Rocky Mountain region. The doctoral in neuroscience program attended by Holmes focuses on how the brain works, with an emphasis on processing of information, behavior, learning and memory. Holmes was one of six pre-thesis Ph.D. students in the program who were awarded a neuroscience training grant from the National Institutes of Health. The grant rewards outstanding neuroscientists who will make major contributions to neurobiology. A syllabus that listed Holmes as a student at the medical school shows he was to have delivered a presentation about microRNA biomarkers. But Holmes struggled, and his own mental health took an ominous turn. In March 2012, he told a classmate he wanted to kill people, and that he would do so "when his life was over," court documents said. Holmes was "denied access to the school after June 12, 2012, after he made threats to a professor," according to court documents. About that time, Holmes was a patient of University of Colorado psychiatrist Lynne Fenton. Fenton was so concerned about Holmes\' behavior that she mentioned it to her colleagues, saying he could be a danger to others, CNN affiliate KMGH-TV reported, citing sources with knowledge of the investigation. Fenton\'s concerns surfaced in early June, sources told the Denver station. Holmes began to fantasize about killing "a lot of people" in early June, nearly six weeks before the shootings, the station reported, citing unidentified sources familiar with the investigation. Holmes\' psychiatrist contacted several members of a "behavioral evaluation and threat assessment" team to say Holmes could be a danger to others, the station reported. At issue was whether to order Holmes held for 72 hours to be evaluated by mental health professionals, the station reported. "Fenton made initial phone calls about engaging the BETA team" in "the first 10 days" of June, but it "never came together" because in the period Fenton was having conversations with team members, Holmes began the process of dropping out of school, a source told KMGH. Defense attorneys have rejected the prosecution\'s assertions that Holmes was barred from campus. Citing statements from the university, Holmes\' attorneys have argued that his access was revoked because that\'s normal procedure when a student drops enrollment. What caused this turn for the worse for Holmes has yet to be clearly detailed. In the months before the shooting, he bought four weapons and more than 6,000 rounds of ammunition, authorities said. Police said he also booby-trapped his third-floor apartment with explosives, but police weren\'t fooled. After Holmes was caught in the cinema parking lot immediately after the shooting, bomb technicians went to the apartment and neutralized the explosives. No one was injured at the apartment building. Nine minutes before Holmes went into the movie theater, he called a University of Colorado switchboard, public defender Brady has said in court. The number he called can be used to get in contact with faculty members during off hours, Brady said. Court documents have also revealed that investigators have obtained text messages that Holmes exchanged with someone before the shooting. That person was not named, and the content of the texts has not been made public. According to The New York Times, Holmes sent a text message to a fellow graduate student, a woman, about two weeks before the shooting. She asked if he had left Aurora yet, reported the newspaper, which didn\'t identify her. No, he had two months left on his lease, Holmes wrote back, according to the Times. He asked if she had heard of "dysphoric mania," a form of bipolar disorder marked by the highs of mania and the dark and sometimes paranoid delusions of major depression. The woman asked if the disorder could be managed with treatment. "It was," Holmes wrote her, according to the Times. But he warned she should stay away from him "because I am bad news," the newspaper reported. It was her last contact with Holmes. After the shooting, Holmes\' family issued a brief statement: "Our hearts go out to those who were involved in this tragedy and to the families and friends of those involved," they said, without giving any information about their son. Since then, prosecutors have refused to offer a plea deal to Holmes. For Holmes, "justice is death," said Brauchler, the district attorney. In December, Holmes\' parents, who will be attending the trial, issued another statement: They asked that their son\'s life be spared and that he be sent to an institution for mentally ill people for the rest of his life, if he\'s found not guilty by reason of insanity. "He is not a monster," Robert and Arlene Holmes wrote, saying the death penalty is "morally wrong, especially when the condemned is mentally ill." "He is a human being gripped by a severe mental illness," the parents said. The matter will be settled by the jury. CNN\'s Ana Cabrera and Sara Weisfeldt contributed to this report from Denver."""

input_ids = tokenizer(article, return_tensors="pt").input_ids
output_ids = model.generate(input_ids)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
'''
from transformers import BigBirdForMultipleChoice
# model = TFAutoModelForSeq2SeqLM.from_pretrained("allenai/longformer-base-4096")
import pytorch_lightning as pl
from transformers import BigBirdForTokenClassification



sum = [{'sum':'sssssss'},{'sum1':'1sssssss'}]
# print(sum[0]['sum'])
section = ['Introduction', 'Related Work', 'Related Work', 'Related Work', 'Experiments', 'Discussion']
sections = []
print("----------")
for i in section:
    # print(i)
    sections.append(i)

import numpy as np
# print(np.unique(sections))

# from datasets import load_dataset
# raw_datasets = load_dataset('csv', data_files=['./tmp3/b_dataset_2_train.csv'])
# print(raw_datasets)
contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                       "could've": "could have", "couldn't": "could not",

                       "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                       "hasn't": "has not", "haven't": "have not",

                       "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                       "I'm": "I am", "I've": "I have", "i'd": "i would",

                       "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would",

                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                       "let's": "let us", "ma'am": "madam",

                       "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have",

                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                       "needn't've": "need not have", "o'clock": "of the clock",

                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have",

                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                       "she'll've": "she will have", "she's": "she is",

                       "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                       "so've": "so have", "so's": "so as",

                       "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                       "there'd": "there would",

                       "there'd've": "there would have", "there's": "there is", "here's": "here is",
                       "they'd": "they would", "they'd've": "they would have",

                       "they'll": "they will", "they'll've": "they will have", "they're": "they are",
                       "they've": "they have", "to've": "to have",

                       "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                       "we'll've": "we will have", "we're": "we are",

                       "we've": "we have", "weren't": "were not", "what'll": "what will",
                       "what'll've": "what will have", "what're": "what are",

                       "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                       "where'd": "where did", "where's": "where is",

                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                       "who've": "who have",

                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                       "won't've": "will not have",

                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                       "y'all": "you all",

                       "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                       "y'all've": "you all have",

                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                       "you'll've": "you will have",

                       "you're": "you are", "you've": "you have"}

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

Introductions = []
Introductions_without_stopwords = []
Abstracts = []

T_I_FILE = "./test/article_3_train_{}.txt"
T_A_FILE = "./test/summary_3_train_{}.txt"
for year in range(1,5):
    with open(T_I_FILE.format(year), "rt", encoding='utf-8') as TEXT_I_FILE:
        for i in TEXT_I_FILE.readlines():
            Introduction_Text = ''.join(i.rstrip(''))
            Introductions.append(Introduction_Text)
            Introductions_without_stopwords.append(
                ' '.join([w for w in Introduction_Text.split() if w not in stop_words]))
        TEXT_I_FILE.close()
    with open(T_A_FILE.format(year), "rt", encoding='utf-8', errors='ignore') as TEXT_A_FILE:
        for i in TEXT_A_FILE.readlines():
            Abstract_Text = ''.join(i.rstrip())
            Abstracts.append(Abstract_Text)
        TEXT_A_FILE.close()
print(len(Abstracts), len(Introductions), len(Introductions_without_stopwords))
data = pd.DataFrame(
        {'article': Introductions,
         'summary': Abstracts})
data.to_csv('./test/b_dataset_3_train.csv', index=False)
'''
python  run_summarization_longformer.py --num_workers 12  --save_prefix eval_long16k_nooverlap_large --model_path facebook/bart-large-xsum --max_input_len 16368 --batch_size 2 --grad_accum 4 --grad_ckpt   --attention_mode sliding_chunks_no_overlap --attention_window 340 --val_every 0.333333333  --debug --resume summarization/run_long16k_nooverlap_large/_ckpt_epoch_3_v1.ckpt  --val_percent_check 1.0 --disable_checkpointing

python run_summarization.py --model_name_or_path facebook/bart-large-xsum --do_train --do_eval --train_file ./a_dataset_1_train.csv --validation_file ./a_dataset_1_test.csv --text_column Article --summary_column Summary --source_prefix "summarize: " --output_dir ./tmp/tst-summarization --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --overwrite_output_dir --predict_with_generate

python run_summarization.py --model_name_or_path t5-small --do_train --do_eval --train_file ./dataset3/b_dataset_1_train.csv --validation_file ./dataset3/b_dataset_1_test.csv --text_column document --summary_column summary --source_prefix "summarize: " --output_dir ./tmp/t5-small-dataset-1 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --overwrite_output_dir --predict_with_generate

python run_summarization.py --model_name_or_path t5-small --do_train --do_eval --train_file ./dataset3/b_dataset_1_train.csv --validation_file ./dataset3/b_dataset_1_test.csv --source_prefix "summarize: " --output_dir ./tmp/t5-small-dataset-1 --per_device_train_batch_size=6 --per_device_eval_batch_size=6 --overwrite_output_dir --predict_with_generate

python run_summarization.py --model_name_or_path t5-large --do_train --do_eval --train_file ./dataset3/b_dataset_1_train.csv --validation_file ./dataset3/b_dataset_1_test.csv --source_prefix "summarize: " --output_dir ./tmp/t5-large-dataset-1 --per_device_train_batch_size=6 --per_device_eval_batch_size=6 --overwrite_output_dir --predict_with_generate

python run_summarization.py --model_name_or_path nsi319/legal-led-base-16384 --do_train --do_eval --train_file ./dataset3/b_dataset_1_train.csv --validation_file ./dataset3/b_dataset_1_test.csv --source_prefix "" --output_dir ./tmp/led-dataset-1 --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --overwrite_output_dir --predict_with_generate


'''