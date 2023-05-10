import streamlit as st
import subprocess
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_splitter import SentenceSplitter, split_text_into_sentences
from parrot import Parrot
# import torch
import warnings
warnings.filterwarnings("ignore")

st.title("My Streamlit App")
algorithm = st.selectbox("Select an algorithm", ("Model 1 : Document level", "Model 2 : Pegasus Google", "Model 3 : Parrot"))
text = st.text_area("Enter some text:")
button_pressed = st.button("Run Commands")
# convert text to text file


if button_pressed:
    if algorithm == "Model 1 : Document level":
        with open("C:\\Users\\laksh\\OneDrive\\Desktop\\DL_Ops-Project-main\\our_data\\story.txt", 'w') as file:
            file.write(text)

        # Run subprocess commands
        commands = [
            ["python", "C:\\Users\\laksh\\OneDrive\\Desktop\\DL_Ops-Project-main\\eval\\coherence.py", "--inference", "--pretrain_model", "albert-base-v2", "--save_file", "C:\\Users\\laksh\\OneDrive\\Desktop\\DL_Ops-Project-main\\our_data", "--text_file", "C:\\Users\\laksh\\OneDrive\\Desktop\\DL_Ops-Project-main\\our_data\\story.txt"],
            ["python", "C:\\Users\\laksh\\OneDrive\\Desktop\\DL_Ops-Project-main\\preprocess.py", "--source", "C:\\Users\\laksh\\OneDrive\\Desktop\\DL_Ops-Project-main\\our_data\\story.txt", "--graph", "C:\\Users\\laksh\\OneDrive\\Desktop\\DL_Ops-Project-main\\our_data\\story.txt.graph", "--vocab", "C:\\Users\\laksh\\OneDrive\\Desktop\\DL_Ops-Project-main\\our_data\\vocab.share", "--save_file", "C:\\Users\\laksh\\OneDrive\\Desktop\\DL_Ops-Project-main\\our_data\\sent.pt"],
            ["python", "C:\\Users\\laksh\\OneDrive\\Desktop\\DL_Ops-Project-main\\generator.py", "--cuda_num", "0", "--file", "C:\\Users\\laksh\\OneDrive\\Desktop\\DL_Ops-Project-main\\our_data\\sent.pt", "--max_tokens", "10000", "--vocab", "C:\\Users\\laksh\\OneDrive\\Desktop\\DL_Ops-Project-main\\our_data\\vocab.share", "--decode_method", "greedy", "--beam", "5", "--model_path", "C:\\Users\\laksh\\OneDrive\\Desktop\\DL_Ops-Project-main\\our_data\\model.pkl", "--output_path", "C:\\Users\\laksh\\OneDrive\\Desktop\\DL_Ops-Project-main\\our_data\\output", "--max_length", "300"]
        ]

        for command in commands:
            subprocess.run(command)

        # Read the output file
        with open("C:\\Users\\laksh\\OneDrive\\Desktop\\DL_Ops-Project-main\\our_data\\output\\result.txt", 'r') as file:
            output_text = file.read()

        st.text_area("Your Paraphrased Output is:", output_text)
    
    
    elif algorithm == "Model 2 : Pegasus Google":
        model_name = 'tuner007/pegasus_paraphrase'
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

        def get_response(input_text,num_return_sequences):
            batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
            translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
            tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
            return tgt_text
        
        splitter = SentenceSplitter(language='en')

        sentence_list = splitter.split(text)
        # print(sentence_list)
        paraphrase = []
        for i in sentence_list:
            a = get_response(i,1)
            paraphrase.append(a)
        
        paraphrase2 = [' '.join(x) for x in paraphrase]
        
        paraphrase3 = [' '.join(x for x in paraphrase2) ]
        paraphrased_text = str(paraphrase3).strip('[]').strip("'")
        st.text_area("Your Paraphrased Output is:", paraphrased_text)
    
    else:
        parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
        splitter = SentenceSplitter(language='en')
        sentence_list = splitter.split(text)
        paraphrase = []
        for phrase in sentence_list:
            # print("*"*75)
            # print("Input_phrase: ", phrase)
            # print("*"*75)
            para_phrases = parrot.augment(input_phrase=phrase, use_gpu=False)
            # for para_phrase in para_phrases:
            #     print(para_phrase)
            #     print("")
            paraphrase.append(para_phrases[-1][0])   
        paraphrase2 = [' '.join(x) for x in paraphrase]
        
        paraphrase3 = [' '.join(x for x in paraphrase2) ]
        paraphrased_text = str(paraphrase3).strip('[]').strip("'")
        st.text_area("Your Paraphrased Output is:", paraphrased_text) 
