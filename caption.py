import matplotlib.pyplot as plt
import random
import json
from scipy import ndimage
import numpy as np
from copy import deepcopy
from PIL import Image
import IPython.display
from math import floor
import torch
import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more
import torch.optim as optim  
import torchvision.transforms.functional as TF
import torchvision
from torchvision import datasets, models, transforms
import string


is_cuda = torch.cuda.is_available()
#is_cuda
if(is_cuda):
    USE_GPU = True
else:
    USE_GPU = False
ENDWORD = '<END>'
STARTWORD = '<START>'
PADWORD = '<PAD>'
HEIGHT = 299
WIDTH = 299
INPUT_EMBEDDING = 300
HIDDEN_SIZE = 300
OUTPUT_EMBEDDING = 300

CAPTION_FILE = 'caption_datasets/dataset_flickr8k.json'
IMAGE_DIR = 'Flicker8k_Dataset/'

inception =models.inception_v3(pretrained=True)
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
num_ftrs = inception.fc.in_features
num_ftrs

set_parameter_requires_grad(inception, True)
num_ftrs = inception.fc.in_features
inception.fc = nn.Linear(num_ftrs,INPUT_EMBEDDING)
inception.load_state_dict(torch.load('models/inception_epochs_40.pth'))

class Flickr8KImageCaptionDataset:
    
    def __init__(self):
        
        all_data = json.load(open('caption_datasets/dataset_flickr8k.json', 'r'))
        all_data=all_data['images']
        
        self.training_data = []
        self.test_data = []
        self.w2i = {ENDWORD: 0, STARTWORD: 1}
        self.word_frequency = {ENDWORD: 0, STARTWORD: 0}
        self.i2w = {0: ENDWORD, 1: STARTWORD}
        self.tokens = 2 #END is default
        self.batch_index = 0
        
        for data in all_data:
            if(data['split']=='train'):
                self.training_data.append(data)
            else:
                self.test_data.append(data)
                
            for sentence in data['sentences']:
                for token in sentence['tokens']:
                    if(token not in self.w2i.keys()):
                        self.w2i[token] = self.tokens
                        self.i2w[self.tokens] = token
                        self.tokens +=1
                        self.word_frequency[token] = 1
                    else:
                        self.word_frequency[token] += 1
                        
    def image_to_tensor(self,filename):
        image = Image.open(filename)
        image = TF.resize(img=image, size=(HEIGHT,WIDTH))
       
        image = TF.to_tensor(pic=image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        return torch.unsqueeze(image,0)

    
    def return_train_batch(self): #size of 1 always
        
        #np.random.shuffle(self.training_data)
        for index in range(len(self.training_data)):
        
        #index = np.random.randint(len(self.training_data))
            sentence_index = np.random.randint(len(self.training_data[index]['sentences']))
            output_sentence_tokens = deepcopy(self.training_data[index]['sentences'][sentence_index]['tokens'])
            output_sentence_tokens.append(ENDWORD) #corresponds to end word

            image = self.image_to_tensor('Flicker8k_Dataset/'+self.training_data[index]['filename'])


            yield image, list(map(lambda x: self.w2i[x], output_sentence_tokens)), output_sentence_tokens, index
    
    def convert_tensor_to_word(self, output_tensor):
        output = F.log_softmax(output_tensor.detach().squeeze(), dim=0).numpy()
        return self.i2w[np.argmax(output)]
    
    def convert_sentence_to_tokens(self, sentence):
        
        tokens = sentence.split(" ")
        converted_tokens= list(map(lambda x: self.w2i[x], tokens))
        converted_tokens.append(self.w2i[ENDWORD])
        return converted_tokens
    
    def caption_image_greedy(self, net, image_filename, max_words=15): #non beam search, no temperature implemented
        net.eval()
        inception.eval()
        image_tensor = self.image_to_tensor(image_filename)
        hidden=None
        embedding=None
        words = []
    
        input_token = STARTWORD
        input_tensor = torch.tensor(self.w2i[input_token]).type(torch.LongTensor)
        
        
        for i in range(max_words):
            
            if(i==0):
                out, hidden=net(input_tensor, hidden=image_tensor, process_image=True)
            else:
                out, hidden=net(input_tensor, hidden)
                
            word = self.convert_tensor_to_word(out)
            
            input_token = self.w2i[word]
            input_tensor = torch.tensor(input_token).type(torch.LongTensor)
            
            
            if(word==ENDWORD):
                break
            else:
                words.append(word)

        return ' '.join(words)
    
    
    def forward_beam(self, net, hidden, process_image, partial_sentences, sentences, topn_words=5, max_sentences=10):
        
        max_words = 50
        
        hidden_index = {}
        
        while(sentences<max_sentences):
            
            #print("Sentences: ",sentences)
            
            new_partial_sentences = []
            new_partial_sentences_logp = []
            new_partial_avg_logp= []
            
            if(len(partial_sentences[-1][0])>max_words):
                break
            
            
        
            for partial_sentence in partial_sentences:

                input_token = partial_sentence[0][-1]

                input_tensor = torch.tensor(self.w2i[input_token]).type(torch.FloatTensor)

                if(partial_sentence[0][-1]==STARTWORD):
                    out, hidden=net(input_tensor, hidden, process_image=True)
                else:
                    out, hidden=net(input_tensor, torch.tensor(hidden_index[input_token]))

                #take first topn words and add as children to root

                out = F.log_softmax(out.detach().squeeze(), dim=0).numpy()
                out_indexes = np.argsort(out)[::-1][:topn_words]

        
                for out_index in out_indexes:
                    
                    if(self.i2w[out_index]==ENDWORD):
                        sentences=sentences+1
                        
                    else:
                    
                        total_logp = float(out[out_index]) + partial_sentence[1]
                        new_partial_sentences_logp.append(total_logp)
                        new_partial_sentences.append([np.concatenate((partial_sentence[0], [self.i2w[out_index]])),total_logp])
                        
                        len_words = len(new_partial_sentences[-1][0])
                        
                        
                        
                        new_partial_avg_logp.append(total_logp/len_words)
                        
                        #print(self.i2w[out_index])
                        
                        hidden_index[self.i2w[out_index]] = deepcopy(hidden.detach().numpy())
                
        
            
                
            #select topn partial sentences
        
            top_indexes = np.argsort(new_partial_sentences_logp)[::-1][:topn_words]
            
            
            
            
            
            
        
        
            new_partial_sentences = np.array(new_partial_sentences)[top_indexes]
        
            #print("New partial sentences (topn):", new_partial_sentences)
        
            partial_sentences = new_partial_sentences
        
            
        return partial_sentences
    
    def caption_image_beam_search(self, net, image_filename, topn_words=10, max_sentences=10):
        
        net.eval()
        inception.eval()
        image_tensor = self.image_to_tensor(image_filename)
        hidden=None
        embedding=None
        words = []
    
        
        sentences = 0
        partial_sentences = [[[STARTWORD], 0.0]]
        #root_id = hash(input_token) #for start word
        #nodes = {}
        
        #nodes[root_id] = Node(root_id, [STARTWORD, 0], None)
       
        partial_sentences = self.forward_beam(net, image_tensor, True, partial_sentences, sentences,  topn_words, max_sentences)
    
        logp = []
        
        joined_sentences = []
    
        for partial_sentence in partial_sentences:
            
        
            joined_sentences.append([' '.join(partial_sentence[0][1:]),partial_sentence[1]])
            
            
                
        return joined_sentences

    def print_beam_caption(self, net, train_filename,num_captions=0):
        beam_sentences = f.caption_image_beam_search(net,train_filename)
        if(num_captions==0):
            num_captions=len(beam_sentences)
        for sentence in beam_sentences[:num_captions]:
            print(sentence[0]+" [",sentence[1], "]")
        
class IC_V6(nn.Module):
    
    #V2: Fed image vector directly as hidden and fed words generated as iputs back to LSTM
    #V3: Added an embedding layer between words input and GRU/LSTM
    
    def __init__(self, token_dict_size):
        super(IC_V6, self).__init__()
        
        #Input is an image of height 500, and width 500
        
        
        self.embedding_size = INPUT_EMBEDDING
        self.hidden_state_size = HIDDEN_SIZE
        self.token_dict_size = token_dict_size
        self.output_size = OUTPUT_EMBEDDING
        
        self.batchnorm = nn.BatchNorm1d(self.embedding_size)
        
        self.input_embedding = nn.Embedding(self.token_dict_size, self.embedding_size)
        
        self.embedding_dropout = nn.Dropout(p=0.22)
        self.gru_layers = 3
        
        self.gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_state_size, num_layers=self.gru_layers, dropout=0.22)
        self.linear = nn.Linear(self.hidden_state_size, self.output_size)
        self.out = nn.Linear(self.output_size, token_dict_size)
        
    def forward(self, input_tokens, hidden, process_image=False, use_inception=True):
        
        
        
        if(USE_GPU):
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        if(process_image):
            if(use_inception):
                inp=self.embedding_dropout(inception(hidden))
            else:
                inp=hidden
            #inp=self.batchnorm(inp)
            hidden=torch.zeros((self.gru_layers,1, self.hidden_state_size))
        else:
            inp=self.embedding_dropout(self.input_embedding(input_tokens.view(1).type(torch.LongTensor).to(device)))
            #inp=self.batchnorm(inp)
            
        
        hidden = hidden.view(self.gru_layers,1,-1)
        inp = inp.view(1,1,-1)
        
        out, hidden = self.gru(inp, hidden)
        out = self.out(self.linear(out))
        
        
        
        return out, hidden
    
f = Flickr8KImageCaptionDataset()   
net = IC_V6(f.tokens)
net.load_state_dict(torch.load('models/epochs_40_loss_2_841_v6.pth'))
#net.eval()
train_filename = '/Users/prasadgujar16/Desktop/searching/my-images/a.jpg'
print("Greedy caption:", f.caption_image_greedy(net,train_filename))
ok = f.caption_image_greedy(net,train_filename)
file = open('caption.txt', 'w')
file.write(ok)
file.close()
#print("")
#print("Beam caption:")
#f.print_beam_caption(net, train_filename, 3)
#pil_im = Image.open(train_filename, 'r')
#plt.figure()
#plt.imshow(np.asarray(pil_im))
#plt.show()