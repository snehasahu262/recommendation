
# required libraries - numpy, pandas, pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random

# initilizing device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#loading data 

new_interaction= pd.read_csv('new_interaction.csv',index_col=0)


#displaying all coloumns present in the data
new_interaction.columns

#selecting the coloumns
new_interaction=new_interaction[["_id_x","_id_y","int_score"]]

ratings = new_interaction.rename(columns = {"_id_x": "userId", 
                                  "_id_y":"movieId","int_score":"rating"})
    
# getting the three column names from a pandas dataframe
user_col, item_col, rating_col = ratings.columns

# this function returns a python dictionary
# which maps each id to a corresponding index value
def list_2_dict(id_list:list):
    d={}
    for id, index in zip(id_list, range(len(id_list))):
        d[id] = index
    return d


# splits ratings dataframe to training and validation dataframes
def get_data(ratings, valid_pct:float = 0.2):
    # shuffle the indexes
    ln = random.sample(range(0, len(ratings)), len(ratings))
    
    # split based on the given validation set percentage 
    part = int(len(ln)*valid_pct)
    
    valid_index = ln[0:part]
    train_index = ln[part:]
    valid = ratings.iloc[valid_index]
    train = ratings.iloc[train_index]
    return [train,valid]

# get a batch -> (user, item and rating arrays) from the dataframe
def get_batch(ratings, start:int, end:int):
    return ratings[user_col][start:end].values, ratings[item_col][start:end].values, ratings[rating_col][start:end].values

# get list of unique user ids
users = sorted(list(set(ratings[user_col].values)))

# get list of unique item ids
items = sorted(list(set(ratings[item_col].values)))

# generate dict of correponding indexes for the user ids
user2idx = list_2_dict(users)

# generate dict of correponding indexes for the item ids
item2idx = list_2_dict(items)

# neural net based on Embedding matrices
class EmbeddingModel(nn.Module):
    def __init__(self, n_factors, n_users, n_items, y_range, initialise = 0.01):
        super().__init__()
        self.y_range = y_range
        self.u_weight = nn.Embedding(n_users, n_factors)
        self.i_weight = nn.Embedding(n_items, n_factors)
        self.u_bias = nn.Embedding(n_users, 1)
        self.i_bias = nn.Embedding(n_items, 1)
        
        # initialise the weights of the embeddings
        self.u_weight.weight.data.uniform_(-initialise, initialise)
        self.i_weight.weight.data.uniform_(-initialise, initialise)
        self.u_bias.weight.data.uniform_(-initialise, initialise)
        self.i_bias.weight.data.uniform_(-initialise, initialise)

    def forward(self, users, items):
        # dot multiply the weights for the given user_id and item_id
        dot = self.u_weight(users)* self.i_weight(items)
        
        # sum the result of dot multiplication above and add both the bias terms
        res = dot.sum(1) + self.u_bias(users).squeeze() + self.i_bias(items).squeeze()
        
        # return the output in the given range
        return torch.sigmoid(res) * (self.y_range[1]-self.y_range[0]) + self.y_range[0]
    
# create a model object
# y_range has been extended(0-11) than required(1-10) to make the
# values lie in the linear region of the sigmoid function
model = EmbeddingModel(10, len(users), len(items), [0,11], initialise = 0.01).to(device)

# split the data, returns a list [train, valid]
data = get_data(ratings, 0.3)

# loss = mean((target_rating - predicted_rating)**2)
loss_function = nn.MSELoss()

# optimizer function will update the weights of the Neural Net
optimizer = optim.SGD(model.parameters(), lr=2e-3, momentum=0.9)

# batch size for each input
bs = 128

def train(epochs, bs):
    for epoch in range(epochs):
        
        # training the model
        i=0
        total_loss = 0.0
        ct = 0
        while i < len(data[0]):
            x1,x2,y = get_batch(data[0],i,i+bs)
            i+=bs
            ct+=1
            user_ids = torch.LongTensor([user2idx[u] for u in x1]).to(device)
            item_ids = torch.LongTensor([item2idx[b] for b in x2]).to(device)
            y = torch.Tensor(y).to(device)
            # disregard/zero the gradients from previous computation
            model.zero_grad() 
            preds = model(user_ids,item_ids)
            loss = loss_function(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= ct
        
        # getting the loss on validation set
        i = 0
        total_val_loss = 0.0
        cv=0
        m = model.eval() # setting the model to evaluation mode
        while i < len(data[1]):
            x11,x21,y1 = get_batch(data[1],i,i+bs)
            i+=bs
            cv+=1
            user_ids = torch.LongTensor([user2idx[u] for u in x11]).to(device)
            item_ids = torch.LongTensor([item2idx[b] for b in x21]).to(device)
            y1 = torch.Tensor(y1).to(device)
            preds = m(user_ids,item_ids)
            loss = loss_function(preds, y1)
            total_val_loss += loss.item()
        total_val_loss /= cv
        
        print('epoch', epoch+1, '   train loss', "%.3f" % total_loss, 
              '   valid loss', "%.3f" % total_val_loss)
        
train(20,128)

def recommend_item_for_user(model, user_id):
    m = model.eval().cpu()
    user_ids = torch.LongTensor([user2idx[u] for u in [user_id]*len(items)])
    item_ids = torch.LongTensor([item2idx[b] for b in items])
    remove = set(ratings[ratings[user_col] == user_id][item_col].values)
    preds = m(user_ids,item_ids).detach().numpy()
    pred_item = [(p,b) for p,b in sorted(zip(preds,items), reverse = True) if b not in remove]
    return pred_item

recommend_item_for_user3=recommend_item_for_user(model,"59bd3c43b137a16aac26f79c")
recommend_item_for_user2=recommend_item_for_user3[0:10]
recommend_item_for_user1=pd.DataFrame(recommend_item_for_user2)
    
#ranaming the coloumns   
recommend_item_for_user1 = recommend_item_for_user1.rename(columns = {0: "intrest_score", 
                                  1:"celeb_id"}) 

#loading the celebrity feature data to map Celebrity id with Celebrity name
Celeb_Features_df= pd.read_pickle("Celebrity_Features_df.pkl")
Celeb_Features_df=Celeb_Features_df[["_id","celebrityName"]]

    
out = (recommend_item_for_user1.merge(Celeb_Features_df, left_on='celeb_id', right_on='_id')
          .reindex(columns=['intrest_score','celeb_id', 'celebrityName']))

#printing the result out 
print(out) 
print(len(recommend_item_for_user1))
 

