
# coding: utf-8

# In[1]:

import pandas as pd
from scipy.spatial.distance import cosine


# In[2]:

rating = pd.read_csv("lastfm-matrix-germany.csv")


# In[3]:

rating


# ## Item Based Collaborative Filtering Approach

# In[4]:

item_data = rating.drop('user',1)


# ## Creating a Similarity Matrix using Cosine Similarity

# In[5]:

sim_mat = pd.DataFrame(index=item_data.columns,columns=item_data.columns)


# In[6]:

sim_mat


# In[7]:

for i in range(0,len(sim_mat.columns)) :
    # Looping through the columns for each column
    for j in range(0,len(sim_mat.columns)) :
      # Filling in placeholder with cosine similarities
      sim_mat.ix[i,j] = 1-cosine(item_data.ix[:,i],item_data.ix[:,j])


# In[8]:

sim_mat


# In[9]:

data_neighbours = pd.DataFrame(index=sim_mat.columns,columns=range(1,11))


# In[10]:

for i in range(0,len(sim_mat.columns)):
    data_neighbours.ix[i,:10] = sim_mat.ix[0:,i].order(ascending=False)[:10].index


# In[11]:

data_neighbours.ix[:,2:]


# ## User Based Collaborative Filtering Approach

# #### The entire process for implementing a User Baased Collaborating Recommendation system is as follows:
# #### --> Constructing an Item Based similarity matrix
# #### --> Checking the items associated with the user
# #### --> For each item the user is associated with, get the top X neighbours
# #### --> Get the associations of the user for each neighbour
# #### --> Calculate a similarity score using formula
# #### --> Recommend the items with the highest score

# In[13]:

user_sim_data = pd.DataFrame(index=rating.index,columns=rating.columns)


# In[14]:

user_sim_data.ix[:,:1] = rating.ix[:,:1]


# In[15]:

user_sim_data


# In[12]:

def getScore(history, similarities):
   return sum(history*similarities)/sum(similarities)


# In[16]:

for i in range(0,len(user_sim_data.index)):
    for j in range(1,len(user_sim_data.columns)):
        user = user_sim_data.index[i]
        song = user_sim_data.columns[j]
 
        if rating.ix[i][j] == 1:
            user_sim_data.ix[i][j] = 0
        else:
            song_top_names = data_neighbours.ix[song][1:10]
            song_top_sims = sim_mat.ix[song].order(ascending=False)[1:10]
            user_assoc = item_data.ix[user,song_top_names]
 
            user_sim_data.ix[i][j] = getScore(user_assoc,song_top_sims)


# In[17]:

user_sim_data


# In[18]:

data_recommend = pd.DataFrame(index=user_sim_data.index, columns=['user','1','2','3','4','5','6'])
data_recommend.ix[0:,0] = user_sim_data.ix[:,0]


# In[20]:

for i in range(0,len(user_sim_data.index)):
    data_recommend.ix[i,1:] = user_sim_data.ix[i,:].order(ascending=False).ix[1:7,].index.transpose()


# In[21]:

data_recommend


# In[ ]:



