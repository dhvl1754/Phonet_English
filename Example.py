#!/usr/bin/env python
# coding: utf-8

# In[19]:


get_ipython().system(' git clone https://github.com/dhvl1754/Phonet_English')


# In[13]:


from Phonet_English.Phonet_English import Phonet_English


# In[14]:


phon_Eng=Phonet_English(["all"])


# In[18]:


file_audio="sentence.wav"
df = phon_Eng.get_PLLR(file_audio)
df


# In[ ]:




