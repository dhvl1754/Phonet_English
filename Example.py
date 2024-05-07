#Installation
git clone https://github.com/dhvl1754/Phonet_English'

#Importing library
from Phonet_English.Phonet_English import Phonet_English

phon_Eng=Phonet_English(["all"])
file_audio="sentence.wav"
df = phon_Eng.get_PLLR(file_audio)
print(df)





