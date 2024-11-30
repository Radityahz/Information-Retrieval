normal_list = {"gue": "saya", "gua": "saya", "aku": "saya", 
"aq": "saya", "lagi": "sedang"}
text = "aq lagi di jalan nih"
text_normal = []
for t in text.split(" "):
    text_normal.append(normal_list[t] if t in normal_list else t)
print (text_normal)