LABEL_CHOICES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
LABEL_CHOICES_LIST = [str(i) for i in LABEL_CHOICES]
encode_maps = {}
decode_maps = {}
for i, char in enumerate(LABEL_CHOICES, 1):
    encode_maps[char] = i
    decode_maps[i] = char

print(LABEL_CHOICES_LIST[40], LABEL_CHOICES_LIST[13])
