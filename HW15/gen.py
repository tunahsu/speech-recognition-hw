with open('utf8-ZhuYin.map', 'r', encoding='utf8') as f:
     lines = f.readlines()

with open('ZhuYin-utf8.map', 'w', encoding='utf-8') as f:
    hs = set()
    ws = dict()
    for line in lines:
        s = line.split(' ')
        zhus = s[1].replace('\n', '').split('/')
        ws.update({s[0]: zhus})
        for zhu in zhus:
            hs.add(zhu[0])
    for h in sorted(hs):
        J個格式好奇怪怎麼長成這樣子喵 = []
        f.write('{} '.format(h))
        for key, values in ws.items():
            for value in values:
                if h == value[0]:
                    J個格式好奇怪怎麼長成這樣子喵.append('{} {}\n'.format(key, key))
                    f.write('{} '.format(key))
                    break
        f.write('\n')
        f.writelines(J個格式好奇怪怎麼長成這樣子喵)
